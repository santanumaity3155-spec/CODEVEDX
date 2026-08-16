"""
Module 3: Reusable Intent Prediction / Inference Module
=======================================================
Internal Helpdesk Chatbot - Intent Classification

This module provides the *production* interface for predicting helpdesk
intents from natural-language questions.  It consumes the artifacts produced
by ``src/train_intent_model.py``:

    models/intent_classifier_pipeline.pkl   <-- PRODUCTION artifact (single sklearn Pipeline)
    models/intent_classifier.pkl            <-- classifier component copy
    models/tfidf_vectorizer.pkl             <-- TF-IDF vectorizer component copy

The PRODUCTION model is the complete sklearn ``Pipeline``::

    Pipeline([
        ("tfidf", TfidfVectorizer(...)),        # fitted ONLY on training data
        ("classifier", <best classifier>)      # selected via macro-F1
    ])

IMPORTANT - preprocessing consistency
-------------------------------------
Prediction input text passes through the SAME text cleaning used while
building the training features (``NLPPreprocessor.clean_text`` -> the
``clean_question`` column).  The TF-IDF vectorizer was fitted on
``clean_question`` strings, so every prediction is cleaned exactly the same
way before it is transformed.  No different/custom preprocessing is
reproduced here.

Confidence handling
-------------------
* If the estimator exposes ``predict_proba`` the returned ``confidence`` is a
  genuine probability and ``probabilities`` (a class->probability mapping) is
  included.
* If the estimator does NOT expose ``predict_proba`` (e.g. a raw LinearSVM)
  the confidence is derived from the ``decision_function`` margin as a
  bounded, monotonic *confidence score* in (0,1] and is explicitly flagged
  with ``confidence_source == "decision_function_margin"`` so it is never
  confused with a calibrated probability.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:  # pragma: no cover - import safety
    sys.path.insert(0, str(SRC_DIR))

from nlp_preprocessor import NLPPreprocessor  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"

DEFAULT_PIPELINE_PATH = MODEL_DIR / "intent_classifier_pipeline.pkl"
DEFAULT_CLASSIFIER_PATH = MODEL_DIR / "intent_classifier.pkl"
DEFAULT_VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"

# Lightweight in-process cache so repeated predictions do not re-read the
# serialized model from disk on every call (off by default in tests).
_model_cache: Dict[str, "IntentClassifier"] = {}


class IntentClassifier:
    """Wraps a fitted prediction pipeline with the shared text preprocessing."""

    def __init__(
        self,
        pipeline: Any,
        classes: List[str],
        preprocessor: Optional[NLPPreprocessor] = None,
    ) -> None:
        self.pipeline = pipeline
        self.classes = list(classes)
        self.preprocessor = preprocessor or NLPPreprocessor(PROJECT_ROOT)

    # ------------------------------------------------------------------ load
    @classmethod
    def load(
        cls,
        pipeline_path: Optional[Path] = None,
        classifier_path: Optional[Path] = None,
        vectorizer_path: Optional[Path] = None,
    ) -> "IntentClassifier":
        """
        Load the production model from disk.

        Priority:
          1. ``models/intent_classifier_pipeline.pkl`` (single Pipeline).
          2. ``intent_classifier.pkl`` + ``tfidf_vectorizer.pkl`` (components)
             combined into a Pipeline at load time.

        Raises FileNotFoundError when no serialized model exists.
        """
        import joblib

        pipeline_path = Path(pipeline_path or DEFAULT_PIPELINE_PATH)
        classifier_path = Path(classifier_path or DEFAULT_CLASSIFIER_PATH)
        vectorizer_path = Path(vectorizer_path or DEFAULT_VECTORIZER_PATH)

        if pipeline_path.exists():
            pipeline = joblib.load(pipeline_path)
        elif classifier_path.exists() and vectorizer_path.exists():
            from sklearn.pipeline import Pipeline

            classifier = joblib.load(classifier_path)
            vectorizer = joblib.load(vectorizer_path)
            pipeline = Pipeline([("tfidf", vectorizer), ("classifier", classifier)])
        else:
            raise FileNotFoundError(
                "No serialized intent-classification model found. Expected one of:\n"
                f"  {pipeline_path}\n"
                f"  ({classifier_path} + {vectorizer_path})\n"
                "Run `python src/train_intent_model.py` first."
            )

        classes = getattr(pipeline, "classes_", None)
        if classes is None:
            classifier = pipeline.named_steps.get("classifier")
            classes = getattr(classifier, "classes_", None)
        if classes is None:
            raise ValueError(
                "Loaded model does not expose `classes_`; cannot predict intents."
            )
        return cls(pipeline=pipeline, classes=list(classes))

    # ------------------------------------------------------------- preprocessing
    def preprocess_text(self, text: Any) -> str:
        """
        Clean raw user text using the SAME pipeline used at training time.

        Training features were built from the ``clean_question`` column of
        ``data/processed/faq_nlp_ready.csv``, i.e. the output of
        ``NLPPreprocessor.clean_text``.  Every prediction input is therefore
        run through the same ``clean_text`` step before the TF-IDF transform.
        """
        return self.preprocessor.clean_text(text)

    # ----------------------------------------------------------------- predict
    def predict_intent(self, text: Any) -> str:
        """
        Predict a single intent label.

        Args:
            text: Raw natural-language question.

        Returns:
            Predicted intent label (snake_case string).
        """
        cleaned = self.preprocess_text(text)
        return str(self.pipeline.predict([cleaned])[0])

    def predict_proba(self, text: Any) -> Optional[np.ndarray]:
        """
        Return the class probability vector when the estimator supports it.

        Args:
            text: Raw natural-language question.

        Returns:
            Numpy array of probabilities aligned with ``self.classes``, or
            ``None`` when the estimator cannot produce probabilities.
        """
        if not hasattr(self.pipeline, "predict_proba"):
            return None
        cleaned = self.preprocess_text(text)
        return np.asarray(self.pipeline.predict_proba([cleaned])[0])

    def predict_with_confidence(self, text: Any) -> Dict[str, Any]:
        """
        Predict an intent together with a confidence estimate.

        Returns a dictionary with at least::

            {
                "text": <original input>,
                "intent": <predicted intent>,
                "confidence": <float in [0,1]>
            }

        When the estimator supports ``predict_proba``, ``probabilities``
        (class -> probability mapping) and ``confidence_source ==
        "predict_proba"`` are included.  Otherwise the confidence is a bounded
        decision-margin score and ``confidence_source ==
        "decision_function_margin"`` (explicitly NOT a probability).
        """
        cleaned = self.preprocess_text(text)

        if hasattr(self.pipeline, "predict_proba"):
            proba = np.asarray(self.pipeline.predict_proba([cleaned])[0])
            best_idx = int(np.argmax(proba))
            return {
                "text": text,
                "intent": str(self.classes[best_idx]),
                "confidence": round(float(proba[best_idx]), 6),
                "probabilities": {
                    str(cls_name): round(float(p), 6)
                    for cls_name, p in zip(self.classes, proba)
                },
                "confidence_source": "predict_proba",
            }

        # No native probabilities: use the decision-function margin to build a
        # bounded, monotonic *confidence score* (not a calibrated probability).
        decision = np.asarray(self.pipeline.decision_function([cleaned])[0])
        best_idx = int(np.argmax(decision))
        ordered = np.sort(decision)[::-1]
        margin = float(ordered[0] - ordered[1]) if ordered.size > 1 else float(ordered[0])
        confidence = float(1.0 / (1.0 + np.exp(-margin)))  # sigmoid(margin) in (0,1)
        return {
            "text": text,
            "intent": str(self.classes[best_idx]),
            "confidence": round(confidence, 6),
            "confidence_source": "decision_function_margin",
            "note": "confidence is a bounded decision-score, not a calibrated probability",
        }

    def predict_batch(self, texts: List[Any]) -> List[Dict[str, Any]]:
        """
        Predict intents for a list of questions.

        Args:
            texts: List of raw questions.

        Returns:
            List of dictionaries with keys ``text``, ``intent``, ``confidence``
            (see ``predict_with_confidence``).
        """
        return [self.predict_with_confidence(t) for t in texts]


# --------------------------------------------------------------- module-level API
def load_model(use_cache: bool = True) -> IntentClassifier:
    """
    Load the production intent classifier.

    Args:
        use_cache: When True (default) a light in-process cache avoids
            repeated disk reads.  Pass False to force a fresh load from disk.

    Returns:
        An ``IntentClassifier`` ready for predictions.
    """
    cache_key = str(DEFAULT_PIPELINE_PATH)
    if use_cache and cache_key in _model_cache:
        return _model_cache[cache_key]
    model = IntentClassifier.load()
    if use_cache:
        _model_cache[cache_key] = model
    return model


def predict_intent(text: Any, model: Optional[IntentClassifier] = None) -> str:
    """Module-level single-intent prediction convenience wrapper."""
    predictor = model or load_model()
    return predictor.predict_intent(text)


def predict_with_confidence(
    text: Any, model: Optional[IntentClassifier] = None
) -> Dict[str, Any]:
    """Module-level single-prediction (with confidence) convenience wrapper."""
    predictor = model or load_model()
    return predictor.predict_with_confidence(text)


def predict_batch(
    texts: List[Any], model: Optional[IntentClassifier] = None
) -> List[Dict[str, Any]]:
    """Module-level batch prediction convenience wrapper."""
    predictor = model or load_model()
    return predictor.predict_batch(texts)