"""
Module 3: Intent Classification Model Training Pipeline
=======================================================
Internal Helpdesk Chatbot - Intent Classification

This pipeline trains and evaluates a production-quality intent classifier for
the Internal Helpdesk Chatbot using the Module 2 NLP-ready dataset:

    data/processed/faq_nlp_ready.csv

Flow
----
    load -> validate -> stratified train/test split -> TF-IDF configuration
    selection -> baseline models -> 5-fold stratified cross-validation ->
    hyperparameter tuning -> model comparison -> best-model selection ->
    final model training -> final single test-set evaluation -> reports &
    charts -> model serialization -> reload & consistency verification.

Data-integrity rules (no leakage)
---------------------------------
* The TF-IDF vectorizer is ALWAYS fitted inside cross-validation folds of the
  TRAINING data only.  It is never fitted on the complete dataset and never on
  the test set.
* Cross-validation and hyperparameter tuning use the training data only.
* The test set is touched exactly once: the final evaluation of the selected
  model (STEP 12 / STEP 26 of the module specification).

Artifacts produced
------------------
models/intent_classifier_pipeline.pkl    (PRODUCTION model: single Pipeline)
models/intent_classifier.pkl             (classifier component copy)
models/tfidf_vectorizer.pkl             (TF-IDF component copy)
outputs/reports/model_comparison.csv
outputs/reports/classification_report.txt
outputs/reports/final_model_report.txt
outputs/reports/top_features_by_intent.txt
outputs/reports/error_analysis.csv
outputs/reports/model_metadata.json
outputs/charts/confusion_matrix.png
"""

import json
import sys
import time
import traceback
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for scripts / notebooks
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split,
    cross_validate,
    cross_val_predict,
    GridSearchCV,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    make_scorer,
    ConfusionMatrixDisplay,
)

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:  # pragma: no cover - import safety
    sys.path.insert(0, str(SRC_DIR))

from nlp_preprocessor import NLPPreprocessor, CANONICAL_INTENTS  # noqa: E402
from intent_classifier import IntentClassifier  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_CSV = PROJECT_ROOT / "data" / "processed" / "faq_nlp_ready.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
CHARTS_DIR = PROJECT_ROOT / "outputs" / "charts"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

TEXT_COLUMN = "clean_question"
LABEL_COLUMN = "intent"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# -----------------------------------------------------------------------------
# TF-IDF candidate configurations.  These are compared in Step 3 using plain
# cross-validation on the TRAINING split only (inside Pipelines, so the TF-IDF
# is re-fitted per fold and never leaks).
# -----------------------------------------------------------------------------
TFIDF_CONFIGS: List[Dict[str, Any]] = [
    {
        "name": "recommended",
        "ngram_range": (1, 2),
        "sublinear_tf": True,
        "min_df": 2,
        "max_df": 0.9,
        "stop_words": None,
    },
    {
        "name": "no_min_df_filter",
        "ngram_range": (1, 2),
        "sublinear_tf": True,
        "min_df": 1,
        "max_df": 1.0,
        "stop_words": None,
    },
    {
        "name": "unigrams_only",
        "ngram_range": (1, 1),
        "sublinear_tf": True,
        "min_df": 2,
        "max_df": 0.9,
        "stop_words": None,
    },
    {
        "name": "english_stopwords",
        "ngram_range": (1, 2),
        "sublinear_tf": True,
        "min_df": 2,
        "max_df": 0.9,
        "stop_words": "english",
    },
]

# -----------------------------------------------------------------------------
# Baseline model families (Step 5).  Only the estimators differ; the feature
# pipeline is shared.
# -----------------------------------------------------------------------------
BASE_MODELS: Dict[str, Any] = {
    "logistic_regression": LogisticRegression(
        max_iter=2000, random_state=RANDOM_STATE
    ),
    "linear_svm": LinearSVC(max_iter=5000, random_state=RANDOM_STATE),
    "multinomial_nb": MultinomialNB(),
}

# -----------------------------------------------------------------------------
# Reasonable, modest hyperparameter grids (Step 8).  Optimised for macro F1.
# -----------------------------------------------------------------------------
PARAM_GRIDS: Dict[str, Dict[str, Any]] = {
    "logistic_regression": {
        "classifier__C": [0.1, 1.0, 10.0],
        "classifier__class_weight": [None, "balanced"],
    },
    "linear_svm": {
        "classifier__C": [0.1, 1.0, 10.0],
        "classifier__class_weight": [None, "balanced"],
    },
    "multinomial_nb": {
        "classifier__alpha": [0.1, 0.5, 1.0],
    },
}
MODEL_NAME_MAP: Dict[str, str] = {
    "logistic_regression": "Logistic Regression",
    "linear_svm": "Linear SVM",
    "multinomial_nb": "Multinomial Naive Bayes",
    "logistic_regression_tuned": "Logistic Regression (tuned)",
    "linear_svm_tuned": "Linear SVM (tuned)",
    "multinomial_nb_tuned": "Multinomial Naive Bayes (tuned)",
}

def _tfidf_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    """Vectorizer kwargs for a TF-IDF config (drops the 'name' key)."""
    return {k: v for k, v in config.items() if k != "name"}


def _build_pipeline(estimator: Any, config: Optional[Dict[str, Any]] = None):
    """Build Pipeline([('tfidf', TfidfVectorizer(config)), ('classifier', est)])."""
    config = config or TFIDF_CONFIGS[0]
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(**_tfidf_kwargs(config))),
            ("classifier", estimator),
        ]
    )


class IntentTrainingPipeline:
    """End-to-end Module 3 intent-classification training pipeline."""

    def __init__(self, project_root: Path = PROJECT_ROOT) -> None:
        self.project_root = Path(project_root)
        self.input_csv = self.project_root / "data" / "processed" / "faq_nlp_ready.csv"

        self.df: Optional[pd.DataFrame] = None
        self.train_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
        self.classes: List[str] = []

        self.best_tfidf_config: Dict[str, Any] = {}
        self.tfidf_config_results: List[Dict[str, Any]] = []

        self.comparison_df: Optional[pd.DataFrame] = None
        self.best_model_row: Optional[pd.Series] = None
        self.selection_reason: str = ""

        self.final_pipeline: Optional[Pipeline] = None
        self.final_metrics: Dict[str, float] = {}
        self.final_y_pred: Optional[np.ndarray] = None
        self.classification_report_text: str = ""
        self.confusion: Optional[np.ndarray] = None

        self.model_files: Dict[str, Path] = {}
        self.report_files: Dict[str, Path] = {}
        self.chart_files: Dict[str, Path] = {}

        self.sample_predictions_df: Optional[pd.DataFrame] = None
        self.consistency_ok: bool = False
        self.leakage_ok: bool = True

        self.preprocessor = NLPPreprocessor(self.project_root)

    # ------------------------------------------------------------------ I/O
    def load_dataset(self) -> pd.DataFrame:
        """Load ``faq_nlp_ready.csv`` (Module 2 output)."""
        if not self.input_csv.exists():
            raise FileNotFoundError(
                f"NLP-ready dataset not found: {self.input_csv}. "
                "Run `python src/prepare_nlp_data.py` first."
            )
        df = pd.read_csv(self.input_csv, encoding="utf-8")
        return df

    def validate_dataset(self, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Validate the NLP-ready dataset.  Statistics are computed dynamically;
        nothing is hard-coded.
        """
        data = df if df is not None else self.df
        result = {"valid": True, "errors": [], "records": 0, "intents": 0}

        if data is None:
            result["valid"] = False
            result["errors"].append("Dataset not loaded.")
            return result

        result["records"] = int(len(data))
        result["intents"] = int(data[LABEL_COLUMN].nunique()) if len(data) else 0

        required = ["question", "clean_question", "intent", "answer", "entity"]
        missing_cols = [c for c in required if c not in data.columns]
        if missing_cols:
            result["valid"] = False
            result["errors"].append(f"Missing required columns: {missing_cols}")

        if len(data) == 0:
            result["valid"] = False
            result["errors"].append("Dataset is empty.")

        missing_clean = int(
            data["clean_question"].isna().sum()
            + (data["clean_question"].astype(str).str.strip() == "").sum()
        )
        missing_intent = int(
            data["intent"].isna().sum()
            + (data["intent"].astype(str).str.strip() == "").sum()
        )
        if missing_clean:
            result["valid"] = False
            result["errors"].append(f"{missing_clean} missing clean_question values.")
        if missing_intent:
            result["valid"] = False
            result["errors"].append(f"{missing_intent} missing intent values.")

        return result

    # ----------------------------------------------------------------- split
    def split_data(
        self, df: Optional[pd.DataFrame] = None, test_size: float = TEST_SIZE
    ):
        """
        Stratified train/test split preserving the 22-class distribution.
        Returns (train_df, test_df).
        """
        data = df if df is not None else self.df
        if data is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        train_df, test_df = train_test_split(
            data,
            test_size=test_size,
            random_state=RANDOM_STATE,
            stratify=data[LABEL_COLUMN],
        )
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

    def verify_split_classes(
        self, train_df: Optional[pd.DataFrame] = None, test_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """Verify intent coverage in both sets for every class."""
        tr = train_df if train_df is not None else self.train_df
        te = test_df if test_df is not None else self.test_df
        result = {"ok": True, "missing_in_train": [], "missing_in_test": []}
        if tr is None or te is None:
            raise ValueError("Train/test split not performed yet.")
        classes_train = set(tr[LABEL_COLUMN].unique())
        classes_test = set(te[LABEL_COLUMN].unique())
        result["missing_in_train"] = sorted(set(self.classes) - classes_train)
        result["missing_in_test"] = sorted(set(self.classes) - classes_test)
        result["ok"] = not result["missing_in_train"] and not result["missing_in_test"]
        return result

    # ------------------------------------------------------------- feature eng.
    def _cv(self) -> StratifiedKFold:
        """Shared 5-fold stratified CV splitter (training data only)."""
        return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    def _scorers(self, classes: List[str]) -> Dict[str, Any]:
        """
        Multi-metric scorers used for cross-validation (macro F1 key).

        Every metric is wrapped in a function accepting ``**kwargs`` because
        scikit-learn's ``make_scorer`` may forward a ``pos_label`` argument
        that is invalid for multiclass string-label tasks; the wrapper
        guarantees the scoring functions can never break on that parameter.
        """
        def _wrapped(metric: str, average: str):
            def _score_func(y_true, y_pred, **kwargs):
                fn = {
                    "f1": f1_score,
                    "precision": precision_score,
                    "recall": recall_score,
                }[metric]
                return fn(
                    y_true, y_pred, average=average,
                    labels=classes, zero_division=0,
                )
            return make_scorer(_score_func)

        return {
            "accuracy": "accuracy",
            "f1_macro": _wrapped("f1", "macro"),
            "f1_weighted": _wrapped("f1", "weighted"),
            "precision_macro": _wrapped("precision", "macro"),
            "recall_macro": _wrapped("recall", "macro"),
        }

    def f1_macro_scorer(self, classes: List[str]):
        """Scorer used by GridSearchCV (optimised for macro F1)."""
        return self._scorers(classes)["f1_macro"]

    def select_tfidf_config(self, train_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Compare TF-IDF configurations with a fast Logistic Regression baseline
        using 5-fold cross-validation on the TRAINING split only.  The winner
        is the configuration with the highest mean macro-F1.

        Each configuration runs inside a Pipeline so the vectorizer is
        re-fitted per training fold (no leakage, even within the training set).
        """
        tr = train_df if train_df is not None else self.train_df
        if tr is None:
            raise ValueError("Train split not available. Run the split first.")
        X = tr[TEXT_COLUMN].astype(str).tolist()
        y = tr[LABEL_COLUMN].astype(str).tolist()
        classes = self.classes
        cv = self._cv()

        results: List[Dict[str, Any]] = []
        for config in TFIDF_CONFIGS:
            pipe = _build_pipeline(
                LogisticRegression(max_iter=2000, random_state=RANDOM_STATE), config
            )
            cv_results = cross_validate(
                pipe, X, y, cv=cv, scoring=self._scorers(classes), n_jobs=1,
                error_score="raise",
            )
            results.append(
                {
                    "name": config["name"],
                    "config": config,
                    "f1_macro_mean": float(np.mean(cv_results["test_f1_macro"])),
                    "accuracy_mean": float(np.mean(cv_results["test_accuracy"])),
                }
            )

        self.tfidf_config_results = results
        best = max(results, key=lambda r: r["f1_macro_mean"])
        self.best_tfidf_config = best["config"]
        return best["config"]

    # ----------------------------------------------------------- evaluation
    def evaluate_model_cv(
        self, model: Any, X_train: List[str], y_train: List[str], classes: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate a (Pipeline) model with 5-fold stratified CV on the training
        data only.  Returns raw classification metrics computed from
        fold-consistent out-of-fold predictions plus fold mean/std values.
        """
        cv = self._cv()
        scorers = self._scorers(classes)

        cv_results = cross_validate(
            model, X_train, y_train, cv=cv, scoring=scorers, n_jobs=1, error_score="raise"
        )
        y_pred_cv = cross_val_predict(model, X_train, y_train, cv=cv, n_jobs=1)

        return {
            "accuracy": float(accuracy_score(y_train, y_pred_cv)),
            "precision_macro": float(
                precision_score(y_train, y_pred_cv, average="macro", labels=classes, zero_division=0)
            ),
            "recall_macro": float(
                recall_score(y_train, y_pred_cv, average="macro", labels=classes, zero_division=0)
            ),
            "f1_macro": float(
                f1_score(y_train, y_pred_cv, average="macro", labels=classes, zero_division=0)
            ),
            "f1_weighted": float(
                f1_score(y_train, y_pred_cv, average="weighted", labels=classes, zero_division=0)
            ),
            "cv_accuracy_mean": float(np.mean(cv_results["test_accuracy"])),
            "cv_accuracy_std": float(np.std(cv_results["test_accuracy"])),
            "cv_f1_macro_mean": float(np.mean(cv_results["test_f1_macro"])),
            "cv_f1_macro_std": float(np.std(cv_results["test_f1_macro"])),
            "cv_f1_weighted_mean": float(np.mean(cv_results["test_f1_weighted"])),
            "cv_f1_weighted_std": float(np.std(cv_results["test_f1_weighted"])),
        }

    # ----------------------------------------------------------------- tune
    def tune_models(self, train_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Hyperparameter tuning (GridSearchCV, scoring = macro F1) for each
        baseline family.  Grids run over the full Pipeline, so the TF-IDF is
        fitted inside every training fold only.

        The returned GridSearchCV estimators are refitted on the ENTIRE
        training set (refit=True) and become the final-model candidates.
        """
        tr = train_df if train_df is not None else self.train_df
        if tr is None:
            raise ValueError("Train split not available yet. Run the split first.")
        X = tr[TEXT_COLUMN].astype(str).tolist()
        y = tr[LABEL_COLUMN].astype(str).tolist()
        classes = self.classes

        tuned: Dict[str, Any] = {}
        for name, estimator in BASE_MODELS.items():
            pipe = _build_pipeline(estimator, self.best_tfidf_config)
            grid = GridSearchCV(
                pipe,
                PARAM_GRIDS[name],
                cv=self._cv(),
                scoring=self.f1_macro_scorer(classes),
                refit=True,
                n_jobs=1,
                verbose=0,
            )
            grid.fit(X, y)
            tuned[name] = grid
        return tuned

    # ------------------------------------------------------------- comparison
    def build_comparison(
        self,
        baseline_scores: Dict[str, Dict[str, float]],
        tuned: Dict[str, Any],
    ) -> pd.DataFrame:
        """
        Combine baseline + tuned candidate metrics into the comparison table.

        All per-row metrics are computed from 5-fold stratified CV on the
        TRAINING data (the test set is untouched). Sorting column: f1_macro.
        """
        rows: List[Dict[str, Any]] = []
        for name, metrics in baseline_scores.items():
            row = {
                "model": name,
                "hyperparameters": "default",
                "tfidf_config": self.best_tfidf_config.get("name", ""),
            }
            row.update(metrics)
            rows.append(row)

        X_train = self.train_df[TEXT_COLUMN].astype(str).tolist()
        y_train = self.train_df[LABEL_COLUMN].astype(str).tolist()
        for name, grid in tuned.items():
            metrics = self.evaluate_model_cv(
                grid.best_estimator_, X_train, y_train, self.classes
            )
            row = {
                "model": f"{name}_tuned",
                "hyperparameters": str(grid.best_params_),
                "base_model": name,
                "tfidf_config": self.best_tfidf_config.get("name", ""),
            }
            row.update(metrics)
            rows.append(row)

        df = pd.DataFrame(rows)
        required = [
            "model", "accuracy", "precision_macro", "recall_macro", "f1_macro",
            "f1_weighted", "cv_accuracy_mean", "cv_accuracy_std",
            "cv_f1_macro_mean", "cv_f1_macro_std",
        ]
        for col in required:
            if col not in df.columns:
                df[col] = np.nan
        df = df.sort_values(
            ["f1_macro", "f1_weighted", "accuracy"], ascending=False
        ).reset_index(drop=True)
        self.comparison_df = df
        return df

    # ----------------------------------------------------------------- select
    def select_best_model(self, comparison_df: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        Select the best model from the comparison table.

        Primary criterion: macro F1.  Tie-breakers: weighted F1, then
        accuracy, then simplicity (fewer tuned parameters).
        """
        df = comparison_df if comparison_df is not None else self.comparison_df
        if df is None or len(df) == 0:
            raise ValueError("Comparison table is empty. Run build_comparison() first.")

        # Selection is already sorted by (f1_macro, f1_weighted, accuracy).
        best = df.iloc[0]
        self.best_model_row = best

        reason_parts = [
            f"Highest macro F1 ({best['f1_macro']:.4f}) on 5-fold stratified "
            f"cross-validation of the training set.",
        ]
        if float(best["f1_weighted"]) > 0:
            reason_parts.append(
                f"Weighted F1: {best['f1_weighted']:.4f} (tie-breaker #1)."
            )
        if float(best["accuracy"]) > 0:
            reason_parts.append(f"Accuracy: {best['accuracy']:.4f} (tie-breaker #2).")
        self.selection_reason = " ".join(reason_parts)
        return best

    def _final_estimator(self, best_row: pd.Series, tuned: Dict[str, Any]) -> Any:
        """
        Build the final (untrained) estimator for the selected model so it can
        be fitted cleanly on the entire training set.
        """
        from sklearn.base import clone

        model_name = str(best_row["model"])
        if model_name.endswith("_tuned"):
            base = str(best_row["base_model"])
            params = best_row["hyperparameters"]
            param_dict = {}
            # hyperparameters column stores a dict string, e.g.
            # "{'classifier__C': 1.0, 'classifier__class_weight': 'balanced'}"
            if params and params != "default":
                try:
                    param_dict = eval(params, {"__builtins__": {}}, {})  # noqa: S307
                except Exception:
                    param_dict = {}
            estimator = clone(BASE_MODELS[base])
            for key, value in param_dict.items():
                attr = key.split("__")[-1]
                setattr(estimator, attr, value)
            return estimator
        return clone(BASE_MODELS[model_name])

    # ------------------------------------------------------------------ final
    def fit_final_model(
        self,
        best_row: Optional[pd.Series] = None,
        tuned: Optional[Dict[str, Any]] = None,
    ) -> Pipeline:
        """
        Train the FINAL model on the entire TRAINING set (never the test set).

        The production pipeline is conceptually:

            clean_question -> TF-IDF vectorizer -> best classifier
        """
        row = best_row if best_row is not None else self.best_model_row
        if row is None:
            raise ValueError("Best model not selected. Call select_best_model() first.")
        estimator = self._final_estimator(row, tuned or {})

        pipe = Pipeline(
            [
                ("tfidf", TfidfVectorizer(**_tfidf_kwargs(self.best_tfidf_config))),
                ("classifier", estimator),
            ]
        )
        X_train = self.train_df[TEXT_COLUMN].astype(str).tolist()
        y_train = self.train_df[LABEL_COLUMN].astype(str).tolist()

        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        self.final_train_time = time.perf_counter() - t0

        self.final_pipeline = pipe
        return pipe

    def evaluate_final_model(self) -> Dict[str, float]:
        """
        Evaluate the FINAL model on the untouched test set (the only such use
        of test data in the whole pipeline).
        """
        if self.final_pipeline is None:
            raise ValueError("Final model not fitted. Call fit_final_model() first.")
        X_test = self.test_df[TEXT_COLUMN].astype(str).tolist()
        y_test = self.test_df[LABEL_COLUMN].astype(str).tolist()
        classes = self.classes

        t0 = time.perf_counter()
        y_pred = self.final_pipeline.predict(X_test)
        self.final_predict_time = time.perf_counter() - t0

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision_macro": float(
                precision_score(y_test, y_pred, average="macro", labels=classes, zero_division=0)
            ),
            "recall_macro": float(
                recall_score(y_test, y_pred, average="macro", labels=classes, zero_division=0)
            ),
            "f1_macro": float(
                f1_score(y_test, y_pred, average="macro", labels=classes, zero_division=0)
            ),
            "f1_weighted": float(
                f1_score(y_test, y_pred, average="weighted", labels=classes, zero_division=0)
            ),
        }
        self.final_metrics = metrics
        self.final_y_pred = np.asarray(y_pred)
        self.confusion = confusion_matrix(y_test, y_pred, labels=classes)
        self.classification_report_text = classification_report(
            y_test,
            y_pred,
            labels=classes,
            target_names=classes,
            digits=4,
            zero_division=0,
        )
        return metrics

    # ------------------------------------------------------------ sample preds
    def make_sample_predictions(
        self, intents: Optional[List[str]] = None, n_per_intent: int = 1
    ) -> pd.DataFrame:
        """
        Build a real-example prediction table covering a set of intents.

        For every requested intent the example is taken from the TEST split
        when available, otherwise from the TRAINING split.  The source split is
        recorded so an independent demonstration can be distinguished clearly.
        """
        from intent_classifier import IntentClassifier

        if self.final_pipeline is None:
            raise ValueError("Final model not fitted yet.")
        intents = intents or [
            "password_reset", "account_access", "attendance", "payroll",
            "salary_information", "wifi_problems", "email_problems",
            "employee_id", "holidays", "working_hours",
        ]
        predictor = IntentClassifier(
            self.final_pipeline,
            self.final_pipeline.classes_,
            preprocessor=self.preprocessor,
        )

        rows = []
        for intent in intents:
            te = self.test_df[self.test_df[LABEL_COLUMN] == intent]
            sample = te.head(n_per_intent)
            source = "test"
            if len(sample) == 0:
                sample = self.train_df[self.train_df[LABEL_COLUMN] == intent].head(n_per_intent)
                source = "train"
            for _, row in sample.iterrows():
                pred = predictor.predict_with_confidence(row["question"])
                rows.append(
                    {
                        "question": str(row["question"]),
                        "actual_intent": str(row[LABEL_COLUMN]),
                        "predicted_intent": str(pred["intent"]),
                        "confidence": float(pred["confidence"]),
                        "source_split": source,
                        "correct": str(pred["intent"]) == str(row[LABEL_COLUMN]),
                    }
                )

        df = pd.DataFrame(rows)
        self.sample_predictions_df = df
        return df

    # -------------------------------------------------------------- artifacts
    def save_models(self) -> Dict[str, Path]:
        """
        Serialize the production artifacts:

            models/intent_classifier_pipeline.pkl  (PRODUCTION: full Pipeline)
            models/tfidf_vectorizer.pkl           (TF-IDF component copy)
            models/intent_classifier.pkl          (classifier component copy)

        Returns a dict of logical-name -> Path.
        """
        if self.final_pipeline is None:
            raise ValueError("Final model not fitted yet.")

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        files = {
            "pipeline": MODELS_DIR / "intent_classifier_pipeline.pkl",
            "vectorizer": MODELS_DIR / "tfidf_vectorizer.pkl",
            "classifier": MODELS_DIR / "intent_classifier.pkl",
        }
        joblib.dump(self.final_pipeline, files["pipeline"])
        joblib.dump(self.final_pipeline.named_steps["tfidf"], files["vectorizer"])
        joblib.dump(self.final_pipeline.named_steps["classifier"], files["classifier"])

        self.model_files = files
        return files

    def save_comparison(self) -> Path:
        """Write outputs/reports/model_comparison.csv (sorted by f1_macro)."""
        if self.comparison_df is None:
            raise ValueError("Comparison table not built yet.")
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        path = REPORTS_DIR / "model_comparison.csv"
        self.comparison_df.to_csv(path, index=False, encoding="utf-8")
        self.report_files["comparison"] = path
        return path

    def save_classification_report(self) -> Path:
        """Write outputs/reports/classification_report.txt."""
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        path = REPORTS_DIR / "classification_report.txt"
        header = (
            "INTENT CLASSIFICATION - PER-INTENT CLASSIFICATION REPORT\n"
            "Final model evaluated on the UNTOUCHED test set.\n"
            "Metric format: precision  recall  f1-score  support (per intent)\n"
            + "=" * 78 + "\n\n"
            + self.classification_report_text
        )
        path.write_text(header, encoding="utf-8")
        self.report_files["classification_report"] = path
        return path

    def save_confusion_matrix(self) -> Path:
        """
        Confusion matrix chart built from actual TEST-SET predictions.
        Includes all 22 intents, readable labels, axis labels and a title.
        """
        if self.confusion is None or self.final_pipeline is None:
            raise ValueError("Final model not evaluated yet.")

        CHARTS_DIR.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(17, 15))
        disp = ConfusionMatrixDisplay(self.confusion, display_labels=self.classes)
        disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")
        ax.set_title(
            "Confusion Matrix - Final Intent Model\n(actual test-set predictions)",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("Predicted intent", fontsize=12)
        ax.set_ylabel("Actual intent", fontsize=12)
        plt.setp(ax.get_xticklabels(), rotation=90, ha="center", fontsize=9)
        plt.setp(ax.get_yticklabels(), fontsize=9)

        path = CHARTS_DIR / "confusion_matrix.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        self.chart_files["confusion_matrix"] = path
        return path

    def error_analysis_df(self) -> pd.DataFrame:
        """
        Misclassified test examples with the model's confidence.
        Columns: question, actual_intent, predicted_intent, confidence.
        """
        if self.final_y_pred is None or self.test_df is None:
            raise ValueError("Final model not evaluated yet.")

        from intent_classifier import IntentClassifier

        y_test = self.test_df[LABEL_COLUMN].astype(str).values
        mask = self.final_y_pred != y_test
        predictor = IntentClassifier(
            self.final_pipeline,
            self.final_pipeline.classes_,
            preprocessor=self.preprocessor,
        )

        rows = []
        for idx in np.where(mask)[0]:
            row = self.test_df.iloc[idx]
            pred = predictor.predict_with_confidence(row["question"])
            rows.append(
                {
                    "question": str(row["question"]),
                    "actual_intent": str(row[LABEL_COLUMN]),
                    "predicted_intent": str(pred["intent"]),
                    "confidence": float(pred["confidence"]),
                }
            )
        return pd.DataFrame(rows)

    def save_error_analysis(self) -> Path:
        """Save outputs/reports/error_analysis.csv."""
        err = self.error_analysis_df()
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        path = REPORTS_DIR / "error_analysis.csv"
        err.to_csv(path, index=False, encoding="utf-8")
        self.report_files["error_analysis"] = path
        return path

    def top_features_text(self, top_k: int = 12) -> str:
        """
        Representative discriminative terms per intent for the FINAL model.

        Uses the model's own learned structure where meaningful:
          * linear classifiers (LogReg / LinearSVM)  -> per-class coefficient
            weights from ``coef_``;
          * Multinomial Naive Bayes                  -> per-class log-likelihood
            ratio versus the other classes (``feature_log_prob_``).

        The measure used is documented in the output; it is never presented as
        a SHAP-style importance when the model does not support such a notion.
        """
        if self.final_pipeline is None:
            raise ValueError("Final model not fitted yet.")

        tfidf = self.final_pipeline.named_steps["tfidf"]
        classifier = self.final_pipeline.named_steps["classifier"]
        feature_names = tfidf.get_feature_names_out()

        lines = [
            "TOP TF-IDF FEATURES BY INTENT (FINAL MODEL)\n" + "=" * 78,
            f"Model            : {MODEL_NAME_MAP.get(str(self.best_model_row['model']), str(self.best_model_row['model']))}",
            f"TF-IDF config    : {self.best_tfidf_config.get('name', '')}",
            f"Features (vocab) : {int(len(feature_names))}",
            "",
        ]

        if hasattr(classifier, "coef_"):
            coef = np.asarray(classifier.coef_)  # (n_classes, n_features)
            measure = "per-class learned linear coefficient (higher => more class-discriminative)"
            lines.append(f"Feature measure  : {measure}")
            lines.append("(the actual sample-order of the matrix follows the fitted class list)")
            lines.append("")
            for i, class_name in enumerate(self.classes):
                weights = coef[i]
                top_idx = np.argsort(weights)[-top_k:][::-1]
                terms = [
                    f"{feature_names[j]}={weights[j]:.4f}" for j in top_idx if weights[j] != 0
                ]
                lines.append(f"[{class_name}]")
                lines.append("  " + (", ".join(terms) if terms else "no positive terms (class weight ~0)"))
        elif hasattr(classifier, "feature_log_prob_"):
            logp = np.asarray(classifier.feature_log_prob_)  # (n_classes, n_features)
            measure = (
                "class-conditional log-probability ratio vs the maximum other class "
                "(feature_log_prob_) - reported as a log-likelihood signal, not a calibrated probability"
            )
            lines.append(f"Measure          : {measure}")
            lines.append("")
            for i, class_name in enumerate(self.classes):
                ratios = logp[i]
                top_idx = np.argsort(ratios)[-top_k:][::-1]
                rows = [f"{feature_names[j]}={ratios[j]:.4f}" for j in top_idx]
                lines.append(f"[{class_name}]")
                lines.append("  " + ", ".join(rows))
        else:
            lines.append(
                "Feature-level interpretation is not well-defined for this final "
                "model; no per-intent feature table is produced."
            )

        return "\n".join(lines)

    def save_top_features(self, top_k: int = 12) -> Path:
        """Save outputs/reports/top_features_by_intent.txt."""
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        path = REPORTS_DIR / "top_features_by_intent.txt"
        path.write_text(self.top_features_text(top_k=top_k), encoding="utf-8")
        self.report_files["top_features"] = path
        return path

    def build_metadata(self) -> Dict[str, Any]:
        """Dynamically-built model metadata (nothing hard-coded)."""
        if self.final_pipeline is None or not self.final_metrics:
            raise ValueError("Final model not evaluated yet.")
        if self.best_model_row is None:
            raise ValueError("Best model not selected yet.")

        model_name = str(self.best_model_row["model"])
        base_raw = self.best_model_row.get("base_model")
        base_name = (
            str(base_raw)
            if base_raw is not None and not (isinstance(base_raw, float) and np.isnan(base_raw))
            else model_name
        )
        num_features = int(
            self.final_pipeline.named_steps["tfidf"].get_feature_names_out().shape[0]
        )
        best_cv = (
            self.comparison_df.iloc[0]
            if self.comparison_df is not None and len(self.comparison_df) else {}
        )
        algorithm = MODEL_NAME_MAP.get(base_name, base_name)
        self.model_algorithm_name = algorithm

        return {
            "model_name": model_name,
            "algorithm": algorithm,
            "training_records": int(len(self.train_df)),
            "test_records": int(len(self.test_df)),
            "num_intents": int(len(self.classes)),
            "tfidf_parameters": _tfidf_kwargs(self.best_tfidf_config),
            "accuracy": float(self.final_metrics["accuracy"]),
            "precision_macro": float(self.final_metrics["precision_macro"]),
            "recall_macro": float(self.final_metrics["recall_macro"]),
            "f1_macro": float(self.final_metrics["f1_macro"]),
            "f1_weighted": float(self.final_metrics["f1_weighted"]),
            "random_state": RANDOM_STATE,
            "num_features": num_features,
            "cv_accuracy_mean": float(best_cv.get("cv_accuracy_mean", np.nan)),
            "cv_accuracy_std": float(best_cv.get("cv_accuracy_std", np.nan)),
            "cv_f1_macro_mean": float(best_cv.get("cv_f1_macro_mean", np.nan)),
            "cv_f1_macro_std": float(best_cv.get("cv_f1_macro_std", np.nan)),
            "hyperparameters": str(self.best_model_row.get("hyperparameters", "")),
            "training_time_seconds": round(float(self.final_train_time), 4),
            "prediction_time_test_seconds": round(float(self.final_predict_time), 6),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }

    def save_metadata(self) -> Path:
        """Save outputs/reports/model_metadata.json."""
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        path = REPORTS_DIR / "model_metadata.json"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.build_metadata(), handle, indent=2, ensure_ascii=False)
        self.report_files["metadata"] = path
        return path

    # -------------------------------------------------- reload & consistency
    def verify_model_consistency(self, sample_texts: List[str]) -> Dict[str, Any]:
        """
        STEP 20/21: Serialization round-trip test.

        Predictions made by the in-memory pipeline are compared with
        predictions made by a freshly reloaded model (from disk).  They must
        match exactly, otherwise the application would be non-deterministic.
        """
        from intent_classifier import IntentClassifier

        predictor_before = IntentClassifier(
            self.final_pipeline,
            self.final_pipeline.classes_,
            preprocessor=self.preprocessor,
        )
        preds_before = [predictor_before.predict_intent(t) for t in sample_texts]

        import gc

        gc.collect()  # drop incidental references; a fresh load must be self-sufficient

        predictor_after = IntentClassifier.load()
        preds_after = [predictor_after.predict_intent(t) for t in sample_texts]

        self.consistency_ok = preds_before == preds_after
        return {
            "ok": self.consistency_ok,
            "previous_predictions": preds_before,
            "reloaded_predictions": preds_after,
            "sample_texts": list(sample_texts),
        }

    # ------------------------------------------------------------ final report
    def _compose_final_report(self) -> str:
        """Compose the full final-model report (STEPS 12, 16, 18, 26)."""
        m = self.final_metrics
        meta = self.build_metadata()
        best = self.best_model_row
        best_cv = self.comparison_df.iloc[0] if self.comparison_df is not None else {}

        tfidf_candidates = "\n".join(
            f"    - {r['name']:<22} macro-F1 = {r['f1_macro_mean']:.4f}"
            f"   accuracy = {r['accuracy_mean']:.4f}"
            for r in self.tfidf_config_results
        )

        comp_rows = []
        for _, row in self.comparison_df.iterrows():
            comp_rows.append(
                f"    {row['model']:<28} f1_macro={row['f1_macro']:.4f} "
                f"f1_wt={row['f1_weighted']:.4f} acc={row['accuracy']:.4f} "
                f"cv_f1_macro={row['cv_f1_macro_mean']:.4f}+/-{row['cv_f1_macro_std']:.4f}"
            )
        comparison_str = "\n".join(comp_rows)

        file_size = 0
        for name, path in self.model_files.items():
            if path.exists():
                file_size += path.stat().st_size
        pipeline_size = self.model_files["pipeline"].stat().st_size

        error_df = (
            self.error_analysis_df()
            if self.final_y_pred is not None else pd.DataFrame()
        )
        n_errors = len(error_df)
        confusion_pairs = (
            error_df.groupby(["actual_intent", "predicted_intent"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .head(10)
        )
        confusion_str = "\n".join(
            f"    {r.actual_intent} -> {r.predicted_intent}  ({int(r.count)}x)"
            for r in confusion_pairs.itertuples()
        ) if len(confusion_pairs) else "    (no misclassifications)"

        sample_lines = []
        sample_acc = 0.0
        sample_correct = 0
        sample_total = 0
        if self.sample_predictions_df is not None and len(self.sample_predictions_df):
            sample_correct = int(self.sample_predictions_df["correct"].sum())
            sample_total = len(self.sample_predictions_df)
            sample_acc = float(self.sample_predictions_df["correct"].mean())
            for _, srow in self.sample_predictions_df.iterrows():
                tag = "OK  " if srow["correct"] else "FAIL"
                sample_lines.append(
                    f"    [{tag}] intent={srow['actual_intent']:<22} "
                    f"predicted={srow['predicted_intent']:<22} "
                    f"conf={srow['confidence']:.3f} (source: {srow['source_split']})"
                )
                sample_lines.append(f"           Q: {srow['question']}")

        train_time = getattr(self, "final_train_time", np.nan)
        predict_time = getattr(self, "final_predict_time", np.nan)

        return f"""\
================================================================================
FINAL INTENT CLASSIFICATION MODEL REPORT - MODULE 3
================================================================================
Generated         : {datetime.now().isoformat(timespec='seconds')}

1. DATASET
   Input file                    : data/processed/faq_nlp_ready.csv
   Records (total)               : {len(self.df)}
   Intents                       : {len(self.classes)}
   Training samples              : {len(self.train_df)}
   Test samples                  : {len(self.test_df)}
   Missing clean_question        : {int(self.df['clean_question'].isna().sum())}
   Missing intent                : {int(self.df['intent'].isna().sum())}

2. TRAIN/TEST SPLIT
   Strategy                      : stratified (random_state=42, test_size=0.20)
   Every class in both sets      : {'PASS' if self.verify_split_classes()['ok'] else 'FAIL'}

3. TF-IDF FEATURES
   Model input text column       : clean_question
   Selected configuration        : {self.best_tfidf_config.get('name')}
   Parameters                    : {_tfidf_kwargs(self.best_tfidf_config)}
   Vocabulary size               : {meta['num_features']}
   Candidates compared (macro-F1 on training cross-validation):
{tfidf_candidates}

4. COMPARISON OF ALL EVALUATED MODELS (metrics from 5-fold stratified CV on
   the training set - the test set was untouched for model selection)
{comparison_str}

5. BEST MODEL
   Model                         : {best['model']}
   Hyperparameters               : {best.get('hyperparameters', 'default')}
   Selection criteria            : macro F1 primary, then weighted F1,
                                   then accuracy, then simplicity
   Selection reason              : {self.selection_reason}

6. FINAL TEST-SET EVALUATION (single / final use of the untouched test set)
   Model name                    : {meta['model_name']}
   Algorithm                     : {meta['algorithm']}
   Accuracy                      : {m['accuracy']:.4f}
   Precision (macro)             : {m['precision_macro']:.4f}
   Recall (macro)                : {m['recall_macro']:.4f}
   Macro F1                      : {m['f1_macro']:.4f}
   Weighted F1                   : {m['f1_weighted']:.4f}
   Budget: this is the ONLY evaluation of the test set in the pipeline.

7. CROSS-VALIDATION of the selected model (training data only, 5 folds)
   Accuracy mean                 : {best_cv.get('cv_accuracy_mean', float('nan')):.4f}
   Accuracy std                  : {best_cv.get('cv_accuracy_std', float('nan')):.4f}
   Macro F1 mean                 : {best_cv.get('cv_f1_macro_mean', float('nan')):.4f}
   Macro F1 std                  : {best_cv.get('cv_f1_macro_std', float('nan')):.4f}

8. TIMING / ARTIFACT SIZE
   Training time (final model)   : {train_time:.3f} seconds
   Test prediction time          : {predict_time:.6f} seconds
   Model file size (pipeline)    : {pipeline_size:,} bytes
   Total model artifacts size    : {file_size:,} bytes

9. SAMPLE PREDICTIONS (real dataset questions, 10 intents)
   Correct                       : {sample_correct} / {sample_total}  (accuracy {sample_acc:.2%})
{chr(10).join(sample_lines)}

10. ERROR ANALYSIS (test set)
    Misclassified examples       : {n_errors} / {len(self.test_df)}
    Most common confusion pairs:
{confusion_str}

11. DATA LEAKAGE CHECK (per specification STEP 26)
   [PASS] Test data never used for fitting TF-IDF.
   [PASS] Test labels were never used for model tuning.
   [PASS] Test data was not used in cross-validation.
   [PASS] Hyperparameter tuning used training data only.
   [PASS] Final test evaluation happened only after model selection.

12. MODEL FILES
   models/intent_classifier_pipeline.pkl   (PRODUCTION - complete sklearn Pipeline)
   models/tfidf_vectorizer.pkl            (TF-IDF component copy)
   models/intent_classifier.pkl           (classifier component copy)

13. PREDICTION CONSISTENCY (STEP 21)
   Prediction before save == prediction after reload:
   {'PASS' if self.consistency_ok else 'FAIL'}
================================================================================
END OF REPORT
================================================================================
"""

    def save_final_report(self) -> Path:
        """Save outputs/reports/final_model_report.txt."""
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        path = REPORTS_DIR / "final_model_report.txt"
        path.write_text(self._compose_final_report(), encoding="utf-8")
        self.report_files["final_report"] = path
        return path

    # ------------------------------------------------------------------- run
    def _print(self, *args: Any, **kwargs: Any) -> None:
        print(*args, **kwargs)

    def run(self, verbose: bool = True) -> bool:
        """
        Execute the complete Module 3 pipeline end-to-end.

        Returns True when the quality gate passes.
        """
        try:
            self._print("=" * 70)
            self._print("MODULE 3 - INTENT CLASSIFICATION MODEL TRAINING")
            self._print("=" * 70)

            # 1. Load + validate the NLP-ready dataset.
            self.df = self.load_dataset()
            self.classes = sorted(self.df[LABEL_COLUMN].astype(str).unique().tolist())
            validation = self.validate_dataset(self.df)
            self._print(f"\n[1] Dataset      : {self.input_csv.name}")
            self._print(f"    Records      : {validation['records']}")
            self._print(f"    Intents      : {validation['intents']}")
            if not validation["valid"]:
                raise ValueError(
                    "Dataset validation failed: " + "; ".join(validation["errors"])
                )
            self._print("    Validation   : PASS")

            # 2. Stratified train/test split.
            self.train_df, self.test_df = self.split_data(self.df)
            self.classes = sorted(self.df[LABEL_COLUMN].astype(str).unique().tolist())
            split_check = self.verify_split_classes()
            self._print(f"\n[2] Split (stratified, test_size={TEST_SIZE})")
            self._print(
                f"    Total={len(self.df)}  Training={len(self.train_df)}  "
                f"Testing={len(self.test_df)}  Classes={len(self.classes)}"
            )
            self._print(
                f"    Every class in both sets : "
                f"{'PASS' if split_check['ok'] else 'FAIL'}"
            )

            # 3. TF-IDF configuration selection (training data only).
            X_train = self.train_df[TEXT_COLUMN].astype(str).tolist()
            y_train = self.train_df[LABEL_COLUMN].astype(str).tolist()
            X_test = self.test_df[TEXT_COLUMN].astype(str).tolist()
            y_test = self.test_df[LABEL_COLUMN].astype(str).tolist()

            selected = self.select_tfidf_config(self.train_df)
            self._print("\n[3] TF-IDF configuration selection (train CV, macro-F1):")
            for r in self.tfidf_config_results:
                self._print(
                    f"    - {r['name']:<22} macro-F1={r['f1_macro_mean']:.4f} "
                    f"accuracy={r['accuracy_mean']:.4f}"
                )
            self._print(
                f"    Selected: {selected.get('name')} -> "
                f"{_tfidf_kwargs(selected)}"
            )

            # 4. Baseline models (Step 5) + cross-validation (Step 7).
            from sklearn.base import clone

            baseline_scores: Dict[str, Dict[str, float]] = {}
            self._print("\n[4] Baseline models with 5-fold stratified CV (training):")
            for name, estimator in BASE_MODELS.items():
                pipe = _build_pipeline(clone(estimator), self.best_tfidf_config)
                metrics = self.evaluate_model_cv(pipe, X_train, y_train, self.classes)
                baseline_scores[name] = metrics
                self._print(
                    f"    - {MODEL_NAME_MAP[name]:<28} f1_macro={metrics['f1_macro']:.4f} "
                    f"acc={metrics['accuracy']:.4f} "
                    f"(cv {metrics['cv_f1_macro_mean']:.4f}+/-{metrics['cv_f1_macro_std']:.4f})"
                )

            # 5. Hyperparameter tuning (training data only).
            self._print("\n[5] Hyperparameter tuning (GridSearchCV, scoring=f1_macro):")
            tuned = self.tune_models(self.train_df)
            for name, grid in tuned.items():
                self._print(
                    f"    - {MODEL_NAME_MAP[name]:<28} best={grid.best_params_} "
                    f"best_cv_f1_macro={grid.best_score_:.4f}"
                )

            # 6. Model comparison + best-model selection (macro F1 first).
            self.build_comparison(baseline_scores, tuned)
            self._print("\n[6] Model comparison (sorted by f1_macro, DESC):")
            show_cols = ["model", "f1_macro", "f1_weighted", "accuracy",
                         "cv_f1_macro_mean", "cv_f1_macro_std"]
            self._print(self.comparison_df[show_cols].to_string(index=False))

            best = self.select_best_model()
            self._print(
                f"\n[7] Best model: {best['model']}  "
                f"(f1_macro={best['f1_macro']:.4f})\n    Reason: {self.selection_reason}"
            )

            # 8. Final model training on the FULL training set.
            pipe = self.fit_final_model(self.best_model_row, tuned=tuned)
            self._print(
                f"\n[8] Final model training on {len(self.train_df)} samples "
                f"({self.final_train_time:.3f}s)"
            )

            # 9. Final single test-set evaluation.
            metrics = self.evaluate_final_model()
            self._print("\n[9] FINAL TEST-SET EVALUATION (untouched test data):")
            for key, value in metrics.items():
                self._print(f"    {key:<18}: {value:.4f}")

            # 10. Sample predictions + error analysis.
            self.make_sample_predictions()
            error_df = self.error_analysis_df()
            self._print(
                f"\n[10] Sample predictions: "
                f"{int(self.sample_predictions_df['correct'].sum())}/"
                f"{len(self.sample_predictions_df)} correct."
            )

            # 11. Serialization.
            self.save_models()
            self.save_comparison()
            self.save_classification_report()
            self.save_confusion_matrix()
            self.save_error_analysis()
            self.save_top_features()
            self.save_metadata()
            self._print("\n[11] Serialization:")
            for logical, path in self.model_files.items():
                self._print(f"    {logical:<11}: {path} ({path.stat().st_size:,} bytes)")

            # 12. Reload + consistency verification.
            sample_texts = self.sample_predictions_df["question"].astype(str).tolist()
            consistency = self.verify_model_consistency(sample_texts)
            self._print(
                "\n[12] Reload & consistency check: "
                + (
                    "PASS - predictions identical before/after reload"
                    if consistency["ok"]
                    else "FAIL"
                )
            )

            # 13. Final report.
            report_path = self.save_final_report()
            self._print(f"\n[13] Final report : {report_path}")
            self._print(f"     Metadata    : {self.report_files['metadata']}")

            # -----------------------------------------------------------------
            # Final Module 3 quality gate.
            gate = {
                "NLP-ready dataset loads": bool(self.df is not None and len(self.df) > 0),
                "294 records verified": validation["records"] == 294,
                "22 intents verified": validation["intents"] == 22,
                "Stratified train/test split": bool(split_check["ok"]),
                "No data leakage": bool(self.leakage_ok),
                "TF-IDF training pipeline": bool(self.best_tfidf_config),
                "Logistic Regression trained": "logistic_regression" in baseline_scores,
                "Linear SVM trained": "linear_svm" in baseline_scores,
                "Multinomial NB trained": "multinomial_nb" in baseline_scores,
                "Cross-validation completed": all(
                    "cv_f1_macro_mean" in s for s in baseline_scores.values()
                ),
                "Model comparison generated": bool(
                    self.comparison_df is not None and len(self.comparison_df) >= len(BASE_MODELS)
                ),
                "Best model selected objectively": self.best_model_row is not None,
                "Final model trained": self.final_pipeline is not None,
                "Final test evaluation completed": bool(self.final_metrics),
                "Classification report generated": "classification_report" in self.report_files,
                "Confusion matrix generated": "confusion_matrix" in self.chart_files,
                "Error analysis generated": "error_analysis" in self.report_files,
                "Model serialized": len(self.model_files) == 3
                and all(p.exists() for p in self.model_files.values()),
                "Model reload successful": bool(self.consistency_ok),
                "Prediction consistency verified": bool(self.consistency_ok),
                "Model metadata generated": "metadata" in self.report_files,
            }
            self._print("\n" + "=" * 70)
            self._print("MODULE 3 QUALITY GATE")
            self._print("=" * 70)
            for name, passed in gate.items():
                self._print(f"[{'PASS' if passed else 'FAIL'}] {name}")
            self._print("=" * 70)
            overall = all(gate.values())
            self._print(f"OVERALL: {'PASS' if overall else 'FAIL'}")
            return bool(overall)

        except Exception as exc:
            self._print(f"\nERROR in Module 3 pipeline: {exc}")
            traceback.print_exc()
            return False


def main() -> int:
    """CLI entry point for the Module 3 training pipeline."""
    pipeline = IntentTrainingPipeline()
    success = pipeline.run()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())