"""Test suite for Module 3: Intent Classification (real behaviour, 23 tests).

These tests exercise ACTUAL functionality (never ``assert True``):

  1.  Module imports
  2.  Input dataset loads
  3.  Required columns exist
  4.  Train/test split works
  5.  Stratification works
  6.  TF-IDF fitting works
  7.  Models train successfully
  8.  Model comparison generated
  9.  Best model selected
  10. Model saved
  11. Vectorizer/pipeline saved
  12. Model reload works (+ prediction consistency before/after reload)
  13. Single prediction works
  14. Batch prediction works
  15. Prediction output contains intent
  16. Confidence handling works
  17. All predicted intents belong to valid intents
  18. Final report exists
  19. Confusion matrix exists
  20. Classification report exists
  21. Error analysis exists
  22. Prediction preprocessing matches training (same clean_question)
  23. No leakage: vectorizer fitted on training data only
"""

import json
import sys
from pathlib import Path

import pytest
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nlp_preprocessor import CANONICAL_INTENTS  # noqa: E402
from train_intent_model import (  # noqa: E402
    IntentTrainingPipeline, BASE_MODELS, TEXT_COLUMN, LABEL_COLUMN,
)
from intent_classifier import IntentClassifier  # noqa: E402

INPUT_CSV = PROJECT_ROOT / "data" / "processed" / "faq_nlp_ready.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
CHARTS_DIR = PROJECT_ROOT / "outputs" / "charts"

REQUIRED_COLUMNS = ["question", "clean_question", "intent", "answer", "entity"]
COMPARISON_COLUMNS = [
    "model", "accuracy", "precision_macro", "recall_macro", "f1_macro",
    "f1_weighted", "cv_accuracy_mean", "cv_accuracy_std",
    "cv_f1_macro_mean", "cv_f1_macro_std",
]


@pytest.fixture(scope="session")
def pipeline():
    return IntentTrainingPipeline(PROJECT_ROOT)


@pytest.fixture(scope="session")
def df(pipeline):
    data = pipeline.load_dataset()
    assert len(data) > 0, "Module 2 NLP-ready dataset is empty"
    return data


@pytest.fixture(scope="session")
def model():
    # Production model loaded once per session (documented artifact).
    return IntentClassifier.load()


# 1. Module imports -----------------------------------------------------------
def test_module_imports():
    assert IntentClassifier is not None
    assert IntentTrainingPipeline is not None
    assert {"logistic_regression", "linear_svm", "multinomial_nb"} <= set(BASE_MODELS)
    assert LABEL_COLUMN == "intent"
    assert TEXT_COLUMN == "clean_question"

    import intent_classifier

    for fn in ("load_model", "predict_intent", "predict_with_confidence", "predict_batch"):
        assert callable(getattr(intent_classifier, fn)), f"missing {fn}"


# 2. Input dataset loads ------------------------------------------------------
def test_input_dataset_loads(pipeline):
    data = pipeline.load_dataset()
    assert len(data) > 0, "Dataset loads empty"
    assert data["intent"].nunique() > 1


# 3. Required columns exist ---------------------------------------------------
def test_required_columns_exist(df):
    for col in REQUIRED_COLUMNS:
        assert col in df.columns, f"missing column {col}"


# 4. Train/test split works ---------------------------------------------------
def test_train_test_split_works(pipeline, df):
    train_df, test_df = pipeline.split_data(df)
    assert len(train_df) > 0 and len(test_df) > 0
    assert len(train_df) + len(test_df) == len(df)
    assert set(train_df["question"]).isdisjoint(set(test_df["question"]))


# 5. Stratification works -----------------------------------------------------
def test_stratification_works(pipeline, df):
    train_df, test_df = pipeline.split_data(df)
    # Every class must be represented in BOTH sets (all 22 retained).
    assert set(train_df[LABEL_COLUMN]) == set(df[LABEL_COLUMN])
    assert set(test_df[LABEL_COLUMN]) == set(df[LABEL_COLUMN])
    # Per-class train fraction should stay close to 0.8 (~ stratified).
    for cls in df[LABEL_COLUMN].unique():
        total = int((df[LABEL_COLUMN] == cls).sum())
        tr = int((train_df[LABEL_COLUMN] == cls).sum())
        frac = tr / total
        assert 0.5 <= frac <= 1.0, (
            f"class {cls} train fraction {frac:.2f} diverges from stratification"
        )


# 6. TF-IDF fitting works ------------------------------------------------------
def test_tfidf_fitting_works(pipeline, df):
    from sklearn.feature_extraction.text import TfidfVectorizer

    from train_intent_model import TFIDF_CONFIGS

    train_df, test_df = pipeline.split_data(df)
    cfg = {k: v for k, v in TFIDF_CONFIGS[0].items() if k != "name"}
    vec = TfidfVectorizer(**cfg).fit(train_df[TEXT_COLUMN].astype(str))
    X_train = vec.transform(train_df[TEXT_COLUMN].astype(str))
    assert X_train.shape[0] == len(train_df), "vectorizer fit on wrong rows"
    assert X_train.shape[1] > 0, "empty vocabulary"
    # The test split must simply be transformed - never re-fitted.
    X_test = vec.transform(test_df[TEXT_COLUMN].astype(str))
    assert X_test.shape[0] == len(test_df)
    assert X_test.shape[1] == X_train.shape[1]


# 7. Models train successfully ------------------------------------------------
def test_models_train_successfully(pipeline, df):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline

    train_df, _ = pipeline.split_data(df)
    X = train_df[TEXT_COLUMN].astype(str).tolist()
    y = train_df[LABEL_COLUMN].astype(str).tolist()
    for name, estimator in BASE_MODELS.items():
        pipe = Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9)),
                ("clf", estimator),
            ]
        )
        pipe.fit(X, y)
        pred = pipe.predict(X)
        assert pred.shape == (len(X),), f"{name} prediction shape wrong"
        assert set(pred).issubset(set(y)), f"{name} predicted unknown labels"


# 8. Model comparison generated ------------------------------------------------
def test_model_comparison_generated():
    csv_path = REPORTS_DIR / "model_comparison.csv"
    assert csv_path.exists(), "model_comparison.csv missing"
    df = pd.read_csv(csv_path)
    for col in COMPARISON_COLUMNS:
        assert col in df.columns, f"missing comparison column {col}"
    f1 = df["f1_macro"].tolist()
    assert f1 == sorted(f1, reverse=True), "comparison not sorted by f1_macro"


# 9. Best model selected --------------------------------------------------------
def test_best_model_selected():
    cmp = pd.read_csv(REPORTS_DIR / "model_comparison.csv")
    meta = json.loads((REPORTS_DIR / "model_metadata.json").read_text(encoding="utf-8"))
    assert meta["model_name"] == cmp.iloc[0]["model"], (
        "metadata model != best row of comparison table"
    )
    report = (REPORTS_DIR / "final_model_report.txt").read_text(encoding="utf-8")
    assert "Selection reason" in report or "macro F1" in report.lower()


# 10. Model saved ----------------------------------------------------------------
def test_model_saved():
    model_path = MODELS_DIR / "intent_classifier.pkl"
    assert model_path.exists() and model_path.stat().st_size > 0
    import joblib

    estimator = joblib.load(model_path)
    assert hasattr(estimator, "predict"), "saved classifier has no predict()"
    assert hasattr(estimator, "classes_")


# 11. Vectorizer / pipeline saved ----------------------------------------------------
def test_vectorizer_and_pipeline_saved():
    pipeline_path = MODELS_DIR / "intent_classifier_pipeline.pkl"
    vectorizer_path = MODELS_DIR / "tfidf_vectorizer.pkl"
    assert pipeline_path.exists() and pipeline_path.stat().st_size > 0
    assert vectorizer_path.exists() and vectorizer_path.stat().st_size > 0
    import joblib

    vectorizer = joblib.load(vectorizer_path)
    assert hasattr(vectorizer, "vocabulary_")
    pipe = joblib.load(pipeline_path)
    assert hasattr(pipe, "steps") and hasattr(pipe, "classes_")


# 12. Model reload works (+ consistency) -----------------------------------------
def test_model_reload_and_consistency():
    questions = ["How do I reset my password?", "My laptop is very slow."]
    first = IntentClassifier.load()
    preds_before = [first.predict_intent(q) for q in questions]
    # Force a FRESH load from disk (module cache cleared) and re-predict.
    import intent_classifier

    intent_classifier._model_cache.clear()
    fresh = IntentClassifier.load()
    preds_after = [fresh.predict_intent(q) for q in questions]
    assert preds_before == preds_after, "predictions differ before/after reload"


# 13. Single prediction works -------------------------------------------------
def test_single_prediction_works(model):
    intent = model.predict_intent("Where can I find the office location?")
    assert isinstance(intent, str) and intent


# 14. Batch prediction works ----------------------------------------------------
def test_batch_prediction_works(model):
    texts = ["How do I reset my password?", "When is my salary credited?"]
    results = model.predict_batch(texts)
    assert isinstance(results, list)
    assert len(results) == len(texts)
    for result in results:
        assert "intent" in result and result["intent"]


# 15. Prediction output contains intent ------------------------------------------
def test_prediction_output_contains_intent(model):
    pred = model.predict_with_confidence("I cannot access my email account.")
    assert isinstance(pred, dict)
    assert "intent" in pred
    assert pred["intent"]


# 16. Confidence handling works ----------------------------------------------------
def test_confidence_handling_works(model):
    pred = model.predict_with_confidence("What are the working hours?")
    assert 0.0 <= float(pred["confidence"]) <= 1.0
    assert pred["confidence_source"] in {"predict_proba", "decision_function_margin"}
    if pred["confidence_source"] == "predict_proba":
        assert "probabilities" in pred
        probs = pred["probabilities"]
        assert max(probs.values()) == pytest.approx(float(pred["confidence"]), abs=1e-6)
    else:
        # decision-margin confidence must NEVER be labelled as a probability.
        assert "probabilities" not in pred
        assert "note" in pred


# 17. All predicted intents valid ---------------------------------------------------
def test_predicted_intents_valid(model, df):
    sample = df.head(8)["question"].astype(str).tolist()
    results = model.predict_batch(sample)
    valid = set(CANONICAL_INTENTS)
    for result in results:
        assert result["intent"] in valid, f"invalid intent {result['intent']}"


# 18. Final report exists ---------------------------------------------------------------
def test_final_report_exists():
    report = REPORTS_DIR / "final_model_report.txt"
    assert report.exists() and report.stat().st_size > 0
    text = report.read_text(encoding="utf-8")
    for token in ("Macro F1", "Accuracy", "Training samples", "Test samples"):
        assert token in text


# 19. Confusion matrix exists ------------------------------------------------------------
def test_confusion_matrix_exists():
    chart = CHARTS_DIR / "confusion_matrix.png"
    assert chart.exists() and chart.stat().st_size > 0


# 20. Classification report exists ----------------------------------------------------------
def test_classification_report_exists():
    report = REPORTS_DIR / "classification_report.txt"
    assert report.exists() and report.stat().st_size > 0
    text = report.read_text(encoding="utf-8")
    for token in ("precision", "recall", "f1-score", "macro avg", "weighted avg"):
        assert token.lower() in text.lower()


# 21. Error analysis exists -------------------------------------------------------------------
def test_error_analysis_exists():
    csv_path = REPORTS_DIR / "error_analysis.csv"
    assert csv_path.exists() and csv_path.stat().st_size > 0
    df_err = pd.read_csv(csv_path)
    for col in ("question", "actual_intent", "predicted_intent", "confidence"):
        assert col in df_err.columns, f"missing error-analysis column {col}"


# 22. Preprocessing matches training (same clean_question) ---------------------------------
def test_preprocessing_matches_training(df, model):
    from nlp_preprocessor import NLPPreprocessor

    prep = NLPPreprocessor(PROJECT_ROOT)
    for _, row in df.head(3).iterrows():
        cleaned = model.preprocess_text(row["question"])
        assert cleaned == row["clean_question"], (
            "prediction preprocessing differs from training clean_question"
        )
        assert cleaned == prep.clean_text(row["question"])


# 23. No leakage: vectorizer fitted on training data only -------------------------
def test_no_leakage_vectorizer_train_only(pipeline, df):
    import joblib

    train_df, test_df = pipeline.split_data(df)
    vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer.pkl")
    vocabulary = set(vectorizer.get_feature_names_out())

    train_terms = set()
    for q in train_df[TEXT_COLUMN].astype(str):
        train_terms |= set(q.split())
    test_only_terms = set()
    for q in test_df[TEXT_COLUMN].astype(str):
        test_only_terms |= set(q.split())
    test_only_terms -= train_terms  # terms that appear ONLY in the test split

    leaked = test_only_terms & vocabulary
    assert len(leaked) == 0, (
        f"test-only terms leaked into the fitted TF-IDF vocabulary: "
        f"{sorted(leaked)[:10]}"
    )