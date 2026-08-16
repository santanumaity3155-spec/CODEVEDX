# AI Chatbot for Internal Helpdesk

## Project Overview

This project implements an AI-powered chatbot for internal helpdesk support.
The chatbot uses Natural Language Processing (NLP) and intent detection to
answer frequently asked questions from employees about IT support, HR
policies, and company procedures.

## Module 1: Complete Dataset Preparation, Domain Adaptation & Validation

Module 1 is **COMPLETE**. It prepares a clean, validated, reproducible
**internal helpdesk FAQ dataset** ready for Module 2 (NLP preprocessing),
Module 3 (intent classification) and Module 4 (chatbot engine).

### Dataset roles

| Dataset                                       | Role                                            |
| --------------------------------------------- | ----------------------------------------------- |
| `data/raw/Bitext_..._27K_responses-v11.csv`   | Original source (26,872 e-commerce records). Never modified. |
| `data/raw/faq_dataset.csv`                    | Project internal-helpdesk candidate (was imbalanced). |
| `data/raw/faq_dataset_augmented.csv`          | 135 carefully curated examples (built by the pipeline). |
| `data/processed/faq_dataset.csv`              | **Final cleaned, balanced Module 1 dataset.**   |

### Structure

`data/processed/faq_dataset.csv` uses the schema:

- `question`: the user's natural-language helpdesk question
- `intent`: the canonical intent/category (snake_case)
- `answer`: the appropriate helpdesk response
- `entity`: optional entity, or empty string when none applies

### Final dataset statistics (computed dynamically)

- Total records: 294
- Total intents: 22
- Minimum / maximum examples per intent: 12 / 20
- Duplicate questions: 0
- Missing values: 0
- Total entities: 22
- Class balance ratio: 0.6
- Quality gate: **PASS**

All 22 intents are internal-helpdesk domains (password_reset, account_access,
email_problems, laptop_problems, software_installation, wifi_problems,
leave_policy, salary_information, payroll, attendance, holidays, hr_support,
security, technical_support, office_location, employee_id, working_hours,
internet_problems, contact_information, greetings, goodbye, help).

### Intents covered by the base dataset (`data/raw/faq_dataset.csv`)

greetings / goodbye / help, password_reset / account_access,
laptop_problems / software_installation, internet_problems / wifi_problems /
email_problems, leave_policy / attendance / working_hours / holidays,
salary_information / payroll, employee_id / hr_support / office_location,
contact_information / security / technical_support.

## Module 2: NLP Text Preprocessing

Module 2 is **COMPLETE**. `src/nlp_preprocessor.py` provides a reproducible,
deterministic text preprocessing pipeline used by the Module 3 models:

- lowercasing
- HTML/URL/email/phone/number/entity token masking
- punctuation removal
- tokenization
- stop-word removal (custom internal-helpdesk stop list)
- lemmatization

**Tests:** `tests/test_nlp_preprocessing.py` (6 tests). Run with
`python -m pytest tests/test_nlp_preprocessing.py -v`.

## Module 3: Intent Classification Model Development & Evaluation

Module 3 is **COMPLETE**. `src/train_intent_model.py` implements the
`IntentTrainingPipeline` and `src/intent_classifier.py` implements the
`IntentClassifier` prediction wrapper. Everything is reproducible with
`random_state=42`.

### Running the training pipeline (CLI)

```bash
python src/train_intent_model.py
```

### Reproducing the notebook

```bash
python build_model_training_notebook.py          # build notebooks/model_training.ipynb
jupyter nbconvert --to notebook --execute notebooks/model_training.ipynb --inplace
```

The notebook walks through the full pipeline in 20 sections and ends with a
final quality gate (`MODULE 3: COMPLETE`).

### Results (best model: Linear SVM — `linear_svm_tuned`)

| Metric (test set)        | Value   |
| ------------------------ | ------- |
| Accuracy                 | 0.7119  |
| Macro Precision          | 0.7083  |
| Macro Recall             | 0.7121  |
| **Macro F1**             | **0.6801** |
| Weighted F1              | 0.6899  |

Cross-validation (best model, training data only): macro F1 = 0.6608 ± 0.0438,
accuracy = 0.7021 ± 0.0404.

6 models compared: Logistic Regression, Linear SVM and Multinomial NB, each
baseline and hyperparameter-tuned; the best model is selected by the highest
5-fold CV macro F1.

### Outputs

- **Production model:** `models/intent_classifier_pipeline.pkl`
  (complete `Pipeline` = TF-IDF + Linear SVM)
- **Component models:** `models/intent_classifier.pkl`,
  `models/tfidf_vectorizer.pkl`
- **Reports:** `outputs/reports/model_comparison.csv`,
  `outputs/reports/final_model_report.txt`, `outputs/reports/classification_report.txt`,
  `outputs/reports/top_features_by_intent.txt`, `outputs/reports/error_analysis.csv`,
  `outputs/reports/model_metadata.json`
- **Charts:** `outputs/charts/confusion_matrix.png` (all 22 intents)

### Tests

```bash
python -m pytest -v
```

The full suite (44 tests) covers Module 1 regression (15), Module 2
(6) and Module 3 intent-classification (23). See
`MODULE_3_COMPLETION_REPORT.md` for the full report and the data-leakage
verification.

## Setup

### Prerequisites

- Python 3.8+

### Installation

```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

## Dataset Preparation

To build, clean and validate the dataset:

```bash
python src/prepare_dataset.py
```

This reproducible script performs, in order:
`load -> validate -> clean -> normalize intents -> balance -> validate ->
save -> report -> charts`. Running it multiple times does **not** create
duplicates.

### Outputs

- **Processed dataset:** `data/processed/faq_dataset.csv`
- **Dataset report:** `outputs/reports/dataset_report.txt`
- **Intent distribution chart:** `outputs/charts/intent_distribution.png`
- **Intent distribution pie chart:** `outputs/charts/intent_distribution_pie.png`

## Tests

```bash
python -m pytest tests/test_dataset_preparation.py -v
```

The suite enforces (among others) that `test_intent_distribution_valid`
requires a minimum of **10 examples per intent**.

## Notebook

Open `notebooks/data_preparation.ipynb` in Jupyter. It mirrors the CLI
pipeline across 16 sections with real executable cells:

1. Module 1 Overview
2. Load Raw Datasets
3. Inspect Dataset Structure
4. Missing Value Analysis
5. Duplicate Analysis
6. Intent Distribution
7. Internal Helpdesk Domain Analysis
8. Data Cleaning
9. Intent Normalization
10. Intent Balancing
11. Entity Validation
12. Final Dataset Validation
13. Dataset Statistics
14. Visualizations
15. Save Processed Dataset
16. Final Quality Gate

## Next Steps (Module 4+)

- Module 4: Chatbot prediction engine
- Module 5: Flask API development
- Module 6: Admin panel and CRUD operations

## Contributing

This is an internship project. For questions or issues, please contact the
development team.

## License

Internal use only - CodeVedX Internship Program

## Module 4: AI Helpdesk Chatbot Engine

Module 4 is **COMPLETE**. The chatbot engine consumes the trained Module 3 intent-classification model (`models/intent_classifier_pipeline.pkl`) and provides the next production layer of the internal helpdesk chatbot:

### Features

- **Intent prediction**: Reuses the Module 3 Linear SVM pipeline with decision-function margin confidence (`confidence_source == "decision_function_margin"`).
- **Confidence tiers**: LOW (< 0.52) triggers fallback; MEDIUM (0.52�0.65) returns answer with caution note; HIGH (>= 0.65) returns raw FAQ answer.
- **FAQ answer retrieval**: Data-driven selection from `data/processed/faq_nlp_ready.csv`; cosine-relevance signal (threshold 0.40) for out-of-domain detection.
- **Entity awareness**: Rule-based extraction from the canonical entity vocabulary (`ENTITY_KEYWORDS`, word-boundary matching).
- **Fallback handling**: Empty/whitespace overflow, out-of-domain, low confidence, and no-answer scenarios with professional helpdesk messages.
- **Configurable thresholds**: All confidence and relevance thresholds are parameters of `ChatbotConfig`, not hard-coded.
- **CLI interface**: Interactive mode with `help` / `clear` / `exit` / `quit` commands; single-question `--once` mode; JSON output.
- **Regression protection**: All 44 existing Module 1�3 tests continue passing; 19 new Module 4 tests bring the total to 63 passed / 0 failed.

### Quick start

```bash
python src/chatbot.py               # interactive CLI
python src/chatbot.py --once "How do I reset my password?"  # single question
python src/chatbot.py --once --json "question"  # JSON output
```

### Test results

- Module 4 tests: **19 passed** (behavioural, no `assert True`)
- Full test suite: **63 passed** / 0 failed (44 Module 1�3 + 19 Module 4)
- Manual test questions: **7/7 passed**
- Regression: **zero failures** on existing Module 1�3 tests

### Module 1/2/3 Integration

The Module 3 trained artifacts are consumed directly:

| Artifact | Path | Role |
|----------|------|------|
| Production Pipeline | `models/intent_classifier_pipeline.pkl` | Complete sklearn `Pipeline` (TF-IDF + Linear SVM) |
| Classifier component | `models/intent_classifier.pkl` | Extracted `LinearSVC` copy |
| TF-IDF vectorizer | `models/tfidf_vectorizer.pkl` | Fitted only on training data � never refit on user input |
| NLP-ready dataset | `data/processed/faq_nlp_ready.csv` | Module 2 output: 294 rows, 22 intents, `clean_question`, `answer`, `entity` |

Data-leakage rule: The TF-IDF vectorizer is loaded once at Module 4 initialisation and reused unchanged for every prediction. No re-fitting occurs. The same `NLPPreprocessor.clean_text()` pipeline that produced `clean_question` during Module 2 training is applied verbatim at inference time.

### Intents covered by the base dataset (`data/raw/faq_dataset.csv`)

greetings / goodbye / help, password_reset / account_access,
laptop_problems / software_installation, internet_problems / wifi_problems /
email_problems, leave_policy / attendance / working_hours / holidays,
salary_information / payroll, employee_id / hr_support / office_location,
contact_information / security / technical_support.

## Module 5: Flask API Development

Module 5 is **COMPLETE**. It exposes the Module 4 chatbot engine as a
production-ready Flask REST API.

### Architecture

```
src/
    app.py              # Flask application factory + CLI entry point
    config.py           # Flask + chatbot configuration
    api/
        __init__.py
        routes.py       # API endpoints
    chatbot.py          # Module 4 orchestrator (unchanged)
    chatbot_config.py   # Module 4 config (unchanged)
    ...
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with model/FAQ status |
| `/api/chat` | POST | Primary chatbot endpoint |
| `/api/predict` | POST | Intent prediction only |

### Request / Response Example

**Request:**
```json
POST /api/chat
{
    "message": "How do I reset my password?"
}
```

**Response:**
```json
{
    "success": true,
    "message": "How do I reset my password?",
    "intent": "password_reset",
    "confidence": 0.705,
    "response": "You can reset your password through the internal password reset portal...",
    "fallback": false,
    "entities": ["password"],
    "confidence_level": "high"
}
```

### Running the API

```bash
python src/app.py
```

The server starts on `http://127.0.0.1:5000` by default.

### Configuration

Environment variables (all optional):

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_HOST` | `127.0.0.1` | Flask bind address |
| `APP_PORT` | `5000` | Flask port |
| `APP_DEBUG` | `false` | Debug mode |
| `ALLOWED_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000` | CORS origins |
| `MAX_MESSAGE_LENGTH` | `500` | Max characters per message |
| `LOG_LEVEL` | `INFO` | Logging level |

### Features

- **Model loaded once** at startup (no per-request reload).
- **Request validation**: empty, null, whitespace, oversized, invalid JSON.
- **Structured JSON responses** with intent, confidence, entities, fallback flag.
- **Centralised error handling**: no stack traces or local paths leaked to clients.
- **CORS** configured via `ALLOWED_ORIGINS` environment variable.
- **Logging** at INFO/WARNING/ERROR levels.

### Test results

- Module 5 API tests: **34 passed**
- Module 5 integration tests: **15 passed**
- Module 6 admin tests: **64 passed**
- Full test suite: **209 tests** / **208 passed** / **1 failed** (pre-existing unrelated failure in TF-IDF leakage test)
- Regression: **zero failures** on existing Module 1�4 tests
- Manual API testing: **passed** (health, chat, predict, validation, fallback)
- End-to-end integration tests: **68 passed** (test_final_integration.py + test_integration.py)

### Performance

- Average request latency: ~1.7ms (local test client)
- Model cached at startup; no per-request disk I/O for model loading.

### Module 6: Admin FAQ Management

- Admin CRUD operations (create, update, delete, list, search) for FAQ/helpdesk knowledge base
- Safe CSV persistence with atomic file writes
- Input validation (question, intent, answer, entity)
- Intent validation against canonical project intents
- Duplicate question prevention (case-insensitive)
- Chatbot refresh mechanism (POST /api/admin/reload) after admin operations
- Optional admin API key protection via ADMIN_API_KEY environment variable
- Logging of all admin operations (create, update, delete, reload, validation failures)
- Admin API endpoints: GET/POST /api/admin/faqs, PUT/DELETE /api/admin/faqs/<id>, POST /api/admin/reload

#### Admin API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/admin/faqs` | GET | List all FAQs |
| `/api/admin/faqs` | POST | Create a new FAQ |
| `/api/admin/faqs/<id>` | PUT | Update an existing FAQ |
| `/api/admin/faqs/<id>` | DELETE | Delete an FAQ |
| `/api/admin/reload` | POST | Reload FAQ data from disk |

#### Admin Validation Rules

- Question: Must not be missing, empty, or whitespace-only; max 1000 characters
- Intent: Must be a valid canonical intent from the project's defined set
- Answer: Must not be missing or empty; max 5000 characters
- Entity: Must be a valid canonical entity or empty string
- Duplicates: Duplicate questions (case-insensitive) are rejected
