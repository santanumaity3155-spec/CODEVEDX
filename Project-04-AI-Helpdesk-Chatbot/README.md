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