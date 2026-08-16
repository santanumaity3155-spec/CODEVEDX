# MODULE 3 COMPLETION REPORT

**Module:** Intent Classification Model Development & Evaluation
**Project:** Project-04-AI-Helpdesk-Chatbot (AI Chatbot for Internal Helpdesk)

---

## Dataset
- **Records:** 294
- **Intents:** 22
- **Training samples:** 235
- **Test samples:** 59
- **Split:** stratified, `test_size=0.20`, `random_state=42`, `stratify=intent`
- **Every class present in both splits:** PASS

## Models evaluated (macro-F1, 5-fold stratified CV on training set)
| Model | Params | Macro F1 | Weighted F1 | Accuracy | CV macro F1 (mean ± std) |
|---|---|---|---|---|---|
| Linear SVM (tuned) | C=1.0, class_weight=balanced | **0.6828** | 0.6929 | 0.7021 | 0.6608 ± 0.0438 |
| Linear SVM (baseline) | default | 0.6780 | 0.6876 | 0.6979 | 0.6537 ± 0.0506 |
| Log. Regression (tuned) | C=10.0, balanced | 0.6655 | 0.6764 | 0.6809 | 0.6392 ± 0.0709 |
| Log. Regression (baseline) | default | 0.6522 | 0.6587 | 0.6809 | 0.6250 ± 0.0554 |
| Multinomial NB (tuned) | alpha=0.5 | 0.5922 | 0.5984 | 0.6255 | 0.5615 ± 0.0592 |
| Multinomial NB (baseline) | default | 0.5343 | 0.5390 | 0.5702 | 0.4972 ± 0.0667 |

## Best model
- **Model name:** `linear_svm_tuned`
- **Algorithm:** Linear SVM (`LinearSVC`, C=1.0, `class_weight="balanced"`)
- **Reason for selection:** Highest macro F1 (0.6828) on 5-fold stratified
  cross-validation of the training set; tie-breakers weighted F1 (0.6929) and
  accuracy (0.7021) also highest. Selected using training-data-only metrics
  (the test set was not used until the final evaluation).

## Final metrics (single evaluation of the untouched test set)
- **Accuracy:** 0.7119
- **Macro Precision:** 0.7083
- **Macro Recall:** 0.7121
- **Macro F1:** 0.6801
- **Weighted F1:** 0.6899

## Cross-validation (best model, training data only)
- **Accuracy mean:** 0.7021
- **Accuracy std:** 0.0404
- **Macro F1 mean:** 0.6608
- **Macro F1 std:** 0.0438

## Data leakage check
**PASS** — TF-IDF fitted on training data only; test set never used for
cross-validation, tuning, or any model-selection decision; final test
evaluation performed only after model selection (documented in
`outputs/reports/final_model_report.txt`).

## Model files
- `models/intent_classifier_pipeline.pkl` — **PRODUCTION model** (complete
  sklearn `Pipeline`: TF-IDF + Linear SVM); 55,176 bytes
- `models/intent_classifier.pkl` — classifier component copy (49,219 bytes)
- `models/tfidf_vectorizer.pkl` — TF-IDF component copy (6,065 bytes)

## Reports generated
- `outputs/reports/model_comparison.csv` (sorted by `f1_macro` desc)
- `outputs/reports/final_model_report.txt`
- `outputs/reports/classification_report.txt` (per-intent + macro/weighted avg)
- `outputs/reports/top_features_by_intent.txt`
- `outputs/reports/error_analysis.csv`
- `outputs/reports/model_metadata.json`

## Charts generated
- `outputs/charts/confusion_matrix.png` (all 22 intents, actual test predictions)

## Sample predictions
10 real dataset questions across password_reset, account_access, attendance,
payroll, salary_information, wifi_problems, email_problems, employee_id,
holidays, working_hours — **6/10 correct** (all labelled with source split).

## Verification
- **Prediction test:** PASS
- **Model reload test:** PASS
- **Prediction consistency (before/after reload):** PASS
- **Module 1 regression:** PASS (15/15)
- **Module 2 regression:** PASS (6/6)
- **Module 3 tests:** 23 passed / 0 failed
- **Full test suite:** 44 passed / 0 failed
- **Notebook:** `notebooks/model_training.ipynb` created and executed
  successfully (0 errors, `MODULE 3: COMPLETE` gate)

## Problems discovered
1. `MODEL_NAME_MAP` constant was accidentally dropped from the training module
   during initial composition -> `NameError` in the CLI run.
2. `confusion_matrix` was referenced but not imported -> `NameError`.
3. `make_scorer(f1_score, average='macro', labels=...)` forwards a
   `pos_label=1` default that is invalid for multiclass string labels,
   breaking `GridSearchCV` scoring -> all CV metrics became `nan`.
4. nbconvert launches the kernel with cwd = `notebooks/`, so `Path.cwd()`
   did not point at the project root -> `ModuleNotFoundError` in cell 1.
5. A `\n` escape inside a notebook code cell was rendered as a literal
   newline, producing a `SyntaxError: unterminated string literal`.
6. Metadata building read `base_model` of untuned rows as NaN -> algorithm
   name would render as "nan".

## Problems fixed
1. Re-added `MODEL_NAME_MAP` (incl. tuned model names).
2. Added `confusion_matrix` to the sklearn.metrics import.
3. Wrapped every CV scorer in a function that absorbs `**kwargs`; robust for
   macro/weighted F1, precision and recall on 22 string labels.
4. Made notebook root detection robust (`data/processed/faq_nlp_ready.csv`
   probe with fallback to parent directory).
5. Escaped the newline correctly in the notebook builder.
6. Guarded NaN when deriving the algorithm/base name from the comparison row.

## FINAL STATUS
**MODULE 3: COMPLETE**