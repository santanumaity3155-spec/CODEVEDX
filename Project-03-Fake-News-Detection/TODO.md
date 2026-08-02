# Module 2 — Data Preprocessing Tracking

This file tracks progress for **Module 2: Data Preprocessing**.
It is a temporary tracking artifact and will be removed before final submission.

## ✅ Completed (Module 1 — Foundation)

- [x] Verify folder structure
- [x] Verify dataset presence (`data/raw/Fake.csv`, `data/raw/True.csv`)
- [x] Create `.gitignore`, `requirements.txt`, `README.md`, `PROJECT_SUMMARY.md`
- [x] Initialize `src/__init__.py`

## 🔄 Module 2 — Data Preprocessing

- [ ] Create notebook generator script (`notebooks/write_preprocessing_notebook.py`)
- [ ] Generate `notebooks/data_preprocessing.ipynb`
- [ ] Step 1: Project introduction (Markdown)
- [ ] Step 2: Import libraries (pandas, numpy, re, string, pathlib, nltk, warnings)
- [ ] Step 3: Auto-download NLTK resources (punkt, stopwords, wordnet, omw-1.4)
- [ ] Step 4: Load Fake.csv & True.csv (shape, columns, head, tail, info)
- [ ] Step 5: Add labels (Fake → 0, True → 1)
- [ ] Step 6: Merge, shuffle, reset index
- [ ] Step 7: Merged shape & class distribution
- [ ] Step 8: Missing values, duplicates, empty text/title checks
- [ ] Step 9: Handle missing/duplicates/invalid text/whitespace
- [ ] Step 10: Full NLP cleaning (lowercase, HTML, URLs, emails, punctuation, numbers, whitespace, special chars, non-ASCII)
- [ ] Step 11: Tokenization (NLTK)
- [ ] Step 12: Stopword removal (keeping not/no/never)
- [ ] Step 13: Lemmatization (WordNetLemmatizer)
- [ ] Step 14: Create `clean_text` column (originals preserved)
- [ ] Step 15: NLP features (word_count, char_count, sentence_count, avg_word_length)
- [ ] Step 16: Verify preprocessing (original → clean examples)
- [ ] Step 17: Quality checks (no missing, no duplicates, no empty clean_text, balanced labels)
- [ ] Step 18: Save processed dataset (`data/processed/fake_news_dataset.csv`)
- [ ] Step 19: Reload & verify (shape, columns, dtypes)
- [ ] Step 20: Generate preprocessing report (`outputs/reports/preprocessing_report.txt`)
- [ ] Execute notebook end-to-end (nbconvert)
- [ ] Verify CSV + report exist and are correct
- [ ] Update this TODO.md

## ✅ Module 2 Success Criteria

- [ ] `data/processed/fake_news_dataset.csv` created
- [ ] `outputs/reports/preprocessing_report.txt` created
- [ ] Notebook executes top-to-bottom without errors
- [ ] All preprocessing functions work correctly

