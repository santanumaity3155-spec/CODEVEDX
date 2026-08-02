# Module 3 — EDA Tracking

This file tracks progress for **Module 3: Exploratory Data Analysis (EDA)**.
It is a temporary tracking artifact and will be removed before final submission.

## ✅ Completed (Modules 1–2)

- [x] Module 1 — Project Foundation
- [x] Module 2 — Data Preprocessing
  - [x] `data/processed/fake_news_dataset.csv` (43,971 rows × 10 cols)
  - [x] `outputs/reports/preprocessing_report.txt`

## 🔄 Module 3 — Exploratory Data Analysis (EDA)

### Plan Steps (approved)

- [x] 1. Create `notebooks/write_eda_notebook.py` (builder script, Steps 1–20)
- [x] 2. Generate `notebooks/eda.ipynb`
- [x] 3. Execute notebook end-to-end (nbconvert)
- [x] 4. Verify charts in `outputs/charts/` + `outputs/reports/eda_report.txt`
- [x] 5. Update this TODO.md

### Notebook Build Steps

- [x] Create notebook generator script (`notebooks/write_eda_notebook.py`)
- [x] Generate `notebooks/eda.ipynb`
- [x] Step 1: Markdown intro (project overview, purpose of EDA, dataset summary, expected outputs)
- [x] Step 2: Import libraries (pandas, numpy, matplotlib, seaborn, pathlib, warnings, wordcloud, collections)
- [x] Step 3: Load dataset (shape, columns, dtypes, head, tail)
- [x] Step 4: Dataset summary (describe, missing values, duplicates, memory usage)
- [x] Step 5: Target/label analysis (bar + pie chart, percentage distribution)
- [x] Step 6: Text length analysis (word/char/sentence — histogram, boxplot, violin, density)
- [x] Step 7: Avg word length analysis (histogram + boxplot)
- [x] Step 8: Subject analysis (count plot, pie chart, top categories)
- [x] Step 9: Publication date analysis (parse dates, year/month, time series + bar charts)
- [x] Step 10: WordClouds (fake vs real, high-res PNG)
- [x] Step 11: Top 30 common words (fake vs real, horizontal bar charts)
- [x] Step 12: Top 20 bigrams (fake vs real)
- [x] Step 13: Top 20 trigrams (fake vs real)
- [x] Step 14: Correlation analysis (heatmap of numerical features)
- [x] Step 15: Outlier analysis (IQR + boxplots)
- [x] Step 16: Statistical summary (fake vs real comparison)
- [x] Step 17: Interesting insights (auto-printed observations)
- [x] Step 18: Save all charts to `outputs/charts/` (descriptive filenames)
- [x] Step 19: Generate `outputs/reports/eda_report.txt`
- [x] Step 20: Notebook validation (assert charts + report exist)

## ✅ Module 3 Success Criteria

- [x] `notebooks/eda.ipynb` generated and executes top-to-bottom
- [x] All charts saved to `outputs/charts/`
- [x] `outputs/reports/eda_report.txt` generated
- [x] No exceptions during execution

