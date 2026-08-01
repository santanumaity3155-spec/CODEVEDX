# PROJECT SUMMARY — AI Based Fake News Detection Tool

---

## 📌 Project Name

**AI Based Fake News Detection Tool**

---

## 🎓 Internship Name

**CodeVedX Internship Program — AI/ML Track**

---

## 🎯 Project Goal

To build a **machine learning and NLP-powered console application** that automatically detects whether a news article is **Fake** or **True**. The application will classify articles based on their textual content and provide confidence-based predictions to help users assess news credibility.

---

## 📍 Current Status

**Module 1 — Project Foundation: ✅ COMPLETE**

| Deliverable                    | Status     |
|--------------------------------|------------|
| Folder structure verified      | ✅ Done     |
| Dataset presence verified      | ✅ Done     |
| `README.md` created            | ✅ Done     |
| `PROJECT_SUMMARY.md` created   | ✅ Done     |
| `requirements.txt` created     | ✅ Done     |
| `.gitignore` created           | ✅ Done     |
| `src/__init__.py` created      | ✅ Done     |

**Note:** No preprocessing, EDA, model training, prediction logic, or application code is included in Module 1. These are planned for subsequent modules.

---

## 📊 Dataset Information

| Item          | Detail                                                    |
|---------------|-----------------------------------------------------------|
| Location      | `data/raw/`                                               |
| Files         | `Fake.csv`, `True.csv`                                    |
| Format        | CSV                                                       |
| Label Mapping | `Fake` → 1, `True` → 0 (planned for later modules)        |
| Module 1 Rule | Presence verified **only** — files are **not modified**    |

### Expected Columns (to be confirmed during EDA)
- **title**: Headline of the article
- **text**: Full body text of the article
- **subject**: Category/topic
- **date**: Publication date

---

## 📁 Folder Structure

```
Project-03-Fake-News-Detection/
│
├── data/
│   ├── raw/                 # Raw datasets (Fake.csv, True.csv)
│   └── processed/           # Processed datasets (future modules)
│
├── notebooks/               # Jupyter notebooks (future modules)
│
├── models/                  # Trained models (future modules)
│
├── outputs/
│   ├── charts/              # Generated charts
│   ├── reports/             # Analysis reports
│   └── predictions/         # Prediction results
│
├── logs/                    # Application logs
│
├── src/                     # Python source package
│   └── __init__.py
│
├── tests/                   # Unit tests (future modules)
│
├── README.md                # Project documentation
├── PROJECT_SUMMARY.md       # This summary
├── requirements.txt         # Python dependencies
├── .gitignore               # Git ignore rules
└── main.py                  # Application entry point (future modules)
```

---

## 🛣️ Development Roadmap

| Phase  | Module                         | Description                                                  |
|--------|--------------------------------|--------------------------------------------------------------|
| ✅      | **Module 1 — Foundation**      | Project structure, dataset verification, documentation, deps  |
| ⬜      | **Module 2 — Preprocessing**   | Text cleaning, tokenization, stop-word removal, train/test    |
| ⬜      | **Module 3 — EDA**             | Statistics, visualizations, word clouds, insights             |
| ⬜      | **Module 4 — Model Training**  | TF-IDF vectorization, classifier training, evaluation         |
| ⬜      | **Module 5 — Application**     | Console menu, single/batch prediction, reporting, charts      |
| ⬜      | **Module 6 — Testing & Polish**| Unit tests, coverage, final documentation review              |

---

## 🧩 Module Breakdown

### Module 1 — Project Foundation (CURRENT)
- Verify and create the folder structure.
- Verify dataset presence (`Fake.csv`, `True.csv`).
- Create professional project documentation (`README.md`, `PROJECT_SUMMARY.md`).
- Define dependencies (`requirements.txt`).
- Add repository hygiene (`.gitignore`).
- Initialize the source package (`src/__init__.py`).

### Module 2 — Data Preprocessing *(planned)*
- Load and inspect the datasets.
- Clean text (lowercasing, punctuation removal, etc.).
- Remove stop words and tokenize.
- Build the combined labeled dataset and train/test split.

### Module 3 — Exploratory Data Analysis *(planned)*
- Statistical summaries and class distribution.
- Word clouds for fake vs. true articles.
- Subject/topic distribution and date analysis.
- Correlation and feature analysis.

### Module 4 — Model Training *(planned)*
- TF-IDF vectorization.
- Train classifiers (e.g., Logistic Regression, Naive Bayes, Random Forest).
- Evaluate with accuracy, precision, recall, F1, and confusion matrix.
- Save the best model with `joblib`.

### Module 5 — Prediction & Application *(planned)*
- Console menu with data, training, prediction, and export options.
- Single and batch prediction with confidence scores.
- Generate charts and reports into `outputs/`.

### Module 6 — Testing & Polish *(planned)*
- Unit tests for data handling, validation, and prediction.
- Coverage report and quality review.
- Final documentation and GitHub readiness check.

---

## 🔮 Future Work

- Add more advanced NLP techniques (TF-IDF n-grams, word embeddings).
- Experiment with ensemble and deep learning models.
- Add a web interface or API wrapper.
- Incorporate article metadata (subject, date) into modeling.
- Implement automated model retraining and evaluation pipelines.
- Add CI/CD integration for testing and linting.

---

## 👨‍💻 Author

**CodeVedX Intern** — [@santanumaity3155-spec](https://github.com/santanumaity3155-spec)

**Version:** 1.0.0  
**Last Updated:** 2024  
**Status:** Module 1 Complete — Awaiting Module 2

