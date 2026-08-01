# AI Based Fake News Detection Tool

![Python](https://img.shields.io/badge/Python-3.12%2B-blue)
![Status](https://img.shields.io/badge/Status-Foundation%20Ready-yellow)
![License](https://img.shields.io/badge/License-MIT-orange)

A machine learning and NLP-powered console application designed to automatically detect fake news articles. This repository currently contains the **Module 1 — Project Foundation**: a professional, GitHub-ready project structure with verified datasets and core documentation. Subsequent modules will add preprocessing, EDA, model training, prediction, and the console interface.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Features (Planned)](#features-planned)
- [Folder Structure](#folder-structure)
- [Dataset Description](#dataset-description)
- [Technology Stack](#technology-stack)
- [Future Modules](#future-modules)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Screenshots](#screenshots)
- [License](#license)
- [Author](#author)

---

## 📌 Project Overview

The **AI Based Fake News Detection Tool** aims to classify news articles as **Fake** or **True** using natural language processing (NLP) and machine learning techniques. By analyzing the textual content of news articles, the application will provide a confidence-based prediction to help users assess the credibility of news.

The project is developed as part of the **CodeVedX Internship Program** and is built incrementally across multiple modules. This repository represents **Module 1**, which establishes a clean, professional foundation for all future development.

---

## 🎯 Objectives

- Build a machine learning model that classifies news articles as **Fake** or **True**.
- Apply NLP techniques to extract meaningful patterns from news text.
- Provide an easy-to-use console interface for single and batch predictions.
- Generate insightful visualizations and reports during EDA.
- Follow professional software engineering practices: modular architecture, logging, validation, type hints, and testing.
- Deliver an internship-ready, GitHub-ready project with comprehensive documentation.

---

## ✨ Features (Planned)

> ⚠️ **Note:** The features below are *planned* for upcoming modules. They are **not yet implemented** in this foundation stage.

### Planned Core Functionality
- **📄 Data Preprocessing**: Text cleaning, tokenization, stop-word removal, and feature engineering.
- **📊 Exploratory Data Analysis (EDA)**: Statistical summaries, distribution plots, word clouds, and correlation analysis.
- **🤖 Model Training**: Training and evaluating ML classifiers (e.g., Logistic Regression, Naive Bayes, Random Forest) using TF-IDF vectorization.
- **🔮 Prediction Engine**: Single-article and batch prediction with confidence scores.
- **🖥️ Console Application**: Interactive menu for loading data, training, predicting, and exporting results.
- **📈 Reports & Charts**: Automated generation of charts and analysis reports.
- **🧪 Testing Suite**: Unit tests for data handling, validation, and prediction logic.

### Planned Technical Features
- Modular architecture with clear separation of concerns.
- Comprehensive logging and error handling.
- Full type hints and PEP 8 compliance.
- Unit testing with `pytest`.
- Reproducible environment via `requirements.txt`.

---

## 📁 Folder Structure

```
Project-03-Fake-News-Detection/
│
├── data/
│   ├── raw/                 # Raw dataset (Fake.csv, True.csv)
│   └── processed/           # Processed datasets (future modules)
│
├── notebooks/               # Jupyter notebooks for EDA, preprocessing, and training
│
├── models/                  # Trained model artifacts (future modules)
│
├── outputs/
│   ├── charts/              # Generated visualizations
│   ├── reports/             # Analysis reports
│   └── predictions/         # Prediction results
│
├── logs/                    # Application logs
│
├── src/                     # Python source package
│   └── __init__.py
│
├── tests/                   # Unit tests
│
├── README.md                # Project documentation
├── PROJECT_SUMMARY.md       # Internship project summary
├── requirements.txt         # Python dependencies
├── .gitignore               # Git ignore rules
└── main.py                  # Application entry point (future modules)
```

---

## 📊 Dataset Description

The dataset is sourced from the **Fake and real news dataset** and is located in `data/raw/`.

| File       | Description                                        | Format |
|------------|----------------------------------------------------|--------|
| `Fake.csv` | News articles labeled as **Fake** (1)               | CSV    |
| `True.csv` | News articles labeled as **True** (0)               | CSV    |

### Expected Columns
- **title**: Headline of the news article.
- **text**: Full body text of the article.
- **subject**: Category/topic of the article.
- **date**: Publication date.

> **Note:** In Module 1, the datasets are only **verified for presence** and are **not modified or preprocessed**. Preprocessing belongs to a later module.

---

## 🛠️ Technology Stack

| Category            | Technology                            |
|---------------------|---------------------------------------|
| Language            | Python 3.12+                          |
| Data Manipulation   | pandas, numpy                         |
| NLP                 | NLTK                                  |
| Machine Learning    | scikit-learn                          |
| Model Persistence   | joblib                                |
| Visualization       | matplotlib, seaborn, wordcloud        |
| Data Export         | openpyxl                              |
| Testing             | pytest                                |
| Notebooks           | Jupyter Notebook                     |

---

## 🧭 Future Modules

- **Module 2 — Data Preprocessing**: Text cleaning, normalization, and train/test splits.
- **Module 3 — Exploratory Data Analysis (EDA)**: Visualization and statistical analysis of the dataset.
- **Module 4 — Model Training**: TF-IDF vectorization and classifier training/evaluation.
- **Module 5 — Prediction & Application**: Console menu, single/batch prediction, and reporting.
- **Module 6 — Testing & Polish**: Unit tests, documentation review, and final packaging.

---

## 🚀 Installation

### Prerequisites
- Python 3.12 or higher
- `pip` package manager
- Virtual environment (recommended)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/santanumaity3155-spec/CODEVEDX.git
   cd CODEVEDX/Project-03-Fake-News-Detection
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # Linux / macOS
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ How to Run

> **Placeholder** — The application entry point and console interface will be implemented in a later module.

Once the console application is implemented, the expected usage will be:

```bash
python main.py
```

---

## 📸 Screenshots

> **Placeholder** — Screenshots will be added here after the console application and visualizations are implemented in future modules.

---

## 📝 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## 👨‍💻 Author

**CodeVedX Intern**
- GitHub: [@santanumaity3155-spec](https://github.com/santanumaity3155-spec)

---

**Project Status**: Module 1 (Foundation) ✅ — Preprocessing, EDA, and model training are planned for subsequent modules.

