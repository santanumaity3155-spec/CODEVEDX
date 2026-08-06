# AI Based Fake News Detection Tool

![Python](https://img.shields.io/badge/Python-3.12%2B-blue)
![Status](https://img.shields.io/badge/Status-Production%20Ready-green)
![License](https://img.shields.io/badge/License-MIT-orange)
![Tests](https://img.shields.io/badge/Tests-48%2F48%20Passing-brightgreen)

A machine learning and NLP-powered console application designed to automatically detect fake news articles. This tool uses a Random Forest classifier trained on 43,971 news articles with TF-IDF vectorization to achieve 99.68% accuracy in classifying news as Fake or Real.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Features](#features)
- [Architecture](#architecture)
- [Folder Structure](#folder-structure)
- [Dataset Description](#dataset-description)
- [Model Performance](#model-performance)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Console Screenshots](#console-screenshots)
- [Outputs](#outputs)
- [Testing](#testing)
- [Future Improvements](#future-improvements)
- [License](#license)
- [Author](#author)
- [Acknowledgements](#acknowledgements)

---

## 📌 Project Overview

The **AI Based Fake News Detection Tool** is a production-ready machine learning application that classifies news articles as **Fake** or **Real** using natural language processing (NLP) and advanced machine learning techniques. By analyzing the textual content of news articles, the application provides confidence-based predictions to help users assess the credibility of news.

### Key Highlights
- ✅ **99.68% Accuracy** with Random Forest Classifier
- ✅ **43,971 articles** in training dataset
- ✅ **20,000 vocabulary** TF-IDF features
- ✅ **48/48 tests passing** (100% success rate)
- ✅ **Production-ready** with comprehensive logging and error handling
- ✅ **Interactive console** with 10 menu options

---

## 🎯 Objectives

- Build a high-accuracy machine learning model that classifies news articles as **Fake** or **Real**
- Apply NLP techniques (TF-IDF, text cleaning) to extract meaningful patterns from news text
- Provide an easy-to-use console interface for single and batch predictions
- Generate insightful visualizations and reports during EDA
- Follow professional software engineering practices: modular architecture, logging, validation, type hints, and testing
- Deliver an internship-ready, GitHub-ready project with comprehensive documentation

---

## ✨ Features

### Core Functionality
- **📊 Dataset Information**: View comprehensive dataset statistics and class distribution
- **🔮 Single Prediction**: Predict whether a news article is fake or real with confidence scores
- **📦 Batch Prediction**: Process multiple articles from CSV files
- **💾 Export Predictions**: Export results in CSV, JSON, or TXT formats
- **📈 Model Information**: View detailed model performance metrics and statistics
- **📜 Prediction History**: Track and review past predictions
- **🗑️ Clear History**: Manage prediction history storage
- **💻 System Information**: Monitor system resources and application status
- **❓ Help & Guide**: Comprehensive user guide and troubleshooting

### Technical Features
- **Modular Architecture**: Clean separation of concerns with dedicated modules
- **Comprehensive Logging**: Dual logging to console and file with timestamps
- **Input Validation**: Robust validation for all user inputs and files
- **Error Handling**: Graceful error handling with user-friendly messages
- **Type Hints**: Full type annotation for better code maintainability
- **PEP 8 Compliant**: Follows Python coding standards
- **Performance Optimized**: Efficient batch processing and memory management

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Console Application                       │
│                    (Interactive Menu)                        │
└───────────────────────┬─────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│   Predictor  │ │ Data Handler │ │  Validation  │
│   Engine     │ │              │ │   Module     │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                │
       ▼                ▼                ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│   Model      │ │   Dataset    │ │   Config     │
│  (Random     │ │   & History  │ │   & Logger   │
│   Forest)    │ │              │ │              │
└──────────────┘ └──────────────┘ └──────────────┘
       │                │                │
       └────────────────┼────────────────┘
                        │
                        ▼
              ┌──────────────────┐
              │  TF-IDF           │
              │  Vectorizer       │
              └──────────────────┘
```

### Data Flow
1. **Input**: User provides text via console or CSV file
2. **Validation**: Input is validated for length, format, and content
3. **Preprocessing**: Text is cleaned (lowercase, remove URLs, special chars)
4. **Vectorization**: Text is converted to TF-IDF features
5. **Prediction**: Model predicts class and confidence scores
6. **Output**: Results displayed to user and saved to history
7. **Export**: User can export predictions in multiple formats

---

## 📁 Folder Structure

```
Project-03-Fake-News-Detection/
│
├── data/
│   ├── raw/                      # Raw datasets
│   │   ├── Fake.csv              # Fake news articles (21,417 rows)
│   │   └── True.csv              # Real news articles (21,454 rows)
│   └── processed/                # Processed datasets
│       └── fake_news_dataset.csv # Combined dataset (43,971 rows)
│
├── notebooks/                    # Jupyter notebooks
│   ├── data_preprocessing.ipynb  # Data preprocessing notebook
│   ├── eda.ipynb                 # Exploratory Data Analysis
│   └── model_training.ipynb      # Model training and evaluation
│
├── models/                       # Trained model artifacts
│   ├── fake_news_model.pkl       # Random Forest model (~500 MB)
│   └── tfidf_vectorizer.pkl      # TF-IDF vectorizer (~50 MB)
│
├── outputs/
│   ├── charts/                   # Generated visualizations (42 charts)
│   │   ├── wordcloud_fake.png
│   │   ├── wordcloud_real.png
│   │   ├── confusion_matrix_rf.png
│   │   ├── roc_curve.png
│   │   └── ... (38 more charts)
│   ├── predictions/              # Prediction results
│   │   └── prediction_history.csv
│   └── reports/                  # Analysis reports
│       ├── preprocessing_report.txt
│       ├── eda_report.txt
│       └── model_report.txt
│
├── logs/                         # Application logs
│   └── application.log
│
├── src/                          # Python source package
│   ├── __init__.py
│   ├── config.py                 # Configuration and constants
│   ├── logger.py                 # Logging configuration
│   ├── utils.py                  # Utility functions
│   ├── validation.py             # Input validation
│   ├── data_handler.py           # Data operations
│   ├── predictor.py              # Prediction engine
│   └── menu.py                   # Console menu system
│
├── tests/                        # Unit tests
│
├── main.py                       # Application entry point
├── test_module6.py               # Module 6 comprehensive tests
├── README.md                     # This file
├── PROJECT_SUMMARY.md            # Internship project summary
├── CHANGELOG.md                  # Version history
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
└── LICENSE                       # MIT License
```

---

## 📊 Dataset Description

The dataset is sourced from the **Fake and real news dataset** (Kaggle) and contains news articles from various sources.

### Dataset Statistics
- **Total Articles**: 43,971 (21,417 fake, 21,454 real)
- **Columns**: 10 (title, text, subject, date, and preprocessing features)
- **Class Distribution**: Balanced (50.3% fake, 49.7% real)
- **Sources**: Multiple news outlets and fact-checking websites

### Column Description
- **title**: Headline of the news article
- **text**: Full body text of the article
- **subject**: Category/topic of the article
- **date**: Publication date
- **clean_text**: Preprocessed and cleaned text
- **text_length**: Character count of article
- **word_count**: Number of words in article
- **label**: Binary label (0 = Real, 1 = Fake)

### Preprocessing Steps
1. Text cleaning (lowercase, remove URLs, emails, special characters)
2. Stop word removal
3. Tokenization
4. Train/test split (80/20, stratified)
5. TF-IDF vectorization (20,000 features, 1-2 n-grams)

---

## 🎯 Model Performance

### Best Model: Random Forest Classifier

| Metric | Score |
|--------|-------|
| **Accuracy** | 99.68% |
| **Precision** | 99.48% |
| **Recall** | 99.86% |
| **F1 Score** | 99.67% |
| **ROC-AUC** | 99.99% |

### Cross-Validation Results
- **5-Fold CV Accuracy**: 99.69% ± 0.08%
- **Training Samples**: 35,176
- **Test Samples**: 8,795
- **Misclassified**: 28 out of 8,795 (0.32%)

### Model Configuration
- **Algorithm**: Random Forest Classifier
- **Vectorizer**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Vocabulary Size**: 20,000 features
- **N-gram Range**: (1, 2) - unigrams and bigrams
- **Training Date**: 2024

---

## 🛠️ Technology Stack

| Category | Technology | Version |
|----------|-----------|---------|
| **Language** | Python | 3.12+ |
| **Data Manipulation** | pandas, numpy | 2.2.0+, 1.26.0+ |
| **NLP** | NLTK | 3.8.0+ |
| **Machine Learning** | scikit-learn | 1.4.0+ |
| **Model Persistence** | joblib | 1.3.0+ |
| **Visualization** | matplotlib, seaborn, wordcloud | 3.8.0+, 0.13.0+, 1.9.0+ |
| **Data Export** | openpyxl | 3.1.0+ |
| **Testing** | pytest | 8.0.0+ |
| **Notebooks** | Jupyter | 1.0.0+ |
| **System Monitoring** | psutil | 5.9.0+ |

---

## 🚀 Installation

### Prerequisites
- Python 3.12 or higher
- pip package manager
- Virtual environment (recommended)
- 4GB RAM minimum (8GB recommended for model loading)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/santanumaity3155-spec/CODEVEDX.git
   cd CODEVEDX/Project-03-Fake-News-Detection
   ```

2. **Create and activate a virtual environment**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate
   
   # Linux / macOS
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python test_module6.py
   ```

---

## ▶️ Usage

### Starting the Application

```bash
python main.py
```

### Menu Options

1. **View Dataset Information** - Display dataset statistics and class distribution
2. **Predict News** - Single article prediction (paste text or load from file)
3. **Batch Prediction** - Process multiple articles from CSV
4. **Export Predictions** - Export results in CSV, JSON, or TXT format
5. **View Model Information** - Display model performance metrics
6. **View Prediction History** - Review past predictions
7. **Clear Prediction History** - Delete all prediction records
8. **System Information** - View system specs and resource usage
9. **Help** - User guide and troubleshooting
10. **Exit** - Close the application

### Single Prediction Example

```
Enter your choice (1-10): 2

Choose input method:
  1. Paste text directly
  2. Load from .txt file
  0. Back to main menu

Enter choice (0-2): 1

Enter or paste the news text below:
(Press Ctrl+D or Ctrl+Z on a new line when done)

Scientists have confirmed that drinking two cups of coffee every hour 
can make people live up to 200 years old...

======================================================================
PREDICTION RESULT
======================================================================

  ⚠️  PREDICTION: FAKE NEWS
  !********************************************************************!

  Confidence Score     : 97.00%
  Probability (Fake)   : 97.00%
  Probability (Real)   : 3.00%
  Processing Time      : 0.1299 seconds
  Input Length         : 245 characters
  Timestamp            : 2026-08-07 00:01:50
```

### Batch Prediction Example

```bash
# Prepare a CSV file with a 'text' column
# Then select Option 3 from the menu

Enter path to CSV file: data/batch_articles.csv

Total articles to predict: 100
Proceed with batch prediction? (y/n): y

Starting batch prediction...
======================================================================
BATCH PREDICTION RESULTS
======================================================================

  Total Processed    : 100
  Successful         : 100
  Failed             : 0
  Success Rate       : 100.00%

  ✓ Predictions saved to:
    outputs/predictions/batch_predictions_20260807_000031.csv
```

---

## 📸 Console Screenshots

### Main Menu
```
======================================================================
        AI Based Fake News Detection Tool
              Version 1.0.0
======================================================================

  1. View Dataset Information
  2. Predict News
  3. Batch Prediction
  4. Export Predictions
  5. View Model Information
  6. View Prediction History
  7. Clear Prediction History
  8. System Information
  9. Help
 10. Exit

======================================================================
Enter your choice (1-10):
```

### Model Information
```
======================================================================
              MODEL INFORMATION
======================================================================

MODEL DETAILS
----------------------------------------------------------------------
  Model Name           : Random Forest
  Algorithm            : Random Forest Classifier
  Model Type           : RandomForestClassifier
  Training Dataset     : Fake News Dataset (Kaggle)
  Training Date        : 2024

TRAINING STATISTICS
----------------------------------------------------------------------
  Training Samples     : 35,176
  Test Samples         : 8,795
  Vocabulary Size      : 20,000
  N-gram Range         : (1, 2)

PERFORMANCE METRICS
----------------------------------------------------------------------
  Accuracy             : 99.68%
  Precision            : 99.48%
  Recall               : 99.86%
  F1 Score             : 99.67%
  ROC-AUC              : 99.99%
```

---

## 📊 Outputs

### Generated Reports
- **preprocessing_report.txt** - Data preprocessing statistics
- **eda_report.txt** - Exploratory data analysis findings
- **model_report.txt** - Model training and evaluation results
- **test_report.txt** - Comprehensive test results (48 tests)
- **performance_report.txt** - System and model performance metrics

### Generated Charts (42 total)
- **Word Clouds**: Most frequent words in fake vs real news
- **Confusion Matrices**: Model performance visualization
- **ROC/PR Curves**: Classification threshold analysis
- **Feature Importance**: Top predictive words/n-grams
- **Distribution Plots**: Text length, subject, date analysis
- **Model Comparison**: Accuracy, F1, precision, recall comparison

### Prediction Exports
- **CSV Format**: Spreadsheet-compatible with all prediction data
- **JSON Format**: Structured data for web applications
- **TXT Format**: Human-readable formatted report

---

## 🧪 Testing

### Test Suite Overview
- **Total Tests**: 48
- **Passed**: 48 (100%)
- **Failed**: 0 (0%)
- **Execution Time**: ~4 seconds

### Test Categories
1. **Model Loading** (5 tests) - Model and vectorizer verification
2. **Single Prediction** (5 tests) - Fake and real news classification
3. **Batch Prediction** (4 tests) - CSV processing and batch results
4. **Prediction History** (3 tests) - History tracking and retrieval
5. **Export Predictions** (3 tests) - CSV, JSON, TXT export
6. **Clear History** (2 tests) - History management
7. **Edge Cases** (5 tests) - Empty text, special chars, long text
8. **Dataset Info** (4 tests) - Dataset statistics and validation
9. **Validation Functions** (6 tests) - Input validation
10. **Performance** (4 tests) - Speed and memory usage
11. **Model Reload** (4 tests) - Model persistence verification

### Running Tests
```bash
python test_module6.py
```

### Test Output
```
======================================================================
TEST SUMMARY
======================================================================
Total Tests: 48
Passed: 48
Failed: 0
Success Rate: 100.00%
Execution Time: 4.14 seconds
```

---

## 🔮 Future Improvements

### Model Enhancements
- [ ] Implement deep learning models (BERT, RoBERTa, LSTM)
- [ ] Add word embeddings (Word2Vec, GloVe)
- [ ] Incorporate article metadata (subject, date, source)
- [ ] Implement ensemble methods with multiple classifiers
- [ ] Add model retraining pipeline with new data

### Application Features
- [ ] Web interface using Flask or FastAPI
- [ ] REST API for programmatic access
- [ ] Database integration for prediction history
- [ ] User authentication and prediction tracking
- [ ] Real-time news feed integration
- [ ] Multi-language support
- [ ] Browser extension for fact-checking

### Technical Improvements
- [ ] CI/CD pipeline with GitHub Actions
- [ ] Docker containerization
- [ ] Model versioning and A/B testing
- [ ] Advanced caching for faster predictions
- [ ] Distributed processing for large batches
- [ ] Automated model monitoring and drift detection

---

## 📝 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**CodeVedX Intern** - AI/ML Track
- GitHub: [@santanumaity3155-spec](https://github.com/santanumaity3155-spec)
- Project: CodeVedX Internship Program 2024

---

## 🙏 Acknowledgements

- **Dataset**: Fake and real news dataset from Kaggle
- **CodeVedX**: Internship program and mentorship
- **scikit-learn**: Machine learning framework
- **NLTK**: Natural language processing toolkit
- **Open Source Community**: For invaluable tools and libraries

---

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the Help section (Option 9) in the application
- Review logs in `logs/application.log`
- Consult the troubleshooting guide in the Help menu

---

**Project Status**: ✅ **Module 6 Complete - Production Ready**  
**Last Updated**: August 2024  
**Version**: 1.0.0  
**Test Coverage**: 100% (48/48 tests passing)