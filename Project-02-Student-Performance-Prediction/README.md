# Student Performance Prediction System

![Python](https://img.shields.io/badge/Python-3.12%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-green)
![License](https://img.shields.io/badge/License-MIT-orange)

A machine learning-powered console application that predicts student exam performance based on various academic and personal factors. Built with modular architecture, comprehensive testing, and professional code quality standards.

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Dataset Description](#dataset-description)
- [Installation](#installation)
- [Requirements](#requirements)
- [Running the Application](#running-the-application)
- [Screenshots](#screenshots)
- [Machine Learning Workflow](#machine-learning-workflow)
- [Prediction Example](#prediction-example)
- [Testing](#testing)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

### Core Functionality
- **📊 Dataset Management**: View, search, add, update, and delete student records
- **🔍 Advanced Search**: Search students by multiple fields with exact or partial matching
- **🤖 ML Predictions**: Predict exam scores using trained machine learning models
- **📈 Batch Processing**: Process multiple predictions from CSV files
- **💾 Data Export**: Export datasets and predictions in CSV or JSON format
- **📋 Model Information**: View detailed model metrics and feature importances

### Technical Features
- **Modular Architecture**: Clean separation of concerns with dedicated modules
- **Comprehensive Logging**: Detailed application logs for debugging and monitoring
- **Input Validation**: Robust validation for all user inputs
- **Error Handling**: Graceful error handling with user-friendly messages
- **Unit Testing**: 85%+ test coverage with pytest
- **Type Hints**: Full type annotation for better code documentation
- **PEP 8 Compliant**: Follows Python best practices and style guidelines

## 📁 Project Structure

```
Project-02-Student-Performance-Prediction/
├── main.py                # Root-level entry point (run with: python main.py)
├── src/
│   ├── __init__.py
│   ├── config.py          # Centralized configuration with pathlib
│   ├── logger.py          # Logging configuration
│   ├── utils.py           # Utility functions
│   ├── validation.py      # Input validation
│   ├── data_handler.py    # CSV operations
│   ├── predictor.py       # ML model predictions with joblib
│   ├── menu.py            # User interface
│   └── main.py            # Application entry point
├── tests/
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_validation.py
│   ├── test_utils.py
│   ├── test_data_handler.py
│   └── test_predictor.py
├── data/
│   ├── raw/               # Raw dataset
│   └── processed/         # Processed dataset
│       └── student_performance.csv
├── models/
│   └── student_performance_model.pkl
├── outputs/
│   ├── predictions/       # Prediction results
│   ├── charts/            # Visualization charts
│   └── reports/           # Analysis reports
├── logs/
│   └── application.log    # Application logs
├── notebooks/
│   ├── data_preprocessing.ipynb
│   ├── eda.ipynb
│   └── model_training.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

## 📊 Dataset Description

The dataset contains student performance data with the following features:

### Numeric Features
- **Hours_Studied**: Number of hours spent studying (0-50)
- **Attendance**: Class attendance percentage (0-100)
- **Sleep_Hours**: Average sleep hours per day (0-24)
- **Previous_Scores**: Previous exam scores (0-100)
- **Tutoring_Sessions**: Number of tutoring sessions attended (0-10)
- **Physical_Activity**: Hours of physical activity per week (0-10)
- **Exam_Score**: Final exam score (target variable, 0-100)

### Categorical Features
- **Gender**: Male/Female
- **School_Type**: Public/Private
- **Teacher_Quality**: Low/Medium/High
- **Parental_Involvement**: Low/Medium/High
- **Access_to_Resources**: Low/Medium/High
- **Extracurricular_Activities**: Yes/No
- **Motivation_Level**: Low/Medium/High
- **Internet_Access**: Yes/No
- **Family_Income**: Low/Medium/High
- **Peer_Influence**: Negative/Neutral/Positive
- **Learning_Disabilities**: Yes/No
- **Parental_Education_Level**: High School/College/Postgraduate
- **Distance_from_Home**: Near/Moderate/Far

## 🚀 Installation

### Prerequisites
- Python 3.12 or higher
- pip package manager
- Virtual environment (recommended)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/santanumaity3155-spec/CODEVEDX.git
   cd CODEVEDX/Project-02-Student-Performance-Prediction
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 📦 Requirements

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
joblib>=1.3.0
pytest>=7.4.0
pytest-cov>=4.1.0
```

## 🎯 Running the Application

### Start the Application
```bash
python main.py
```

Or alternatively:
```bash
python src/main.py
```

### Main Menu Options

1. **View Dataset Information** - Display dataset statistics and sample records
2. **Search Student** - Search for students by various fields
3. **Add Student Record** - Add new student records to the dataset
4. **Update Student Record** - Modify existing student records
5. **Delete Student Record** - Remove student records
6. **Predict Student Performance** - Single student performance prediction
7. **Batch Prediction** - Predict performance for multiple students from CSV
8. **Export Predictions** - Export dataset or predictions to file
9. **View Model Information** - Display model details and metrics
10. **Exit** - Close the application

## 📸 Screenshots

### Main Menu
```
==============================================================================
                Student Performance Prediction System
                      Version: 1.0.0
          Machine Learning-Powered Student Performance Prediction
                          Author: CodeVedX Intern
==============================================================================

MAIN MENU
------------------------------------------------------------------------------
 1. View Dataset Information
 2. Search Student
 3. Add Student Record
 4. Update Student Record
 5. Delete Student Record
 6. Predict Student Performance
 7. Batch Prediction
 8. Export Predictions
 9. View Model Information
10. Exit
------------------------------------------------------------------------------

Enter your choice (1-10):
```

### Prediction Result
```
==============================================================================
                          PREDICTION RESULT
==============================================================================

Input Features:
--------------------------------------------------------------------------------
Hours_Studied      : 25
Attendance         : 85
Sleep_Hours        : 7
Previous_Scores    : 75
Tutoring_Sessions  : 2
Physical_Activity  : 3

--------------------------------------------------------------------------------
                        PREDICTED EXAM SCORE

                         85.50/100 (A)

--------------------------------------------------------------------------------
Interpretation: Very Good! The student is likely to perform well.
==============================================================================
```

## 🔬 Machine Learning Workflow

### 1. Data Preprocessing
- Handle missing values
- Encode categorical variables (Label Encoding)
- Feature scaling and normalization
- Feature selection and engineering
- Train-test split (80-20)

### 2. Exploratory Data Analysis (EDA)
- Statistical analysis of all features
- Correlation analysis
- Distribution visualization
- Outlier detection and handling
- Feature importance analysis

### 3. Model Training
- Algorithm: Random Forest Regressor
- Train-test split: 80-20
- Cross-validation: 5-fold
- Hyperparameter tuning using GridSearchCV
- Model evaluation using multiple metrics

### 4. Model Deployment
- Serialize trained model using joblib
- Create prediction pipeline
- Integrate with console application
- Performance monitoring and logging

### Model Performance
- **Algorithm**: Random Forest Regressor
- **R² Score**: 0.95+
- **Mean Absolute Error**: < 3.0
- **Root Mean Squared Error**: < 4.0
- **Features Used**: 6 primary features

## 💡 Prediction Example

### Single Prediction
```
Enter student information for prediction
--------------------------------------------------------------------------------

--- Required Information ---

Hours Studied (0-50): 25
Attendance (0-100): 85
Sleep Hours (0-24): 7
Previous Scores (0-100): 75
Tutoring Sessions (0-10): 2
Physical Activity (0-10): 3

Making prediction...

[Prediction Result Displayed]
```

### Batch Prediction
1. Prepare a CSV file with student data
2. Select option 7 from menu
3. Enter CSV file path
4. View results and save predictions

## 🧪 Testing

### Run All Tests
```bash
pytest tests/ -v
```

### Run Tests with Coverage
```bash
pytest tests/ --cov=src --cov-report=html
```

### Run Specific Test File
```bash
pytest tests/test_validation.py -v
```

### Test Coverage
- **Config Module**: 100%
- **Validation Module**: 95%+
- **Utils Module**: 90%+
- **Data Handler**: 85%+
- **Predictor**: 85%+

**Overall Coverage**: 85%+

## 🚀 Future Improvements

### Short Term
- [ ] Add GUI interface using Tkinter or PyQt
- [ ] Implement database storage (SQLite/PostgreSQL)
- [ ] Add more ML algorithms for comparison
- [ ] Generate detailed prediction reports with visualizations
- [ ] Add student performance trends over time

### Medium Term
- [ ] Web application using Flask/FastAPI
- [ ] User authentication and role-based access
- [ ] Real-time prediction API
- [ ] Integration with student information systems
- [ ] Automated model retraining pipeline

### Long Term
- [ ] Deep learning models for improved accuracy
- [ ] Recommendation system for student improvement
- [ ] Mobile application
- [ ] Cloud deployment with auto-scaling
- [ ] Multi-language support

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Code Standards
- Follow PEP 8 style guide
- Add type hints to all functions
- Write docstrings for all modules, classes, and functions
- Include unit tests for new features
- Maintain 85%+ test coverage

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**CodeVedX Intern**
- GitHub: [@santanumaity3155-spec](https://github.com/santanumaity3155-spec)

## 🙏 Acknowledgments

- CodeVedX for the internship opportunity
- Scikit-learn team for the excellent ML library
- The open-source community for inspiration and tools

---

## 📞 Support

For support, please open an issue in the GitHub repository or contact the development team.

**Last Updated**: 2024
**Version**: 1.0.0