# Utility Usage Prediction Tool

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success)

**A production-quality Machine Learning application for predicting utility usage based on historical data**

*Developed for CodeVedX AI/ML Internship*

</div>

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Folder Structure](#folder-structure)
- [Installation](#installation)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [ML Workflow](#ml-workflow)
- [Usage Guide](#usage-guide)
- [Output Examples](#output-examples)
- [Testing](#testing)
- [Future Improvements](#future-improvements)
- [Author](#author)
- [License](#license)

---

## 🎯 Project Overview

The **Utility Usage Prediction Tool** is a comprehensive Machine Learning application designed to predict household utility usage (specifically Global Active Power in kilowatts) based on various electrical parameters and temporal features. This project demonstrates a complete ML pipeline from data preprocessing to model deployment in a user-friendly console application.

### Key Highlights

- **Modular Architecture**: Clean separation of concerns with dedicated modules for configuration, logging, validation, data handling, ML operations, and user interface
- **Production-Ready Code**: Follows PEP8 standards, includes comprehensive error handling, logging, and input validation
- **Machine Learning Pipeline**: Complete workflow from data loading to model training, evaluation, and prediction
- **User-Friendly Interface**: Professional console-based menu system for easy interaction
- **Comprehensive Testing**: Unit tests for all critical modules
- **Well-Documented**: Extensive docstrings, type hints, and detailed README

### Technology Stack

- **Language**: Python 3.12+
- **ML Framework**: Scikit-Learn
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Model Persistence**: Joblib
- **Testing**: Pytest

---

## ✨ Features

### 1. Data Management
- ✅ **Add Records**: Add new utility usage records with validated input
- ✅ **View Records**: Display all records in a formatted table
- ✅ **Search Records**: Search for specific records by ID
- ✅ **Update Records**: Modify existing records with validation
- ✅ **Delete Records**: Remove records with confirmation
- ✅ **Automatic ID Generation**: Unique IDs for each record
- ✅ **Data Backup**: Automatic backup creation before modifications

### 2. Machine Learning
- ✅ **Model Training**: Train Linear Regression models on your data
- ✅ **Model Evaluation**: Comprehensive metrics (MAE, MSE, RMSE, R² Score)
- ✅ **Model Persistence**: Save and load trained models using Joblib
- ✅ **Feature Selection**: 10 carefully selected features for prediction
- ✅ **Batch Prediction**: Make predictions on multiple records
- ✅ **Model Metrics**: View detailed model information and coefficients

### 3. Prediction & Analysis
- ✅ **Single Prediction**: Predict utility usage for custom inputs
- ✅ **Batch Predictions**: Process multiple records at once
- ✅ **Prediction Export**: Save predictions to CSV files
- ✅ **Feature Importance**: Analyze feature coefficients and impact
- ✅ **Visualization**: Generate 5 different charts for analysis

### 4. User Interface
- ✅ **Professional Menu System**: Clean, intuitive console interface
- ✅ **Input Validation**: Comprehensive validation for all user inputs
- ✅ **Error Handling**: Graceful error handling with user-friendly messages
- ✅ **Logging**: Detailed logging to file and console
- ✅ **Progress Indicators**: Visual feedback during operations

### 5. Data Visualization
- ✅ **Histogram**: Distribution of Global Active Power
- ✅ **Correlation Heatmap**: Feature correlation analysis
- ✅ **Actual vs Predicted**: Model performance visualization
- ✅ **Residual Plot**: Error analysis
- ✅ **Feature Distribution**: Box plots for all features

---

## 📁 Folder Structure

```
Project-01-Utility-Usage-Prediction/
│
├── data/
│   ├── raw/                          # Raw data files (gitignored)
│   │   └── household_power_consumption.txt
│   └── processed/                    # Processed data files
│       └── utility_usage.csv         # Main dataset
│
├── notebooks/
│   ├── data_preprocessing.ipynb      # Data preprocessing notebook
│   └── model_training.ipynb          # Model training notebook
│
├── models/
│   └── usage_prediction.pkl          # Trained model file
│
├── outputs/
│   ├── charts/                       # Visualization charts
│   │   ├── histogram_global_active_power.png
│   │   ├── correlation_heatmap.png
│   │   ├── actual_vs_predicted.png
│   │   ├── residual_plot.png
│   │   └── feature_distribution.png
│   ├── predictions/                  # Prediction outputs
│   │   └── model_predictions.csv
│   └── reports/                      # Evaluation reports
│       └── model_evaluation_report.txt
│
├── logs/
│   └── application.log               # Application logs
│
├── src/                              # Source code modules
│   ├── __init__.py                   # Package initialization
│   ├── config.py                     # Configuration settings
│   ├── logger.py                     # Logging configuration
│   ├── validation.py                 # Input validation functions
│   ├── utils.py                      # Utility functions
│   ├── data_handler.py               # Data management (CRUD)
│   ├── predictor.py                  # ML prediction module
│   └── menu.py                       # Menu system
│
├── tests/                            # Unit tests
│   ├── __init__.py
│   ├── test_validation.py            # Validation tests
│   ├── test_data_handler.py          # Data handler tests
│   └── test_predictor.py             # Predictor tests
│
├── main.py                           # Application entry point
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git ignore rules
├── LICENSE                           # MIT License
└── README.md                         # This file
```

---

## 🚀 Installation

### Prerequisites

- Python 3.12 or higher
- pip package manager
- Git (for cloning)

### Step 1: Clone the Repository

```bash
git clone https://github.com/santanumaity3155-spec/CODEVEDX.git
cd CODEVEDX/Project-01-Utility-Usage-Prediction
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -m pytest tests/ -v
```

---

## 📦 Requirements

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.12+ | Programming Language |
| NumPy | >=1.24.0 | Numerical Computing |
| Pandas | >=2.0.0 | Data Manipulation |
| Scikit-Learn | >=1.3.0 | Machine Learning |
| Joblib | >=1.3.0 | Model Persistence |
| Matplotlib | >=3.7.0 | Visualization |
| Seaborn | >=0.12.0 | Statistical Visualization |
| Pytest | >=7.4.0 | Testing Framework |

### Development Dependencies

- **Black**: Code formatting
- **Flake8**: Linting
- **Mypy**: Type checking
- **Jupyter**: Notebook support

---

## 🎮 How to Run

### Running the Application

```bash
# Make sure you're in the project directory
cd Project-01-Utility-Usage-Prediction

# Run the main application
python main.py
```

### Running the Model Training Notebook

```bash
# Start Jupyter Notebook
jupyter notebook

# Navigate to notebooks/model_training.ipynb
# Run all cells to train the model
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_validation.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

---

## 🔄 ML Workflow

### Phase 1: Data Preprocessing

**Notebook**: `notebooks/data_preprocessing.ipynb`

1. **Load Raw Data**: Load household power consumption dataset
2. **Data Exploration**: Analyze structure, missing values, duplicates
3. **Data Cleaning**:
   - Handle missing values (remove or impute)
   - Remove duplicates
   - Convert data types
4. **Feature Engineering**:
   - Extract Year, Month, Day, Hour from datetime
   - Create temporal features
5. **Save Clean Data**: Export to `data/processed/utility_usage.csv`

**Input**: `data/raw/household_power_consumption.txt`  
**Output**: `data/processed/utility_usage.csv`

### Phase 2: Model Training

**Notebook**: `notebooks/model_training.ipynb`

1. **Load Processed Data**: Read cleaned dataset
2. **Dataset Exploration**: Statistical analysis, correlation analysis
3. **Feature Selection**: Select 10 features for training
   - Global_reactive_power
   - Voltage
   - Global_intensity
   - Sub_metering_1, 2, 3
   - Year, Month, Day, Hour
4. **Target Selection**: Global_active_power (kW)
5. **Train-Test Split**: 80% training, 20% testing
6. **Model Training**: Linear Regression
7. **Model Evaluation**:
   - Mean Absolute Error (MAE)
   - Mean Squared Error (MSE)
   - Root Mean Squared Error (RMSE)
   - R² Score
8. **Save Model**: Export to `models/usage_prediction.pkl`
9. **Generate Visualizations**: 5 charts (automatically via application)
10. **Generate Reports**: Evaluation report and predictions CSV (automatically via application)

**Input**: `data/processed/utility_usage.csv`  
**Outputs**:
- `models/usage_prediction.pkl`
- `outputs/charts/*.png` (5 charts)
- `outputs/predictions/model_predictions.csv`
- `outputs/reports/model_evaluation_report.txt`

### Phase 3: Application Usage

**Application**: `main.py`

1. **Launch Application**: Run `python main.py`
2. **Menu Navigation**: Choose from 10 options
3. **Data Management**: Add, view, update, delete records
4. **Model Training**: Train model directly from application
5. **Make Predictions**: Input features and get predictions
6. **Export Results**: Save predictions and reports

---

## 📖 Usage Guide

### Main Menu Options

```
============================================================
                    UTILITY USAGE PREDICTION TOOL
                           Version 1.0.0
                    CodeVedX AI/ML Internship
============================================================

MAIN MENU
============================================================
  1. Add Record
  2. View Records
  3. Search Record
  4. Update Record
  5. Delete Record
  6. Train ML Model
  7. Predict Usage
  8. View Model Metrics
  9. Export Prediction Report
 10. Exit
============================================================
```

### Option 1: Add Record

Add a new utility usage record to the dataset.

**Input Required**:
- Global Reactive Power (kW): 0.0 - 50.0
- Voltage (V): 200.0 - 250.0
- Global Intensity (A): 0.0 - 50.0
- Sub Metering 1 (Wh): 0.0 - 100.0
- Sub Metering 2 (Wh): 0.0 - 100.0
- Sub Metering 3 (Wh): 0.0 - 100.0
- Year: 2000 - Current Year
- Month: 1 - 12
- Day: 1 - 31
- Hour: 0 - 23
- Global Active Power (kW): 0.0 - 10.0

**Output**: Record ID assigned automatically

### Option 2: View Records

Display all records in a formatted table (up to 100 records).

**Output**: Formatted table with all records

### Option 3: Search Record

Search for a specific record by ID.

**Input**: Record ID

**Output**: Detailed record information

### Option 4: Update Record

Update an existing record.

**Input**: Record ID and fields to update

**Output**: Confirmation message

### Option 5: Delete Record

Delete a record from the dataset.

**Input**: Record ID

**Output**: Confirmation message

### Option 6: Train ML Model

Train a new Linear Regression model on the dataset.

**Requirements**:
- Minimum 10 records in dataset
- Dataset must be valid

**Output**:
- Trained model saved to `models/usage_prediction.pkl`
- Training & Testing metrics (MAE, MSE, RMSE, R² Score)
- Charts generated in `outputs/charts/`
- Reports generated in `outputs/reports/`

### Option 7: Predict Usage

Make a prediction for utility usage.

**Input**: 10 feature values

**Output**:
- Predicted Global Active Power (kW)
- Option to save prediction

### Option 8: View Model Metrics

Display detailed model information.

**Output**:
- Model type and path
- Feature coefficients
- Model performance metrics
- Option to view full evaluation report

### Option 9: Export Prediction Report

Export prediction reports to CSV.

**Output**: Exported CSV file

### Option 10: Exit

Exit the application with confirmation.

---

## 📊 Output Examples

### Model Evaluation Report

```
================================================================================
                    MODEL EVALUATION REPORT
================================================================================

Model Information:
  - Model Type: Linear Regression
  - Framework: Scikit-Learn
  - Target Variable: Global_active_power (kW)
  - Features Used: 10

Dataset Information:
  - Total Samples: 2049
  - Training Samples: 1639 (80%)
  - Testing Samples: 410 (20%)
  - Features: 10

Model Performance Metrics:
  Training Set:
    - Mean Absolute Error (MAE):  0.0234 kW
    - Mean Squared Error (MSE):   0.0012
    - Root Mean Squared Error:    0.0346 kW
    - R² Score:                   0.9876

  Testing Set:
    - Mean Absolute Error (MAE):  0.0256 kW
    - Mean Squared Error (MSE):   0.0014
    - Root Mean Squared Error:    0.0374 kW
    - R² Score:                   0.9854

Model Parameters:
  - Coefficients: [0.1234, 0.0567, ...]
  - Intercept: 0.1234

Feature Importance (Coefficients):
    - Global_reactive_power   : +0.123456
    - Voltage                 : -0.234567
    - Global_intensity        : +0.345678
    - Sub_metering_1          : +0.012345
    - Sub_metering_2          : +0.023456
    - Sub_metering_3          : +0.034567
    - Year                    : +0.001234
    - Month                   : -0.002345
    - Day                     : +0.003456
    - Hour                    : -0.004567

Outputs Generated:
  - Model: utility_usage_model.pkl
  - Charts: outputs/charts/
    - histogram_global_active_power.png
    - correlation_heatmap.png
    - actual_vs_predicted.png
    - residual_plot.png
    - feature_distribution.png
  - Predictions: model_predictions.csv

================================================================================
                              END OF REPORT
================================================================================
```

### Prediction Result

```
============================================================
PREDICTION RESULT
============================================================

           Input Features
------------------------------------------------------------
Global_reactive_power        : 0.5000
Voltage                      : 220.0000
Global_intensity             : 10.0000
Sub_metering_1               : 1.0000
Sub_metering_2               : 0.0000
Sub_metering_3               : 5.0000
Year                         : 2023
Month                        : 6
Day                          : 15
Hour                         : 14

           Prediction
------------------------------------------------------------
Predicted Active Power        : 1.2345 kW
============================================================
```

---

## 🧪 Testing

### Test Structure

The project includes comprehensive unit tests for all critical modules:

- **test_validation.py**: 40+ test cases for validation functions
- **test_data_handler.py**: 20+ test cases for CRUD operations
- **test_predictor.py**: 15+ test cases for ML operations

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage report
python -m pytest tests/ --cov=src --cov-report=html

# Run specific test class
python -m pytest tests/test_validation.py::TestValidateInteger -v

# Run with detailed output
python -m pytest tests/ -v --tb=short
```

### Test Coverage

- ✅ Input validation (integers, floats, strings, menu choices)
- ✅ Domain-specific validation (voltage, intensity, power, dates)
- ✅ Data handler CRUD operations
- ✅ CSV loading and saving
- ✅ Record management (add, update, delete, search)
- ✅ Model loading and prediction
- ✅ Input validation for predictions
- ✅ Feature importance extraction
- ✅ Prediction saving

---

## 🔮 Future Improvements

### Short-term Enhancements
- [ ] Support for multiple ML algorithms (Random Forest, XGBoost, Neural Networks)
- [ ] Hyperparameter tuning with GridSearchCV
- [ ] Cross-validation for better model evaluation
- [ ] Model versioning and A/B testing
- [ ] REST API for predictions (FastAPI/Flask)
- [ ] Database integration (SQLite/PostgreSQL)
- [ ] Web interface (Streamlit/Gradio)

### Long-term Enhancements
- [ ] Real-time data ingestion
- [ ] Time series forecasting (ARIMA, LSTM)
- [ ] Anomaly detection in utility usage
- [ ] Personalized recommendations for energy saving
- [ ] Multi-user support with authentication
- [ ] Cloud deployment (AWS/Azure/GCP)
- [ ] Mobile application
- [ ] Integration with smart home devices
- [ ] Dashboard with real-time analytics
- [ ] Automated retraining pipeline

---

## 👨‍💻 Author

**CodeVedX AI/ML Internship**

- GitHub: [@santanumaity3155-spec](https://github.com/santanumaity3155-spec)
- Project Repository: [CODEVEDX](https://github.com/santanumaity3155-spec/CODEVEDX)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **CodeVedX** - For providing the internship opportunity and project framework
- **UCI Machine Learning Repository** - For the Household Power Consumption dataset
- **Scikit-Learn Team** - For the excellent ML framework
- **Python Community** - For the amazing ecosystem of data science tools

---

## 📞 Support

For questions, issues, or contributions:

1. Open an issue in the GitHub repository
2. Check the documentation and troubleshooting guide
3. Review the logs in `logs/application.log`
4. Run tests to verify installation

---

## 🎓 Learning Outcomes

This project demonstrates:

- ✅ Complete ML pipeline implementation
- ✅ Production-quality Python code
- ✅ Modular architecture design
- ✅ Comprehensive testing practices
- ✅ Professional documentation
- ✅ Error handling and logging
- ✅ Input validation and security
- ✅ Data visualization techniques
- ✅ Model evaluation and metrics
- ✅ Console application development
- ✅ Git version control

---

<div align="center">

**Built with ❤️ for CodeVedX AI/ML Internship**

*Last Updated: 2024*

</div>