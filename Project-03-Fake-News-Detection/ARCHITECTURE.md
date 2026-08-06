# SYSTEM ARCHITECTURE — AI Based Fake News Detection Tool

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERACTION                          │
│                    (Console / Terminal)                          │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    main.py (Entry Point)                         │
│              - Initializes paths and imports                     │
│              - Starts the application                            │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    src/menu.py (MenuSystem)                      │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Option 1: View Dataset Information                       │  │
│  │  Option 2: Predict News (Single)                          │  │
│  │  Option 3: Batch Prediction                               │  │
│  │  Option 4: Export Predictions                             │  │
│  │  Option 5: View Model Information                         │  │
│  │  Option 6: View Prediction History                        │  │
│  │  Option 7: Clear Prediction History                       │  │
│  │  Option 8: System Information                             │  │
│  │  Option 9: Help                                           │  │
│  │  Option 10: Exit                                          │  │
│  └───────────────────────────────────────────────────────────┘  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│   Predictor  │   │ Data Handler │   │  Validation  │
│   Engine     │   │              │   │   Module     │
│(predictor.py)│   │(data_handler)│   │(validation)  │
└──────┬───────┘   └──────┬───────┘   └──────┬───────┘
       │                   │                   │
       │                   │                   │
  ┌────┴────┐         ┌────┴────┐         ┌────┴────┐
  │         │         │         │         │         │
  ▼         ▼         ▼         ▼         ▼         ▼
┌────┐   ┌────┐   ┌────┐   ┌────┐   ┌────┐   ┌────┐
│Model│   │Text │   │CSV │   │File│   │Menu│   │Text│
│Load │   │Clean│   │Load│   │Path│   │Chc │   │Val │
└────┘   └────┘   └────┘   └────┘   └────┘   └────┘
       │         │         │         │         │
       └─────────┼─────────┼─────────┼─────────┘
                 │         │         │
                 ▼         ▼         ▼
           ┌─────────────────────────────┐
           │   src/config.py             │
           │  - Paths & Constants        │
           │  - Model Info               │
           │  - Validation Settings      │
           └─────────────┬───────────────┘
                         │
                         ▼
           ┌─────────────────────────────┐
           │   src/logger.py             │
           │  - Dual Logging             │
           │  - Console + File           │
           └─────────────────────────────┘
```

---

## Detailed Component Architecture

### 1. Entry Point Layer

#### `main.py`
**Purpose**: Application entry point and initialization

**Responsibilities**:
- Add src directory to Python path
- Import and call main() from src.main
- Handle import errors gracefully
- Exit with appropriate status code

**Dependencies**: src.main

---

### 2. Core Application Layer

#### `src/main.py`
**Purpose**: Main application controller

**Responsibilities**:
- Initialize application
- Start menu system
- Handle global exceptions
- Coordinate shutdown
- Log application lifecycle events

**Dependencies**: src.menu, src.logger

---

#### `src/menu.py` (MenuSystem)
**Purpose**: Interactive console interface

**Responsibilities**:
- Display main menu (10 options)
- Handle user input and validation
- Route to appropriate handlers
- Manage application state (running/stopped)
- Display formatted output with banners
- Handle keyboard interrupts gracefully

**Key Methods**:
- `display_main_menu()` - Show menu and get choice
- `run()` - Main application loop
- `handle_choice()` - Route to handler
- `view_dataset_info()` - Option 1
- `predict_news()` - Option 2
- `batch_prediction()` - Option 3
- `export_predictions()` - Option 4
- `view_model_info()` - Option 5
- `view_prediction_history()` - Option 6
- `clear_prediction_history()` - Option 7
- `system_information()` - Option 8
- `show_help()` - Option 9
- `exit_application()` - Option 10

**Dependencies**: src.config, src.logger, src.utils, src.validation, src.data_handler, src.predictor

---

### 3. Business Logic Layer

#### `src/predictor.py` (NewsPredictor)
**Purpose**: ML prediction engine

**Responsibilities**:
- Load and manage model and vectorizer
- Clean and preprocess text
- Make single predictions
- Make batch predictions
- Calculate confidence scores
- Track processing time
- Provide model information

**Key Methods**:
- `_load_model()` - Load Random Forest model
- `_load_vectorizer()` - Load TF-IDF vectorizer
- `clean_text()` - Text preprocessing
- `predict()` - Single prediction
- `predict_batch()` - Batch prediction
- `get_model_info()` - Model metadata
- `is_ready()` - Check if model is loaded

**Data Flow**:
```
Input Text → Clean Text → Vectorize → Predict → Confidence Scores → Result
```

**Dependencies**: src.config, src.logger, joblib, numpy

---

#### `src/data_handler.py` (DataHandler)
**Purpose**: Data operations and persistence

**Responsibilities**:
- Load and cache dataset
- Manage prediction history
- Save/load predictions
- Export to multiple formats (CSV, JSON, TXT)
- Load CSV for batch prediction
- Provide dataset information

**Key Methods**:
- `get_dataset_info()` - Dataset statistics
- `load_csv_for_batch()` - Load batch CSV
- `save_predictions()` - Save batch results
- `load_prediction_history()` - Load history
- `save_prediction_history()` - Persist history
- `add_prediction_to_history()` - Add single record
- `clear_prediction_history()` - Delete all history
- `get_prediction_history()` - Get limited history
- `export_predictions()` - Export in multiple formats

**Dependencies**: src.config, src.logger, pandas

---

### 4. Utility Layer

#### `src/validation.py`
**Purpose**: Input validation and sanitization

**Responsibilities**:
- Validate text input (length, content)
- Validate file paths (existence, permissions)
- Validate CSV files (format, columns)
- Validate menu choices (range, type)
- Validate export formats
- Validate history limits

**Key Functions**:
- `validate_text_input()` - Text validation
- `validate_file_path()` - File path validation
- `validate_csv_file()` - CSV validation
- `validate_text_column()` - Column validation
- `validate_model_file()` - Model file validation
- `validate_dataset()` - Dataset validation
- `validate_menu_choice()` - Menu input validation
- `validate_export_format()` - Export format validation
- `validate_history_limit()` - History limit validation

**Dependencies**: src.config

---

#### `src/utils.py`
**Purpose**: Helper functions and utilities

**Responsibilities**:
- Console UI utilities (clear screen, banners)
- Formatting (numbers, percentages, timestamps)
- System information gathering
- Memory usage monitoring
- User input helpers
- Progress bar display

**Key Functions**:
- `clear_screen()` - Clear console
- `pause()` - Wait for user input
- `print_banner()` - Print ASCII banner
- `format_number()` - Format with commas
- `format_percentage()` - Format as percentage
- `format_timestamp()` - Format datetime
- `truncate_text()` - Truncate with ellipsis
- `confirm_action()` - Get y/n confirmation
- `get_system_info()` - System information
- `get_memory_usage()` - Memory statistics
- `display_progress_bar()` - Progress indicator

**Dependencies**: src.config, psutil

---

#### `src/logger.py`
**Purpose**: Logging configuration

**Responsibilities**:
- Configure dual logging (console + file)
- Set log levels and formatters
- Create log directory
- Prevent duplicate handlers
- Log application lifecycle

**Configuration**:
- **File Handler**: Logs to `logs/application.log`
- **Console Handler**: Logs to stdout
- **Log Level**: INFO
- **Format**: Timestamp - Level - Message

**Dependencies**: src.config

---

#### `src/config.py`
**Purpose**: Centralized configuration

**Responsibilities**:
- Define all paths (data, models, outputs, logs)
- Set application constants
- Define model information
- Set validation limits
- Create required directories

**Key Constants**:
- `PROJECT_ROOT` - Base directory
- `DATA_DIR`, `MODEL_DIR`, `OUTPUT_DIR`, `LOG_DIR`
- `MODEL_PATH`, `VECTORIZER_PATH`
- `PREDICTIONS_DIR`, `REPORTS_DIR`, `CHARTS_DIR`
- `APP_NAME`, `APP_VERSION`
- `MIN_TEXT_LENGTH`, `MAX_TEXT_LENGTH`
- `MODEL_INFO` - Model metadata

**Dependencies**: pathlib, os

---

### 5. Data Layer

#### Model Files
- **`models/fake_news_model.pkl`** - Trained Random Forest model
  - Size: ~500 MB
  - Type: RandomForestClassifier
  - Features: 20,000 TF-IDF features
  - Accuracy: 99.68%

- **`models/tfidf_vectorizer.pkl`** - Fitted TF-IDF vectorizer
  - Size: ~50 MB
  - Type: TfidfVectorizer
  - Vocabulary: 20,000 features
  - N-grams: (1, 2)

#### Dataset Files
- **`data/raw/Fake.csv`** - Raw fake news articles
  - Rows: 21,417
  - Columns: 4 (title, text, subject, date)

- **`data/raw/True.csv`** - Raw real news articles
  - Rows: 21,454
  - Columns: 4 (title, text, subject, date)

- **`data/processed/fake_news_dataset.csv`** - Combined dataset
  - Rows: 43,971
  - Columns: 10 (includes clean_text, text_length, word_count, label)

#### Output Files
- **`outputs/predictions/prediction_history.csv`** - Prediction history
- **`outputs/reports/*.txt`** - Analysis reports
- **`outputs/charts/*.png`** - Visualization charts

---

## Data Flow Diagrams

### Single Prediction Flow

```
┌─────────────┐
│   User      │
│  Input      │
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│  Menu System     │
│  (Option 2)      │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  Validation      │
│  - Length check  │
│  - Content check │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  Text Cleaning   │
│  - Lowercase     │
│  - Remove URLs   │
│  - Remove special│
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  TF-IDF Vector   │
│  - Transform     │
│  - 20K features  │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  Random Forest   │
│  - Predict       │
│  - Probabilities │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  Result          │
│  - Prediction    │
│  - Confidence    │
│  - Probabilities │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  Display Result  │
│  Save to History │
│  Log Event       │
└──────────────────┘
```

### Batch Prediction Flow

```
┌─────────────┐
│  CSV File   │
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│  Load CSV        │
│  - Validate      │
│  - Parse         │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  Extract Texts   │
│  - Fill NA       │
│  - Convert to str│
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  For Each Text:  │
│  ┌─────────────┐ │
│  │ Clean Text  │ │
│  └──────┬──────┘ │
│         │        │
│  ┌──────▼──────┐ │
│  │ Vectorize   │ │
│  └──────┬──────┘ │
│         │        │
│  ┌──────▼──────┐ │
│  │ Predict     │ │
│  └──────┬──────┘ │
│         │        │
│  ┌──────▼──────┐ │
│  │ Collect     │ │
│  │ Result      │ │
│  └─────────────┘ │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  Create Results  │
│  DataFrame       │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  Save to CSV     │
│  Add to History  │
│  Display Summary │
└──────────────────┘
```

---

## Module Dependencies

```
┌─────────────────────────────────────────────────────────────┐
│                      Dependencies Graph                      │
└─────────────────────────────────────────────────────────────┘

main.py
  └─> src/main.py
        └─> src/menu.py
              ├─> src/config.py
              ├─> src/logger.py
              ├─> src/utils.py
              ├─> src/validation.py
              ├─> src/data_handler.py
              │     └─> src/config.py
              │     └─> src/logger.py
              └─> src/predictor.py
                    ├─> src/config.py
                    └─> src/logger.py

src/logger.py
  └─> src/config.py

src/utils.py
  └─> src/config.py

src/validation.py
  └─> src/config.py

src/data_handler.py
  ├─> src/config.py
  └─> src/logger.py

src/predictor.py
  ├─> src/config.py
  └─> src/logger.py
```

---

## Technology Stack

### Core Technologies
- **Language**: Python 3.12+
- **ML Framework**: scikit-learn 1.4.0+
- **NLP**: NLTK 3.8.0+
- **Data Processing**: pandas 2.2.0+, numpy 1.26.0+
- **Model Serialization**: joblib 1.3.0+

### Visualization
- **Charts**: matplotlib 3.8.0+, seaborn 0.13.0+
- **Word Clouds**: wordcloud 1.9.0+

### System Monitoring
- **Performance**: psutil 5.9.0+

### Testing
- **Framework**: pytest 8.0.0+
- **Coverage**: pytest-cov 4.1.0+

### Notebooks
- **Environment**: Jupyter 1.0.0+, notebook 7.0.0+

---

## Design Patterns

### 1. Singleton Pattern
- **Usage**: Global instances of predictor, data_handler, logger
- **Benefit**: Single instance throughout application lifecycle

### 2. Factory Pattern
- **Usage**: Dynamic handler selection in menu system
- **Benefit**: Easy addition of new menu options

### 3. Strategy Pattern
- **Usage**: Multiple export formats (CSV, JSON, TXT)
- **Benefit**: Easy addition of new export formats

### 4. Template Method Pattern
- **Usage**: Test suite structure
- **Benefit**: Consistent test execution and reporting

### 5. Facade Pattern
- **Usage**: Menu system simplifies complex subsystems
- **Benefit**: Clean interface for users

---

## Security Considerations

### Input Validation
- All user inputs validated before processing
- File paths checked for existence and permissions
- Text length limits enforced (20-50,000 characters)
- CSV files validated for format and columns

### Error Handling
- Comprehensive try-except blocks
- Graceful degradation on errors
- Detailed logging for debugging
- User-friendly error messages

### Data Safety
- No sensitive data in repository
- Model files excluded from Git
- Prediction history managed with size limits
- Safe file operations with error handling

---

## Performance Considerations

### Memory Management
- Model loaded once at startup
- Dataset cached in memory
- History limited to 10,000 records
- Efficient batch processing

### Speed Optimization
- TF-IDF vectorization pre-computed
- Batch predictions for multiple texts
- Minimal file I/O operations
- Optimized text cleaning regex

### Scalability
- Batch processing supports large files
- History management prevents unbounded growth
- Modular design allows easy extension
- Configurable limits and thresholds

---

## Deployment Architecture

### Development
```
Developer Machine
├── Python 3.12+
├── Virtual Environment
├── All dependencies installed
├── Model files in models/
└── Run: python main.py
```

### Production
```
Production Server
├── Python 3.12+
├── Virtual Environment
├── Dependencies from requirements.txt
├── Model files deployed
├── Logs directory writable
├── Outputs directory writable
└── Run: python main.py
```

### Future: Containerized
```
Docker Container
├── Base: python:3.12-slim
├── Dependencies installed
├── Model files included
├── Exposed ports (if web interface added)
└── CMD: python main.py
```

---

## Monitoring & Logging

### Application Logs
- **Location**: `logs/application.log`
- **Format**: Timestamp - Level - Message
- **Levels**: INFO, WARNING, ERROR, CRITICAL
- **Rotation**: Manual (can be automated)

### Logged Events
- Application startup/shutdown
- Model loading success/failure
- Predictions made (with confidence)
- Errors and exceptions
- User actions (menu selections)
- Data operations (load, save, export)

### Performance Metrics
- Prediction time per article
- Batch processing time
- Memory usage (RSS, VMS)
- System information
- Model loading time

---

## Extension Points

### Adding New Menu Options
1. Add handler method in `MenuSystem` class
2. Add entry to `handlers` dictionary in `handle_choice()`
3. Implement business logic
4. Add validation if needed

### Adding New Export Formats
1. Add format case in `export_predictions()` method
2. Implement format-specific logic
3. Update validation in `validate_export_format()`

### Adding New Models
1. Train model in notebook
2. Save with joblib
3. Update `MODEL_INFO` in config.py
4. Update model loading logic if needed

### Adding New Visualizations
1. Create visualization in EDA notebook
2. Save to `outputs/charts/`
3. Update documentation

---

**Architecture Version**: 1.0.0  
**Last Updated**: August 2024  
**Status**: Production Ready