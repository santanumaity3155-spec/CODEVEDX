# AI Chatbot for Internal Helpdesk

## Project Overview

This project implements an AI-powered chatbot for internal helpdesk support. The chatbot uses Natural Language Processing (NLP) and intent detection to answer frequently asked questions from employees about IT support, HR policies, and company procedures.

## Module 1: Project Setup + FAQ Dataset Preparation

### Objective
Prepare a high-quality FAQ dataset for training an intent classification model.

### Dataset Information

**Location:** data/raw/faq_dataset.csv (raw), data/processed/faq_dataset.csv (processed)

**Structure:**
- question: The FAQ question
- intent: The intent/category of the question
- nswer: The answer to the question
- entity: Optional entity extraction field

### Intents Covered

The dataset includes the following intents:
- greetings / goodbye / help
- password_reset / account_access
- laptop_problems / software_installation
- internet_problems / wifi_problems / email_problems
- leave_policy / attendance / working_hours / holidays
- salary_information / payroll
- employee_id / hr_support / office_location
- contact_information / security / technical_support

### Dataset Statistics
- Total records: 220+
- Unique intents: 22
- Examples per intent: 10-20
- Class balance: Well-balanced

## Setup and Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. Clone the repository:
`ash
git clone <repository-url>
cd Project-04-AI-Helpdesk-Chatbot
`

2. Create virtual environment (optional but recommended):
`ash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
`

3. Install dependencies:
`ash
pip install -r requirements.txt
`

## Dataset Preparation

### Running the Preparation Script

To prepare and validate the dataset:
`ash
python src/prepare_dataset.py
`

This script will:
1. Load the raw FAQ dataset
2. Validate data quality (missing values, duplicates, etc.)
3. Clean and normalize the data
4. Remove duplicate questions
5. Generate statistics and analysis
6. Save the processed dataset to data/processed/faq_dataset.csv`n7. Generate a dataset report at outputs/reports/dataset_report.txt`n8. Create visualization charts at outputs/charts/`n
### Output Files

After running the preparation script, you will have:

- **Processed Dataset:** data/processed/faq_dataset.csv`n- **Dataset Report:** outputs/reports/dataset_report.txt`n- **Intent Distribution Chart:** outputs/charts/intent_distribution.png`n- **Intent Distribution Pie Chart:** outputs/charts/intent_distribution_pie.png`n
## Dataset Quality

The dataset preparation includes:
- Missing value detection and handling
- Duplicate question removal
- Whitespace cleanup
- Intent label normalization
- Data validation
- Quality scoring

## Next Steps (Module 2+)

- Module 2: Intent Classification Model Training
- Module 3: TF-IDF Vectorization and Feature Extraction
- Module 4: Chatbot Prediction Engine
- Module 5: Flask API Development
- Module 6: Admin Panel and CRUD Operations

## Contributing

This is an internship project. For questions or issues, please contact the development team.

## License

Internal use only - CodeVedX Internship Program
