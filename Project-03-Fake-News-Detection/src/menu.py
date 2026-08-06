"""
Menu system for the Fake News Detection Tool.
Provides interactive console interface with all application features.
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

from config import (
    APP_NAME, APP_VERSION, PREDICTIONS_DIR, 
    MIN_TEXT_LENGTH, MAX_TEXT_LENGTH
)
from logger import logger
from utils import (
    clear_screen, pause, print_banner, print_section_header,
    format_number, format_percentage, format_file_size,
    format_timestamp, truncate_text, confirm_action, get_user_input,
    display_progress_bar, get_system_info, get_memory_usage
)
from validation import (
    validate_text_input, validate_file_path, validate_csv_file,
    validate_text_column, validate_menu_choice, validate_export_format,
    validate_history_limit
)
from data_handler import data_handler
from predictor import predictor


class MenuSystem:
    """Main menu system for the application."""
    
    def __init__(self):
        """Initialize the menu system."""
        self.running = False
        self.data_handler = data_handler
        self.predictor = predictor
    
    def display_main_menu(self):
        """Display the main menu and get user choice."""
        clear_screen()
        print_banner(APP_NAME, f"Version {APP_VERSION}")
        
        print("  1. View Dataset Information")
        print("  2. Predict News")
        print("  3. Batch Prediction")
        print("  4. Export Predictions")
        print("  5. View Model Information")
        print("  6. View Prediction History")
        print("  7. Clear Prediction History")
        print("  8. System Information")
        print("  9. Help")
        print(" 10. Exit")
        
        print("\n" + "=" * 70)
        choice = input("Enter your choice (1-10): ").strip()
        
        is_valid, value, message = validate_menu_choice(choice, 1, 10)
        if not is_valid:
            print(f"\n❌ {message}")
            pause()
            return None
        
        return value
    
    def run(self):
        """Run the main application loop."""
        self.running = True
        
        # Check if model is ready
        if not self.predictor.is_ready():
            print_banner("WARNING")
            print("⚠️  Model or vectorizer not loaded properly!")
            print(f"Error: {self.predictor.load_error}")
            print("\nPlease ensure the model files exist in the models/ directory.")
            pause()
            return
        
        logger.info("Application started successfully")
        
        while self.running:
            try:
                choice = self.display_main_menu()
                
                if choice is None:
                    continue
                
                self.handle_choice(choice)
                
            except KeyboardInterrupt:
                print("\n\n⚠️  Interrupted by user")
                if confirm_action("Do you want to exit?"):
                    self.running = False
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {str(e)}", exc_info=True)
                print(f"\n❌ An unexpected error occurred: {str(e)}")
                pause()
        
        self.shutdown()
    
    def handle_choice(self, choice: int):
        """
        Handle user menu choice.
        
        Args:
            choice: Menu choice number
        """
        handlers = {
            1: self.view_dataset_info,
            2: self.predict_news,
            3: self.batch_prediction,
            4: self.export_predictions,
            5: self.view_model_info,
            6: self.view_prediction_history,
            7: self.clear_prediction_history,
            8: self.system_information,
            9: self.show_help,
            10: self.exit_application
        }
        
        handler = handlers.get(choice)
        if handler:
            try:
                handler()
            except Exception as e:
                logger.error(f"Error handling menu choice {choice}: {str(e)}", exc_info=True)
                print(f"\n❌ Error: {str(e)}")
                pause()
        else:
            print(f"\n❌ Invalid choice: {choice}")
            pause()
    
    def view_dataset_info(self):
        """Option 1: Display dataset information."""
        try:
            print_banner("DATASET INFORMATION")
            
            info = self.data_handler.get_dataset_info()
            
            if "error" in info:
                print(f"❌ {info['error']}")
                logger.warning(f"Dataset info error: {info['error']}")
                pause()
                return
            
            print(f"Dataset Name      : {info['name']}")
            print(f"Location          : {info['location']}")
            print(f"\nDimensions:")
            print(f"  Rows            : {format_number(info['rows'])}")
            print(f"  Columns         : {format_number(info['columns'])}")
            print(f"  Memory Usage    : {info['memory_usage_mb']} MB")
            
            print(f"\nClass Distribution:")
            if info['class_distribution']:
                total = sum(info['class_distribution'].values())
                for label, count in info['class_distribution'].items():
                    pct = format_percentage(count / total)
                    label_name = "FAKE NEWS" if label == 0 else "REAL NEWS"
                    print(f"  {label_name:15s} : {format_number(count)} ({pct})")
            else:
                print("  No label information available")
            
            print(f"\nVocabulary Size   : {info['vocabulary_size']}")
            print(f"Column Names      : {', '.join(info['column_names'][:5])}")
            if len(info['column_names']) > 5:
                print(f"                    ... and {len(info['column_names']) - 5} more")
            
            logger.info("Dataset information viewed")
            pause()
            
        except Exception as e:
            logger.error(f"Error viewing dataset info: {str(e)}")
            print(f"\n❌ Error: {str(e)}")
            pause()
    
    def predict_news(self):
        """Option 2: Predict single news article."""
        try:
            print_banner("PREDICT NEWS")
            
            print("Choose input method:")
            print("  1. Paste text directly")
            print("  2. Load from .txt file")
            print("  0. Back to main menu")
            
            method_choice = input("\nEnter choice (0-2): ").strip()
            is_valid, method, _ = validate_menu_choice(method_choice, 0, 2)
            
            if not is_valid:
                print("\n❌ Invalid choice")
                pause()
                return
            
            if method == 0:
                return
            
            text = ""
            
            if method == 1:
                # Direct text input
                print("\n" + "-" * 70)
                print("Enter or paste the news text below:")
                print("(Press Ctrl+D or Ctrl+Z on a new line when done)")
                print("-" * 70)
                
                lines = []
                try:
                    while True:
                        line = input()
                        lines.append(line)
                except EOFError:
                    pass
                
                text = "\n".join(lines)
                
            elif method == 2:
                # Load from file
                print("\n" + "-" * 70)
                file_path = input("Enter path to .txt file: ").strip().strip('"')
                print("-" * 70)
                
                is_valid, message = validate_file_path(file_path)
                if not is_valid:
                    print(f"\n❌ {message}")
                    pause()
                    return
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    print(f"✓ File loaded successfully")
                except Exception as e:
                    print(f"\n❌ Error reading file: {str(e)}")
                    pause()
                    return
            
            # Validate text
            is_valid, message = validate_text_input(text)
            if not is_valid:
                print(f"\n❌ {message}")
                print(f"   Minimum length: {MIN_TEXT_LENGTH} characters")
                print(f"   Maximum length: {format_number(MAX_TEXT_LENGTH)} characters")
                pause()
                return
            
            # Make prediction
            print("\n" + "-" * 70)
            print("Processing...")
            print("-" * 70)
            
            result = self.predictor.predict(text)
            
            if not result['success']:
                print(f"\n❌ Prediction failed: {result['error']}")
                logger.error(f"Prediction failed: {result['error']}")
                pause()
                return
            
            # Display results
            print_banner("PREDICTION RESULT")
            
            # Prediction with visual indicator
            if result['prediction'] == "FAKE NEWS":
                print("  ⚠️  PREDICTION: FAKE NEWS")
                print("  " + "!" * 68)
            else:
                print("  ✓  PREDICTION: REAL NEWS")
                print("  " + "=" * 68)
            
            print(f"\n  Confidence Score     : {format_percentage(result['confidence'])}")
            print(f"  Probability (Fake)   : {format_percentage(result['probability_fake'])}")
            print(f"  Probability (Real)   : {format_percentage(result['probability_real'])}")
            print(f"  Processing Time      : {result['processing_time']:.4f} seconds")
            print(f"  Input Length         : {format_number(result['input_length'])} characters")
            print(f"  Timestamp            : {format_timestamp()}")
            
            # Add to history
            history_data = {
                'timestamp': result['timestamp'],
                'input_length': result['input_length'],
                'prediction': result['prediction'],
                'confidence': f"{result['confidence']:.4f}",
                'probability_fake': f"{result['probability_fake']:.4f}",
                'probability_real': f"{result['probability_real']:.4f}"
            }
            
            self.data_handler.add_prediction_to_history(history_data)
            logger.info(f"Single prediction completed: {result['prediction']}")
            
            print("\n" + "=" * 70)
            pause()
            
        except Exception as e:
            logger.error(f"Error in predict_news: {str(e)}", exc_info=True)
            print(f"\n❌ Error: {str(e)}")
            pause()
    
    def batch_prediction(self):
        """Option 3: Batch prediction from CSV file."""
        try:
            print_banner("BATCH PREDICTION")
            
            print("This feature will predict multiple news articles from a CSV file.")
            print("\nRequirements:")
            print("  - CSV file must contain a text column with news articles")
            print("  - Supported formats: .csv")
            print("\n" + "=" * 70)
            
            # Get file path
            file_path = input("\nEnter path to CSV file: ").strip().strip('"')
            
            if not file_path:
                print("\n❌ File path cannot be empty")
                pause()
                return
            
            # Validate CSV file
            is_valid, message = validate_csv_file(file_path)
            if not is_valid:
                print(f"\n❌ {message}")
                pause()
                return
            
            print(f"\n✓ {message}")
            
            # Load CSV
            df, load_message = self.data_handler.load_csv_for_batch(file_path)
            if df is None:
                print(f"\n❌ {load_message}")
                pause()
                return
            
            # Display columns and ask for text column
            print(f"\nAvailable columns: {', '.join(df.columns.tolist())}")
            text_column = input("\nEnter the name of the text column: ").strip()
            
            is_valid, message = validate_text_column(file_path, text_column)
            if not is_valid:
                print(f"\n❌ {message}")
                pause()
                return
            
            print(f"\n✓ {message}")
            
            # Confirm batch size
            print(f"\nTotal articles to predict: {format_number(len(df))}")
            if not confirm_action("Proceed with batch prediction?"):
                print("\n⚠️  Batch prediction cancelled")
                pause()
                return
            
            # Perform batch prediction
            print("\n" + "-" * 70)
            print("Starting batch prediction...")
            print("-" * 70)
            
            texts = df[text_column].fillna("").astype(str).tolist()
            results = self.predictor.predict_batch(texts)
            
            # Create results DataFrame
            results_df = df.copy()
            results_df['prediction'] = [r.get('prediction', 'ERROR') for r in results]
            results_df['confidence'] = [r.get('confidence', 0.0) for r in results]
            results_df['probability_fake'] = [r.get('probability_fake', 0.0) for r in results]
            results_df['probability_real'] = [r.get('probability_real', 0.0) for r in results]
            results_df['timestamp'] = [r.get('timestamp', '') for r in results]
            
            # Save predictions
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"batch_predictions_{timestamp}.csv"
            success, message = self.data_handler.save_predictions(results_df, filename)
            
            print("\n" + "=" * 70)
            print("BATCH PREDICTION RESULTS")
            print("=" * 70)
            
            successful = sum(1 for r in results if r['success'])
            failed = len(results) - successful
            
            print(f"  Total Processed    : {format_number(len(results))}")
            print(f"  Successful         : {format_number(successful)}")
            print(f"  Failed             : {format_number(failed)}")
            print(f"  Success Rate       : {format_percentage(successful / len(results))}")
            
            if success:
                print(f"\n  ✓ Predictions saved to:")
                print(f"    {message.split(': ')[1]}")
            
            # Show sample results
            print("\n" + "-" * 70)
            print("SAMPLE RESULTS (First 5):")
            print("-" * 70)
            
            for idx, result in enumerate(results[:5], 1):
                if result['success']:
                    print(f"\n{idx}. {result['prediction']} (Confidence: {format_percentage(result['confidence'])})")
                    print(f"   Text: {truncate_text(result.get('input_text', ''), 80)}")
            
            # Add all to history
            print("\n" + "-" * 70)
            print("Saving to prediction history...")
            for result in results:
                if result['success']:
                    history_data = {
                        'timestamp': result['timestamp'],
                        'input_length': result.get('input_length', 0),
                        'prediction': result['prediction'],
                        'confidence': f"{result['confidence']:.4f}",
                        'probability_fake': f"{result['probability_fake']:.4f}",
                        'probability_real': f"{result['probability_real']:.4f}"
                    }
                    self.data_handler.add_prediction_to_history(history_data)
            
            logger.info(f"Batch prediction completed: {successful}/{len(results)} successful")
            print("\n" + "=" * 70)
            pause()
            
        except Exception as e:
            logger.error(f"Error in batch_prediction: {str(e)}", exc_info=True)
            print(f"\n❌ Error: {str(e)}")
            pause()
    
    def export_predictions(self):
        """Option 4: Export predictions to file."""
        try:
            print_banner("EXPORT PREDICTIONS")
            
            # Load prediction history
            history_df = self.data_handler.load_prediction_history()
            
            if len(history_df) == 0:
                print("\n❌ No prediction history available to export")
                print("   Make some predictions first.")
                pause()
                return
            
            print(f"Total predictions available: {format_number(len(history_df))}")
            print("\nSupported export formats:")
            print("  1. CSV")
            print("  2. JSON")
            print("  3. TXT (Human-readable)")
            print("  0. Back to main menu")
            
            format_choice = input("\nEnter format choice (0-3): ").strip()
            is_valid, format_num, _ = validate_menu_choice(format_choice, 0, 3)
            
            if not is_valid or format_num == 0:
                return
            
            format_map = {1: 'csv', 2: 'json', 3: 'txt'}
            format_type = format_map[format_num]
            
            # Ask for custom path or use default
            print(f"\nDefault location: {PREDICTIONS_DIR}")
            custom_path = input("Enter custom output path (or press Enter for default): ").strip().strip('"')
            
            output_path = None
            if custom_path:
                output_path = Path(custom_path)
            
            # Export
            print("\n" + "-" * 70)
            print("Exporting predictions...")
            print("-" * 70)
            
            success, message = self.data_handler.export_predictions(
                history_df, format_type, output_path
            )
            
            if success:
                print(f"\n✓ {message}")
                logger.info(f"Predictions exported: {format_type} format")
            else:
                print(f"\n❌ {message}")
                logger.error(f"Export failed: {message}")
            
            pause()
            
        except Exception as e:
            logger.error(f"Error in export_predictions: {str(e)}", exc_info=True)
            print(f"\n❌ Error: {str(e)}")
            pause()
    
    def view_model_info(self):
        """Option 5: Display model information."""
        try:
            print_banner("MODEL INFORMATION")
            
            info = self.predictor.get_model_info()
            
            print("MODEL DETAILS")
            print("-" * 70)
            print(f"  Model Name           : {info['name']}")
            print(f"  Algorithm            : {info['algorithm']}")
            print(f"  Model Type           : {info.get('model_type', 'N/A')}")
            print(f"  Training Dataset     : {info['training_dataset']}")
            print(f"  Training Date        : {info['training_date']}")
            
            print("\nTRAINING STATISTICS")
            print("-" * 70)
            print(f"  Training Samples     : {format_number(info['training_samples'])}")
            print(f"  Test Samples         : {format_number(info['test_samples'])}")
            print(f"  Vocabulary Size      : {format_number(info['vocabulary_size'])}")
            print(f"  N-gram Range         : {info['ngram_range']}")
            
            print("\nPERFORMANCE METRICS")
            print("-" * 70)
            print(f"  Accuracy             : {format_percentage(info['accuracy'])}")
            print(f"  Precision            : {format_percentage(info['precision'])}")
            print(f"  Recall               : {format_percentage(info['recall'])}")
            print(f"  F1 Score             : {format_percentage(info['f1_score'])}")
            print(f"  ROC-AUC              : {format_percentage(info['roc_auc'])}")
            
            print("\nVECTORIZER INFORMATION")
            print("-" * 70)
            print(f"  Type                 : {info['vectorizer']}")
            print(f"  Vocabulary Size      : {format_number(info['vocabulary_size'])}")
            
            print("\nFILE INFORMATION")
            print("-" * 70)
            print(f"  Model File           : {info.get('model_file_size', 'N/A')}")
            print(f"  Vectorizer File      : {info.get('vectorizer_file_size', 'N/A')}")
            
            print("\nRUNTIME STATUS")
            print("-" * 70)
            print(f"  Model Loaded         : {'✓ Yes' if info['model_loaded'] else '✗ No'}")
            print(f"  Vectorizer Loaded    : {'✓ Yes' if info['vectorizer_loaded'] else '✗ No'}")
            print(f"  Ready to Predict     : {'✓ Yes' if info['ready'] else '✗ No'}")
            
            if not info['ready']:
                print(f"\n  ⚠️  Error: {info.get('load_error', 'Unknown')}")
            
            logger.info("Model information viewed")
            pause()
            
        except Exception as e:
            logger.error(f"Error viewing model info: {str(e)}")
            print(f"\n❌ Error: {str(e)}")
            pause()
    
    def view_prediction_history(self):
        """Option 6: View prediction history."""
        try:
            print_banner("PREDICTION HISTORY")
            
            print("Display options:")
            print("  1. Last 10 predictions")
            print("  2. Last 50 predictions")
            print("  3. All predictions")
            print("  0. Back to main menu")
            
            choice = input("\nEnter choice (0-3): ").strip()
            is_valid, choice_num, _ = validate_menu_choice(choice, 0, 3)
            
            if not is_valid or choice_num == 0:
                return
            
            limit_map = {1: 10, 2: 50, 3: 10000}
            limit = limit_map[choice_num]
            
            history_df = self.data_handler.get_prediction_history(limit)
            
            if len(history_df) == 0:
                print("\n❌ No prediction history available")
                print("   Make some predictions first.")
                pause()
                return
            
            print(f"\nShowing {min(len(history_df), limit)} most recent predictions:")
            print("=" * 70)
            
            # Display history in a formatted table
            for idx, row in history_df.iterrows():
                print(f"\n[{idx + 1}] {row.get('timestamp', 'N/A')}")
                print(f"    Prediction    : {row.get('prediction', 'N/A')}")
                print(f"    Confidence    : {format_percentage(float(row.get('confidence', 0)))}")
                print(f"    Input Length  : {format_number(int(row.get('input_length', 0)))} chars")
                print("-" * 70)
            
            print(f"\nTotal records shown: {len(history_df)}")
            logger.info(f"Prediction history viewed: {len(history_df)} records")
            pause()
            
        except Exception as e:
            logger.error(f"Error viewing prediction history: {str(e)}")
            print(f"\n❌ Error: {str(e)}")
            pause()
    
    def clear_prediction_history(self):
        """Option 7: Clear prediction history."""
        try:
            print_banner("CLEAR PREDICTION HISTORY")
            
            history_df = self.data_handler.load_prediction_history()
            record_count = len(history_df)
            
            if record_count == 0:
                print("\n❌ No prediction history to clear")
                pause()
                return
            
            print(f"\n⚠️  WARNING: This will permanently delete {format_number(record_count)} prediction records!")
            print("   This action cannot be undone.")
            
            if not confirm_action("Are you sure you want to clear all prediction history?"):
                print("\n⚠️  Operation cancelled")
                pause()
                return
            
            print("\nClearing prediction history...")
            success = self.data_handler.clear_prediction_history()
            
            if success:
                print(f"\n✓ Prediction history cleared successfully")
                print(f"  Deleted {format_number(record_count)} records")
                logger.info(f"Prediction history cleared: {record_count} records deleted")
            else:
                print("\n❌ Failed to clear prediction history")
                logger.error("Failed to clear prediction history")
            
            pause()
            
        except Exception as e:
            logger.error(f"Error clearing prediction history: {str(e)}")
            print(f"\n❌ Error: {str(e)}")
            pause()
    
    def system_information(self):
        """Option 8: Display system information."""
        try:
            print_banner("SYSTEM INFORMATION")
            
            # Get system info
            sys_info = get_system_info()
            mem_info = get_memory_usage()
            
            print("APPLICATION INFORMATION")
            print("-" * 70)
            print(f"  Application Name     : {APP_NAME}")
            print(f"  Version              : {APP_VERSION}")
            print(f"  Working Directory    : {sys_info.get('working_directory', 'N/A')}")
            
            print("\nPYTHON ENVIRONMENT")
            print("-" * 70)
            print(f"  Python Version       : {sys_info.get('python_version', 'N/A')}")
            print(f"  Platform             : {sys_info.get('platform', 'N/A')}")
            print(f"  Platform Version     : {sys_info.get('platform_version', 'N/A')}")
            print(f"  Architecture         : {sys_info.get('architecture', 'N/A')}")
            print(f"  Processor            : {sys_info.get('processor', 'N/A')}")
            
            print("\nMEMORY USAGE")
            print("-" * 70)
            print(f"  RSS Memory           : {mem_info['rss_mb']} MB")
            print(f"  VMS Memory           : {mem_info['vms_mb']} MB")
            print(f"  Memory Percent       : {mem_info['percent']}%")
            
            print("\nMODEL STATUS")
            print("-" * 70)
            status = self.predictor.get_status()
            print(f"  Model Status         : {'✓ Loaded' if status['model_loaded'] else '✗ Not Loaded'}")
            print(f"  Vectorizer Status    : {'✓ Loaded' if status['vectorizer_loaded'] else '✗ Not Loaded'}")
            print(f"  Ready to Predict     : {'✓ Yes' if status['ready'] else '✗ No'}")
            
            if status['load_error']:
                print(f"  ⚠️  Error             : {status['load_error']}")
            
            print("\nDIRECTORY STRUCTURE")
            print("-" * 70)
            from config import (
                PROJECT_ROOT, DATA_DIR, MODEL_DIR, OUTPUT_DIR, 
                PREDICTIONS_DIR, LOG_DIR
            )
            print(f"  Project Root         : {PROJECT_ROOT}")
            print(f"  Data Directory       : {DATA_DIR}")
            print(f"  Model Directory      : {MODEL_DIR}")
            print(f"  Output Directory     : {OUTPUT_DIR}")
            print(f"  Predictions Directory: {PREDICTIONS_DIR}")
            print(f"  Log Directory        : {LOG_DIR}")
            
            logger.info("System information viewed")
            pause()
            
        except Exception as e:
            logger.error(f"Error viewing system information: {str(e)}")
            print(f"\n❌ Error: {str(e)}")
            pause()
    
    def show_help(self):
        """Option 9: Display help information."""
        try:
            print_banner("HELP & USER GUIDE")
            
            print("APPLICATION GUIDE")
            print("-" * 70)
            print("""
This application uses machine learning to detect fake news articles.

HOW TO USE:
1. Start by viewing dataset information (Option 1)
2. Make predictions using Option 2 (single) or Option 3 (batch)
3. Export results using Option 4
4. View history and model info as needed

WORKFLOW:
  Predict News → View Results → Export Data → Analyze History
""")
            
            print("SUPPORTED FILE FORMATS")
            print("-" * 70)
            print("""
Input Formats:
  • .txt  - Plain text files for single prediction
  • .csv  - CSV files for batch prediction

Output Formats:
  • .csv  - Comma-separated values (recommended for analysis)
  • .json - JSON format (for web applications)
  • .txt  - Human-readable text format
""")
            
            print("PREDICTION TIPS")
            print("-" * 70)
            print("""
For Best Results:
  • Use complete news articles (minimum 20 characters)
  • Include full context and details
  • Avoid extremely short headlines only
  • Ensure text is in English

Understanding Results:
  • Confidence: Overall certainty of the prediction
  • Probability: Breakdown of fake vs real likelihood
  • Higher confidence (>90%) indicates more reliable predictions
""")
            
            print("TROUBLESHOOTING")
            print("-" * 70)
            print("""
Common Issues:

1. "Model not loaded" error:
   → Ensure models/fake_news_model.pkl exists
   → Ensure models/tfidf_vectorizer.pkl exists
   → Check logs/application.log for details

2. "Text too short" error:
   → Minimum text length is 20 characters
   → Provide complete news article, not just headlines

3. "File not found" error:
   → Check file path is correct
   → Ensure file exists and is accessible
   → Use absolute paths if relative paths fail

4. "CSV column not found" error:
   → Check column name matches exactly (case-sensitive)
   → View available columns in error message
   → Ensure CSV has header row

5. Application crashes:
   → Check logs/application.log for error details
   → Ensure all dependencies are installed
   → Restart the application
""")
            
            print("NEED MORE HELP?")
            print("-" * 70)
            print("""
Check the following files for more information:
  • README.md - Project documentation
  • outputs/reports/model_report.txt - Model performance details
  • logs/application.log - Application logs
""")
            
            logger.info("Help information viewed")
            pause()
            
        except Exception as e:
            logger.error(f"Error showing help: {str(e)}")
            print(f"\n❌ Error: {str(e)}")
            pause()
    
    def exit_application(self):
        """Option 10: Exit the application gracefully."""
        try:
            print_banner("EXIT APPLICATION")
            
            print("Thank you for using the Fake News Detection Tool!")
            print("\nSummary:")
            
            # Show session statistics
            history_df = self.data_handler.load_prediction_history()
            print(f"  Total predictions made: {format_number(len(history_df))}")
            
            if len(history_df) > 0:
                fake_count = len(history_df[history_df['prediction'] == 'FAKE NEWS'])
                real_count = len(history_df[history_df['prediction'] == 'REAL NEWS'])
                print(f"  Fake news detected    : {format_number(fake_count)}")
                print(f"  Real news detected    : {format_number(real_count)}")
            
            print(f"\n  Logs saved to         : logs/application.log")
            print(f"  Predictions saved to  : outputs/predictions/")
            
            print("\n" + "=" * 70)
            print("Goodbye!".center(70))
            print("=" * 70)
            
            logger.info("Application shutdown initiated")
            self.running = False
            
        except Exception as e:
            logger.error(f"Error during exit: {str(e)}")
            print(f"\n❌ Error during exit: {str(e)}")
            self.running = False
    
    def shutdown(self):
        """Perform cleanup and shutdown."""
        try:
            logger.info("=" * 70)
            logger.info("APPLICATION SHUTDOWN")
            logger.info(f"Shutdown Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("=" * 70)
            
            # Close any open resources
            # (pandas DataFrames will be garbage collected)
            
        except Exception as e:
            print(f"Error during shutdown: {str(e)}")


def show_menu():
    """
    Main entry point for the menu system.
    This function is called from main.py
    """
    try:
        menu = MenuSystem()
        menu.run()
    except Exception as e:
        logger.critical(f"Fatal error in menu system: {str(e)}", exc_info=True)
        print(f"\n❌ Fatal error: {str(e)}")
        print("Please check logs/application.log for details.")
        sys.exit(1)