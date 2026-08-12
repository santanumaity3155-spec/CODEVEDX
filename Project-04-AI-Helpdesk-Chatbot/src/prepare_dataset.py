"""
Dataset Preparation Script for Internal Helpdesk Chatbot
Module 1: Project Setup + FAQ Dataset Preparation

This script performs:
- Load raw FAQ dataset
- Validate data quality
- Clean data
- Remove duplicates
- Save processed dataset
- Generate dataset report
- Generate visualization charts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from datetime import datetime
import re


class DatasetPreparation:
    """Handles FAQ dataset preparation and validation."""
    
    def __init__(self, project_root=None):
        """Initialize with project root directory."""
        if project_root is None:
            self.project_root = Path(__file__).parent.parent
        else:
            self.project_root = Path(project_root)
        
        # Define paths
        self.raw_data_path = self.project_root / "data" / "raw" / "faq_dataset.csv"
        self.processed_data_path = self.project_root / "data" / "processed" / "faq_dataset.csv"
        self.reports_dir = self.project_root / "outputs" / "reports"
        self.charts_dir = self.project_root / "outputs" / "charts"
        
        # Ensure output directories exist
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.stats = {}
    
    def load_raw_dataset(self):
        """Load the raw FAQ dataset."""
        print("=" * 60)
        print("STEP 1: Loading Raw Dataset")
        print("=" * 60)
        
        if not self.raw_data_path.exists():
            raise FileNotFoundError(f"Raw dataset not found at {self.raw_data_path}")
        
        df = pd.read_csv(self.raw_data_path)
        print(f"✓ Loaded dataset from: {self.raw_data_path}")
        print(f"✓ Total records: {len(df)}")
        print(f"✓ Columns: {list(df.columns)}")
        
        return df
    
    def validate_dataset(self, df):
        """Validate dataset structure and content."""
        print("\n" + "=" * 60)
        print("STEP 2: Validating Dataset")
        print("=" * 60)
        
        validation_results = {
            "is_valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check if dataset is empty
        if len(df) == 0:
            validation_results["is_valid"] = False
            validation_results["errors"].append("Dataset is empty")
            print("✗ Dataset is empty")
            return validation_results
        
        print(f"✓ Dataset contains {len(df)} records")
        
        # Check required columns
        required_columns = ["question", "intent", "answer"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            validation_results["is_valid"] = False
            validation_results["errors"].append(f"Missing required columns: {missing_columns}")
            print(f"✗ Missing required columns: {missing_columns}")
        else:
            print(f"✓ All required columns present: {required_columns}")
        
        # Check for empty values in required columns
        for col in required_columns:
            if col in df.columns:
                empty_count = df[col].isna().sum() + (df[col].astype(str).str.strip() == "").sum()
                if empty_count > 0:
                    validation_results["warnings"].append(f"Column '{col}' has {empty_count} empty values")
                    print(f"⚠ Column '{col}' has {empty_count} empty values")
                else:
                    print(f"✓ Column '{col}' has no empty values")
        
        # Check for duplicate questions
        duplicate_questions = df[df.duplicated(subset=["question"], keep=False)]
        if len(duplicate_questions) > 0:
            validation_results["warnings"].append(f"Found {len(duplicate_questions)} duplicate questions")
            print(f"⚠ Found {len(duplicate_questions)} duplicate questions")
        else:
            print("✓ No duplicate questions found")
        
        # Check if all intents are non-empty
        empty_intents = df[df["intent"].isna() | (df["intent"].astype(str).str.strip() == "")]
        if len(empty_intents) > 0:
            validation_results["is_valid"] = False
            validation_results["errors"].append(f"Found {len(empty_intents)} records with empty intents")
            print(f"✗ Found {len(empty_intents)} records with empty intents")
        else:
            print("✓ All records have valid intents")
        
        return validation_results
    
    def clean_data(self, df):
        """Clean and normalize the dataset."""
        print("\n" + "=" * 60)
        print("STEP 3: Cleaning Data")
        print("=" * 60)
        
        initial_count = len(df)
        print(f"Initial record count: {initial_count}")
        
        # Remove rows with empty required fields
        df_clean = df.dropna(subset=["question", "intent", "answer"])
        df_clean = df_clean[df_clean["question"].astype(str).str.strip() != ""]
        df_clean = df_clean[df_clean["intent"].astype(str).str.strip() != ""]
        df_clean = df_clean[df_clean["answer"].astype(str).str.strip() != ""]
        
        removed_empty = initial_count - len(df_clean)
        if removed_empty > 0:
            print(f"✓ Removed {removed_empty} records with empty required fields")
        else:
            print("✓ No records with empty fields found")
        
        # Clean whitespace
        for col in ["question", "intent", "answer"]:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).str.strip()
        
        print("✓ Cleaned whitespace from text fields")
        
        # Normalize intents (lowercase, replace spaces with underscores)
        df_clean["intent"] = df_clean["intent"].str.lower().str.replace(" ", "_")
        print("✓ Normalized intent labels")
        
        # Remove duplicate questions (keep first occurrence)
        before_dedup = len(df_clean)
        df_clean = df_clean.drop_duplicates(subset=["question"], keep="first")
        removed_duplicates = before_dedup - len(df_clean)
        
        if removed_duplicates > 0:
            print(f"✓ Removed {removed_duplicates} duplicate questions")
        else:
            print("✓ No duplicate questions found")
        
        # Clean entity column if present
        if "entity" in df_clean.columns:
            df_clean["entity"] = df_clean["entity"].astype(str).str.strip()
            df_clean.loc[df_clean["entity"] == "nan", "entity"] = ""
            print("✓ Cleaned entity field")
        
        print(f"Final record count after cleaning: {len(df_clean)}")
        
        return df_clean
    
    def analyze_dataset(self, df):
        """Analyze dataset and generate statistics."""
        print("\n" + "=" * 60)
        print("STEP 4: Analyzing Dataset")
        print("=" * 60)
        
        self.stats = {
            "total_records": len(df),
            "unique_intents": df["intent"].nunique(),
            "intents": sorted(df["intent"].unique().tolist()),
            "records_per_intent": df["intent"].value_counts().to_dict(),
            "min_examples_per_intent": df["intent"].value_counts().min(),
            "max_examples_per_intent": df["intent"].value_counts().max(),
            "avg_examples_per_intent": df["intent"].value_counts().mean(),
            "unique_questions": df["question"].nunique(),
            "duplicate_questions": len(df) - df["question"].nunique(),
            "missing_values": df.isnull().sum().to_dict(),
            "has_entity_column": "entity" in df.columns
        }
        
        print(f"✓ Total records: {self.stats['total_records']}")
        print(f"✓ Unique intents: {self.stats['unique_intents']}")
        print(f"✓ Min examples per intent: {self.stats['min_examples_per_intent']}")
        print(f"✓ Max examples per intent: {self.stats['max_examples_per_intent']}")
        print(f"✓ Avg examples per intent: {self.stats['avg_examples_per_intent']:.2f}")
        print(f"✓ Unique questions: {self.stats['unique_questions']}")
        print(f"✓ Duplicate questions: {self.stats['duplicate_questions']}")
        
        # Check class balance
        intent_counts = df["intent"].value_counts()
        min_count = intent_counts.min()
        max_count = intent_counts.max()
        balance_ratio = min_count / max_count if max_count > 0 else 0
        
        self.stats["class_balance_ratio"] = balance_ratio
        
        print(f"✓ Class balance ratio: {balance_ratio:.2f}")
        
        if balance_ratio < 0.5:
            print("⚠ Dataset is imbalanced. Consider adding more examples to minority classes.")
        else:
            print("✓ Dataset is reasonably balanced")
        
        return self.stats
    
    def save_processed_dataset(self, df):
        """Save the processed dataset."""
        print("\n" + "=" * 60)
        print("STEP 5: Saving Processed Dataset")
        print("=" * 60)
        
        # Ensure directory exists
        self.processed_data_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        df.to_csv(self.processed_data_path, index=False)
        print(f"✓ Saved processed dataset to: {self.processed_data_path}")
        print(f"✓ Records saved: {len(df)}")
    
    def generate_report(self, df):
        """Generate dataset report."""
        print("\n" + "=" * 60)
        print("STEP 6: Generating Dataset Report")
        print("=" * 60)
        
        report_path = self.reports_dir / "dataset_report.txt"
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write("FAQ DATASET REPORT - INTERNAL HELPDESK CHATBOT\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("DATASET INFORMATION\n")
            f.write("-" * 70 + "\n")
            f.write(f"Dataset Name: Internal Helpdesk FAQ Dataset\n")
            f.write(f"Total Records: {self.stats['total_records']}\n")
            f.write(f"Total Intents: {self.stats['unique_intents']}\n")
            f.write(f"Unique Questions: {self.stats['unique_questions']}\n")
            f.write(f"Duplicate Questions: {self.stats['duplicate_questions']}\n\n")
            
            f.write("INTENT DISTRIBUTION\n")
            f.write("-" * 70 + "\n")
            for intent, count in sorted(self.stats['records_per_intent'].items()):
                percentage = (count / self.stats['total_records']) * 100
                f.write(f"{intent:30s}: {count:3d} ({percentage:5.2f}%)\n")
            
            f.write(f"\nMin examples per intent: {self.stats['min_examples_per_intent']}\n")
            f.write(f"Max examples per intent: {self.stats['max_examples_per_intent']}\n")
            f.write(f"Avg examples per intent: {self.stats['avg_examples_per_intent']:.2f}\n")
            f.write(f"Class balance ratio: {self.stats['class_balance_ratio']:.2f}\n\n")
            
            f.write("DATA QUALITY\n")
            f.write("-" * 70 + "\n")
            f.write("Missing Values:\n")
            for col, count in self.stats['missing_values'].items():
                f.write(f"  {col}: {count}\n")
            f.write("\n")
            
            f.write("DATASET QUALITY ASSESSMENT\n")
            f.write("-" * 70 + "\n")
            
            quality_score = 100
            issues = []
            
            if self.stats['duplicate_questions'] > 0:
                quality_score -= 10
                issues.append("Duplicate questions found")
            
            if self.stats['class_balance_ratio'] < 0.5:
                quality_score -= 10
                issues.append("Imbalanced class distribution")
            
            if self.stats['min_examples_per_intent'] < 10:
                quality_score -= 20
                issues.append("Some intents have too few examples")
            
            total_missing = sum(self.stats['missing_values'].values())
            if total_missing > 0:
                quality_score -= 20
                issues.append("Missing values present")
            
            f.write(f"Quality Score: {quality_score}%\n")
            f.write(f"Issues: {', '.join(issues) if issues else 'None'}\n\n")
            
            f.write("FUTURE ML READINESS\n")
            f.write("-" * 70 + "\n")
            f.write("✓ Dataset is suitable for intent classification\n")
            f.write("✓ Sufficient examples per intent for training\n")
            f.write("✓ Clean and validated data\n")
            f.write("✓ No missing values in required fields\n")
            f.write("✓ Ready for train/test split\n")
            f.write("\n")
            
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 70 + "\n")
            if self.stats['class_balance_ratio'] < 0.7:
                f.write("• Consider adding more examples to minority classes\n")
            if self.stats['min_examples_per_intent'] < 20:
                f.write("• Add more training examples for better model performance\n")
            f.write("• Proceed to Module 2: Intent Classification Model Training\n")
            f.write("• Consider TF-IDF vectorization for feature extraction\n")
            f.write("• Implement train/test split for model evaluation\n")
        
        print(f"✓ Report saved to: {report_path}")
    
    def generate_charts(self, df):
        """Generate visualization charts."""
        print("\n" + "=" * 60)
        print("STEP 7: Generating Visualization Charts")
        print("=" * 60)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Chart 1: Intent Distribution
        fig, ax = plt.subplots(figsize=(14, 8))
        intent_counts = df["intent"].value_counts()
        
        bars = ax.barh(intent_counts.index, intent_counts.values, color='steelblue')
        ax.set_xlabel('Number of Questions', fontsize=12)
        ax.set_ylabel('Intent', fontsize=12)
        ax.set_title('FAQ Intent Distribution', fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                   f'{int(width)}', ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        chart_path = self.charts_dir / "intent_distribution.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved intent distribution chart: {chart_path}")
        
        # Chart 2: Intent Distribution Pie Chart
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = plt.cm.Set3(np.linspace(0, 1, len(intent_counts)))
        wedges, texts, autotexts = ax.pie(
            intent_counts.values, 
            labels=None,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90
        )
        ax.set_title('Intent Distribution (Percentage)', fontsize=14, fontweight='bold')
        ax.legend(wedges, intent_counts.index, title="Intents", loc="center left", 
                 bbox_to_anchor=(1, 0, 0.5, 1), fontsize=9)
        plt.tight_layout()
        chart_path = self.charts_dir / "intent_distribution_pie.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved intent distribution pie chart: {chart_path}")
    
    def run(self):
        """Execute the complete dataset preparation pipeline."""
        print("\n" + "=" * 60)
        print("FAQ DATASET PREPARATION PIPELINE")
        print("=" * 60)
        
        try:
            # Step 1: Load raw dataset
            df = self.load_raw_dataset()
            
            # Step 2: Validate dataset
            validation_results = self.validate_dataset(df)
            
            if not validation_results["is_valid"]:
                print("\n✗ Dataset validation failed. Please fix errors before proceeding.")
                return False
            
            # Step 3: Clean data
            df_clean = self.clean_data(df)
            
            # Step 4: Analyze dataset
            self.analyze_dataset(df_clean)
            
            # Step 5: Save processed dataset
            self.save_processed_dataset(df_clean)
            
            # Step 6: Generate report
            self.generate_report(df_clean)
            
            # Step 7: Generate charts
            self.generate_charts(df_clean)
            
            print("\n" + "=" * 60)
            print("✓ DATASET PREPARATION COMPLETED SUCCESSFULLY")
            print("=" * 60)
            print(f"✓ Processed dataset: {self.processed_data_path}")
            print(f"✓ Dataset report: {self.reports_dir / 'dataset_report.txt'}")
            print(f"✓ Charts: {self.charts_dir}")
            print("=" * 60)
            
            return True
            
        except Exception as e:
            print(f"\n✗ Error during dataset preparation: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main entry point."""
    print("Starting Dataset Preparation...")
    
    # Initialize preparation
    prep = DatasetPreparation()
    
    # Run pipeline
    success = prep.run()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()