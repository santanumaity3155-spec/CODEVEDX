"""
Reports Module for Utility Usage Prediction Tool

This module provides automated report generation functionality including:
- Model Summary Report
- Prediction Report
- Evaluation Report

All reports are automatically saved to outputs/reports/ directory.

Author: CodeVedX AI/ML Internship
"""

from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from config import (
    REPORTS_DIR,
    APP_NAME,
    APP_VERSION,
    MODEL_PATH,
    DATASET_PATH,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    MODEL_TYPE
)
from logger import get_logger


# ========================================
# REPORT GENERATOR CLASS
# ========================================


class ReportGenerator:
    """
    Automated report generator for model and prediction documentation.
    
    Generates 3 types of reports:
    1. Model Summary Report - Overview of model architecture and configuration
    2. Prediction Report - Detailed prediction results
    3. Evaluation Report - Model performance metrics analysis
    
    All reports are saved to outputs/reports/ directory.
    """
    
    def __init__(self, output_dir: Path = REPORTS_DIR):
        """
        Initialize the report generator.
        
        Args:
            output_dir: Directory to save generated reports
        """
        self.output_dir = output_dir
        self.logger = get_logger("ReportGenerator")
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Report generator initialized. Output: {self.output_dir}")
    
    # ========================================
    # REPORT GENERATORS
    # ========================================
    
    def generate_model_summary_report(
        self,
        model_info: Dict[str, Any],
        dataset_info: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[Path], Optional[str]]:
        """
        Generate a comprehensive model summary report.
        
        Args:
            model_info: Dictionary containing model metadata and parameters
            dataset_info: Optional dictionary containing dataset information
        
        Returns:
            Tuple of (success, report_path, error_message)
        """
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            report_lines = [
                "=" * 80,
                f"{'MODEL SUMMARY REPORT':^80}",
                "=" * 80,
                "",
                f"Application: {APP_NAME} v{APP_VERSION}",
                f"Generated: {timestamp}",
                f"Model Type: {MODEL_TYPE}",
                "",
                "-" * 80,
                "MODEL INFORMATION",
                "-" * 80,
                "",
                f"Model Path: {model_info.get('model_path', MODEL_PATH)}",
                f"Model Exists: {model_info.get('file_exists', 'N/A')}",
                f"Model Loaded: {model_info.get('is_loaded', 'N/A')}",
                f"File Size: {model_info.get('file_size', 'N/A')}",
                f"Features Count: {model_info.get('features_count', len(FEATURE_COLUMNS))}",
                f"Target Variable: {model_info.get('target', TARGET_COLUMN)}",
            ]
            
            # Add model coefficients if available
            if "coefficients" in model_info:
                report_lines.extend([
                    "",
                    "-" * 80,
                    "MODEL COEFFICIENTS",
                    "-" * 80,
                    "",
                    f"{'Feature':30s} {'Coefficient':>15s} {'Impact':>15s}",
                    "-" * 62
                ])
                
                features = model_info.get("features", FEATURE_COLUMNS)
                coefficients = model_info["coefficients"]
                
                for feature, coef in zip(features, coefficients):
                    impact = "Positive" if coef > 0 else "Negative"
                    report_lines.append(f"{feature:30s} {coef:>+15.6f} {impact:>15s}")
                
                report_lines.append("-" * 62)
                report_lines.append(
                    f"{'Intercept':30s} {model_info.get('intercept', 0):>+15.6f}"
                )
            
            # Add dataset information
            if dataset_info:
                report_lines.extend([
                    "",
                    "-" * 80,
                    "DATASET INFORMATION",
                    "-" * 80,
                    "",
                    f"Dataset Path: {dataset_info.get('file_path', DATASET_PATH)}",
                    f"Total Records: {dataset_info.get('total_records', 0)}",
                    f"Total Columns: {dataset_info.get('total_columns', 0)}",
                    f"Columns: {', '.join(dataset_info.get('columns', []))}",
                    f"Memory Usage: {dataset_info.get('memory_usage', 'N/A')}",
                    f"Duplicate Rows: {dataset_info.get('duplicate_rows', 0)}",
                    f"Missing Values: {dataset_info.get('missing_values', 0)}",
                ])
            
            # Add features list
            report_lines.extend([
                "",
                "-" * 80,
                "FEATURE COLUMNS",
                "-" * 80,
                "",
            ])
            
            for i, feature in enumerate(FEATURE_COLUMNS, 1):
                report_lines.append(f"  {i:2d}. {feature}")
            
            report_lines.extend([
                "",
                "=" * 80,
                f"{'END OF MODEL SUMMARY REPORT':^80}",
                "=" * 80,
                ""
            ])
            
            # Write report to file
            report_content = "\n".join(report_lines)
            report_path = self.output_dir / "model_summary_report.txt"
            
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_content)
            
            self.logger.info(f"Model summary report saved: {report_path}")
            return True, report_path, None
            
        except Exception as e:
            error_msg = f"Failed to generate model summary report: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return False, None, error_msg
    
    def generate_evaluation_report(
        self,
        metrics: Dict[str, Any]
    ) -> Tuple[bool, Optional[Path], Optional[str]]:
        """
        Generate a comprehensive model evaluation report.
        
        Args:
            metrics: Dictionary containing evaluation metrics:
                - r2_score: R² score
                - mae: Mean Absolute Error
                - mse: Mean Squared Error
                - rmse: Root Mean Squared Error
                - train_samples: Number of training samples
                - test_samples: Number of test samples
                - total_samples: Total number of samples
                - duration: Training duration in seconds
        
        Returns:
            Tuple of (success, report_path, error_message)
        """
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            r2 = metrics.get("r2_score", 0.0)
            mae = metrics.get("mae", 0.0)
            mse = metrics.get("mse", 0.0)
            rmse = metrics.get("rmse", 0.0)
            train_samples = metrics.get("train_samples", 0)
            test_samples = metrics.get("test_samples", 0)
            total_samples = metrics.get("total_samples", 0)
            duration = metrics.get("duration", 0.0)
            
            report_lines = [
                "=" * 80,
                f"{'MODEL EVALUATION REPORT':^80}",
                "=" * 80,
                "",
                f"Application: {APP_NAME} v{APP_VERSION}",
                f"Generated: {timestamp}",
                f"Model Type: {MODEL_TYPE}",
                f"Target Variable: {TARGET_COLUMN}",
                "",
                "-" * 80,
                "DATASET INFORMATION",
                "-" * 80,
                "",
                f"Total Samples: {total_samples}",
                f"Training Samples: {train_samples} ({train_samples / total_samples * 100:.1f}%)" if total_samples > 0 else "Training Samples: N/A",
                f"Testing Samples: {test_samples} ({test_samples / total_samples * 100:.1f}%)" if total_samples > 0 else "Testing Samples: N/A",
                f"Features Used: {len(FEATURE_COLUMNS)}",
                "",
                "-" * 80,
                "MODEL PERFORMANCE METRICS",
                "-" * 80,
                "",
                f"  {'Metric':25s} {'Value':>15s} {'Interpretation':>30s}",
                "-" * 72,
                f"  {'R² Score':25s} {r2:>15.4f} {'Higher is better (max 1.0)':>30s}",
                f"  {'Mean Absolute Error (MAE)':25s} {mae:>15.4f} {'Lower is better':>30s}",
                f"  {'Mean Squared Error (MSE)':25s} {mse:>15.4f} {'Lower is better':>30s}",
                f"  {'Root Mean Squared Error (RMSE)':25s} {rmse:>15.4f} {'Lower is better':>30s}",
                "-" * 72,
                "",
                "-" * 80,
                "MODEL QUALITY ASSESSMENT",
                "-" * 80,
                "",
            ]
            
            # Quality assessment
            if r2 >= 0.90:
                quality = "Excellent"
                quality_desc = "The model explains 90%+ of variance. Highly reliable."
            elif r2 >= 0.80:
                quality = "Good"
                quality_desc = "The model explains 80%+ of variance. Reliable for predictions."
            elif r2 >= 0.60:
                quality = "Fair"
                quality_desc = "The model explains 60%+ of variance. Moderately reliable."
            elif r2 >= 0.40:
                quality = "Poor"
                quality_desc = "The model explains less than 60% of variance. Limited reliability."
            else:
                quality = "Very Poor"
                quality_desc = "The model explains less than 40% of variance. Needs improvement."
            
            report_lines.extend([
                f"  Model Quality: {quality}",
                f"  Assessment: {quality_desc}",
                "",
                f"  Training Duration: {duration:.2f} seconds",
                "",
                "-" * 80,
                "FEATURE IMPORTANCE (COEFFICIENTS)",
                "-" * 80,
                "",
            ])
            
            # Add feature importance if available
            if "feature_importance" in metrics:
                importance = metrics["feature_importance"]
                # Sort by absolute value
                sorted_features = sorted(
                    importance.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )
                
                report_lines.append(f"{'Rank':5s} {'Feature':30s} {'Coefficient':>15s} {'Abs Impact':>15s}")
                report_lines.append("-" * 67)
                
                for rank, (feature, coef) in enumerate(sorted_features, 1):
                    report_lines.append(
                        f"{rank:5d} {feature:30s} {coef:>+15.6f} {abs(coef):>15.6f}"
                    )
            
            report_lines.extend([
                "",
                "-" * 80,
                "OUTPUTS GENERATED",
                "-" * 80,
                "",
                f"  Model: {MODEL_PATH.name}",
                f"  Charts: outputs/charts/",
                f"    - histogram_global_active_power.png",
                f"    - correlation_heatmap.png",
                f"    - actual_vs_predicted.png",
                f"    - residual_plot.png",
                f"    - monthly_consumption.png",
                f"  Predictions: outputs/predictions/",
                f"  Reports: outputs/reports/",
                "",
                "=" * 80,
                f"{'END OF EVALUATION REPORT':^80}",
                "=" * 80,
                ""
            ])
            
            # Write report to file
            report_content = "\n".join(report_lines)
            report_path = self.output_dir / "model_evaluation_report.txt"
            
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_content)
            
            self.logger.info(f"Evaluation report saved: {report_path}")
            return True, report_path, None
            
        except Exception as e:
            error_msg = f"Failed to generate evaluation report: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return False, None, error_msg
    
    def generate_prediction_report(
        self,
        predictions: pd.DataFrame,
        report_name: Optional[str] = None
    ) -> Tuple[bool, Optional[Path], Optional[str]]:
        """
        Generate a prediction report from prediction results.
        
        Args:
            predictions: DataFrame containing predictions
            report_name: Optional custom report name (default: auto-generated)
        
        Returns:
            Tuple of (success, report_path, error_message)
        """
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            if report_name is None:
                timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_name = f"prediction_report_{timestamp_file}"
            
            report_lines = [
                "=" * 80,
                f"{'PREDICTION REPORT':^80}",
                "=" * 80,
                "",
                f"Application: {APP_NAME} v{APP_VERSION}",
                f"Generated: {timestamp}",
                f"Report Name: {report_name}",
                f"Model Type: {MODEL_TYPE}",
                "",
                "-" * 80,
                "PREDICTION SUMMARY",
                "-" * 80,
                "",
                f"Total Predictions: {len(predictions)}",
                f"Features Used: {len(FEATURE_COLUMNS)}",
                "",
            ]
            
            # Add prediction statistics
            if "Predicted_Active_Power" in predictions.columns:
                pred_values = predictions["Predicted_Active_Power"]
                report_lines.extend([
                    "Prediction Statistics:",
                    f"  Mean: {pred_values.mean():.4f} kW",
                    f"  Std: {pred_values.std():.4f} kW",
                    f"  Min: {pred_values.min():.4f} kW",
                    f"  Max: {pred_values.max():.4f} kW",
                    "",
                ])
            
            # Add first few predictions as preview
            report_lines.extend([
                "-" * 80,
                "PREDICTION PREVIEW (First 10 rows)",
                "-" * 80,
                "",
            ])
            
            preview = predictions.head(10)
            
            # Dynamic column width
            col_widths = {}
            for col in preview.columns:
                col_widths[col] = max(
                    len(str(col)),
                    preview[col].astype(str).str.len().max()
                )
                col_widths[col] = min(col_widths[col], 25)  # Cap width
            
            # Header
            header = " | ".join(f"{col:^{col_widths[col]}}" for col in preview.columns)
            report_lines.append(header)
            report_lines.append("-" * min(120, len(header)))
            
            # Rows
            for _, row in preview.iterrows():
                row_str = " | ".join(
                    f"{str(row[col])[:col_widths[col]]:^{col_widths[col]}}"
                    for col in preview.columns
                )
                report_lines.append(row_str)
            
            report_lines.extend([
                "",
                f"... and {len(predictions) - 10} more predictions" if len(predictions) > 10 else "",
                "",
                "=" * 80,
                f"{'END OF PREDICTION REPORT':^80}",
                "=" * 80,
                ""
            ])
            
            # Write report to file
            report_content = "\n".join(report_lines)
            report_path = self.output_dir / f"{report_name}.txt"
            
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_content)
            
            self.logger.info(f"Prediction report saved: {report_path}")
            return True, report_path, None
            
        except Exception as e:
            error_msg = f"Failed to generate prediction report: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return False, None, error_msg
    
    # ========================================
    # UTILITY METHODS
    # ========================================
    
    def get_report_info(self) -> Dict[str, Any]:
        """
        Get information about generated report files.
        
        Returns:
            Dictionary mapping report names to their details
        """
        report_info: Dict[str, Any] = {}
        
        report_files = {
            "model_summary": "model_summary_report.txt",
            "evaluation": "model_evaluation_report.txt"
        }
        
        for report_name, filename in report_files.items():
            file_path = self.output_dir / filename
            report_info[report_name] = {
                "filename": filename,
                "exists": file_path.exists(),
                "path": str(file_path) if file_path.exists() else None,
                "size": f"{file_path.stat().st_size / 1024:.1f} KB" if file_path.exists() else "N/A"
            }
        
        # List all prediction reports
        prediction_reports = sorted(self.output_dir.glob("prediction_report_*.txt"))
        report_info["prediction_reports"] = [
            {
                "filename": f.name,
                "path": str(f),
                "size": f"{f.stat().st_size / 1024:.1f} KB"
            }
            for f in prediction_reports
        ]
        
        return report_info
    
    def list_generated_reports(self) -> List[str]:
        """
        List all generated report files.
        
        Returns:
            List of report filenames
        """
        if not self.output_dir.exists():
            return []
        
        report_files = sorted(self.output_dir.glob("*.txt"))
        return [f.name for f in report_files]


# ========================================
# CONVENIENCE FUNCTIONS
# ========================================


def get_report_generator() -> ReportGenerator:
    """
    Get a ReportGenerator instance.
    
    Returns:
        ReportGenerator instance
    """
    return ReportGenerator()


def generate_evaluation_report(metrics: Dict[str, Any]) -> Tuple[bool, Optional[Path], Optional[str]]:
    """
    Generate evaluation report (convenience function).
    
    Args:
        metrics: Dictionary containing evaluation metrics
    
    Returns:
        Tuple of (success, report_path, error_message)
    """
    generator = get_report_generator()
    return generator.generate_evaluation_report(metrics)


def generate_model_summary_report(
    model_info: Dict[str, Any],
    dataset_info: Optional[Dict[str, Any]] = None
) -> Tuple[bool, Optional[Path], Optional[str]]:
    """
    Generate model summary report (convenience function).
    
    Args:
        model_info: Dictionary containing model metadata
        dataset_info: Optional dataset information
    
    Returns:
        Tuple of (success, report_path, error_message)
    """
    generator = get_report_generator()
    return generator.generate_model_summary_report(model_info, dataset_info)

