"""
Charts Module for Utility Usage Prediction Tool

This module provides automated chart generation functionality including:
- Distribution Plot (Histogram of target variable)
- Correlation Heatmap
- Actual vs Predicted scatter plot
- Residual Plot
- Monthly Consumption line plot

All charts are automatically saved to outputs/charts/ directory.

Author: CodeVedX AI/ML Internship
"""

from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving to file
import matplotlib.pyplot as plt
import seaborn as sns

from config import CHARTS_DIR, TARGET_COLUMN, FEATURE_COLUMNS
from logger import get_logger


# ========================================
# PLOTTING STYLE CONFIGURATION
# ========================================

# Set professional style for all charts
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

# Chart dimensions
CHART_WIDTH: int = 12
CHART_HEIGHT: int = 8
DPI: int = 150


# ========================================
# CHART GENERATOR CLASS
# ========================================


class ChartGenerator:
    """
    Automated chart generator for model evaluation and data analysis.
    
    Generates 5 types of charts:
    1. Distribution Plot - Histogram of target variable
    2. Correlation Heatmap - Feature correlation matrix
    3. Actual vs Predicted - Scatter plot of actual vs predicted values
    4. Residual Plot - Distribution and pattern of residuals
    5. Monthly Consumption - Average monthly consumption trend
    
    All charts are saved to outputs/charts/ directory.
    """
    
    def __init__(self, output_dir: Path = CHARTS_DIR):
        """
        Initialize the chart generator.
        
        Args:
            output_dir: Directory to save generated charts
        """
        self.output_dir = output_dir
        self.logger = get_logger("ChartGenerator")
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Chart generator initialized. Output: {self.output_dir}")
    
    # ========================================
    # PUBLIC METHODS
    # ========================================
    
    def generate_all_charts(
        self,
        df: pd.DataFrame,
        y_true: Optional[np.ndarray] = None,
        y_pred: Optional[np.ndarray] = None
    ) -> Dict[str, Path]:
        """
        Generate all 5 chart types and save to output directory.
        
        Args:
            df: DataFrame containing the data
            y_true: Optional array of true target values (for model evaluation charts)
            y_pred: Optional array of predicted values (for model evaluation charts)
        
        Returns:
            Dictionary mapping chart names to file paths
        """
        generated: Dict[str, Path] = {}
        
        try:
            # Generate distribution plot
            dist_path = self.plot_distribution(df)
            if dist_path:
                generated["distribution"] = dist_path
            
            # Generate correlation heatmap
            heatmap_path = self.plot_correlation_heatmap(df)
            if heatmap_path:
                generated["correlation_heatmap"] = heatmap_path
            
            # Generate actual vs predicted (if available)
            if y_true is not None and y_pred is not None:
                avp_path = self.plot_actual_vs_predicted(y_true, y_pred)
                if avp_path:
                    generated["actual_vs_predicted"] = avp_path
                
                # Generate residual plot
                residual_path = self.plot_residuals(y_true, y_pred)
                if residual_path:
                    generated["residual_plot"] = residual_path
            
            # Generate monthly consumption (if date columns exist)
            if "Year" in df.columns and "Month" in df.columns:
                monthly_path = self.plot_monthly_consumption(df)
                if monthly_path:
                    generated["monthly_consumption"] = monthly_path
            
            self.logger.info(f"Generated {len(generated)} charts: {list(generated.keys())}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate charts: {str(e)}", exc_info=True)
        
        return generated
    
    # ========================================
    # INDIVIDUAL CHART GENERATORS
    # ========================================
    
    def plot_distribution(self, df: pd.DataFrame) -> Optional[Path]:
        """
        Generate distribution plot (histogram) for the target variable.
        
        Args:
            df: DataFrame containing the target column
        
        Returns:
            Path to saved chart or None on failure
        """
        try:
            if TARGET_COLUMN not in df.columns:
                self.logger.warning(f"Target column '{TARGET_COLUMN}' not found in DataFrame")
                return None
            
            fig, axes = plt.subplots(1, 2, figsize=(CHART_WIDTH, CHART_HEIGHT // 2))
            
            target_data = df[TARGET_COLUMN].dropna()
            
            # Histogram with KDE
            sns.histplot(
                target_data,
                kde=True,
                bins=50,
                color="steelblue",
                edgecolor="white",
                alpha=0.7,
                ax=axes[0]
            )
            axes[0].set_title(f"Distribution of {TARGET_COLUMN}", fontsize=14, fontweight="bold")
            axes[0].set_xlabel(f"{TARGET_COLUMN} (kW)", fontsize=12)
            axes[0].set_ylabel("Frequency", fontsize=12)
            axes[0].axvline(
                target_data.mean(),
                color="red",
                linestyle="dashed",
                linewidth=2,
                label=f"Mean: {target_data.mean():.2f}"
            )
            axes[0].axvline(
                target_data.median(),
                color="green",
                linestyle="dashed",
                linewidth=2,
                label=f"Median: {target_data.median():.2f}"
            )
            axes[0].legend(fontsize=10)
            
            # Box plot
            sns.boxplot(
                x=target_data,
                color="steelblue",
                width=0.4,
                ax=axes[1]
            )
            axes[1].set_title(f"Box Plot of {TARGET_COLUMN}", fontsize=14, fontweight="bold")
            axes[1].set_xlabel(f"{TARGET_COLUMN} (kW)", fontsize=12)
            
            # Add statistics text
            stats_text = (
                f"Count: {len(target_data)}\n"
                f"Mean: {target_data.mean():.4f}\n"
                f"Std: {target_data.std():.4f}\n"
                f"Min: {target_data.min():.4f}\n"
                f"Max: {target_data.max():.4f}"
            )
            axes[1].text(
                0.95, 0.95, stats_text,
                transform=axes[1].transAxes,
                fontsize=10,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
            )
            
            plt.tight_layout()
            
            # Save chart
            file_path = self.output_dir / "histogram_global_active_power.png"
            plt.savefig(file_path, dpi=DPI, bbox_inches="tight")
            plt.close(fig)
            
            self.logger.info(f"Distribution chart saved: {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Failed to generate distribution plot: {str(e)}")
            plt.close("all")
            return None
    
    def plot_correlation_heatmap(self, df: pd.DataFrame) -> Optional[Path]:
        """
        Generate correlation heatmap for numeric features.
        
        Args:
            df: DataFrame with numeric columns
        
        Returns:
            Path to saved chart or None on failure
        """
        try:
            # Select numeric columns
            numeric_df = df.select_dtypes(include=[np.number])
            
            if numeric_df.empty:
                self.logger.warning("No numeric columns found for correlation heatmap")
                return None
            
            # Compute correlation matrix
            corr_matrix = numeric_df.corr()
            
            # Create mask for upper triangle
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            
            # Create figure
            fig_height = max(CHART_HEIGHT, len(corr_matrix) * 0.6)
            fig, ax = plt.subplots(figsize=(CHART_WIDTH, fig_height))
            
            # Draw heatmap
            sns.heatmap(
                corr_matrix,
                mask=mask,
                annot=True,
                fmt=".2f",
                cmap="RdBu_r",
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"},
                ax=ax
            )
            
            ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
            ax.set_xlabel("Features", fontsize=12)
            ax.set_ylabel("Features", fontsize=12)
            
            # Rotate tick labels
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)
            
            plt.tight_layout()
            
            # Save chart
            file_path = self.output_dir / "correlation_heatmap.png"
            plt.savefig(file_path, dpi=DPI, bbox_inches="tight")
            plt.close(fig)
            
            self.logger.info(f"Correlation heatmap saved: {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Failed to generate correlation heatmap: {str(e)}")
            plt.close("all")
            return None
    
    def plot_actual_vs_predicted(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Optional[Path]:
        """
        Generate Actual vs Predicted scatter plot.
        
        Args:
            y_true: Array of true values
            y_pred: Array of predicted values
        
        Returns:
            Path to saved chart or None on failure
        """
        try:
            fig, ax = plt.subplots(figsize=(CHART_WIDTH, CHART_HEIGHT))
            
            # Scatter plot
            ax.scatter(
                y_true, y_pred,
                alpha=0.5,
                color="steelblue",
                edgecolors="white",
                linewidth=0.5,
                s=50
            )
            
            # Perfect prediction line
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax.plot(
                [min_val, max_val],
                [min_val, max_val],
                "r--",
                linewidth=2,
                label="Perfect Prediction"
            )
            
            ax.set_title("Actual vs Predicted Values", fontsize=14, fontweight="bold")
            ax.set_xlabel(f"Actual {TARGET_COLUMN} (kW)", fontsize=12)
            ax.set_ylabel(f"Predicted {TARGET_COLUMN} (kW)", fontsize=12)
            
            # Calculate and display R²
            from sklearn.metrics import r2_score
            r2 = r2_score(y_true, y_pred)
            
            ax.text(
                0.05, 0.95,
                f"R² = {r2:.4f}",
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
            )
            
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save chart
            file_path = self.output_dir / "actual_vs_predicted.png"
            plt.savefig(file_path, dpi=DPI, bbox_inches="tight")
            plt.close(fig)
            
            self.logger.info(f"Actual vs Predicted chart saved: {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Failed to generate actual vs predicted plot: {str(e)}")
            plt.close("all")
            return None
    
    def plot_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Optional[Path]:
        """
        Generate residual analysis plot.
        
        Includes:
        - Residuals vs Predicted scatter plot
        - Residual distribution histogram
        - Q-Q plot for normality check
        
        Args:
            y_true: Array of true values
            y_pred: Array of predicted values
        
        Returns:
            Path to saved chart or None on failure
        """
        try:
            # Calculate residuals
            residuals = y_true - y_pred
            
            fig, axes = plt.subplots(2, 2, figsize=(CHART_WIDTH, CHART_HEIGHT))
            
            # 1. Residuals vs Predicted
            axes[0, 0].scatter(
                y_pred, residuals,
                alpha=0.5,
                color="steelblue",
                edgecolors="white",
                linewidth=0.5
            )
            axes[0, 0].axhline(y=0, color="red", linestyle="--", linewidth=2)
            axes[0, 0].set_title("Residuals vs Predicted", fontsize=12, fontweight="bold")
            axes[0, 0].set_xlabel("Predicted Values (kW)", fontsize=10)
            axes[0, 0].set_ylabel("Residuals (kW)", fontsize=10)
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Residual distribution histogram
            sns.histplot(
                residuals,
                kde=True,
                bins=30,
                color="steelblue",
                edgecolor="white",
                alpha=0.7,
                ax=axes[0, 1]
            )
            axes[0, 1].axvline(
                x=0, color="red", linestyle="--", linewidth=2,
                label=f"Mean: {residuals.mean():.4f}"
            )
            axes[0, 1].set_title("Residual Distribution", fontsize=12, fontweight="bold")
            axes[0, 1].set_xlabel("Residuals (kW)", fontsize=10)
            axes[0, 1].set_ylabel("Frequency", fontsize=10)
            axes[0, 1].legend(fontsize=10)
            
            # 3. Q-Q plot for normality
            from scipy import stats
            stats.probplot(residuals, dist="norm", plot=axes[1, 0])
            axes[1, 0].set_title("Q-Q Plot (Normality Check)", fontsize=12, fontweight="bold")
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Residual statistics text
            axes[1, 1].axis("off")
            stats_text = (
                "Residual Statistics:\n"
                f"Count: {len(residuals)}\n"
                f"Mean: {residuals.mean():.6f}\n"
                f"Std: {residuals.std():.6f}\n"
                f"Min: {residuals.min():.6f}\n"
                f"Max: {residuals.max():.6f}\n"
                f"Skewness: {pd.Series(residuals).skew():.4f}\n"
                f"Kurtosis: {pd.Series(residuals).kurtosis():.4f}"
            )
            axes[1, 1].text(
                0.1, 0.5, stats_text,
                fontsize=12,
                verticalalignment="center",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8)
            )
            
            plt.tight_layout()
            
            # Save chart
            file_path = self.output_dir / "residual_plot.png"
            plt.savefig(file_path, dpi=DPI, bbox_inches="tight")
            plt.close(fig)
            
            self.logger.info(f"Residual plot saved: {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Failed to generate residual plot: {str(e)}")
            plt.close("all")
            return None
    
    def plot_monthly_consumption(self, df: pd.DataFrame) -> Optional[Path]:
        """
        Generate monthly average consumption line plot.
        
        Args:
            df: DataFrame with 'Year', 'Month', and target column
        
        Returns:
            Path to saved chart or None on failure
        """
        try:
            required_cols = ["Year", "Month", TARGET_COLUMN]
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                self.logger.warning(f"Missing columns for monthly plot: {missing}")
                return None
            
            # Create datetime column and aggregate monthly
            df_plot = df.dropna(subset=[TARGET_COLUMN]).copy()
            df_plot["Date"] = pd.to_datetime(
                df_plot[["Year", "Month"]].assign(Day=1)
            )
            
            # Group by month and calculate mean consumption
            monthly_avg = (
                df_plot.groupby("Date")[TARGET_COLUMN]
                .mean()
                .reset_index()
                .sort_values("Date")
            )
            
            fig, ax = plt.subplots(figsize=(CHART_WIDTH, CHART_HEIGHT))
            
            # Line plot with fill
            ax.plot(
                monthly_avg["Date"],
                monthly_avg[TARGET_COLUMN],
                color="steelblue",
                linewidth=2,
                marker="o",
                markersize=6,
                label=f"Average {TARGET_COLUMN}"
            )
            
            ax.fill_between(
                monthly_avg["Date"],
                monthly_avg[TARGET_COLUMN],
                alpha=0.2,
                color="steelblue"
            )
            
            ax.set_title("Average Monthly Consumption Trend", fontsize=14, fontweight="bold")
            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel(f"Average {TARGET_COLUMN} (kW)", fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Format x-axis dates
            fig.autofmt_xdate(rotation=45)
            
            plt.tight_layout()
            
            # Save chart
            file_path = self.output_dir / "monthly_consumption.png"
            plt.savefig(file_path, dpi=DPI, bbox_inches="tight")
            plt.close(fig)
            
            self.logger.info(f"Monthly consumption chart saved: {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Failed to generate monthly consumption plot: {str(e)}")
            plt.close("all")
            return None
    
    # ========================================
    # UTILITY METHODS
    # ========================================
    
    def get_chart_info(self) -> Dict[str, Any]:
        """
        Get information about generated chart files.
        
        Returns:
            Dictionary mapping chart names to their details
        """
        chart_info: Dict[str, Any] = {}
        
        chart_files = {
            "distribution": "histogram_global_active_power.png",
            "correlation_heatmap": "correlation_heatmap.png",
            "actual_vs_predicted": "actual_vs_predicted.png",
            "residual_plot": "residual_plot.png",
            "monthly_consumption": "monthly_consumption.png"
        }
        
        for chart_name, filename in chart_files.items():
            file_path = self.output_dir / filename
            chart_info[chart_name] = {
                "filename": filename,
                "exists": file_path.exists(),
                "path": str(file_path) if file_path.exists() else None,
                "size": f"{file_path.stat().st_size / 1024:.1f} KB" if file_path.exists() else "N/A"
            }
        
        return chart_info
    
    def list_generated_charts(self) -> List[str]:
        """
        List all generated chart files.
        
        Returns:
            List of chart filenames
        """
        if not self.output_dir.exists():
            return []
        
        chart_files = sorted(self.output_dir.glob("*.png"))
        return [f.name for f in chart_files]


# ========================================
# CONVENIENCE FUNCTIONS
# ========================================


def get_chart_generator() -> ChartGenerator:
    """
    Get a ChartGenerator instance.
    
    Returns:
        ChartGenerator instance
    """
    return ChartGenerator()


def generate_all_charts(
    df: pd.DataFrame,
    y_true: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None
) -> Dict[str, Path]:
    """
    Generate all charts (convenience function).
    
    Args:
        df: DataFrame containing the data
        y_true: Optional array of true target values
        y_pred: Optional array of predicted values
    
    Returns:
        Dictionary mapping chart names to file paths
    """
    generator = get_chart_generator()
    return generator.generate_all_charts(df, y_true, y_pred)

