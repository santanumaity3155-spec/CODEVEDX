"""
Module 2: NLP Preprocessing Pipeline for Internal Helpdesk Chatbot

Consumes the Module 1 dataset (data/processed/faq_dataset.csv) and produces a
clean, reusable NLP representation:

    data/processed/faq_nlp_ready.csv
    data/processed/faq_nlp_ready.json
    outputs/reports/nlp_preprocessing_report.txt
    outputs/charts/token_length_distribution.png
    outputs/charts/entity_frequency.png
    outputs/charts/question_length_distribution.png

Flow: load -> validate input -> preprocess -> validate output ->
save csv/json -> statistics -> report -> charts -> before/after examples ->
quality gate.

This module does NOT train or save any machine-learning model. Module 3 will
consume the NLP-ready outputs produced here.
"""

import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import matplotlib

matplotlib.use("Agg")  # non-interactive backend for scripts
import matplotlib.pyplot as plt

# Make `from nlp_preprocessor import ...` work whether run as a script or when
# the package dir is not already on sys.path.
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nlp_preprocessor import (  # noqa: E402
    NLPPreprocessor,
    REQUIRED_COLUMNS,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXPECTED_RECORDS = 294
EXPECTED_INTENTS = 22
MIN_EXAMPLES_PER_INTENT = 12

class NLPPreparationPipeline:
    """End-to-end Module 2 NLP preprocessing pipeline."""

    def __init__(self, project_root: Path = PROJECT_ROOT) -> None:
        self.project_root = Path(project_root)
        self.input_csv = self.project_root / "data" / "processed" / "faq_dataset.csv"
        self.output_csv = self.project_root / "data" / "processed" / "faq_nlp_ready.csv"
        self.output_json = self.project_root / "data" / "processed" / "faq_nlp_ready.json"
        self.report_path = (
            self.project_root / "outputs" / "reports" / "nlp_preprocessing_report.txt"
        )
        self.charts_dir = self.project_root / "outputs" / "charts"
        self.preprocessor = NLPPreprocessor(self.project_root)

        self.input_df: pd.DataFrame | None = None
        self.output_df: pd.DataFrame | None = None
        self.stats: dict = {}
        self.validation: dict = {}
        self.report: dict = {}

    # ------------------------------------------------------------------
    def _load_input(self) -> pd.DataFrame:
        """Load the Module 1 dataset with descriptive errors."""
        if not self.input_csv.exists():
            raise FileNotFoundError(
                f"Module 1 dataset not found: {self.input_csv}. "
                "Run `python src/prepare_dataset.py` first."
            )
        df = self.preprocessor.load_dataset(self.input_csv)
        return df

    def _validate_input(self, df: pd.DataFrame) -> dict:
        """Validate the Module 1 dataset before preprocessing."""
        result = {"valid": True, "errors": []}
        if len(df) != EXPECTED_RECORDS:
            result["errors"].append(
                f"Expected {EXPECTED_RECORDS} records, found {len(df)}"
            )
        if df["intent"].nunique() != EXPECTED_INTENTS:
            result["errors"].append(
                f"Expected {EXPECTED_INTENTS} intents, found {df['intent'].nunique()}"
            )
        if int(df["intent"].value_counts().min()) < MIN_EXAMPLES_PER_INTENT:
            result["errors"].append(
                f"Minimum examples per intent ({int(df['intent'].value_counts().min())}) "
                f"is below {MIN_EXAMPLES_PER_INTENT}"
            )
        for col in REQUIRED_COLUMNS:
            missing = int(df[col].isna().sum()) + int(
                (df[col].astype(str).str.strip() == "").sum()
            )
            if missing:
                result["errors"].append(f"Column '{col}' has {missing} missing values")
        if result["errors"]:
            result["valid"] = False
        return result

    def _generate_report(self, df: pd.DataFrame, input_df: pd.DataFrame) -> Path:
        """Write the NLP preprocessing report to outputs/reports/."""
        stats = self.stats
        lines = []
        lines.append("NLP PREPROCESSING REPORT - Module 2")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 60)

        missing_values = sum(
            int(df[col].isna().sum())
            + int((df[col].astype(str).str.strip() == "").sum())
            for col in REQUIRED_COLUMNS
        )

        lines.append("\nDataset Statistics")
        lines.append("-" * 60)
        lines.append(f"Input records            : {len(input_df)}")
        lines.append(f"Output records           : {len(df)}")
        lines.append(f"Number of intents        : {stats['total_intents']}")
        lines.append(f"Number of entities       : {stats['unique_entities']}")

        lines.append("\nQuestion Statistics")
        lines.append("-" * 60)
        lines.append(f"Average question length  : {stats['avg_question_length']}")
        lines.append(f"Minimum question length  : {stats['min_question_length']}")
        lines.append(f"Maximum question length  : {stats['max_question_length']}")
        lines.append(f"Average token count      : {stats['avg_token_count']}")
        lines.append(f"Minimum token count      : {stats['min_token_count']}")
        lines.append(f"Maximum token count      : {stats['max_token_count']}")

        lines.append("\nVocabulary Statistics")
        lines.append("-" * 60)
        lines.append(f"Raw vocabulary size      : {stats['raw_vocabulary_size']}")
        lines.append(
            f"Lemmatized vocabulary size: {stats['lemmatized_vocabulary_size']}"
        )

        lines.append("\nIntent Statistics")
        lines.append("-" * 60)
        lines.append(f"Minimum examples/intent  : {stats['min_intent_examples']}")
        lines.append(f"Maximum examples/intent  : {stats['max_intent_examples']}")
        lines.append("Intent distribution:")
        for intent, count in sorted(
            stats["intent_distribution"].items(), key=lambda kv: (-kv[1], kv[0])
        ):
            lines.append(f"  {intent}: {count}")

        lines.append("\nEntity Statistics")
        lines.append("-" * 60)
        lines.append(f"Number of unique entities: {stats['unique_entities']}")
        lines.append("Entity frequency:")
        for entity, count in sorted(
            stats["entity_frequency"].items(), key=lambda kv: (-kv[1], kv[0])
        ):
            lines.append(f"  {entity}: {count}")

        lines.append("\nPreprocessing Validation")
        lines.append("-" * 60)
        lines.append(f"Missing values                 : {missing_values}")
        lines.append(f"Duplicate questions            : {stats['duplicate_questions']}")
        lines.append(f"Duplicate clean questions      : {stats['duplicate_clean_questions']}")
        lines.append(f"Records with empty clean_quest : {stats['empty_records']}")
        lines.append(
            f"Record count consistency        : "
            f"{'OK' if len(df) == len(input_df) else 'MISMATCH'}"
        )

        lines.append("\nFinal Quality Gate")
        lines.append("-" * 60)
        for name, passed in self.report["checks"].items():
            lines.append(f"[{'PASS' if passed else 'FAIL'}] {name}")
        lines.append("-" * 60)
        lines.append(
            "MODULE 2 QUALITY GATE: "
            f"{'PASS' if self.report['overall_pass'] else 'FAIL'}"
        )
        lines.append("")

        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text("\n".join(lines), encoding="utf-8")
        return self.report_path

    def _generate_charts(self, df: pd.DataFrame) -> list:
        """Generate Module 2 charts from the actual processed data."""
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        paths = []

        # 1. Token length distribution
        token_counts = df["tokens"].apply(len)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(
            token_counts,
            bins=range(0, int(token_counts.max()) + 2),
            color="#4C72B0",
            edgecolor="white",
        )
        ax.set_xlabel("Number of tokens per question")
        ax.set_ylabel("Number of questions")
        ax.set_title("Token Length Distribution")
        p = self.charts_dir / "token_length_distribution.png"
        fig.tight_layout()
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths.append(p)

        # 2. Entity frequency
        entities = df["entity"].astype(str).str.strip()
        entity_counts = entities[entities != ""].value_counts()
        fig, ax = plt.subplots(figsize=(9, 6))
        entity_counts.plot(kind="bar", ax=ax, color="#55A868")
        ax.set_xlabel("Entity")
        ax.set_ylabel("Frequency")
        ax.set_title("Entity Frequency")
        ax.tick_params(axis="x", rotation=45)
        p = self.charts_dir / "entity_frequency.png"
        fig.tight_layout()
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths.append(p)

        # 3. Question length distribution
        question_lens = df["clean_question"].astype(str).str.len()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(question_lens, bins=30, color="#C44E52", edgecolor="white")
        ax.set_xlabel("Number of characters in cleaned question")
        ax.set_ylabel("Number of questions")
        ax.set_title("Question Length Distribution")
        p = self.charts_dir / "question_length_distribution.png"
        fig.tight_layout()
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths.append(p)

        return paths

    def _before_after_examples(self, df: pd.DataFrame, n: int = 5) -> None:
        """Print real before/after examples from different intents."""
        intents = df["intent"].unique()
        chosen = []
        for intent in sorted(intents):
            row = df[df["intent"] == intent].iloc[0]
            chosen.append(row)
            if len(chosen) >= n:
                break

        print("\n--- BEFORE / AFTER EXAMPLES ---")
        for row in chosen:
            print("\nOriginal:")
            print(" ", row["question"])
            print("Clean:")
            print(" ", row["clean_question"])
            print("Tokens:")
            print(" ", row["tokens"])
            print("Filtered Tokens:")
            print(" ", row["filtered_tokens"])
            print("Lemmatized Tokens:")
            print(" ", row["lemmatized_tokens"])
            print("Intent:")
            print(" ", row["intent"])
            print("Entity:")
            print(" ", row["entity"])

    def run(self) -> bool:
        """Execute the Module 2 pipeline end-to-end."""
        try:
            print("=" * 60)
            print("MODULE 2: NLP PREPROCESSING PIPELINE")
            print("=" * 60)

            # 1. Load Module 1 dataset.
            self.input_df = self._load_input()
            print(f"Loaded Module 1 dataset : {len(self.input_df)} records")

            # 2. Validate input.
            input_validation = self._validate_input(self.input_df)
            if not input_validation["valid"]:
                for err in input_validation["errors"]:
                    print(f"  [FAIL] {err}")
                raise ValueError("Module 1 dataset failed validation: " + "; ".join(input_validation["errors"]))
            print("Input validation        : PASS")

            # 3. Preprocess.
            self.output_df = self.preprocessor.preprocess_dataset(self.input_df)
            print(f"Preprocessed            : {len(self.output_df)} records")

            # 4. Validate output.
            self.validation = self.preprocessor.validate_preprocessed_data(self.output_df)
            if not self.validation["is_valid"]:
                for err in self.validation["errors"]:
                    print(f"  [FAIL] {err}")
                raise ValueError("NLP-ready data failed validation: " + "; ".join(self.validation["errors"]))

            # 5. Save CSV + JSON.
            csv_path, json_path = self.preprocessor.save_nlp_dataset(
                self.output_df, self.project_root / "data" / "processed"
            )
            print(f"Saved CSV  : {csv_path}")
            print(f"Saved JSON : {json_path}")

            # 6. Statistics.
            self.stats = self.preprocessor.generate_nlp_statistics(self.output_df)
            print(
                f"Stats      : {self.stats['total_questions']} records, "
                f"{self.stats['total_intents']} intents, "
                f"{self.stats['unique_entities']} entities"
            )

            # 7. Charts.
            chart_paths = self._generate_charts(self.output_df)
            for chart in chart_paths:
                print(f"Chart      : {chart}")

            # 8. Before/after examples.
            self._before_after_examples(self.output_df, n=5)

            # 9. Quality gate (report artifact treated as present; file is
            #    written immediately below so the flag reflects reality).
            self.report = self.preprocessor.compute_quality_gate(
                self.input_df,
                self.output_df,
                self.validation,
                self.stats,
                artifacts={
                    "csv": csv_path.exists(),
                    "json": json_path.exists(),
                    "report": True,
                    "charts": all(p.exists() for p in chart_paths),
                },
            )

            # 10. Write the report, then reconcile the report flag with the
            #     actual file state (so OVERALL reflects reality).
            report_path = self._generate_report(self.output_df, self.input_df)
            self.report["checks"]["NLP report generated"] = bool(
                report_path.exists() and report_path.stat().st_size > 0
            )
            self.report["overall_pass"] = all(self.report["checks"].values())
            print(f"Report     : {report_path}")

            print("\n--- QUALITY GATE ---")
            for name, passed in self.report["checks"].items():
                print(f"[{'PASS' if passed else 'FAIL'}] {name}")
            print("-" * 40)
            print(f"OVERALL: {'PASS' if self.report['overall_pass'] else 'FAIL'}")

            print("\n" + "=" * 60)
            print(
                "MODULE 2 QUALITY GATE: "
                + ("PASS" if self.report["overall_pass"] else "FAIL")
            )
            print("=" * 60)
            return bool(self.report["overall_pass"])

        except Exception as exc:  # descriptive, no silent catches
            import traceback

            print(f"ERROR in Module 2 pipeline: {exc}")
            traceback.print_exc()
            return False


def main() -> int:
    """CLI entry point for the Module 2 pipeline."""
    pipeline = NLPPreparationPipeline()
    success = pipeline.run()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
