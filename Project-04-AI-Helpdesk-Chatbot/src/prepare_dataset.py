"""
Dataset Preparation Script for Internal Helpdesk Chatbot
Module 1: Complete Dataset Preparation, Domain Adaptation & Validation

Pipeline:
    load raw (faq_dataset.csv + Bitext) -> inspect -> search Bitext for
    internal-helpdesk matches -> load curated augmentation -> combine ->
    clean -> normalize intents -> remove duplicates -> validate intent
    balance -> validate -> save processed dataset -> generate report ->
    generate charts.

Key decisions
-------------
- The Bitext dataset is the ORIGINAL SOURCE dataset and is never modified.
- data/raw/faq_dataset.csv is the project internal-helpdesk candidate.
  After inspection it was found to be CLEAN but IMBALANCED (12 of 22
  intents had fewer than the required 10 examples) -> it is incomplete.
- Under-represented intents are augmented with carefully curated examples
  stored in src/augmentation_data.py.  A transparent copy is written to
  data/raw/faq_dataset_augmented.csv.
- Entities: an empty string means "no entity".

Author / project: Project-04-AI-Helpdesk-Chatbot (Module 1)
"""

import sys
import re
import json
from pathlib import Path
from datetime import datetime
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")  # non-interactive backend for scripts
import matplotlib.pyplot as plt

# Make `from augmentation_data import ...` work whether run as a script or
# when the package dir is not already on sys.path.
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from augmentation_data import AUGMENTED_DATA  # noqa: E402

# ---------------------------------------------------------------------------
# Schema / vocabulary
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS = ["question", "intent", "answer", "entity"]

# Canonical intent labels (consistent snake_case, internal-helpdesk only).
CANONICAL_INTENTS = [
    "account_access", "attendance", "contact_information", "email_problems",
    "employee_id", "goodbye", "greetings", "help", "holidays", "hr_support",
    "internet_problems", "laptop_problems", "leave_policy", "office_location",
    "password_reset", "payroll", "salary_information", "security",
    "software_installation", "technical_support", "wifi_problems",
    "working_hours",
]

# Canonical entity values used across the dataset. Empty string = no entity.
CANONICAL_ENTITIES = [
    "email", "account", "password", "laptop", "wifi", "software",
    "leave", "salary", "payroll", "internet", "working_hours",
    "employee_id", "holiday", "attendance", "hr", "security",
    "contact", "location", "it_support", "greeting", "goodbye", "help",
]

# Preferred entity per intent (used for augmented rows).
INTENT_ENTITY_MAP = {
    "account_access": "account", "attendance": "attendance",
    "contact_information": "contact", "email_problems": "email",
    "employee_id": "employee_id", "goodbye": "goodbye",
    "greetings": "greeting", "help": "help", "holidays": "holiday",
    "hr_support": "hr", "internet_problems": "internet",
    "laptop_problems": "laptop", "leave_policy": "leave",
    "office_location": "location", "password_reset": "password",
    "payroll": "payroll", "salary_information": "salary",
    "security": "security", "software_installation": "software",
    "technical_support": "it_support", "wifi_problems": "wifi",
    "working_hours": "working_hours",
}

# Quality thresholds
MIN_EXAMPLES_PER_INTENT = 10
PREFERRED_EXAMPLES_PER_INTENT = 12

# Bitext keyword scan: internal-helpdesk intent -> (Bitext intents,
# question keywords) that could plausibly match. Used only for the
# documented Bitext suitability scan (the Bitext set is never modified).
BITEXT_SEARCH_MAP = {
    "password_reset": {
        "bitext_intents": ["recover_password"],
        "keywords": ["password", "pin", "passcode", "forgot", "login"],
    },
    "account_access": {
        "bitext_intents": ["create_account", "edit_account", "delete_account",
                           "registration_problems", "switch_account"],
        "keywords": ["account", "login", "access", "registered"],
    },
    "contact_information": {
        "bitext_intents": ["contact_customer_service", "contact_human_agent"],
        "keywords": ["contact", "phone", "email", "reach", "talk"],
    },
}
class DatasetPreparation:
    """Load, clean, validate, balance and export the internal-helpdesk FAQ set.

    All paths are resolved relative to `project_root` so the pipeline is
    reproducible from any machine without hard-coded absolute paths.
    """

    def __init__(self, project_root=None):
        if project_root is None:
            self.project_root = Path(__file__).resolve().parent.parent
        else:
            self.project_root = Path(project_root)

        self.raw_data_path = self.project_root / "data" / "raw" / "faq_dataset.csv"
        self.bitext_path = (
            self.project_root / "data" / "raw"
            / "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
        )
        self.augmented_data_path = self.project_root / "data" / "raw" / "faq_dataset_augmented.csv"
        self.processed_data_path = self.project_root / "data" / "processed" / "faq_dataset.csv"
        self.reports_dir = self.project_root / "outputs" / "reports"
        self.charts_dir = self.project_root / "outputs" / "charts"

        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        (self.project_root / "data" / "processed").mkdir(parents=True, exist_ok=True)

        # Metadata collected during the run (all computed dynamically).
        self.stats = {}
        self.augmentation_log = []
        self.intent_normalization_map = {}
        self.bitext_scan = {}

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def load_raw_dataset(self):
        """Load data/raw/faq_dataset.csv (the project candidate dataset)."""
        if not self.raw_data_path.exists():
            raise FileNotFoundError(f"Raw dataset not found: {self.raw_data_path}")
        df = pd.read_csv(self.raw_data_path, encoding="utf-8-sig")
        df.columns = [c.strip() for c in df.columns]
        return df

    def load_bitext_dataset(self):
        """Load the Bitext source dataset (never modified)."""
        if not self.bitext_path.exists():
            raise FileNotFoundError(f"Bitext dataset not found: {self.bitext_path}")
        df = pd.read_csv(self.bitext_path, encoding="utf-8-sig")
        df.columns = [c.strip() for c in df.columns]
        return df

    def load_augmented_data(self):
        """Build a DataFrame from the curated augmentation module."""
        rows = [dict(question=q, intent=i, answer=a, entity=e)
                for (q, i, a, e) in AUGMENTED_DATA]
        return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)

    # ------------------------------------------------------------------
    # Bitext suitability scan (recorded, not used to relabel intents)
    # ------------------------------------------------------------------
    def scan_bitext_for_helpdesk_examples(self, bitext_df, augmented_df):
        """Search the Bitext dataset for semantically matching internal-helpdesk
        examples.  Results are logged in the report.  The Bitext set is
        e-commerce / customer-support oriented, so no matches are reused for the
        under-represented helpdesk intents, but the scan is documented.
        """
        scan = {}
        for intent, spec in BITEXT_SEARCH_MAP.items():
            candidates = bitext_df[
                bitext_df["intent"].astype(str).str.strip().isin(spec["bitext_intents"])
            ]
            questions = (
                candidates["instruction"].astype(str).str.lower()
                if "instruction" in candidates.columns else pd.Series(dtype=str)
            )
            matched = 0
            for kw in spec["keywords"]:
                matched += questions.str.contains(re.escape(kw), regex=True, na=False).sum()
            scan[intent] = {
                "bitext_intents": spec["bitext_intents"],
                "candidate_rows": int(len(candidates)),
                "keyword_matches": int(matched),
                "reused_count": 0,
                "reason": (
                    "Bitext rows are customer-support (orders/payments/accounts) "
                    "and are not semantically appropriate for internal helpdesk. "
                    "Curated augmentation used instead."
                ),
            }
        self.bitext_scan = scan
        return scan

    # ------------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------------
    def _safe_str(self, value):
        if pd.isna(value):
            return ""
        return str(value)
# ------------------------------------------------------------------
    # Cleaning
    # ------------------------------------------------------------------
    def clean_text(self, text):
        """Normalize question / answer text without destroying information.

        Keeps meaningful punctuation; only collapses excess whitespace,
        strips leading/trailing spaces and removes malformed control chars.
        Acts as a string (not tokenization - tokenization is Module 2).
        """
        s = self._safe_str(text)
        # Normalise unicode (e.g. smart quotes to their ASCII-safe forms).
        s = s.replace("\u2019", "'").replace("\u2018", "'")
        s = s.replace("\u201c", '"').replace("\u201d", '"')
        s = s.replace("\u2013", "-").replace("\u2014", "-")
        s = s.replace("\u00a0", " ")  # non-breaking space
        # Remove malformed control chars (keep newline/tab out of question text).
        s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", s)
        # Collapse repeated whitespace.
        s = re.sub(r"\s+", " ", s)
        return s.strip()

    def clean_questions(self, df):
        """Strip / normalise question text and drop empty questions."""
        df = df.copy()
        df["question"] = df["question"].apply(self.clean_text)
        df = df[df["question"] != ""].reset_index(drop=True)
        return df

    def clean_answers(self, df):
        """Strip answer text and drop empty / malformed answers."""
        df = df.copy()
        df["answer"] = df["answer"].apply(self.clean_text)
        df = df[df["answer"] != ""].reset_index(drop=True)
        return df

    def clean_entities(self, df):
        """Normalise the entity column: strip text; empty string means none."""
        df = df.copy()
        df["entity"] = df["entity"].apply(self._safe_str).apply(
            lambda s: s.strip() if s.strip() else ""
        )
        return df

    # ------------------------------------------------------------------
    # Intent normalization
    # ------------------------------------------------------------------
    def normalize_intent(self, intent):
        """Return the canonical snake_case intent label, or '' if invalid."""
        s = self._safe_str(intent).strip().lower()
        s = re.sub(r"\s+", "_", s)
        s = re.sub(r"[^a-z0-9_]", "", s)
        s = re.sub(r"_+", "_", s).strip("_")
        return s

    def normalize_intents(self, df):
        """Normalize intent labels and log any that changed from the raw form."""
        df = df.copy()
        mapping = {}
        for idx, raw in df["intent"].items():
            norm = self.normalize_intent(raw)
            if norm not in CANONICAL_INTENTS:
                # Some legitimately-used synonyms -> canonical merge.
                alias_map = {
                    "laptop_problem": "laptop_problems",
                    "laptop_issue": "laptop_problems",
                    "email_problem": "email_problems",
                    "wifi_problem": "wifi_problems",
                    "internet_problem": "internet_problems",
                    "software_install": "software_installation",
                    "salary": "salary_information",
                }
                norm = alias_map.get(norm, norm)
            key = str(raw).strip()
            if norm and norm != key:
                mapping[key] = norm
            df.at[idx, "intent"] = norm
        df = df[df["intent"] != ""].reset_index(drop=True)
        df = df[df["intent"].isin(CANONICAL_INTENTS)].reset_index(drop=True)
        self.intent_normalization_map = {k: v for k, v in mapping.items()}
        return df
# ------------------------------------------------------------------
    # Combine + balance
    # ------------------------------------------------------------------
    def combine(self, raw_df, aug_df):
        """Combine the raw base dataset with the curated augmentation."""
        combined = pd.concat([raw_df, aug_df], ignore_index=True)
        combined = combined[REQUIRED_COLUMNS].copy()
        return combined

    def remove_duplicates(self, df):
        """Remove duplicate questions across exact / case-insensitive /
        whitespace-normalized representations. Keep first occurrence.
        """
        df = df.copy()
        before = len(df)
        # Exact duplicates
        df = df.drop_duplicates(subset=["question"], keep="first")
        # Case-insensitive duplicates
        df["_q_lower"] = df["question"].str.lower()
        df = df.drop_duplicates(subset=["_q_lower"], keep="first")
        # Whitespace-normalized duplicates
        df["_q_norm"] = df["_q_lower"].str.replace(r"\s+", " ", regex=True).str.strip()
        df = df.drop_duplicates(subset=["_q_norm"], keep="first")
        df = df.drop(columns=["_q_lower", "_q_norm"]).reset_index(drop=True)
        self.stats["duplicates_removed"] = before - len(df)
        return df

    def validate_entities(self, df):
        """Check the entity column. Empty string = no entity (valid). Any
        non-empty value must be in the canonical entity set."""
        invalid = {}
        for idx, ent in df["entity"].items():
            e = str(ent).strip()
            if e and e not in CANONICAL_ENTITIES:
                invalid[idx] = e
        return invalid

    def fill_missing_entities(self, df, aug_df):
        """Backfill empty entities using the intent->entity map, but only for
        rows that came from the curated augmentation (tracked by question)."""
        aug_questions = set(aug_df["question"].astype(str))
        df = df.copy()
        for idx, row in df.iterrows():
            ent = str(row["entity"]).strip()
            if not ent and row["question"] in aug_questions:
                df.at[idx, "entity"] = INTENT_ENTITY_MAP.get(row["intent"], "")
        return df

    def ensure_intent_balance(self, df):
        """Verify every intent has at least MIN_EXAMPLES_PER_INTENT examples.

        The curated augmentation set below already brings every intent to
        >= PREFERRED_EXAMPLES_PER_INTENT. This guard re-checks and logs any
        intent it still needs to flag."""
        counts = df["intent"].value_counts().to_dict()
        below = {k: v for k, v in counts.items() if v < MIN_EXAMPLES_PER_INTENT}
        self.stats["intents_below_minimum"] = below
        return below

    # ------------------------------------------------------------------
    # Validation helpers (for report + quality gate)
    # ------------------------------------------------------------------
    def _missing_check(self, df, col):
        return int((pd.isna(df[col]) | (df[col].astype(str).str.strip() == "")).sum())

    def validate_dataset(self, df):
        """Return a dict of boolean results for the Module 1 quality gate."""
        res = {}
        res["columns_present"] = all(c in df.columns for c in REQUIRED_COLUMNS)
        res["not_empty"] = len(df) > 0
        res["missing_questions"] = self._missing_check(df, "question")
        res["missing_intents"] = self._missing_check(df, "intent")
        res["missing_answers"] = self._missing_check(df, "answer")
        res["duplicate_questions"] = int(df.duplicated(subset=["question"], keep=False).sum())
        res["q_ok"] = res["missing_questions"] == 0
        res["i_ok"] = res["missing_intents"] == 0
        res["a_ok"] = res["missing_answers"] == 0
        res["dup_ok"] = res["duplicate_questions"] == 0
        res["intents_valid"] = set(df["intent"]) <= set(CANONICAL_INTENTS)
        below = self.ensure_intent_balance(df)
        res["balance_ok"] = len(below) == 0
        res["invalid_entities"] = self.validate_entities(df)
        res["entity_ok"] = len(res["invalid_entities"]) == 0
        res["min_examples"] = int(df["intent"].value_counts().min())
        return res
# ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------
    def save_augmented_raw_csv(self, aug_df):
        """Write the curated augmentation to data/raw/ as a transparent copy.
        This is an ADDITION to raw data (never overwrites the base set)."""
        self.augmented_data_path.parent.mkdir(parents=True, exist_ok=True)
        aug_df.to_csv(self.augmented_data_path, index=False, encoding="utf-8")
        return self.augmented_data_path

    def save_processed_dataset(self, df):
        """Write the final clean Module 1 dataset to data/processed/."""
        self.processed_data_path.parent.mkdir(parents=True, exist_ok=True)
        df = df[REQUIRED_COLUMNS].copy()
        df.to_csv(self.processed_data_path, index=False, encoding="utf-8")
        return self.processed_data_path

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    def generate_report(self, df, raw_df, bitext_df, aug_df, validation):
        """Generate outputs/reports/dataset_report.txt with dynamically
        computed statistics (nothing hard-coded)."""
        report_path = self.reports_dir / "dataset_report.txt"

        counts = df["intent"].value_counts()
        min_examples = int(counts.min())
        max_examples = int(counts.max())
        avg_examples = round(float(counts.mean()), 2)
        num_intents = int(counts.shape[0])
        class_balance = round(min_examples / max_examples, 2) if max_examples else 0.0
        below = {k: v for k, v in counts.items() if v < MIN_EXAMPLES_PER_INTENT}
        entities = df["entity"].astype(str).fillna("").replace("", pd.NA).dropna()
        num_entities = int(entities.nunique())
        lines = []
        lines.append("=" * 76)
        lines.append("PROJECT 4 - MODULE 1 DATASET REPORT")
        lines.append("=" * 76)
        lines.append("")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        lines.append("DATASET OVERVIEW")
        lines.append("-" * 76)
        lines.append(f"Total Records: {len(df)}")
        lines.append(f"Total Intents: {num_intents}")
        lines.append(f"Total Entities: {num_entities}")
        lines.append("")
        lines.append("MISSING VALUES")
        lines.append("-" * 76)
        lines.append(f"Missing Questions: {validation['missing_questions']}")
        lines.append(f"Missing Intents: {validation['missing_intents']}")
        lines.append(f"Missing Answers: {validation['missing_answers']}")
        lines.append("")
        lines.append(f"Duplicate Questions: {validation['duplicate_questions']}")
        lines.append("")
        lines.append("INTENT BALANCE")
        lines.append("-" * 76)
        lines.append(f"Minimum Examples Per Intent: {min_examples}")
        lines.append(f"Maximum Examples Per Intent: {max_examples}")
        lines.append(f"Average Examples Per Intent: {avg_examples}")
        lines.append(f"Intents Below Minimum ({MIN_EXAMPLES_PER_INTENT}): "
                     f"{', '.join(sorted(below)) if below else 'None'}")
        lines.append(f"Class Balance Ratio: {class_balance}")
        lines.append("")
        lines.append("INTENT DISTRIBUTION")
        lines.append("-" * 76)
        for intent in sorted(counts.index):
            lines.append(f"{intent:<24} : {int(counts[intent]):>3}  "
                         f"({float(counts[intent]) / len(df) * 100:5.2f}%)")
        lines.append("")
        lines.append("INTERNAL HELPDESK DOMAIN VALIDATION")
        lines.append("-" * 76)
        valid_intents = set(df["intent"]) <= set(CANONICAL_INTENTS)
        lines.append(f"All intents are canonical internal-helpdesk intents: "
                     f"{'PASS' if valid_intents else 'FAIL'}")
        lines.append(f"All entities are canonical: "
                     f"{'PASS' if len(validation['invalid_entities']) == 0 else 'FAIL'}")
        lines.append("")

        lines.append("SOURCE DATASETS")
        lines.append("-" * 76)
        lines.append(f"Original Bitext records : {len(bitext_df)}")
        lines.append(f"Base faq_dataset.csv    : {len(raw_df)}")
        lines.append(f"Curated augmented       : {len(aug_df)}")
        for intent, info in self.bitext_scan.items():
            lines.append(f"  Bitext scan [{intent}]: {info['candidate_rows']} candidate rows, "
                         f"{info['keyword_matches']} keyword matches, "
                         f"{info['reused_count']} reused.")
        lines.append("")
        lines.append("AUGMENTATION LOG")
        lines.append("-" * 76)
        for q, intent in self.augmentation_log:
            lines.append(f"  [augmented] ({intent}) {q}")
        lines.append("")
        lines.append("INTENT NORMALIZATION MAP")
        lines.append("-" * 76)
        if self.intent_normalization_map:
            for k, v in self.intent_normalization_map.items():
                lines.append(f"  {k!r} -> {v!r}")
            lines.append("")
        else:
            lines.append("  (no intent relabeling occurred)")
            lines.append("")

        lines.append("QUALITY GATE")
        lines.append("-" * 76)
        records = [
            ("Required columns exist",
             validation["columns_present"] and validation["not_empty"]),
            ("No missing questions", validation["q_ok"]),
            ("No missing intents", validation["i_ok"]),
            ("No missing answers", validation["a_ok"]),
            ("No duplicate questions", validation["dup_ok"]),
            ("Minimum 10 examples per intent", validation["balance_ok"]),
            ("Intent labels valid", validation["intents_valid"]),
            ("Entity validation passed", validation["entity_ok"]),
        ]
        all_pass = all(p for _, p in records)
        for name, passed in records:
            lines.append(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        lines.append("")
        lines.append(f"Quality Gate: {'PASS' if all_pass else 'FAIL'}")
        lines.append("")
        lines.append("=" * 76)
        lines.append("MODULE 1: COMPLETE" if all_pass else "MODULE 1: NOT COMPLETE")
        lines.append("=" * 76)

        self.reports_dir.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return report_path
# ------------------------------------------------------------------
    # Charts
    # ------------------------------------------------------------------
    def generate_charts(self, df):
        """Render intent distribution bar + pie charts (png)."""
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        counts = df["intent"].value_counts().sort_values()

        # Bar chart
        fig, ax = plt.subplots(figsize=(10, 8))
        bars = ax.barh(counts.index, counts.values, color="steelblue")
        ax.set_xlabel("Number of Examples", fontsize=11)
        ax.set_ylabel("Intent", fontsize=11)
        ax.set_title("Intent Distribution (processed dataset)", fontsize=13,
                     fontweight="bold")
        for b in bars:
            ax.text(b.get_width() + 0.15, b.get_y() + b.get_height() / 2,
                    str(int(b.get_width())), va="center", fontsize=9)
        # Mark the minimum threshold
        ax.axvline(MIN_EXAMPLES_PER_INTENT, color="red", linestyle="--", linewidth=1)
        ax.text(MIN_EXAMPLES_PER_INTENT, len(counts) - 0.5, "  min=10",
                color="red", fontsize=8)
        fig.tight_layout()
        bar_path = self.charts_dir / "intent_distribution.png"
        fig.savefig(bar_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Pie chart
        fig2, ax2 = plt.subplots(figsize=(9, 8))
        counts_sorted = df["intent"].value_counts().sort_values(ascending=False)
        colors = plt.cm.Set3(np.linspace(0, 1, len(counts_sorted)))
        wedges, _texts, autotexts = ax2.pie(
            counts_sorted.values, labels=None, autopct="%1.1f%%",
            colors=colors, startangle=90)
        ax2.legend(wedges, [f"{i} ({counts_sorted[i]})" for i in counts_sorted.index],
                   title="Intents", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1),
                   fontsize=9)
        ax2.set_title("Intent Distribution (%) - processed dataset", fontsize=13,
                      fontweight="bold")
        fig2.tight_layout()
        pie_path = self.charts_dir / "intent_distribution_pie.png"
        fig2.savefig(pie_path, dpi=150, bbox_inches="tight")
        plt.close(fig2)

        return [bar_path, pie_path]
# ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------
    def run(self, verbose=True):
        """Run the complete Module 1 pipeline. Returns True on success."""
        def log(msg):
            if verbose:
                print(msg)

        try:
            log("=" * 60)
            log("MODULE 1: DATASET PREPARATION")
            log("=" * 60)

            # 1. Load
            raw = self.load_raw_dataset()
            bitext = self.load_bitext_dataset()
            aug = self.load_augmented_data()
            log(f"Loaded raw faq_dataset.csv  : {len(raw)} records")
            log(f"Loaded Bitext dataset        : {len(bitext)} records")
            log(f"Loaded curated augmentation  : {len(aug)} records")

            # 2. Inspect / validate raw
            raw_valid = {
                "rows": len(raw),
                "columns": list(raw.columns),
                "missing": {c: int(raw[c].isna().sum()) for c in raw.columns},
                "dup_rows": int(raw.duplicated().sum()),
                "dup_q": int(raw.duplicated(subset=["question"]).sum()),
                "intents": int(raw["intent"].nunique()),
            }
            self.stats["raw"] = raw_valid

            # 3. Bitext suitability scan
            self.scan_bitext_for_helpdesk_examples(bitext, aug)

            # 4. Clean
            df = self.clean_questions(raw)
            df = self.clean_answers(df)
            df = self.clean_entities(df)

            # 5. Normalize intents
            df = self.normalize_intents(df)

            # 6. Combine base + augmentation
            df = self.combine(df, aug)

            # 7. Backfill entities for augmented rows, then re-clean
            df = self.fill_missing_entities(df, aug)
            df = self.clean_questions(df)
            df = self.clean_answers(df)
            df = self.clean_entities(df)

            # 8. Remove duplicates
            df = self.remove_duplicates(df)

            # 9. Save augmented raw copy (transparency)
            aug_path = self.save_augmented_raw_csv(aug)

            # 10. Record augmentation log
            aug_questions = set(aug["question"].astype(str))
            self.augmentation_log = [
                (q, i) for q, i, _a, _e in AUGMENTED_DATA
            ]
            # Synchronize with final dataset presence
            final_qs = set(df["question"].astype(str))

            # 11. Validate
            validation = self.validate_dataset(df)

            # 12. Save processed + report + charts
            proc_path = self.save_processed_dataset(df)
            report_path = self.generate_report(df, raw, bitext, aug, validation)
            chart_paths = self.generate_charts(df)

            # 13. Summary
            log("\n--- SUMMARY ---")
            log(f"Processed records       : {len(df)}")
            log(f"Intents                 : {df['intent'].nunique()}")
            log(f"Min examples per intent : {validation['min_examples']}")
            log(f"Missing questions       : {validation['missing_questions']}")
            log(f"Duplicate questions     : {validation['duplicate_questions']}")
            log(f"Processed dataset       : {proc_path}")
            log(f"Report                  : {report_path}")
            log(f"Charts                  : {', '.join(str(p) for p in chart_paths)}")
            log(f"\nQuality Gate: {'PASS' if all([
                validation['columns_present'], validation['not_empty'],
                validation['q_ok'], validation['i_ok'], validation['a_ok'],
                validation['dup_ok'], validation['balance_ok'],
                validation['intents_valid'], validation['entity_ok'],
            ]) else 'FAIL'}")
            return True

        except Exception as e:
            import traceback
            log(f"ERROR in Module 1 pipeline: {e}")
            traceback.print_exc()
            return False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    """CLI entry point."""
    prep = DatasetPreparation()
    success = prep.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
