"""Build notebooks/data_preparation.ipynb with real executable cells."""
import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "notebooks" / "data_preparation.ipynb"

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


# ---------------------------------------------------------------------------
md("""# Data Preparation Notebook — Module 1
### AI Chatbot for Internal Helpdesk (Project-04)

This notebook walks through **Module 1: Complete Dataset Preparation, Domain
Adaptation and Validation**. It performs the same pipeline as
`src/prepare_dataset.py`:

1. Module 1 Overview
2. Load Raw Datasets
3. Inspect Dataset Structure
4. Missing Value Analysis
5. Duplicate Analysis
6. Intent Distribution
7. Internal Helpdesk Domain Analysis
8. Data Cleaning
9. Intent Normalization
10. Intent Balancing
11. Entity Validation
12. Final Dataset Validation
13. Dataset Statistics
14. Visualizations
15. Save Processed Dataset
16. Final Quality Gate
""")

# ---------------------------------------------------------------------------
md("## 1. Module 1 Overview")

md("""### Objective
Produce a clean, validated, reproducible **internal helpdesk FAQ dataset**
suitable for Module 2 (NLP preprocessing), Module 3 (intent classification)
and Module 4 (chatbot engine).

### Inputs
- **Bitext dataset** (`data/raw/Bitext_...csv`) — original source, 26,872
  e-commerce records. Never modified.
- **`data/raw/faq_dataset.csv`** — project internal-helpdesk candidate,
  159 records, 22 intents. Clean but imbalanced.
- **`src/augmentation_data.py`** — 135 curated examples that bring every
  intent to >= 12 examples.

### Outputs
- `data/processed/faq_dataset.csv` — final clean dataset.
- `outputs/reports/dataset_report.txt` — data quality report.
- `outputs/charts/intent_distribution.png` + `intent_distribution_pie.png`.
""")

code("""import sys
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np

# Project root = the working directory for this notebook.
PROJECT_ROOT = Path.cwd().resolve()
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from prepare_dataset import DatasetPreparation, REQUIRED_COLUMNS, \\
    CANONICAL_INTENTS, CANONICAL_ENTITIES, AUGMENTED_DATA, \\
    MIN_EXAMPLES_PER_INTENT

prep = DatasetPreparation(PROJECT_ROOT)
print("Project root:", PROJECT_ROOT)
print("Required columns:", REQUIRED_COLUMNS)
print("Canonical intents:", len(CANONICAL_INTENTS), "| Canonical entities:", len(CANONICAL_ENTITIES))
print("Curated augmentation entries:", len(AUGMENTED_DATA))
""")

# ---------------------------------------------------------------------------
md("## 2. Load Raw Datasets")

code("""# Bitext (original source) + base faq_dataset.csv + curated augmentation
bitext = prep.load_bitext_dataset()
raw = prep.load_raw_dataset()
aug = prep.load_augmented_data()

print("Bitext dataset       :", bitext.shape)
print("Base faq_dataset.csv :", raw.shape)
print("Curated augmentation :", aug.shape)
""")

code("""raw.head()""")

# ---------------------------------------------------------------------------
md("## 3. Inspect Dataset Structure")

code("""print("=== Bitext ===")
print("Columns:", list(bitext.columns))
print("N unique intents:", bitext["intent"].nunique())
print()
print("=== faq_dataset.csv ===")
print("Columns:", list(raw.columns))
print("dtypes:\\n", raw.dtypes)
print()
print("=== Curated augmentation ===")
print("Columns:", list(aug.columns))
print("dtypes:\\n", aug.dtypes)
""")

# ---------------------------------------------------------------------------
md("## 4. Missing Value Analysis")

code("""print("=== Bitext missing ===")
print(bitext.isna().sum().to_dict())
print()
print("=== faq_dataset.csv missing ===")
print(raw.isna().sum().to_dict())
print()
print("=== Augmentation missing ===")
print(aug.isna().sum().to_dict())
""")

code("""for col in REQUIRED_COLUMNS:
    n_missing = raw[col].isna().sum()
    n_empty = (raw[col].astype(str).str.strip() == "").sum()
    print(f"{col}: NaN={n_missing}  empty=''={n_empty}")
""")

# ---------------------------------------------------------------------------
md("## 5. Duplicate Analysis")

code("""print("Bitext duplicate rows:", bitext.duplicated().sum())
print("Base duplicate rows  :", raw.duplicated().sum())
print("Base duplicate questions:", raw.duplicated(subset=["question"]).sum())
print("Augmentation duplicate questions:", aug.duplicated(subset=["question"]).sum())
""")
# ---------------------------------------------------------------------------
md("## 6. Intent Distribution")

code("""base_dist = raw["intent"].value_counts()
print("Intent distribution in faq_dataset.csv (before balancing):")
print(base_dist.to_string())
print()
print("Intents below minimum (< 10):",
      sorted(base_dist[base_dist < MIN_EXAMPLES_PER_INTENT].index.tolist()))
""")

# ---------------------------------------------------------------------------
md("## 7. Internal Helpdesk Domain Analysis")

code("""base_intents = set(raw["intent"].astype(str).str.strip())
not_canonical = base_intents - set(CANONICAL_INTENTS)
print("Base intents not in canonical set:", sorted(not_canonical) if not_canonical else "None")
required = set(CANONICAL_INTENTS)
print("Required intents:", len(required))
print("Present intents :", len(base_intents))
print("Missing intents :", sorted(required - base_intents) or "None")
""")

# ---------------------------------------------------------------------------
md("## 8. Data Cleaning")

code("""cleaned = prep.clean_questions(raw)
cleaned = prep.clean_answers(cleaned)
cleaned = prep.clean_entities(cleaned)
print("Records after cleaning:", len(cleaned))
print()
print("Sample clean question:", repr(cleaned["question"].iloc[0]))
print("Sample clean answer  :", repr(cleaned["answer"].iloc[0]))
""")

# ---------------------------------------------------------------------------
md("## 9. Intent Normalization")

code("""cleaned = prep.normalize_intents(cleaned)
print("Intent normalization map:", prep.intent_normalization_map or "None")
print("Unique intents after normalization:", cleaned["intent"].nunique())
print("Intents present:", sorted(cleaned["intent"].unique()))
""")

# ---------------------------------------------------------------------------
md("## 10. Intent Balancing")

code("""# Combine cleaned base set with curated augmentation and dedupe.
combined = prep.combine(cleaned, aug)
combined = prep.fill_missing_entities(combined, aug)
combined = prep.clean_questions(combined)
combined = prep.clean_answers(combined)
combined = prep.clean_entities(combined)
combined = prep.remove_duplicates(combined)

dist = combined["intent"].value_counts()
print("Total records after balancing:", len(combined))
print()
print("Final intent distribution:")
print(dist.to_string())
print()
print("Min examples per intent:", int(dist.min()))
print("Intents below minimum:", prep.ensure_intent_balance(combined) or "None")
""")

# ---------------------------------------------------------------------------
md("## 11. Entity Validation")

code("""invalid_entities = prep.validate_entities(combined)
print("Invalid (non-canonical) entities:", invalid_entities or "None")
print()
ents = combined["entity"].astype(str).str.strip()
print("Entity counts (non-empty):")
print(ents[ents != ""].value_counts().to_string())
""")
# ---------------------------------------------------------------------------
md("## 12. Final Dataset Validation")

code("""validation = prep.validate_dataset(combined)
checks = {
    "columns present": validation["columns_present"],
    "not empty": validation["not_empty"],
    "no missing questions": validation["q_ok"],
    "no missing intents": validation["i_ok"],
    "no missing answers": validation["a_ok"],
    "no duplicate questions": validation["dup_ok"],
    "min 10 per intent": validation["balance_ok"],
    "intents valid": validation["intents_valid"],
    "entity validation": validation["entity_ok"],
}
for name, ok in checks.items():
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
""")

# ---------------------------------------------------------------------------
md("## 13. Dataset Statistics")

code("""stats = {
    "total_records": len(combined),
    "total_intents": combined["intent"].nunique(),
    "total_entities": combined["entity"].astype(str).str.strip().replace("", np.nan).dropna().nunique(),
    "missing_questions": validation["missing_questions"],
    "missing_intents": validation["missing_intents"],
    "missing_answers": validation["missing_answers"],
    "duplicate_questions": validation["duplicate_questions"],
    "min_per_intent": int(combined["intent"].value_counts().min()),
    "max_per_intent": int(combined["intent"].value_counts().max()),
    "avg_per_intent": round(float(combined["intent"].value_counts().mean()), 2),
}
for k, v in stats.items():
    print(f"{k}: {v}")
""")

# ---------------------------------------------------------------------------
md("## 14. Visualizations")

code("""from IPython.display import Image, display
chart_paths = prep.generate_charts(combined)
for p in chart_paths:
    display(Image(filename=str(p)))
""")

# ---------------------------------------------------------------------------
md("## 15. Save Processed Dataset")

code("""proc_path = prep.save_processed_dataset(combined)
aug_path = prep.save_augmented_raw_csv(aug)

print("Saved processed dataset  :", proc_path)
print("Saved augmented raw copy :", aug_path)

final = pd.read_csv(proc_path)
print("\\nProcessed dataset shape:", final.shape)
print("Columns:", list(final.columns))
""")

# ---------------------------------------------------------------------------
md("## 16. Final Quality Gate")

code("""report_path = prep.generate_report(combined, raw, bitext, aug, validation)
print("Report written to:", report_path)

all_pass = all([
    validation["columns_present"], validation["not_empty"], validation["q_ok"],
    validation["i_ok"], validation["a_ok"], validation["dup_ok"],
    validation["balance_ok"], validation["intents_valid"],
    validation["entity_ok"],
])
print()
print("=" * 46)
print("MODULE 1: COMPLETE" if all_pass else "MODULE 1: NOT COMPLETE")
print("=" * 46)
""")

nb.cells = cells
NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
NOTEBOOK.write_text(nbf.writes(nb) + "\n", encoding="utf-8")
print("Notebook written:", NOTEBOOK)