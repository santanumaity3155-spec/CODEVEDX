from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = PROJECT_ROOT / "logs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

MOVIES_PATH = RAW_DATA_DIR / "ml-25m" / "movies.csv"
RATINGS_PATH = RAW_DATA_DIR / "ml-25m" / "ratings.csv"
TAGS_PATH = RAW_DATA_DIR / "ml-25m" / "tags.csv"
LINKS_PATH = RAW_DATA_DIR / "ml-25m" / "links.csv"
GENOME_SCORES_PATH = RAW_DATA_DIR / "ml-25m" / "genome-scores.csv"
GENOME_TAGS_PATH = RAW_DATA_DIR / "ml-25m" / "genome-tags.csv"

DATASET_NAME = "ml-25m"
