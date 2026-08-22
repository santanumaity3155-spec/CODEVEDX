"""Central configuration for the Smart Recommendation System project.

All paths are resolved relative to the project root so that modules,
scripts, notebooks, and tests behave identically regardless of the
caller's working directory.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHARTS_DIR = OUTPUTS_DIR / "charts"
REPORTS_DIR = OUTPUTS_DIR / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

MOVIES_PATH = RAW_DATA_DIR / "ml-25m" / "movies.csv"
RATINGS_PATH = RAW_DATA_DIR / "ml-25m" / "ratings.csv"
TAGS_PATH = RAW_DATA_DIR / "ml-25m" / "tags.csv"
LINKS_PATH = RAW_DATA_DIR / "ml-25m" / "links.csv"
GENOME_SCORES_PATH = RAW_DATA_DIR / "ml-25m" / "genome-scores.csv"
GENOME_TAGS_PATH = RAW_DATA_DIR / "ml-25m" / "genome-tags.csv"

DATASET_NAME = "ml-25m"

# Processed dataset file names (Module 1 outputs).
PROCESSED_MOVIES_PATH = PROCESSED_DATA_DIR / "movies_clean.csv"
PROCESSED_RATINGS_PATH = PROCESSED_DATA_DIR / "ratings_clean.csv"
PROCESSED_TAGS_PATH = PROCESSED_DATA_DIR / "tags_clean.csv"
PROCESSED_LINKS_PATH = PROCESSED_DATA_DIR / "links_clean.csv"
PROCESSED_MOVIES_FEATURES_PATH = PROCESSED_DATA_DIR / "movies_features_base.csv"

# Report/chart outputs.
QUALITY_REPORT_TXT_PATH = REPORTS_DIR / "dataset_quality_report.txt"
QUALITY_REPORT_JSON_PATH = REPORTS_DIR / "dataset_quality_report.json"

# Expected raw schemas. Column order is not enforced; only presence.
EXPECTED_SCHEMAS = {
    "movies": ("movieId", "title", "genres"),
    "ratings": ("userId", "movieId", "rating", "timestamp"),
    "tags": ("userId", "movieId", "tag", "timestamp"),
    "links": ("movieId", "imdbId", "tmdbId"),
    "genome_scores": ("movieId", "tagId", "relevance"),
    "genome_tags": ("tagId", "tag"),
}

# MovieLens 25M rating domain: half-star scale from 0.5 to 5.0.
RATING_MIN = 0.5
RATING_MAX = 5.0
RATING_STEP = 0.5

# MovieLens uses this literal when a movie has no genres at all.
NO_GENRES_LABEL = "(no genres listed)"

# Fixed random seed so any stochastic step stays reproducible.
RANDOM_SEED = 42

# Minimum number of ratings a movie must have to appear in the
# top-popular-movies report (popularity = rating count).
TOP_MOVIES_MIN_RATINGS = 50
TOP_MOVIES_REPORT_SIZE = 100

LOG_FILE_NAME = "pipeline.log"
