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

# ---------------------------------------------------------------------------
# Module 2 - Content feature engineering outputs
# ---------------------------------------------------------------------------
# Module 2 builds ML-ready movie representations from the cleaned MovieLens
# catalog and user tags (processed Module 1 data only - never raw files).
PROCESSED_MOVIE_CONTENT_FEATURES_PATH = (
    PROCESSED_DATA_DIR / "movie_content_features.csv"
)
PROCESSED_MOVIE_GENRE_FEATURES_PATH = (
    PROCESSED_DATA_DIR / "movie_genre_features.csv"
)
PROCESSED_MOVIE_TFIDF_PATH = PROCESSED_DATA_DIR / "movie_tfidf.npz"
PROCESSED_MOVIE_FEATURE_INDEX_PATH = (
    PROCESSED_DATA_DIR / "movie_feature_index.csv"
)

# Fitted TF-IDF vectorizer (joblib pickle). Loaded independently by Module 3.
MOVIE_TFIDF_VECTORIZER_PATH = MODELS_DIR / "movie_tfidf_vectorizer.pkl"

# Module 2 reports.
FEATURE_ENGINEERING_REPORT_JSON_PATH = (
    REPORTS_DIR / "feature_engineering_report.json"
)
FEATURE_ENGINEERING_REPORT_TXT_PATH = (
    REPORTS_DIR / "feature_engineering_report.txt"
)

# TF-IDF configuration.
# IMPORTANT: the vectorizer is fit ONLY on the movie-content corpus so the
# vocabulary leaks no information from ratings, predictions or evaluation data.
TFIDF_NGRAM_RANGE = (1, 2)       # unigrams + bigrams capture multi-word tags/terms
TFIDF_MIN_DF = 2                 # drop terms appearing in fewer than 2 documents
TFIDF_MAX_DF = 0.90              # drop corpus-frequent terms (generic noise)
TFIDF_SUBLINEAR_TF = True        # dampen term-frequency saturation (1 + log(tf))
TFIDF_MAX_FEATURES = 20000        # cap vocabulary size for bounded memory
TFIDF_NORM = "l2"                # cosine-friendly row normalization
TFIDF_DTYPE = "float32"          # half-precision sparse matrix to limit memory

# Token pattern: Unicode alphanumeric runs with internal hyphens preserved so
# MovieLens genres/tags such as "sci-fi" and "film-noir" stay single terms and
# non-ASCII movie titles (Arabic/Greek/Cyrillic) still yield content tokens.
# `[^\W_]` is a Unicode-aware "word char except underscore", so meaningful
# foreign-language title words are retained instead of producing empty documents.
TFIDF_TOKEN_PATTERN = r"[^\W_]+(?:-[^\W_]+)*"

# ---------------------------------------------------------------------------
# Module 3 - Recommendation engine (similarity, recommender, evaluation)
# ---------------------------------------------------------------------------
# Report outputs (same conventions as Modules 1-2: JSON + TXT pairs).
RECOMMENDATION_REPORT_JSON_PATH = (
    REPORTS_DIR / "recommendation_quality_report.json"
)
RECOMMENDATION_REPORT_TXT_PATH = REPORTS_DIR / "recommendation_quality_report.txt"
EVALUATION_REPORT_JSON_PATH = REPORTS_DIR / "evaluation_report.json"
EVALUATION_REPORT_TXT_PATH = REPORTS_DIR / "evaluation_report.txt"
MODULE3_QUALITY_GATE_JSON_PATH = REPORTS_DIR / "module3_quality_gate_report.json"
MODULE3_QUALITY_GATE_TXT_PATH = REPORTS_DIR / "module3_quality_gate_report.txt"

# Default number of recommendations returned per request.
DEFAULT_TOP_K = 10

# Quality control: a movie must have at least this many ratings to be
# recommended. Prevents obscure titles with a handful of interactions from
# dominating similarity rankings. Set to 0 (or None) to disable filtering.
# Aligned with the Module 1 popularity-report threshold for consistency.
RECOMMENDATION_MIN_MOVIE_RATINGS = 50

# Number of seed movies showcased in the recommendation-quality report.
# Seeds are resolved dynamically by searching real catalog titles
# (case-insensitive substring match, most-rated match wins).
RECOMMENDATION_REPORT_NUM_SEEDS = 5
RECOMMENDATION_SEED_TITLE_HINTS = (
    "Toy Story",
    "Braveheart",
    "Jurassic Park",
    "Matrix",
    "Back to the Future",
)

# Chunk size (rows) when streaming the large ratings_clean.csv for
# per-movie popularity aggregation.
POPULARITY_CHUNKSIZE = 5_000_000

# Dtypes used when reading the processed ratings dataset.
PROCESSED_RATINGS_DTYPE = {
    "userId": "int32",
    "movieId": "int32",
    "rating": "float32",
    "timestamp": "int64",
}

# --- Offline evaluation configuration ---------------------------------------
# Protocol (leakage-free, documented in src/evaluation.py):
#   * interactions are split PER USER by timestamp;
#   * the most recent `EVALUATION_TEST_FRACTION` share (at least
#     `EVALUATION_MIN_TEST_ITEMS`) of each evaluated user's interactions
#     becomes the held-out ground truth ("test");
#   * all earlier interactions ("train") drive the user profile vector and
#     the already-seen exclusion set;
#   * a test movie counts as relevant when its rating is
#     >= EVALUATION_LIKE_THRESHOLD (a "liked" movie).
EVALUATION_K_VALUES = (5, 10)
EVALUATION_LIKE_THRESHOLD = 4.0
EVALUATION_TEST_FRACTION = 0.2
EVALUATION_MIN_TEST_ITEMS = 1
EVALUATION_MIN_USER_RATINGS = 5      # users with fewer interactions are skipped
EVALUATION_MAX_USERS = 1500          # cap evaluated users for bounded runtime
EVALUATION_PROGRESS_LOG_EVERY = 250  # log progress every N evaluated users
TFIDF_TOKEN_PATTERN = r"[^\W_]+(?:-[^\W_]+)*"
