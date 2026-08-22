# Project 05 — Smart Recommendation System

A movie recommendation system built on the **MovieLens 25M** dataset.
The project is developed in staged modules; **Module 1 (data foundation,
exploration & preprocessing), Module 2 (content feature engineering) and
Module 3 (content-based recommendation engine + offline evaluation) are
complete**. Later modules will add personalization and a final application
layer.

## Objective

Build a production-quality, reproducible data foundation that later
recommendation modules can rely on: robust loading, schema validation,
profiling, data-quality checks (missing values, duplicates, referential
integrity), documented cleaning, statistical analysis, visualizations,
processed datasets, and automated tests — followed by ML-ready **content
features** (normalized documents, multi-hot genres, aggregated tags, sparse
TF-IDF matrix, feature index, serialized vectorizer).

## Dataset

- Source: [MovieLens 25M](https://grouplens.org/datasets/movielens/25m/) (ml-25m)
- Location: `data/raw/ml-25m/` (**read-only** - never modified by code or tests)

| File | Contents |
|---|---|
| `movies.csv` | movieId, title, genres |
| `ratings.csv` | userId, movieId, rating (0.5-5.0 half-star), timestamp |
| `tags.csv` | userId, movieId, tag, timestamp |
| `links.csv` | movieId, imdbId, tmdbId |
| `genome-scores.csv` | movieId, tagId, relevance |
| `genome-tags.csv` | tagId, tag |

## Module 1 scope (complete)

1. Dataset discovery and validation
2. Robust, memory-conscious loading with efficient dtypes (`src/data_loader.py`)
3. Schema validation for all six files (`src/data_quality.py`)
4. Profiling: rows/columns/dtypes/memory/unique values per dataset
5. Missing-value analysis (per column)
6. Duplicate analysis with domain-appropriate rules:
   - exact rating repeats removed; legitimate user re-ratings retained
   - duplicate titles across distinct movieIds reported, not removed
7. Referential integrity: orphan records quantified (ratings/tags/links/genome vs. movies; genome scores vs. genome tags)
8. Rating-domain validation (0.5-5.0 half-star grid)
9. Cleaning with fully documented rules and removal counts (`src/data_preprocessor.py`)
10. Analyses: ratings, users, movies/popularity, genres, tags (`src/analysis.py`)
11. Visualizations (`src/visualization.py`, 5 charts)
12. Quality reports in TXT + JSON (`src/reporting.py`)
13. Executed exploration notebook (`notebooks/module_1_data_exploration.ipynb`)
14. Automated test suite (`tests/`, pytest)

Not in Module 1: TF-IDF/similarity models, collaborative filtering,
evaluation metrics, personalization, UI.

## Module 2 scope (complete)

Module 2 converts the cleaned MovieLens catalog (`movies_clean.csv`) plus
user tags (`tags_clean.csv`) into machine-learning-ready content features for
the recommendation engine (`src/feature_engineering.py`).

1. Deterministic text normalization (`normalize_text`) - lowercase, whitespace,
   punctuation, missing values; preserves meaningful title/genre/tag words
   including hyphenated terms (`sci-fi`) and non-ASCII titles.
2. Multi-label genre features (`build_genre_features`) - one deterministic
   multi-hot column per distinct genre, vocabulary **derived from the data**
   (never hardcoded), `(no genres listed)` handled structurally.
3. Tag aggregation (`aggregate_tags`) - tags grouped by `movieId`, normalized,
   blank/empty values dropped, deduplicated per movie, deterministically
   ordered (global frequency desc, then alphabetical).
4. Content documents (`build_content_documents`) - one row per movie combining
   stripped-title tokens + normalized genres + aggregated tags; movies without
   tags still get valid documents from title + genres.
5. TF-IDF representation (`fit_tfidf`, scikit-learn `TfidfVectorizer`) fit
   **only on the movie corpus** (no ratings / evaluation leakage).
6. Sparse feature matrix (`movie_tfidf.npz`, scipy CSR, `float32`) - the full
   matrix is never densified.
7. Feature index (`movie_feature_index.csv`) - deterministic contiguous
   `movieId -> row_index` mapping aligned with matrix rows.
8. Vectorizer serialization (`models/movie_tfidf_vectorizer.pkl`, joblib) with
   independent `load_feature_vectorizer()` (verified to reproduce the matrix).
9. Feature quality validation (12 checks: uniqueness, coverage, no NaN/Inf,
   sparsity-friendly shapes, determinism, tagless-movie vectors).
10. Reports in `outputs/reports/feature_engineering_report.{json,txt}`.

Not in Module 2: cosine similarity, nearest-neighbor / personalized
recommendations, collaborative filtering, QA/UI/ranking - those belong to
later modules.

### Feature engineering architecture

```
movies_clean.csv + tags_clean.csv
        |
        v
text/genre normalization -> genre multi-hot -> aggregated tags
        |
        v
  content document per movie (movie_content_features.csv)
        |
        v
  TF-IDF vectorizer (fit on movie corpus only)
        |
        v
  sparse CSR matrix (62423 x V, float32) + feature index + vectorizer.pkl
        |
        v
  validation + reports (JSON/TXT)
```

### Module 2 TF-IDF configuration

Chosen for the 62k-movie corpus; defined in `src/config.py`:

| Parameter     | Value      | Rationale |
|---|---|---|
| `ngram_range` | `(1, 2)`   | unigrams + bigrams capture multi-word tag/term phrases |
| `min_df`      | `2`        | drop terms seen in a single document (noise) |
| `max_df`      | `0.90`     | drop corpus-frequent generic terms |
| `max_features`| `20_000`   | bounded memory for the sparse matrix |
| `sublinear_tf`| `True`     | dampen term-frequency saturation (`1 + log(tf)`) |
| `norm`        | `"l2"`     | cosine-friendly row normalization (used by Module 3) |
| `dtype`       | `float32`  | halves dense-format memory footprint |
| `token_pattern` | `[^\W_]+(?:-[^\W_]+)*` | Unicode-aware; keeps `sci-fi`, `film-noir`, and foreign-language title words |

Sparsity is ~99.9% (62,423 x 20,000 matrix with ~1.25M non-zero values stored
in ~6.5 MiB).

## How to run preprocessing

```bash
pip install -r requirements.txt
python -m src.pipeline
```

The pipeline is deterministic and idempotent; rerunning it on the same
raw data reproduces equivalent outputs. It refuses to finalize if any
critical integrity check fails (final gate PASS/FAIL).

## How to run Module 2 (content feature engineering)

Requires the Module 1 processed outputs (`data/processed/movies_clean.csv`,
`tags_clean.csv`):

```bash
python -m src.feature_engineering
```

The feature pipeline is deterministic and idempotent: same processed inputs
always produce the same movie ordering, feature index, vocabulary and a
numerically equal TF-IDF matrix. It refuses to finalize unless every feature
quality check passes (12/12, gate PASS).

## How to run Module 3 (recommendation engine)

Requires the Module 1 processed outputs and the Module 2 feature artifacts:

```bash
python -m src.pipeline --module 3                 # full run incl. regression tests
python -m src.pipeline --module 3 --skip-regression-tests   # faster iteration
python scripts/smoke_test_recommendations.py      # real-data demo
```

The pipeline is deterministic: same inputs produce identical recommendations,
feature-index ordering and reports. It loads the Module 2 artifacts as-is
(the vectorizer is verified, never re-fit), generates sample recommendations,
runs the leakage-free offline evaluation, executes the Module 3 quality
checks (optionally the full pytest regression suite) and writes JSON + TXT
reports under `outputs/reports/`.

Programmatic use:

```python
from src.recommender import ContentRecommender

recommender = ContentRecommender.from_artifacts()
recs = recommender.recommend_similar_movies(1, top_k=10)
# columns: rank, movieId, title, genres, similarity
```

### Module 3 design notes

* **Similarity** - exact cosine over the sparse TF-IDF rows; the 62,423 x
  20,000 matrix is never densified (one dense score vector per query).
  Deterministic ordering: descending similarity, ties broken by ascending
  `movieId`. The seed movie can never be returned; unknown IDs raise
  `MovieNotFoundError` at engine level and degrade gracefully (empty result)
  at the recommender level.
* **Quality control** - movies with fewer than
  `RECOMMENDATION_MIN_MOVIE_RATINGS` ratings (configurable) are excluded from
  candidates; movies with missing titles/genres are never recommended.
  Optional per-call filters: `genres`, `min_year`, `max_year`, `min_ratings`.
* **Offline evaluation protocol (leakage-free)** - for each evaluated user the
  interactions are split by timestamp: the newest
  `EVALUATION_TEST_FRACTION` share (at least `EVALUATION_MIN_TEST_ITEMS`)
  becomes held-out ground truth; all earlier interactions form the history.
  The content profile is the mean TF-IDF vector of history movies rated >=
  `EVALUATION_LIKE_THRESHOLD`; every history movie is excluded as a candidate;
  relevant items are liked test movies only. Metrics: Precision@K,
  Recall@K, HitRate@K, MAP@K, NDCG@K for each configured K.
* **Personalization** remains a placeholder (`src/personalization.py`) and is
  scheduled for a later module, per the original module roadmap.

## How to run tests

```bash
python -m pytest tests -q            # full suite (Modules 1-3)
python -m pytest tests/test_similarity_engine.py tests/test_recommender.py \
    tests/test_evaluation.py tests/test_recommendation_pipeline.py -v  # Module 3 only
```

Tests never modify `data/raw` or the real `data/processed` / `models`
files. Synthetic end-to-end pipeline tests run inside temporary directories
only; real-data tests are read-only. A baseline SHA-256 manifest
(`tests/baseline_raw_manifest.json`) proves raw files remain unchanged.

## Generated outputs

Processed datasets (`data/processed/`):

- `movies_clean.csv` - cleaned catalog (invalid titles dropped, missing genres -> `(no genres listed)`)
- `ratings_clean.csv` - valid-scale ratings, no exact duplicates, no orphan movieIds
- `tags_clean.csv` - trimmed, deduplicated tags (missing timestamps retained as `<NA>`)
- `links_clean.csv` - deduplicated links (missing external IDs retained as `<NA>`)
- `movies_features_base.csv` - recommendation-ready base table (movieId, title, release_year, genres, num_genres, binary genre flags) - no ML features yet
- `README.txt` - provenance notes

Module 2 feature artifacts:

- `movie_content_features.csv` - one row per movie: movieId, title, release_year,
  genres, normalized_genres, num_genres, aggregated_tags, num_tags, content_text
- `movie_genre_features.csv` - multi-hot genre flags (one column per distinct genre)
- `movie_tfidf.npz` - sparse TF-IDF matrix (scipy CSR, float32, never densified)
- `movie_feature_index.csv` - deterministic contiguous `movieId -> row_index` map

Fitted model (`models/`):

- `movie_tfidf_vectorizer.pkl` - fitted `TfidfVectorizer` (joblib); load with
  `load_feature_vectorizer()` and verify with `verify_loaded_vectorizer()`.

Genome files are validated but intentionally **not duplicated** into
`data/processed` (~435 MiB); later modules read them from `data/raw`
when needed.

Charts (`outputs/charts/`):

- `rating_distribution.png`
- `user_activity_distribution.png`
- `movie_popularity_distribution.png`
- `genre_distribution.png`
- `top_tags.png`

Reports (`outputs/reports/`):

- `dataset_quality_report.txt` - human-readable full report
- `dataset_quality_report.json` - machine-readable metrics for later modules
- `feature_engineering_report.txt` - Module 2 feature report (human-readable)
- `feature_engineering_report.json` - Module 2 metrics (movies, documents, genres,
  tags, TF-IDF vocabulary/matrix/sparsity, vectorizer config, quality checks)
- `recommendation_quality_report.txt` / `.json` - Module 3 sample recommendations
  for real seed movies plus the recommendation quality checks
- `evaluation_report.txt` / `.json` - Module 3 offline evaluation protocol,
  evaluated-user counts and Precision@K / Recall@K / HitRate@K / MAP@K /
  NDCG@K metrics
- `module3_quality_gate_report.txt` / `.json` - final Module 3 gate verdict
  (checks + regression-test outcome)

## Project structure

```
Project-05-Smart-Recommendation-System/
├── data/
│   ├── raw/ml-25m/          # original MovieLens files (READ-ONLY)
│   └── processed/           # generated cleaned datasets + Module 2 features
├── models/                  # fitted TF-IDF vectorizer (Module 2)
├── notebooks/               # module_1_data_exploration.ipynb
├── src/
│   ├── config.py            # paths & constants (incl. TF-IDF configuration)
│   ├── logging_config.py    # console + rotating file logging
│   ├── data_loader.py       # robust CSV loaders
│   ├── data_quality.py      # profiling/schema/duplicates/integrity checks
│   ├── data_preprocessor.py # documented cleaning rules
│   ├── analysis.py          # ratings/users/movies/genres/tags analyses
│   ├── visualization.py     # chart generation
│   ├── reporting.py         # TXT + JSON quality reports
│   ├── pipeline.py          # Module 1 + Module 3 entry points (--module {1,3})
│   ├── similarity_engine.py    # Module 3 exact cosine top-K over TF-IDF rows
│   ├── recommender.py          # Module 3 public recommend_similar_movies API
│   ├── evaluation.py           # Module 3 ranking metrics + leakage-free evaluator
│   └── personalization.py      # placeholder (later module)
│   └── feature_engineering.py  # Module 2 content features (python -m src.feature_engineering)
├── tests/                   # pytest suite (Modules 1-3) + hash manifest
├── logs/                    # pipeline.log (rotating)
├── outputs/
│   ├── charts/
│   └── reports/             # quality + feature engineering reports
├── app/                     # reserved for the final UI module
├── requirements.txt
└── README.md
```

Module 3 consumed the reserved placeholders `similarity_engine.py`,
`recommender.py` and `evaluation.py`, building on the artifacts generated by
Modules 1-2 (`movie_tfidf.npz`, `movie_feature_index.csv`, content documents
and the fitted vectorizer). `personalization.py` remains a placeholder for
the next module.
