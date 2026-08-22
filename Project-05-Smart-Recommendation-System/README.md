# Project 05 — Smart Recommendation System

A movie recommendation system built on the **MovieLens 25M** dataset.
The project is developed in staged modules; **Module 1 (data foundation,
exploration & preprocessing) and Module 2 (content feature engineering) are
complete**. Later modules will add similarity/recommendation algorithms,
evaluation, personalization, and a final application layer.

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

## How to run tests

```bash
python -m pytest tests -v            # full suite (Module 1 + Module 2)
python -m pytest tests/test_feature_engineering.py -v   # Module 2 only
python -m pytest tests -q            # quiet gate
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
│   ├── pipeline.py          # Module 1 end-to-end entry point
│   └── feature_engineering.py  # Module 2 content features (python -m src.feature_engineering)
├── tests/                   # pytest suite (Module 1 + Module 2) + hash manifest
├── logs/                    # pipeline.log (rotating)
├── outputs/
│   ├── charts/
│   └── reports/             # quality + feature engineering reports
├── app/                     # reserved for the final UI module
├── requirements.txt
└── README.md
```

Placeholder modules reserved for later stages (`similarity_engine.py`,
`recommender.py`, `personalization.py`, `evaluation.py`) are intentionally
untouched in Modules 1-2; Module 3 will consume the artifacts generated here
(`movie_tfidf.npz`, `movie_feature_index.csv`, content documents, and the
loaded vectorizer).
