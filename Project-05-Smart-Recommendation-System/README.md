# Project 05 — Smart Recommendation System

A movie recommendation system built on the **MovieLens 25M** dataset.
The project is developed in staged modules; **Module 1 (data foundation,
exploration & preprocessing) is complete**. Later modules will add
feature engineering, similarity/recommendation algorithms, evaluation,
personalization, and a final application layer.

## Objective

Build a production-quality, reproducible data foundation that later
recommendation modules can rely on: robust loading, schema validation,
profiling, data-quality checks (missing values, duplicates, referential
integrity), documented cleaning, statistical analysis, visualizations,
processed datasets, and automated tests.

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

## How to run preprocessing

```bash
pip install -r requirements.txt
python -m src.pipeline
```

The pipeline is deterministic and idempotent; rerunning it on the same
raw data reproduces equivalent outputs. It refuses to finalize if any
critical integrity check fails (final gate PASS/FAIL).

## How to run tests

```bash
python -m pytest tests -v
```

Tests never modify `data/raw`. Synthetic end-to-end pipeline tests run
inside temporary directories only. A baseline SHA-256 manifest
(`tests/baseline_raw_manifest.json`) proves raw files remain unchanged.

## Generated outputs

Processed datasets (`data/processed/`):

- `movies_clean.csv` - cleaned catalog (invalid titles dropped, missing genres -> `(no genres listed)`)
- `ratings_clean.csv` - valid-scale ratings, no exact duplicates, no orphan movieIds
- `tags_clean.csv` - trimmed, deduplicated tags (missing timestamps retained as `<NA>`)
- `links_clean.csv` - deduplicated links (missing external IDs retained as `<NA>`)
- `movies_features_base.csv` - recommendation-ready base table (movieId, title, release_year, genres, num_genres, binary genre flags) - no ML features yet
- `README.txt` - provenance notes

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

## Project structure

```
Project-05-Smart-Recommendation-System/
├── data/
│   ├── raw/ml-25m/          # original MovieLens files (READ-ONLY)
│   └── processed/           # generated cleaned datasets
├── models/                  # reserved for later modules
├── notebooks/               # module_1_data_exploration.ipynb
├── src/
│   ├── config.py            # paths & constants
│   ├── logging_config.py    # console + rotating file logging
│   ├── data_loader.py       # robust CSV loaders
│   ├── data_quality.py      # profiling/schema/duplicates/integrity checks
│   ├── data_preprocessor.py # documented cleaning rules
│   ├── analysis.py          # ratings/users/movies/genres/tags analyses
│   ├── visualization.py     # chart generation
│   ├── reporting.py         # TXT + JSON quality reports
│   └── pipeline.py          # end-to-end entry point (python -m src.pipeline)
├── tests/                   # pytest suite + raw-data hash manifest
├── logs/                    # pipeline.log (rotating)
├── outputs/
│   ├── charts/
│   └── reports/
├── app/                     # reserved for the final UI module
├── requirements.txt
└── README.md
```

Placeholder modules reserved for later stages (`similarity_engine.py`,
`recommender.py`, `personalization.py`, `feature_engineering.py`,
`evaluation.py`) are intentionally untouched in Module 1.
