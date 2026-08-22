"""Shared pytest fixtures for Module 1 tests.

Test isolation rules enforced here:

* Tests NEVER write into ``data/raw``.
* Synthetic end-to-end tests run entirely inside ``tmp_path`` by
  monkeypatching ``src.config`` path constants.
* Heavy real-dataset loads are session-scoped so they happen once.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402
from src.data_loader import (  # noqa: E402
    load_movies,
    load_ratings,
)


# ---------------------------------------------------------------------------
# Real raw data (session-scoped: expensive files loaded once, read-only)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def movies_full() -> pd.DataFrame:
    return load_movies()


@pytest.fixture(scope="session")
def ratings_full() -> pd.DataFrame:
    return load_ratings()


@pytest.fixture(scope="session")
def baseline_raw_manifest() -> dict:
    import json

    manifest_path = Path(__file__).parent / "baseline_raw_manifest.json"
    with open(manifest_path, encoding="utf-8-sig") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Deterministic synthetic data (valid per MovieLens domain rules)
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_movies_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "movieId": pd.array([1, 2, 3, 4], dtype="int32"),
            "title": pd.array(
                ["Alpha (2001)", "Beta (1999)", "Gamma (2010)", "Delta (2020)"],
                dtype="string",
            ),
            "genres": pd.array(
                ["Action|Comedy", "Drama", "(no genres listed)", "Action|IMAX"],
                dtype="string",
            ),
        }
    )


@pytest.fixture
def sample_ratings_df() -> pd.DataFrame:
    # user 1 re-rates movie 1 legitimately (different timestamps);
    # everything stays on the 0.5-5.0 half-star grid.
    return pd.DataFrame(
        {
            "userId": pd.array([10, 10, 11, 12, 12, 13], dtype="int32"),
            "movieId": pd.array([1, 1, 2, 3, 4, 1], dtype="int32"),
            "rating": pd.array([4.5, 4.0, 3.0, 5.0, 2.5, 1.0], dtype="float32"),
            "timestamp": pd.array(
                [1000000000, 1100000000, 1200000000, 1300000000, 1400000000, 1500000000],
                dtype="int64",
            ),
        }
    )


@pytest.fixture
def sample_tags_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "userId": pd.array([10, 10, 11], dtype="int32"),
            "movieId": pd.array([1, 1, 2], dtype="int32"),
            "tag": pd.array(["sci-fi", "classic ", "thought-provoking"], dtype="string"),
            "timestamp": pd.array([100, 200, pd.NA], dtype="Int64"),
        }
    )


@pytest.fixture
def sample_links_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "movieId": pd.array([1, 2, 3], dtype="int32"),
            "imdbId": pd.array([1111111, 2222222, pd.NA], dtype="Int64"),
            "tmdbId": pd.array([862, 8844, 999], dtype="Int64"),
        }
    )


@pytest.fixture
def sample_genome_tags_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tagId": pd.array([1, 2, 3], dtype="int16"),
            "tag": pd.array(["007", "action", "zombie"], dtype="string"),
        }
    )


@pytest.fixture
def sample_genome_scores_df(sample_genome_tags_df: pd.DataFrame) -> pd.DataFrame:
    tag_ids = sample_genome_tags_df["tagId"].tolist()
    return pd.DataFrame(
        {
            "movieId": pd.array([1] * len(tag_ids) + [2] * len(tag_ids), dtype="int32"),
            "tagId": pd.array(tag_ids * 2, dtype="int16"),
            "relevance": pd.array([0.05, 0.5, 0.95] * 2, dtype="float32"),
        }
    )

# ---------------------------------------------------------------------------
# Module 2 fixtures (content feature engineering)
# ---------------------------------------------------------------------------

MOVIE_CATALOG = pd.DataFrame(
    {
        "movieId": pd.array([1, 2, 3, 4, 5, 6], dtype="int32"),
        "title": pd.array(
            [
                "The Matrix (1999)",
                "Toy Story (1995)",
                "Jurassic Park (1993)",
                "The Godfather (1972)",
                "The Godfather: Part II (1974)",
                "Stand By Me (1986)",
            ],
            dtype="string",
        ),
        "genres": pd.array(
            [
                "Action|Sci-Fi|Thriller",
                "Animation|Children|Comedy",
                "Adventure|Sci-Fi",
                "Crime|Drama",
                "Crime|Drama",
                "Adventure|Drama",
            ],
            dtype="string",
        ),
    }
)

MOVIE_TAGS = pd.DataFrame(
    {
        "userId": pd.array([10, 10, 11, 11, 12, 12, 13, 13, 14, 15], dtype="int32"),
        "movieId": pd.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5], dtype="int32"),
        "tag": pd.array(
            [
                "sci-fi",
                "futuristic",
                "pixar",
                "animation",
                "dinosaurs",
                "Sci-Fi  ",
                "mafia",
                "classic",
                "godfather",
                "  ",
            ],
            dtype="string",
        ),
    }
)


@pytest.fixture
def feature_movies_df() -> pd.DataFrame:
    """Synthetic cleaned movie catalog (6 movies, movie 6 has no tags)."""
    return MOVIE_CATALOG.copy()


@pytest.fixture
def feature_tags_df() -> pd.DataFrame:
    """Synthetic cleaned tags; includes a duplicate-case and a blank tag."""
    return MOVIE_TAGS.copy()


@pytest.fixture
def feature_paths(tmp_path: Path, monkeypatch) -> dict:
    """Redirect Module 2 inputs/outputs and TF-IDF tuning to a temp dir.

    Small synthetic corpora need a relaxed ``min_df`` so the vocabulary is
    non-empty; the production defaults (min_df=2, max_df=0.90) stay untouched.
    """
    processed = tmp_path / "processed"
    models = tmp_path / "models"
    reports = tmp_path / "reports"
    processed.mkdir()
    models.mkdir()
    reports.mkdir()

    monkeypatch.setattr(config, "PROCESSED_MOVIES_PATH", processed / "movies_clean.csv")
    monkeypatch.setattr(config, "PROCESSED_TAGS_PATH", processed / "tags_clean.csv")
    monkeypatch.setattr(
        config, "PROCESSED_MOVIE_CONTENT_FEATURES_PATH",
        processed / "movie_content_features.csv",
    )
    monkeypatch.setattr(
        config, "PROCESSED_MOVIE_GENRE_FEATURES_PATH",
        processed / "movie_genre_features.csv",
    )
    monkeypatch.setattr(config, "PROCESSED_MOVIE_TFIDF_PATH", processed / "movie_tfidf.npz")
    monkeypatch.setattr(
        config, "PROCESSED_MOVIE_FEATURE_INDEX_PATH",
        processed / "movie_feature_index.csv",
    )
    monkeypatch.setattr(config, "MODELS_DIR", models)
    monkeypatch.setattr(
        config, "MOVIE_TFIDF_VECTORIZER_PATH", models / "movie_tfidf_vectorizer.pkl"
    )
    monkeypatch.setattr(config, "REPORTS_DIR", reports)
    monkeypatch.setattr(
        config, "FEATURE_ENGINEERING_REPORT_JSON_PATH",
        reports / "feature_engineering_report.json",
    )
    monkeypatch.setattr(
        config, "FEATURE_ENGINEERING_REPORT_TXT_PATH",
        reports / "feature_engineering_report.txt",
    )
    monkeypatch.setattr(config, "TFIDF_MIN_DF", 1)
    monkeypatch.setattr(config, "TFIDF_MAX_DF", 1.0)
    return {"processed": processed, "models": models, "reports": reports}


@pytest.fixture
def feature_pipeline(feature_paths, feature_movies_df, feature_tags_df):
    """A FeatureEngineeringPipeline wired to temp inputs/outputs."""
    from src.feature_engineering import FeatureEngineeringPipeline

    feature_movies_df.to_csv(config.PROCESSED_MOVIES_PATH, index=False)
    feature_tags_df.to_csv(config.PROCESSED_TAGS_PATH, index=False)
    return FeatureEngineeringPipeline()