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


# ---------------------------------------------------------------------------
# Module 3 fixtures (similarity engine / recommender / evaluation / pipeline)
# ---------------------------------------------------------------------------
_MODULE3_CATALOG = pd.DataFrame(
    {
        "movieId": pd.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype="int32"),
        "title": pd.array(
            [
                "Space Buzz Adventure (1995)",
                "Ocean Finding Adventure (2003)",
                "Safari King Musical (1994)",
                "Hard Target Tower (1988)",
                "Lethal Night Cops (1987)",
                "Bus Speed Run (1994)",
                "Cosmos Nature Wonders (2010)",
                "Haunted Manor Night (1999)",
                "Paris Love Cafe (2001)",
            ],
            dtype="string",
        ),
        "genres": pd.array(
            [
                "Animation|Children|Comedy",
                "Animation|Children|Comedy",
                "Animation|Children|Musical",
                "Action|Thriller",
                "Action|Crime|Thriller",
                "Action|Thriller",
                "Documentary",
                "Horror",
                "Comedy|Romance",
            ],
            dtype="string",
        ),
        "release_year": pd.array(
            [1995, 2003, 1994, 1988, 1987, 1994, 2010, 1999, 2001], dtype="Int32"
        ),
    }
)

_MODULE3_TAGS = pd.DataFrame(
    {
        "userId": pd.array([20, 20, 21, 21, 22, 22, 23, 23, 24], dtype="int32"),
        "movieId": pd.array([1, 2, 1, 3, 4, 5, 6, 4, 7], dtype="int32"),
        "tag": pd.array(
            [
                "pixar",
                "pixar ocean",
                "space animation",
                "disney musical",
                "action explosion",
                "action cops",
                "action bus",
                "thriller",
                "nature documentary",
            ],
            dtype="string",
        ),
    }
)

_MODULE3_RATINGS = pd.DataFrame(
    {
        # Timestamps increase within every user; the newest interaction of
        # each user becomes the evaluation hold-out (see src/evaluation.py).
        "userId": pd.array(
            [10, 10, 10, 10, 10, 10,
             11, 11, 11, 11, 11, 11,
             12, 12,
             13, 13, 13, 13, 13, 13,
             14, 14, 14,
             15, 15, 15, 15],
            dtype="int32",
        ),
        "movieId": pd.array(
            [4, 5, 9, 1, 2, 3,
             7, 8, 9, 4, 5, 6,
             1, 8,
             1, 4, 7, 8, 9, 6,
             2, 3, 5,
             7, 8, 9, 4],
            dtype="int32",
        ),
        "rating": pd.array(
            [1.0, 1.0, 2.0, 5.0, 4.5, 4.0,
             2.0, 1.5, 1.0, 5.0, 4.5, 4.0,
             4.0, 3.0,
             5.0, 4.0, 4.5, 2.0, 4.0, 3.0,
             4.5, 4.0, 4.0,
             2.0, 2.0, 3.0, 4.0],
            dtype="float32",
        ),
        "timestamp": pd.array(
            list(range(1000, 1006))    # u10: 6 interactions
            + list(range(2000, 2006))  # u11: 6
            + list(range(3000, 3002))  # u12: 2
            + list(range(4000, 4006))  # u13: 6
            + list(range(5000, 5003))  # u14: 3
            + list(range(6000, 6004)),  # u15: 4
            dtype="int64",
        ),
    }
)


@pytest.fixture
def module3_workspace(tmp_path: Path, monkeypatch) -> dict:
    """Full Module 3 sandbox: genuine synthetic Module 2 artifacts + ratings.

    Redirects every ``src.config`` path used by Modules 1-3 into ``tmp_path``,
    builds a small but real set of feature artifacts through the
    :class:`FeatureEngineeringPipeline`, and writes a synthetic cleaned
    ratings CSV. Nothing touches ``data/raw`` or real processed outputs.
    """
    from src.feature_engineering import FeatureEngineeringPipeline

    processed = tmp_path / "processed"
    models = tmp_path / "models"
    reports = tmp_path / "reports"
    for folder in (processed, models, reports):
        folder.mkdir(parents=True)

    monkeypatch.setattr(config, "PROCESSED_MOVIES_PATH", processed / "movies_clean.csv")
    monkeypatch.setattr(config, "PROCESSED_TAGS_PATH", processed / "tags_clean.csv")
    monkeypatch.setattr(
        config, "PROCESSED_RATINGS_PATH", processed / "ratings_clean.csv"
    )
    monkeypatch.setattr(
        config, "PROCESSED_MOVIE_CONTENT_FEATURES_PATH",
        processed / "movie_content_features.csv",
    )
    monkeypatch.setattr(
        config, "PROCESSED_MOVIE_GENRE_FEATURES_PATH",
        processed / "movie_genre_features.csv",
    )
    monkeypatch.setattr(
        config, "PROCESSED_MOVIE_TFIDF_PATH", processed / "movie_tfidf.npz"
    )
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
    monkeypatch.setattr(
        config, "RECOMMENDATION_REPORT_JSON_PATH",
        reports / "recommendation_quality_report.json",
    )
    monkeypatch.setattr(
        config, "RECOMMENDATION_REPORT_TXT_PATH",
        reports / "recommendation_quality_report.txt",
    )
    monkeypatch.setattr(
        config, "EVALUATION_REPORT_JSON_PATH", reports / "evaluation_report.json"
    )
    monkeypatch.setattr(
        config, "EVALUATION_REPORT_TXT_PATH", reports / "evaluation_report.txt"
    )
    monkeypatch.setattr(
        config, "MODULE3_QUALITY_GATE_JSON_PATH",
        reports / "module3_quality_gate_report.json",
    )
    monkeypatch.setattr(
        config, "MODULE3_QUALITY_GATE_TXT_PATH",
        reports / "module3_quality_gate_report.txt",
    )
    # Tiny corpora need a relaxed vocabulary floor and a reachable
    # popularity bar (each synthetic movie gets few ratings).
    monkeypatch.setattr(config, "TFIDF_MIN_DF", 1)
    monkeypatch.setattr(config, "TFIDF_MAX_DF", 1.0)
    monkeypatch.setattr(config, "RECOMMENDATION_MIN_MOVIE_RATINGS", 2)
    monkeypatch.setattr(config, "EVALUATION_MIN_USER_RATINGS", 5)

    _MODULE3_CATALOG.to_csv(processed / "movies_clean.csv", index=False)
    _MODULE3_TAGS.to_csv(processed / "tags_clean.csv", index=False)
    FeatureEngineeringPipeline().run()
    _MODULE3_RATINGS.to_csv(processed / "ratings_clean.csv", index=False)

    return {
        "processed": processed,
        "models": models,
        "reports": reports,
        "catalog": _MODULE3_CATALOG.copy(),
        "ratings": _MODULE3_RATINGS.copy(),
    }


@pytest.fixture
def small_engine(module3_workspace: dict):
    """SimilarityEngine over the synthetic workspace artifacts."""
    import pandas as pd

    from src.feature_engineering import load_sparse_matrix
    from src.similarity_engine import SimilarityEngine

    matrix = load_sparse_matrix(config.PROCESSED_MOVIE_TFIDF_PATH)
    index = pd.read_csv(config.PROCESSED_MOVIE_FEATURE_INDEX_PATH)
    return SimilarityEngine(matrix, index)


@pytest.fixture
def small_recommender(module3_workspace: dict, small_engine):
    """ContentRecommender wired to the synthetic artifacts."""
    from src.recommender import ContentRecommender, compute_popularity_stats

    popularity = compute_popularity_stats(module3_workspace["ratings"])
    return ContentRecommender(
        engine=small_engine,
        catalog=module3_workspace["catalog"],
        popularity=popularity,
        min_movie_ratings=2,
    )