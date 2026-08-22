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
