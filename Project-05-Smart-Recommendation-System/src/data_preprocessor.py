"""Cleaning and preprocessing pipeline for Module 1.

Every rule applied here is documented in :class:`DataPreprocessor`
method docstrings and is reported with exact removed/retained counts so
the quality report can show precisely what happened to each dataset.
Raw files are never modified; cleaned outputs go to ``data/processed``.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src import config
from src.logging_config import get_logger

logger = get_logger(__name__)

_YEAR_RE = re.compile(r"\((\d{4})\)\s*$")


def extract_release_year(title: str) -> int | None:
    """Extract the trailing ``(YYYY)`` release year from a movie title."""
    if not isinstance(title, str):
        return None
    match = _YEAR_RE.search(title.strip())
    return int(match.group(1)) if match else None


class DataPreprocessor:
    """Applies documented cleaning rules and records removal counts.

    Rules summary (all removals are reported, never silent):

    movies
        - Strip surrounding whitespace from ``title``/``genres``.
        - Drop records with missing ``movieId`` or blank title (invalid).
        - Missing genres become ``"(no genres listed)"``.
        - Duplicate ``movieId`` -> keep first occurrence.
    ratings
        - Drop rows missing userId/movieId/rating/timestamp (invalid).
        - Drop ratings outside the 0.5-5.0 half-star grid (invalid).
        - Drop exact repeats of (userId, movieId, timestamp), keep first.
        - Repeated (userId, movieId) at different timestamps are legitimate
          MovieLens re-ratings and are RETAINED.
    tags
        - Trim whitespace; drop null or empty-after-trim tags.
        - Missing timestamps are retained as <NA> and reported.
        - Exact repeats of (userId, movieId, tag, timestamp) removed.
    links
        - Missing imdbId/tmdbId are legitimate; retained as <NA>.
        - Duplicate ``movieId`` -> keep first.
    """

    def __init__(self) -> None:
        self.cleaning_log: list[dict] = []
        self._removed_total = 0
        self._retained_total = 0

    # ------------------------------------------------------------------
    def _record(self, dataset: str, rules: list[tuple[str, int]], retained: int) -> None:
        for rule, removed in rules:
            self.cleaning_log.append(
                {"dataset": dataset, "rule": rule, "records_removed": int(removed),
                 "records_retained": int(retained)}
            )
            self._removed_total += int(removed)
            logger.info("[%s] %s -> removed %d, retained %d",
                        dataset, rule, removed, retained)
        self._retained_total += int(retained)

    @staticmethod
    def _reset(df: pd.DataFrame) -> pd.DataFrame:
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    def clean_movies(self, movies: pd.DataFrame) -> pd.DataFrame:
        df = movies.copy()

        missing_key = df["movieId"].isna()
        df["title"] = df["title"].str.strip()
        invalid_title = df["title"].isna() | (df["title"] == "")
        drop_mask = missing_key | invalid_title
        n_invalid = int(drop_mask.sum())
        df = df[~drop_mask]

        no_genres = df["genres"].isna() | (df["genres"].astype("string").str.strip() == "")
        n_no_genre = int(no_genres.sum())
        df.loc[no_genres, "genres"] = config.NO_GENRES_LABEL
        df["genres"] = df["genres"].str.strip()

        dup_id = df.duplicated(subset=["movieId"], keep="first")
        n_dup = int(dup_id.sum())
        df = df[~dup_id]

        df["movieId"] = df["movieId"].astype("int32")
        df["title"] = df["title"].astype("string")
        df["genres"] = df["genres"].astype("string")
        df = self._reset(df)
        self._record(
            "movies",
            [
                ("drop missing movieId or blank title", n_invalid),
                ("fill missing genres with '(no genres listed)'", 0),
                ("duplicate movieId kept-first removal", n_dup),
            ],
            len(df),
        )
        return df

    # ------------------------------------------------------------------
    def clean_ratings(
        self,
        ratings: pd.DataFrame,
        valid_movie_ids: pd.Index | set | None = None,
    ) -> pd.DataFrame:
        df = ratings.copy()
        original_len = len(df)

        required_nulls = df[["userId", "movieId", "rating", "timestamp"]].isna().any(axis=1)
        n_missing = int(required_nulls.sum())

        r = df["rating"].astype("float64")
        in_range = r.between(
            config.RATING_MIN, config.RATING_MAX, inclusive="both"
        ).fillna(False)
        half_steps = (r - config.RATING_MIN) / config.RATING_STEP
        on_grid = np.isclose(
            half_steps.to_numpy(), np.round(half_steps.to_numpy()), equal_nan=False
        )
        valid_rating = in_range.to_numpy() & on_grid & r.notna().to_numpy()
        n_bad_rating = int(((~valid_rating) & (~required_nulls.to_numpy())).sum())

        drop_mask = required_nulls | ~pd.Series(valid_rating, index=df.index)
        df = df[~drop_mask]

        exact_dup = df.duplicated(subset=["userId", "movieId", "timestamp"], keep="first")
        n_exact_dup = int(exact_dup.sum())
        df = df[~exact_dup]

        n_orphans = 0
        if valid_movie_ids is not None:
            orphan = ~df["movieId"].isin(valid_movie_ids)
            n_orphans = int(orphan.sum())
            df = df[~orphan]

        df["userId"] = df["userId"].astype("int32")
        df["movieId"] = df["movieId"].astype("int32")
        df["rating"] = df["rating"].astype("float32")
        df["timestamp"] = df["timestamp"].astype("int64")
        df = self._reset(df)
        self._record(
            "ratings",
            [
                ("drop rows missing required fields", n_missing),
                ("drop out-of-range / off-grid ratings", n_bad_rating),
                ("exact duplicate (user,movie,timestamp) kept-first removal", n_exact_dup),
                ("orphan movieId removal vs. catalog", n_orphans),
                ("retain legitimate repeated user-movie re-ratings", 0),
            ],
            len(df),
        )
        assert original_len >= len(df), "cleaning must never add rating rows"
        return df

    # ------------------------------------------------------------------
    def clean_tags(self, tags: pd.DataFrame) -> pd.DataFrame:
        df = tags.copy()

        missing_core = df[["userId", "movieId", "tag"]].isna().any(axis=1)
        n_missing = int(missing_core.sum())
        df = df[~missing_core]

        df["tag"] = df["tag"].str.strip()
        empty_tag = df["tag"].isna() | (df["tag"] == "")
        n_empty = int(empty_tag.sum())
        df = df[~empty_tag]

        n_missing_ts = int(df["timestamp"].isna().sum())

        dup = df.duplicated(subset=["userId", "movieId", "tag", "timestamp"], keep="first")
        n_dup = int(dup.sum())
        df = df[~dup]

        df["userId"] = df["userId"].astype("int32")
        df["movieId"] = df["movieId"].astype("int32")
        df["tag"] = df["tag"].astype("string")
        df["timestamp"] = df["timestamp"].astype("Int64")
        df = self._reset(df)
        self._record(
            "tags",
            [
                ("drop rows missing userId/movieId/tag", n_missing),
                ("drop empty-after-trim tags", n_empty),
                ("retain missing timestamps as <NA>", 0),
                ("exact duplicate (user,movie,tag,timestamp) kept-first removal", n_dup),
            ],
            len(df),
        )
        logger.info("[tags] retained %d rows with missing timestamp", n_missing_ts)
        return df

    # ------------------------------------------------------------------
    def clean_links(
        self,
        links: pd.DataFrame,
        valid_movie_ids: pd.Index | set | None = None,
    ) -> pd.DataFrame:
        df = links.copy()

        missing_id = df["movieId"].isna()
        n_missing = int(missing_id.sum())
        df = df[~missing_id]

        dup = df.duplicated(subset=["movieId"], keep="first")
        n_dup = int(dup.sum())
        df = df[~dup]

        n_missing_external = int(df[["imdbId", "tmdbId"]].isna().any(axis=1).sum())

        n_orphans = 0
        if valid_movie_ids is not None:
            orphan = ~df["movieId"].isin(valid_movie_ids)
            n_orphans = int(orphan.sum())
            df = df[~orphan]

        df["movieId"] = df["movieId"].astype("int32")
        df["imdbId"] = df["imdbId"].astype("Int64")
        df["tmdbId"] = df["tmdbId"].astype("Int64")
        df = self._reset(df)
        self._record(
            "links",
            [
                ("drop rows missing movieId", n_missing),
                ("duplicate movieId kept-first removal", n_dup),
                ("retain missing external IDs (<NA>)", 0),
                ("orphan movieId removal vs. catalog", n_orphans),
            ],
            len(df),
        )
        logger.info("[links] %d rows have at least one missing external ID",
                    n_missing_external)
        return df

    # ------------------------------------------------------------------
    def build_movies_features_base(self, movies_clean: pd.DataFrame) -> pd.DataFrame:
        """Recommendation-ready base table (no ML features yet).

        Columns: movieId, title, release_year, genres, num_genres plus one
        binary indicator per distinct genre token (including
        "(no genres listed)"). Pure deterministic normalization only.
        """
        df = movies_clean.copy()
        years = df["title"].map(extract_release_year).astype("Int32")

        genre_lists = df["genres"].str.split("|")
        num_genres = (
            genre_lists.map(lambda g: 0 if g == [config.NO_GENRES_LABEL]
                            else len(g)).astype("int16")
        )

        feature_df = pd.DataFrame({
            "movieId": df["movieId"],
            "title": df["title"],
            "release_year": years,
            "genres": df["genres"],
            "num_genres": num_genres,
        })

        all_tokens = sorted({t for lst in genre_lists for t in lst})
        for token in all_tokens:
            safe_name = token.replace(" ", "_").replace("-", "_")
            feature_df[f"genre_{safe_name}"] = genre_lists.map(lambda g, t=token: t in g).astype("int8")

        self._record(
            "movies_features_base",
            [("deterministic normalization (no rows changed)", 0)],
            len(feature_df),
        )
        return feature_df

    # ------------------------------------------------------------------
    @property
    def total_removed(self) -> int:
        return self._removed_total

    @property
    def total_retained(self) -> int:
        return self._retained_total

    def cleaning_summary(self) -> dict:
        """JSON-serializable summary of every applied rule."""
        per_dataset: dict[str, dict[str, int]] = {}
        for entry in self.cleaning_log:
            ds = entry["dataset"]
            stats = per_dataset.setdefault(ds, {"records_removed": 0})
            stats["records_removed"] += entry["records_removed"]
        return {
            "rules": self.cleaning_log,
            "per_dataset": per_dataset,
            "total_removed": self._removed_total,
        }
