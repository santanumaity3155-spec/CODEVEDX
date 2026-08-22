"""Exploratory analyses used by the Module 1 reports and charts.

Every function takes a dataframe and returns plain Python types so
results can be embedded in JSON reports without conversion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Ratings
# ---------------------------------------------------------------------------

def analyze_ratings(ratings: pd.DataFrame) -> dict:
    """Core descriptive statistics of the rating column."""
    r = ratings["rating"].astype("float64")
    stats = {
        "total_ratings": int(len(ratings)),
        "min": round(float(r.min()), 4),
        "max": round(float(r.max()), 4),
        "mean": round(float(r.mean()), 4),
        "median": round(float(r.median()), 4),
        "std": round(float(r.std()), 4),
    }
    logger.info(
        "Rating stats: n=%d, mean=%.3f, median=%.3f, std=%.3f",
        stats["total_ratings"], stats["mean"], stats["median"], stats["std"],
    )
    return stats


def rating_distribution(ratings: pd.DataFrame) -> dict:
    """Count and percentage per half-star rating value 0.5 ... 5.0.

    The full grid is always reported (zero counts included) so charts do
    not fabricate or drop buckets.
    """
    grid = np.arange(config.RATING_MIN, config.RATING_MAX + config.RATING_STEP / 2,
                     config.RATING_STEP)
    counts = ratings["rating"].value_counts()
    total = max(len(ratings), 1)

    star_counts = {}
    for value in grid:
        value = float(np.round(value, 1))
        n = int(counts.get(value, 0))
        stars = int(round(value))
        key = f"{stars}-star" if float(stars) == value else f"{value}-star"
        star_counts[key] = {"count": n, "pct": round(100.0 * n / total, 3)}

    logger.info("Rating distribution computed across %d grid values", len(grid))
    return {
        "counts_by_value": {f"{float(np.round(v, 1))}": int(counts.get(float(np.round(v, 1)), 0))
                            for v in grid},
        "star_summary": star_counts,
    }


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

def analyze_users(ratings: pd.DataFrame) -> dict:
    """Ratings-per-user statistics plus activity segments."""
    per_user = ratings.groupby("userId", observed=True).size()

    high_activity_threshold = float(per_user.quantile(0.99))
    low_activity_threshold = float(per_user.quantile(0.01))

    result = {
        "unique_users": int(per_user.size),
        "ratings_per_user_min": int(per_user.min()),
        "ratings_per_user_max": int(per_user.max()),
        "ratings_per_user_mean": round(float(per_user.mean()), 4),
        "ratings_per_user_median": float(per_user.median()),
        "highly_active_users_ge_99th_pct": int((per_user >= high_activity_threshold).sum()),
        "high_activity_threshold_99th_pct": round(high_activity_threshold, 2),
        "low_activity_users_le_1st_pct": int((per_user <= low_activity_threshold).sum()),
        "low_activity_threshold_1st_pct": round(low_activity_threshold, 2),
    }
    logger.info(
        "User activity: %d users, mean %.1f, median %d ratings/user, "
        "%d highly active (>= %.0f), %d low activity (<= %.0f)",
        result["unique_users"],
        result["ratings_per_user_mean"],
        result["ratings_per_user_median"],
        result["highly_active_users_ge_99th_pct"],
        high_activity_threshold,
        result["low_activity_users_le_1st_pct"],
        low_activity_threshold,
    )
    return result


def ratings_per_user_series(ratings: pd.DataFrame) -> pd.Series:
    """Series indexed by userId with the number of ratings per user."""
    return ratings.groupby("userId", observed=True).size().sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Movies / popularity
# ---------------------------------------------------------------------------

def analyze_movie_popularity(ratings: pd.DataFrame, movies: pd.DataFrame | None = None) -> dict:
    """Popularity measured by rating count (never by average alone)."""
    per_movie = ratings.groupby("movieId", observed=True).size()
    mean_per_movie = ratings.groupby("movieId", observed=True)["rating"].mean()

    result = {
        "unique_movies_rated": int(per_movie.size),
        "ratings_per_movie_min": int(per_movie.min()),
        "ratings_per_movie_max": int(per_movie.max()),
        "ratings_per_movie_mean": round(float(per_movie.mean()), 4),
        "ratings_per_movie_median": float(per_movie.median()),
    }

    if movies is not None:
        top = (
            per_movie.rename("rating_count")
            .to_frame()
            .join(mean_per_movie.rename("mean_rating"))
            .join(movies.set_index("movieId")[["title"]])
            .sort_values(["rating_count", "mean_rating"], ascending=[False, False])
            .head(config.TOP_MOVIES_REPORT_SIZE)
        )
        top = top[top["rating_count"] >= min(config.TOP_MOVIES_MIN_RATINGS,
                                             int(per_movie.min()))]
        result["top_movies_by_rating_count"] = [
            {
                "rank": i + 1,
                "movieId": int(mid),
                "title": str(row["title"]),
                "rating_count": int(row["rating_count"]),
                "mean_rating": round(float(row["mean_rating"]), 4),
            }
            for i, (mid, row) in enumerate(top.iterrows())
        ]

    logger.info(
        "Movie popularity: %d movies rated, mean %.1f ratings/movie, max %d",
        result["unique_movies_rated"],
        result["ratings_per_movie_mean"],
        result["ratings_per_movie_max"],
    )
    return result


def ratings_per_movie_series(ratings: pd.DataFrame) -> pd.Series:
    """Series indexed by movieId with the number of ratings per movie."""
    return ratings.groupby("movieId", observed=True).size()


# ---------------------------------------------------------------------------
# Genres
# ---------------------------------------------------------------------------

def split_genres(genres: pd.Series) -> pd.Series:
    """Split pipe-separated genre strings into lists.

    ``(no genres listed)`` is preserved as a single pseudo-genre token.
    """
    def _split(value: str) -> list[str]:
        if not isinstance(value, str) or not value.strip():
            return [config.NO_GENRES_LABEL]
        parts = [g.strip() for g in value.split("|") if g.strip()]
        return parts or [config.NO_GENRES_LABEL]

    return genres.map(_split)


def explode_genres(movies: pd.DataFrame) -> pd.DataFrame:
    """Long-format frame: one row per (movieId, genre)."""
    exploded = movies[["movieId"]].copy()
    exploded["genre"] = split_genres(movies["genres"])
    exploded = exploded.explode("genre", ignore_index=True)
    return exploded


def analyze_genres(movies: pd.DataFrame) -> dict:
    """Movies-per-genre counts and percentages."""
    total_movies = len(movies)
    long = explode_genres(movies)
    counts = long["genre"].value_counts()
    no_genre_count = int((movies["genres"] == config.NO_GENRES_LABEL).sum())

    genres_list = [g for g in counts.index if g != config.NO_GENRES_LABEL]

    avg_genres = float(long.loc[long["genre"] != config.NO_GENRES_LABEL]
                       .groupby("movieId").size().reindex(movies["movieId"], fill_value=0)
                       .mean()) if total_movies else 0.0

    result = {
        "total_movies": int(total_movies),
        "movies_with_no_genre": no_genre_count,
        "avg_genres_per_movie": round(avg_genres, 4),
        "num_distinct_genres": int(len(genres_list)),
        "movies_by_genre": {
            str(g): {"count": int(c), "pct_of_movies": round(100.0 * c / max(total_movies, 1), 3)}
            for g, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        },
    }
    logger.info(
        "Genres: %d distinct genres, %d movies without genres (%.2f%%), "
        "avg %.2f genres/movie",
        result["num_distinct_genres"],
        no_genre_count,
        100.0 * no_genre_count / max(total_movies, 1),
        result["avg_genres_per_movie"],
    )
    return result


# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------

def analyze_tags(tags: pd.DataFrame) -> dict:
    """Tag frequency statistics (no advanced NLP at this stage)."""
    tag_text = tags["tag"]
    normalized = tag_text.str.lower().str.strip()
    counts = normalized.value_counts()
    unique_tags = int(normalized.nunique())
    per_user = tags["userId"].nunique()
    per_movie = tags["movieId"].nunique()

    result = {
        "total_tag_records": int(len(tags)),
        "unique_users_tagging": int(per_user),
        "unique_movies_tagged": int(per_movie),
        "unique_tags_case_insensitive": unique_tags,
        "most_frequent_tags": {
            str(t): int(c) for t, c in counts.head(20).items()
        },
    }
    logger.info(
        "Tags: %d records, %d unique tags (case-insensitive)",
        result["total_tag_records"],
        unique_tags,
    )
    return result


def tag_frequency_series(tags: pd.DataFrame) -> pd.Series:
    """Case-insensitive tag frequency series sorted descending."""
    normalized = tags["tag"].str.lower().str.strip()
    return normalized.value_counts()
