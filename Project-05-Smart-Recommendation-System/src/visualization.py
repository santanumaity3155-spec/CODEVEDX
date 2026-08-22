"""Chart generation for Module 1 (matplotlib, Agg backend).

All charts are saved under ``outputs/charts`` and use only real
computed data.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src.logging_config import get_logger  # noqa: E402

logger = get_logger(__name__)

_DPI = 150


def _ensure_charts_dir() -> Path:
    config.CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    return config.CHARTS_DIR


def plot_rating_distribution(ratings: pd.DataFrame) -> Path:
    """Bar chart of rating counts across the half-star grid."""
    counts = ratings["rating"].value_counts().sort_index()
    labels = [f"{v:.1f}" for v in counts.index]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, counts.values, color="#4c72b0", edgecolor="white")
    ax.bar_label(bars, fmt=lambda v: f"{int(v):,}", fontsize=8, padding=2)
    ax.set_title("Rating Distribution (MovieLens 25M)")
    ax.set_xlabel("Rating value")
    ax.set_ylabel("Number of ratings")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.margins(y=0.12)
    fig.tight_layout()

    out = _ensure_charts_dir() / "rating_distribution.png"
    fig.savefig(out, dpi=_DPI)
    plt.close(fig)
    logger.info("Saved chart %s", out)
    return out


def plot_user_activity_distribution(ratings: pd.DataFrame) -> Path:
    """Histogram of ratings-per-user (log-scaled y axis)."""
    per_user = ratings.groupby("userId", observed=True).size()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(per_user.values, bins=50, color="#55a868", edgecolor="white")
    ax.set_yscale("log")
    ax.set_title(f"User Activity Distribution (n={len(per_user):,} users)")
    ax.set_xlabel("Ratings per user")
    ax.set_ylabel("Users (log scale)")
    ax.axvline(float(per_user.mean()), color="#c44e52", linestyle="--", linewidth=1.2,
               label=f"mean = {per_user.mean():,.0f}")
    ax.axvline(float(per_user.median()), color="#8172b3", linestyle=":", linewidth=1.2,
               label=f"median = {per_user.median():,.0f}")
    ax.legend()
    fig.tight_layout()

    out = _ensure_charts_dir() / "user_activity_distribution.png"
    fig.savefig(out, dpi=_DPI)
    plt.close(fig)
    logger.info("Saved chart %s", out)
    return out


def plot_movie_popularity_distribution(ratings: pd.DataFrame) -> Path:
    """Histogram of ratings-per-movie (log-scaled y axis)."""
    per_movie = ratings.groupby("movieId", observed=True).size()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(per_movie.values, bins=50, color="#dd8452", edgecolor="white")
    ax.set_yscale("log")
    ax.set_title(f"Movie Popularity Distribution (n={len(per_movie):,} movies rated)")
    ax.set_xlabel("Ratings per movie")
    ax.set_ylabel("Movies (log scale)")
    ax.axvline(float(per_movie.mean()), color="#c44e52", linestyle="--", linewidth=1.2,
               label=f"mean = {per_movie.mean():,.1f}")
    ax.axvline(float(per_movie.median()), color="#8172b3", linestyle=":", linewidth=1.2,
               label=f"median = {per_movie.median():,.0f}")
    ax.legend()
    fig.tight_layout()

    out = _ensure_charts_dir() / "movie_popularity_distribution.png"
    fig.savefig(out, dpi=_DPI)
    plt.close(fig)
    logger.info("Saved chart %s", out)
    return out


def plot_genre_distribution(genre_stats: dict) -> Path:
    """Horizontal bar chart: number of movies per genre."""
    items = list(genre_stats["movies_by_genre"].items())
    genres = [name for name, _ in items]
    values = [info["count"] for _, info in items]

    fig, ax = plt.subplots(figsize=(9, max(5, 0.32 * len(genres))))
    colors = ["#937860" if g == config.NO_GENRES_LABEL else "#4c72b0" for g in genres]
    bars = ax.barh(genres[::-1], values[::-1], color=colors[::-1], edgecolor="white")
    ax.bar_label(bars, fmt=lambda v: f"{int(v):,}", fontsize=8, padding=2)
    ax.set_title("Movies per Genre")
    ax.set_xlabel("Number of movies")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.margins(x=0.12)
    fig.tight_layout()

    out = _ensure_charts_dir() / "genre_distribution.png"
    fig.savefig(out, dpi=_DPI)
    plt.close(fig)
    logger.info("Saved chart %s", out)
    return out


def plot_top_tags(tag_freq: pd.Series, top_n: int = 20) -> Path:
    """Horizontal bar chart of the most frequent tags."""
    top = tag_freq.head(top_n)

    fig, ax = plt.subplots(figsize=(9, max(5, 0.32 * len(top))))
    bars = ax.barh(list(top.index)[::-1], list(top.values)[::-1],
                   color="#55a868", edgecolor="white")
    ax.bar_label(bars, fmt=lambda v: f"{int(v):,}", fontsize=8, padding=2)
    ax.set_title(f"Top {len(top)} Tags by Frequency")
    ax.set_xlabel("Tag occurrences")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.margins(x=0.12)
    fig.tight_layout()

    out = _ensure_charts_dir() / "top_tags.png"
    fig.savefig(out, dpi=_DPI)
    plt.close(fig)
    logger.info("Saved chart %s", out)
    return out
