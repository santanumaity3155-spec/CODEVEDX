"""Data-quality checks: profiling, schema, missing values, duplicates,
referential integrity, and domain validation.

All functions return plain Python types (dict/list/int/float) so the
results can be serialized directly into JSON reports.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src import config
from src.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Generic profiling
# ---------------------------------------------------------------------------

def profile_dataset(df: pd.DataFrame, label: str) -> dict:
    """Return a structural profile of a dataframe."""
    profile = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "column_names": [str(c) for c in df.columns],
        "dtypes": {str(c): str(t) for c, t in df.dtypes.items()},
        "memory_mb": round(float(df.memory_usage(deep=True).sum() / 1024**2), 2),
        "missing_values": {str(c): int(n) for c, n in df.isna().sum().items() if n > 0},
        "unique_counts": {
            str(c): int(df[c].nunique()) for c in df.columns
        },
        "duplicate_rows": int(df.duplicated().sum()),
    }
    logger.info(
        "Profiled '%s': %d rows, %d cols, %.2f MiB, %d columns with missing values, "
        "%d duplicate rows",
        label,
        profile["rows"],
        profile["columns"],
        profile["memory_mb"],
        len(profile["missing_values"]),
        profile["duplicate_rows"],
    )
    return profile


def validate_schema(df: pd.DataFrame, dataset_name: str) -> dict:
    """Validate that all required columns exist (order not enforced)."""
    required = list(config.EXPECTED_SCHEMAS[dataset_name])
    missing = [c for c in required if c not in df.columns]
    extra = [c for c in df.columns if c not in required]
    result = {
        "dataset": dataset_name,
        "required_columns": required,
        "missing_columns": missing,
        "unexpected_extra_columns": extra,
        "valid": not missing,
    }
    if missing:
        logger.warning("Schema check FAILED for '%s': missing %s", dataset_name, missing)
    else:
        logger.info("Schema check passed for '%s'", dataset_name)
    return result


# ---------------------------------------------------------------------------
# Missing-value analysis
# ---------------------------------------------------------------------------

def analyze_missing_values(df: pd.DataFrame, label: str) -> dict:
    """Per-column missing counts and percentages for every column."""
    total = max(len(df), 1)
    report = {}
    for col in df.columns:
        n = int(df[col].isna().sum())
        report[str(col)] = {
            "missing": n,
            "pct": round(100.0 * n / total, 4),
        }
    any_missing = sum(v["missing"] for v in report.values())
    logger.info("Missing-value scan for '%s': %d missing cells overall", label, any_missing)
    return {"total_missing_cells": int(any_missing), "by_column": report}


# ---------------------------------------------------------------------------
# Duplicate analysis (domain-appropriate rules; nothing removed blindly)
# ---------------------------------------------------------------------------

def find_duplicates(df: pd.DataFrame, subset: list[str], rule: str) -> dict:
    """Count duplicate records for a given key ``subset`` under ``rule``.

    ``rule`` is only documentation here; callers decide what to remove.
    """
    if not set(subset).issubset(df.columns):
        raise KeyError(f"Duplicates rule '{rule}': columns {subset} not fully present.")
    mask = df.duplicated(subset=subset, keep="first")
    dup_rows = df[mask]
    return {
        "key": subset,
        "rule": rule,
        "duplicate_records": int(mask.sum()),
        "example_keys": [
            tuple(str(v) for v in row)
            for row in dup_rows[subset].head(5).itertuples(index=False)
        ],
    }


def analyze_movie_duplicates(movies: pd.DataFrame) -> dict:
    """Duplicate movieId (must be 0 after cleaning) and duplicate titles.

    Duplicate *titles* with different movieIds are legitimate in MovieLens
    (re-releases / re-issued entries); they are reported but never removed.
    """
    by_id = find_duplicates(movies, ["movieId"], "duplicate movieId: keep first")
    by_title = find_duplicates(
        movies, ["title"], "duplicate title with distinct movieId: legitimate, retain"
    )
    # Titles duplicated on identical movieId are already covered by by_id.
    return {"duplicate_movieId": by_id, "duplicate_title": by_title}


def analyze_rating_duplicates(ratings: pd.DataFrame) -> dict:
    """Exact repeats vs. legitimate repeated interactions.

    - Exact duplicates (same user, movie, timestamp): data errors.
    - Repeated (userId, movieId) pairs with different timestamps are
      legitimate MovieLens re-ratings and must be preserved.
    """
    exact = find_duplicates(
        ratings, ["userId", "movieId", "timestamp"],
        "exact repeat (user+movie+timestamp): invalid, remove",
    )
    repeated_interactions = find_duplicates(
        ratings, ["userId", "movieId"],
        "re-rating of same movie at different time: legitimate, retain",
    )
    return {
        "exact_duplicates": exact,
        "repeated_user_movie_interactions": repeated_interactions,
    }


def analyze_tag_duplicates(tags: pd.DataFrame) -> dict:
    """Exact tag repeats (user+movie+tag+timestamp) are data errors."""
    return {
        "exact_duplicates": find_duplicates(
            tags, ["userId", "movieId", "tag", "timestamp"],
            "exact repeat: invalid, remove",
        )
    }


def analyze_link_duplicates(links: pd.DataFrame) -> dict:
    """movieId must be unique in links.csv."""
    return {
        "duplicate_movieId": find_duplicates(links, ["movieId"], "keep first"),
    }


def analyze_genome_score_duplicates(genome_scores: pd.DataFrame) -> dict:
    """(movieId, tagId) must be unique in genome-scores.csv."""
    return {
        "duplicate_movie_tag_pairs": find_duplicates(
            genome_scores, ["movieId", "tagId"], "keep first"
        ),
    }


# ---------------------------------------------------------------------------
# Referential integrity
# ---------------------------------------------------------------------------

def check_orphans(
    child: pd.DataFrame,
    child_column: str,
    parent: pd.DataFrame,
    parent_column: str,
    relationship: str,
) -> dict:
    """Quantify orphan records in ``child`` whose key is absent from ``parent``."""
    parent_keys = pd.Index(parent[parent_column].unique())
    child_keys = child[child_column]
    orphan_mask = ~child_keys.isin(parent_keys)
    orphan_count = int(orphan_mask.sum())

    sample = (
        child.loc[orphan_mask, child_column].dropna().astype("int64").head(10).astype(int).tolist()
        if orphan_count and pd.api.types.is_numeric_dtype(child[child_column])
        else []
    )

    result = {
        "relationship": relationship,
        "child_rows": int(len(child)),
        "orphan_records": orphan_count,
        "orphan_pct": round(100.0 * orphan_count / max(len(child), 1), 4),
        "orphan_key_examples": sample,
    }
    level = "WARNING" if orphan_count else "INFO"
    logger.log(
        getattr(logging, level),
        "%s: %d/%d orphan records (%.4f%%)",
        relationship,
        orphan_count,
        len(child),
        result["orphan_pct"],
    )
    return result


def analyze_referential_integrity(
    movies: pd.DataFrame,
    ratings: pd.DataFrame | None = None,
    tags: pd.DataFrame | None = None,
    links: pd.DataFrame | None = None,
    genome_scores: pd.DataFrame | None = None,
    genome_tags: pd.DataFrame | None = None,
) -> dict:
    """Run every documented foreign-key relationship check."""
    checks = []
    if ratings is not None:
        checks.append(check_orphans(ratings, "movieId", movies, "movieId",
                                    "ratings.movieId -> movies.movieId"))
    if tags is not None:
        checks.append(check_orphans(tags, "movieId", movies, "movieId",
                                    "tags.movieId -> movies.movieId"))
    if genome_scores is not None:
        checks.append(check_orphans(genome_scores, "movieId", movies, "movieId",
                                    "genome_scores.movieId -> movies.movieId"))
        if genome_tags is not None:
            checks.append(check_orphans(genome_scores, "tagId", genome_tags, "tagId",
                                        "genome_scores.tagId -> genome_tags.tagId"))
    if links is not None:
        checks.append(check_orphans(links, "movieId", movies, "movieId",
                                    "links.movieId -> movies.movieId"))

    total_orphans = int(sum(c["orphan_records"] for c in checks))
    return {"checks": checks, "total_orphan_records": total_orphans}


# ---------------------------------------------------------------------------
# Domain validation
# ---------------------------------------------------------------------------

def validate_rating_domain(ratings: pd.DataFrame) -> dict:
    """Check ratings fall on the MovieLens 0.5-5.0 half-star scale."""
    r = ratings["rating"].astype("float64")
    valid_range = r.between(
        config.RATING_MIN, config.RATING_MAX, inclusive="both"
    ).fillna(False)
    half_steps = (r - config.RATING_MIN) / config.RATING_STEP
    on_grid = pd.Series(
        np.isclose(
            half_steps.to_numpy(), np.round(half_steps.to_numpy()), equal_nan=False
        ),
        index=r.index,
    )

    invalid_range = int((~valid_range & r.notna()).sum())
    off_grid = int((valid_range & ~on_grid & r.notna()).sum())
    result = {
        "min_allowed": config.RATING_MIN,
        "max_allowed": config.RATING_MAX,
        "out_of_range": invalid_range,
        "off_half_star_grid": off_grid,
        "invalid_total": int(invalid_range + off_grid),
        "valid": invalid_range == 0 and off_grid == 0,
    }
    if not result["valid"]:
        logger.warning(
            "Rating domain violations found: %d out-of-range, %d off-grid",
            invalid_range,
            off_grid,
        )
    else:
        logger.info("All ratings lie within the valid MovieLens scale")
    return result


def validate_id_columns(df: pd.DataFrame, id_columns: list[str], label: str) -> dict:
    """IDs must be non-null, positive integers."""
    issues = {}
    for col in id_columns:
        s = df[col]
        non_null = s.dropna()
        negative = int((non_null < 0).sum()) if pd.api.types.is_numeric_dtype(non_null) else 0
        non_integer = int((non_null % 1 != 0).sum()) if pd.api.types.is_numeric_dtype(non_null) else 0
        missing = int(s.isna().sum())
        if missing or negative or non_integer:
            issues[col] = {"missing": missing, "negative": negative, "non_integer": non_integer}
    result = {"dataset": label, "checked": id_columns, "issues": issues,
              "valid": not issues}
    if issues:
        logger.warning("ID validation issues in '%s': %s", label, issues)
    else:
        logger.info("ID validation passed for '%s' (%s)", label, ", ".join(id_columns))
    return result


def summarize_quality_checks(*check_groups: dict) -> dict:
    """Combine multiple check-result dicts into one quality_checks block."""
    combined: dict = {}
    for group in check_groups:
        combined.update(group)
    return combined
