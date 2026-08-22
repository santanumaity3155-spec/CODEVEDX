"""Robust, memory-conscious loaders for the MovieLens 25M dataset.

Every public function returns a :class:`pandas.DataFrame` loaded with
efficient dtypes and validated against the expected schema. Failures
raise descriptive, actionable exceptions derived from
:class:`DataLoadError` instead of letting low-level parser errors
escape.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import config
from src.logging_config import get_logger

logger = get_logger(__name__)


class DataLoadError(Exception):
    """Base class for all data-loading failures."""


class MissingFileError(DataLoadError):
    """Raised when the requested CSV file does not exist."""


class MalformedCSVError(DataLoadError):
    """Raised when a CSV file cannot be parsed (corrupt or empty)."""


class SchemaError(DataLoadError):
    """Raised when required columns are missing from a dataset."""


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise MissingFileError(
            f"Dataset file not found: '{path}'. "
            f"Verify the {config.DATASET_NAME} archive was extracted under "
            f"'{config.RAW_DATA_DIR}' and that the path is correct."
        )


def _validate_columns(df: pd.DataFrame, required: tuple[str, ...], label: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise SchemaError(
            f"Dataset '{label}' is missing required column(s): {missing}. "
            f"Found columns: {list(df.columns)}."
        )


def load_csv(
    path: str | Path,
    *,
    dtype: dict[str, str] | None = None,
    required_columns: tuple[str, ...] = (),
    label: str | None = None,
) -> pd.DataFrame:
    """Load a single CSV file with validation and clear error reporting.

    Parameters
    ----------
    path:
        Path to the CSV file.
    dtype:
        Optional per-column dtypes passed to :func:`pandas.read_csv`.
    required_columns:
        Columns that must exist; a :class:`SchemaError` is raised otherwise.
    label:
        Human-friendly dataset name used in logs and error messages.

    Returns
    -------
    pd.DataFrame
        The loaded dataframe (never a copy of cached state).
    """
    path = Path(path)
    label = label or path.name
    _require_file(path)

    logger.info("Loading '%s' from %s ...", label, path)
    try:
        df = pd.read_csv(path, dtype=dtype, encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise DataLoadError(
            f"Failed to decode '{path}' as UTF-8 ({exc}). "
            "The file may be corrupted or use an unexpected encoding."
        ) from exc
    except pd.errors.EmptyDataError as exc:
        raise MalformedCSVError(f"CSV file '{path}' contains no data.") from exc
    except pd.errors.ParserError as exc:
        raise MalformedCSVError(
            f"CSV file '{path}' is malformed and could not be parsed: {exc}"
        ) from exc
    except ValueError as exc:
        raise DataLoadError(
            f"Could not load '{path}' with the requested dtypes: {exc}. "
            "This usually means the file contains unexpected values "
            "(e.g., text in a numeric column or missing IDs)."
        ) from exc

    if df.empty:
        raise DataLoadError(f"CSV file '{path}' loaded successfully but is empty (0 rows).")

    _validate_columns(df, required_columns, label)

    mem_mb = df.memory_usage(deep=True).sum() / 1024**2
    logger.info(
        "Loaded '%s': %d rows x %d columns (%.1f MiB in memory)",
        label,
        len(df),
        df.shape[1],
        mem_mb,
    )
    return df


# ---------------------------------------------------------------------------
# Per-dataset loaders. Dtypes are chosen for the known MovieLens 25M ID
# ranges while remaining wide enough to never truncate values.
# ---------------------------------------------------------------------------

_MOVIES_DTYPE = {"movieId": "int32", "title": "string", "genres": "string"}
_RATINGS_DTYPE = {
    "userId": "int32",
    "movieId": "int32",
    "rating": "float32",
    "timestamp": "int64",
}
_TAGS_DTYPE = {
    "userId": "int32",
    "movieId": "int32",
    "tag": "string",
    "timestamp": "Int64",
}
_LINKS_DTYPE = {"movieId": "int32", "imdbId": "Int64", "tmdbId": "Int64"}
_GENOME_SCORES_DTYPE = {"movieId": "int32", "tagId": "int16", "relevance": "float32"}
_GENOME_TAGS_DTYPE = {"tagId": "int16", "tag": "string"}


def load_movies(path: str | Path | None = None) -> pd.DataFrame:
    """Load ``movies.csv`` (movieId, title, genres)."""
    path = config.MOVIES_PATH if path is None else path
    return load_csv(
        path,
        dtype=_MOVIES_DTYPE,
        required_columns=config.EXPECTED_SCHEMAS["movies"],
        label="movies",
    )


def load_ratings(path: str | Path | None = None) -> pd.DataFrame:
    """Load ``ratings.csv`` (userId, movieId, rating, timestamp)."""
    path = config.RATINGS_PATH if path is None else path
    return load_csv(
        path,
        dtype=_RATINGS_DTYPE,
        required_columns=config.EXPECTED_SCHEMAS["ratings"],
        label="ratings",
    )


def load_tags(path: str | Path | None = None) -> pd.DataFrame:
    """Load ``tags.csv`` (userId, movieId, tag, timestamp)."""
    path = config.TAGS_PATH if path is None else path
    return load_csv(
        path,
        dtype=_TAGS_DTYPE,
        required_columns=config.EXPECTED_SCHEMAS["tags"],
        label="tags",
    )


def load_links(path: str | Path | None = None) -> pd.DataFrame:
    """Load ``links.csv`` (movieId, imdbId, tmdbId); external IDs may be null."""
    path = config.LINKS_PATH if path is None else path
    return load_csv(
        path,
        dtype=_LINKS_DTYPE,
        required_columns=config.EXPECTED_SCHEMAS["links"],
        label="links",
    )


def load_genome_scores(path: str | Path | None = None) -> pd.DataFrame:
    """Load ``genome-scores.csv`` (movieId, tagId, relevance)."""
    path = config.GENOME_SCORES_PATH if path is None else path
    return load_csv(
        path,
        dtype=_GENOME_SCORES_DTYPE,
        required_columns=config.EXPECTED_SCHEMAS["genome_scores"],
        label="genome_scores",
    )


def load_genome_tags(path: str | Path | None = None) -> pd.DataFrame:
    """Load ``genome-tags.csv`` (tagId, tag)."""
    path = config.GENOME_TAGS_PATH if path is None else path
    return load_csv(
        path,
        dtype=_GENOME_TAGS_DTYPE,
        required_columns=config.EXPECTED_SCHEMAS["genome_tags"],
        label="genome_tags",
    )
