"""Content feature engineering for the Smart Recommendation System (Module 2).

Converts the cleaned MovieLens catalog (title, genres) together with user tags
into machine-learning-ready representations:

    cleaned movies + tags
        -> text / genre normalization
        -> feature construction (content documents + multi-hot genres)
        -> TF-IDF vectorization (scikit-learn)
        -> sparse feature matrix (scipy.sparse)
        -> feature index (movieId -> row index)
        -> serialized vectorizer (joblib)
        -> quality report (JSON + TXT)

Module 2 consumes ONLY the Module 1 processed outputs in ``data/processed/``
(``movies_clean.csv`` and ``tags_clean.csv``). It never reads from
``data/raw`` and never modifies it.

The pipeline is fully deterministic: re-running on the same processed inputs
reproduces identical movie ordering, vocabulary, feature index and a
numerically equal sparse TF-IDF matrix.

Run from the project root with::

    python -m src.feature_engineering
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from src import config
from src.analysis import split_genres
from src.data_loader import DataLoadError, MissingFileError, load_csv
from src.data_preprocessor import extract_release_year
from src.logging_config import get_logger
from src.pipeline import display_path, save_csv
from src.reporting import write_json_report

logger = get_logger(__name__)


class FeatureEngineeringError(Exception):
    """Raised for any Module 2 feature-engineering failure."""


# Token regex: alphanumeric runs with internal hyphens preserved so MovieLens
# genres/tags such as "sci-fi" and "film-noir" stay single terms. All other
# punctuation (pipes, commas, parentheses, quotes) is treated as a separator.
_TOKEN_RE = re.compile(config.TFIDF_TOKEN_PATTERN)
_YEAR_RE = re.compile(r"\s*\(\d{4}\)\s*$")
_NO_GENRES = config.NO_GENRES_LABEL


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------
def normalize_text(text: object) -> str:
    """Deterministically normalize free text into lowercased, single-spaced tokens.

    * ``None`` / NaN / non-string values become an empty string
    * lowercases the input
    * extracts tokens (alphanumeric runs, internal hyphens preserved)
    * joins tokens with single spaces

    The transform is idempotent and intentionally non-aggressive: it keeps
    meaningful movie words and genre/tag tokens while removing formatting noise.
    """
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""
    try:
        if pd.isna(text):
            return ""
    except (TypeError, ValueError):
        pass
    tokens = _TOKEN_RE.findall(str(text).lower())
    return " ".join(tokens)


def strip_release_year(title: object) -> str:
    """Remove the trailing ``(YYYY)`` release-year suffix from a title.

    The year is retained separately as ``release_year`` metadata; dropping it
    from the content document keeps TF-IDF tokens focused on semantic terms
    rather than a numeric suffix shared by many titles.
    """
    if not isinstance(title, str):
        return ""
    return _YEAR_RE.sub("", title).strip()


def _safe_genre_column(token: str) -> str:
    """Map a raw genre token to a stable boolean-flag column name.

    Mirrors the Module 1 ``movies_features_base`` convention (spaces and
    hyphens become underscores) so this representation stays aligned with it.
    """
    safe = token.replace(" ", "_").replace("-", "_")
    return f"genre_{safe}"


def split_normalized_genres(genres_series: pd.Series) -> pd.Series:
    """Return a Series of lists of normalized genre tokens (real genres only).

    Reuses :func:`src.analysis.split_genres` for pipe-splitting and the
    ``(no genres listed)`` handling, then lowercases each token (keeping
    internal hyphens). The ``(no genres listed)`` pseudo-genre is excluded
    from the returned token lists; it is still represented as a structural
    flag by :func:`build_genre_features`.
    """
    lists = split_genres(genres_series)

    def _norm(values: list[str]) -> list[str]:
        return [normalize_text(g) for g in values if g != _NO_GENRES]

    return lists.map(_norm)
# ---------------------------------------------------------------------------
# Tag aggregation
# ---------------------------------------------------------------------------
def aggregate_tags(tags_df: pd.DataFrame | None) -> pd.DataFrame:
    """Aggregate and normalize tags per movie.

    Returns a DataFrame with one row per movieId that has at least one usable
    tag, with columns:

    * ``movieId`` (int32)
    * ``aggregated_tags`` (str) - space-joined *unique* normalized tags, ordered
      by global frequency (desc) then alphabetically for deterministic output.
    * ``num_tags`` (int8) - count of unique tags

    Tags are deduplicated within a movie so the same term is not repeated.
    Missing / empty / null tags are dropped. Movies with no tags simply do not
    appear here and receive an empty string downstream.

    Parameters
    ----------
    tags_df:
        Cleaned tags (``tags_clean.csv``) with at least ``movieId`` and ``tag``
        columns, or ``None`` when no tag source is available.
    """
    empty = pd.DataFrame(
        {
            "movieId": pd.array([], dtype="int32"),
            "aggregated_tags": pd.array([], dtype="string"),
            "num_tags": pd.array([], dtype="int16"),
        }
    )
    if tags_df is None or len(tags_df) == 0:
        return empty
    work = tags_df[["movieId", "tag"]].dropna(subset=["movieId", "tag"]).copy()
    if len(work) == 0:
        return empty
    work["movieId"] = work["movieId"].astype("int32")
    work["tag"] = work["tag"].map(normalize_text)
    work = work[work["tag"] != ""]  # drop empty after normalization
    if len(work) == 0:
        return empty

    global_freq = work["tag"].value_counts()

    def _ordered(tags: pd.Series) -> list[str]:
        unique_tags = list(dict.fromkeys(tags.tolist()))  # dedupe, first-seen order
        unique_tags.sort(key=lambda tag: (-int(global_freq.get(tag, 0)), tag))
        return unique_tags

    records = []
    for movie_id, group in work.groupby("movieId", sort=True):
        unique_tags = _ordered(group["tag"])
        records.append(
            {
                "movieId": int(movie_id),
                "aggregated_tags": " ".join(unique_tags),
                "num_tags": int(len(unique_tags)),
            }
        )
    agg = pd.DataFrame(records, columns=["movieId", "aggregated_tags", "num_tags"])
    agg["movieId"] = agg["movieId"].astype("int32")
    agg["num_tags"] = agg["num_tags"].astype("int16")
    agg = agg.sort_values("movieId", kind="mergesort").reset_index(drop=True)
    logger.info(
        "Aggregated tags for %d movies (%d unique tags)",
        len(agg),
        int(work["tag"].nunique()),
    )
    return agg


def compute_tag_statistics(tags_df: pd.DataFrame | None) -> dict:
    """Return JSON-safe tag statistics computed on the cleaned tag set."""
    if tags_df is None or len(tags_df) == 0:
        return {
            "total_tag_records": 0,
            "valid_tag_records": 0,
            "unique_tags": 0,
            "unique_users_tagging": 0,
            "unique_movies_tagged": 0,
            "avg_unique_tags_per_movie": 0.0,
            "most_frequent_tags": {},
        }
    total = int(len(tags_df))
    work = tags_df.dropna(subset=["movieId", "tag"]).copy()
    work["tag"] = work["tag"].map(normalize_text)
    work = work[work["tag"] != ""]
    counts = work["tag"].value_counts()
    per_movie = (
        work.groupby("movieId")["tag"].nunique() if len(work) else pd.Series(dtype="int64")
    )
    stats = {
        "total_tag_records": total,
        "valid_tag_records": int(len(work)),
        "unique_tags": int(counts.size),
        "unique_users_tagging": int(work["userId"].nunique()) if "userId" in work else 0,
        "unique_movies_tagged": int(work["movieId"].nunique()),
        "avg_unique_tags_per_movie": round(float(per_movie.mean()), 4) if len(per_movie) else 0.0,
        "most_frequent_tags": {str(k): int(v) for k, v in counts.head(20).items()},
    }
    logger.info(
        "Tag statistics: %d records, %d unique tags across %d movies",
        stats["total_tag_records"],
        stats["unique_tags"],
        stats["unique_movies_tagged"],
    )
    return stats
# ---------------------------------------------------------------------------
# Genre features
# ---------------------------------------------------------------------------
def build_genre_features(movies_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Build a deterministic multi-hot genre representation from movie genres.

    Returns ``(genre_df, genre_vocab)`` where ``genre_df`` has one row per
    movie (``movieId`` plus one ``genre_<token>`` boolean flag per distinct
    genre token, including ``(no genres listed)``) and ``genre_vocab`` is the
    sorted list of genre tokens. Column naming mirrors Module 1
    ``movies_features_base`` for consistency.
    """
    genre_lists = split_genres(movies_df["genres"])
    genre_vocab = sorted({token for tokens in genre_lists for token in tokens})
    records: dict = {"movieId": movies_df["movieId"].astype("int32").to_numpy()}
    lists = genre_lists.tolist()
    for token in genre_vocab:
        col = _safe_genre_column(token)
        records[col] = np.array([int(token in tokens) for tokens in lists], dtype="int8")
    genre_df = pd.DataFrame(records)
    genre_df = genre_df.sort_values("movieId", kind="mergesort").reset_index(drop=True)
    logger.info(
        "Genre features: %d distinct genres -> %d multi-hot columns for %d movies",
        len(genre_vocab),
        genre_df.shape[1] - 1,
        len(genre_df),
    )
    return genre_df, genre_vocab


# ---------------------------------------------------------------------------
# Content documents
# ---------------------------------------------------------------------------
def build_content_documents(
    movies_df: pd.DataFrame, tag_agg: pd.DataFrame
) -> pd.DataFrame:
    """Build one deterministic content document per movie.

    Columns: movieId, title, release_year, genres, normalized_genres,
    num_genres, aggregated_tags, num_tags, content_text.

    ``content_text`` combines stripped-title tokens, normalized genre tokens
    and normalized aggregated tag tokens (single-spaced, lowercased). Movies
    without tags still receive valid documents built from title + genres.
    """
    df = movies_df[["movieId", "title", "genres"]].copy()
    df["movieId"] = df["movieId"].astype("int32")
    df["release_year"] = movies_df["title"].map(extract_release_year).astype("Int32")

    genre_lists = split_genres(movies_df["genres"])
    norm_genre_lists = genre_lists.map(
        lambda values: [normalize_text(g) for g in values if g != _NO_GENRES]
    ).tolist()
    df["normalized_genres"] = ["|".join(tokens) for tokens in norm_genre_lists]
    df["num_genres"] = pd.array([len(tokens) for tokens in norm_genre_lists], dtype="int16")

    # Left-join normalized tags; movies without tags get empty strings.
    if len(tag_agg):
        df = df.merge(
            tag_agg[["movieId", "aggregated_tags", "num_tags"]],
            on="movieId",
            how="left",
        )
    else:
        df["aggregated_tags"] = pd.array([""] * len(df), dtype="object")
        df["num_tags"] = pd.array([0] * len(df), dtype="int16")
    df["aggregated_tags"] = df["aggregated_tags"].fillna("")
    df["num_tags"] = df["num_tags"].fillna(0).astype("int16")

    titles = movies_df["title"].tolist()
    title_tokens = [normalize_text(strip_release_year(t)).split() for t in titles]
    tag_tokens = [str(s).split() for s in df["aggregated_tags"].tolist()]
    content_text = [
        " ".join(t + g + tag)
        for t, g, tag in zip(title_tokens, norm_genre_lists, tag_tokens)
    ]
    df["content_text"] = content_text

    columns = [
        "movieId", "title", "release_year", "genres", "normalized_genres",
        "num_genres", "aggregated_tags", "num_tags", "content_text",
    ]
    df = df[columns].sort_values("movieId", kind="mergesort").reset_index(drop=True)
    logger.info(
        "Built %d content documents (%d with tags, %d without)",
        len(df),
        int((df["num_tags"] > 0).sum()),
        int((df["num_tags"] == 0).sum()),
    )
    return df
# ---------------------------------------------------------------------------
# TF-IDF + feature index
# ---------------------------------------------------------------------------
def fit_tfidf(
    documents: list[str],
) -> tuple[TfidfVectorizer, sparse.csr_matrix]:
    """Fit a TF-IDF vectorizer on the movie-content corpus only.

    The vectorizer never sees ratings, predictions or evaluation data, so the
    vocabulary leaks no information from later pipeline stages. The returned
    matrix is CSR-encoded ``float32`` to keep memory bounded for the 62k-movie
    corpus.
    """
    if len(documents) == 0:
        raise FeatureEngineeringError("Cannot fit TF-IDF on an empty corpus.")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        token_pattern=config.TFIDF_TOKEN_PATTERN,
        ngram_range=config.TFIDF_NGRAM_RANGE,
        min_df=config.TFIDF_MIN_DF,
        max_df=config.TFIDF_MAX_DF,
        max_features=config.TFIDF_MAX_FEATURES,
        sublinear_tf=config.TFIDF_SUBLINEAR_TF,
        norm=config.TFIDF_NORM,
        dtype=np.dtype(config.TFIDF_DTYPE),
    )
    try:
        matrix = vectorizer.fit_transform(documents).tocsr()
    except Exception as exc:  # noqa: BLE001 - wrapped into a pipeline error
        raise FeatureEngineeringError(f"TF-IDF vectorization failed: {exc}") from exc
    if len(vectorizer.vocabulary_) == 0:
        raise FeatureEngineeringError(
            "TF-IDF vocabulary is empty after fitting. The corpus may not "
            "contain enough distinct terms for the configured min_df/max_df."
        )
    logger.info(
        "TF-IDF fit: vocabulary=%d features, matrix=%s, nnz=%d",
        len(vectorizer.vocabulary_),
        matrix.shape,
        matrix.nnz,
    )
    return vectorizer, matrix


def build_feature_index(movie_ids: pd.Series | list[int]) -> pd.DataFrame:
    """Build a deterministic movieId -> contiguous row-index mapping.

    Ordering is ascending by movieId (deterministic); ``row_index`` runs
    0..N-1 with no gaps. The mapping aligns with the TF-IDF matrix rows.
    """
    ids = sorted({int(m) for m in movie_ids})
    index = pd.DataFrame(
        {
            "movieId": pd.array(ids, dtype="int32"),
            "row_index": pd.array(np.arange(len(ids), dtype="int32")),
        }
    )
    return index


def validate_feature_artifacts(
    movies_df: pd.DataFrame,
    content_df: pd.DataFrame,
    matrix: sparse.spmatrix,
    vectorizer: TfidfVectorizer,
    feature_index: pd.DataFrame,
) -> list[dict]:
    """Run the Module 2 feature-quality checks (Step 13).

    Returns a list of ``{"check", "passed", "detail"}`` records covering:
    per-movie feature rows, ID uniqueness, matrix/index row counts matching the
    movie count, absence of NaN/Inf, non-empty vocabulary, unique contiguous
    row indices, deterministic ordering, content-discriminating vectors, and
    valid vectors for movies that lack tags.
    """
    checks: list[dict] = []
    n_movies = int(len(movies_df))

    def _record(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    # 1. Every movie has a feature row.
    _record(
        "every_movie_has_feature_row",
        len(content_df) == n_movies,
        f"movies={n_movies} content_rows={len(content_df)}",
    )
    # 2. Movie IDs unique in content documents.
    _record("movie_ids_unique_in_content", bool(content_df["movieId"].is_unique))
    # 3. Feature matrix row count equals movie count.
    _record(
        "matrix_rows_equal_movie_count",
        matrix.shape[0] == n_movies,
        f"matrix_rows={matrix.shape[0]}",
    )
    # 4. Feature index row count equals movie count.
    _record(
        "feature_index_rows_equal_movie_count",
        len(feature_index) == n_movies,
        f"index_rows={len(feature_index)}",
    )
    # 5. No NaN in feature matrix.
    _record("matrix_no_nan", not bool(np.isnan(matrix.data).any()))
    # 6. No infinite values in feature matrix.
    _record("matrix_no_inf", not bool(np.isinf(matrix.data).any()))
    # 7. Vectorizer vocabulary non-empty.
    _record(
        "vocabulary_non_empty",
        len(vectorizer.vocabulary_) > 0,
        f"vocab={len(vectorizer.vocabulary_)}",
    )
    # 8. Every feature row maps to exactly one movie (unique movieId in index).
    _record("feature_index_unique_movieid", bool(feature_index["movieId"].is_unique))
    # 9. Deterministic ordering (ascending movieId).
    ids = feature_index["movieId"].tolist()
    _record("ordering_sorted_ascending", ids == sorted(ids))
    # 10. Contiguous row indices 0..N-1.
    row_idx = feature_index["row_index"].to_numpy()
    _record(
        "indices_contiguous",
        len(row_idx) == n_movies and list(row_idx) == list(range(n_movies)),
    )
    # 11. Movies with different content can produce different vectors.
    texts = content_df["content_text"].astype(str).tolist()
    diff_pair = None
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if texts[i] != texts[j]:
                diff_pair = (i, j)
                break
        if diff_pair:
            break
    if diff_pair is not None:
        i, j = diff_pair
        different = not np.array_equal(matrix.getrow(i).toarray(), matrix.getrow(j).toarray())
        _record("different_content_different_vectors", different)
    else:
        _record("different_content_different_vectors", False, "all content identical")
    # 12. Movies without tags still receive valid (non-empty) content vectors.
    no_tag = content_df[content_df["aggregated_tags"].astype(str).str.len() == 0]
    _record(
        "movies_without_tags_have_content",
        len(no_tag) == 0 or bool((no_tag["content_text"].astype(str).str.len() > 0).all()),
        f"no_tag_movies={len(no_tag)}",
    )
    return checks
# ---------------------------------------------------------------------------
# Serialization & loading
# ---------------------------------------------------------------------------
def save_sparse_matrix(matrix: sparse.spmatrix, path: str | Path) -> Path:
    """Persist the sparse TF-IDF matrix to an efficient scipy ``.npz`` archive.

    The matrix stays sparse end-to-end; it is NEVER densified here. A ``.npz``
    suffix is appended when the given path lacks one.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    target = path if path.suffix.lower() == ".npz" else Path(f"{path}.npz")
    try:
        sparse.save_npz(target, matrix.tocsr())
    except Exception as exc:  # noqa: BLE001 - wrapped into a pipeline error
        raise FeatureEngineeringError(
            f"Failed to serialize sparse matrix to '{target}': {exc}"
        ) from exc
    size_mb = target.stat().st_size / 1024**2
    logger.info(
        "Saved sparse matrix %s (nnz=%d, %.2f MiB) to %s",
        matrix.shape,
        matrix.nnz,
        size_mb,
        target,
    )
    return target


def load_sparse_matrix(path: str | Path) -> sparse.csr_matrix:
    """Load a serialized sparse feature matrix (``.npz``) as CSR."""
    path = Path(path)
    if not path.is_file():
        raise MissingFileError(
            f"Sparse matrix file not found: '{path}'. "
            "Run 'python -m src.feature_engineering' first."
        )
    try:
        matrix = sparse.load_npz(path).tocsr()
    except Exception as exc:  # noqa: BLE001 - wrap into a pipeline error
        raise FeatureEngineeringError(
            f"Failed to load sparse matrix from '{path}': {exc}"
        ) from exc
    logger.info("Loaded sparse matrix %s (nnz=%d) from %s", matrix.shape, matrix.nnz, path)
    return matrix


def save_feature_vectorizer(vectorizer: TfidfVectorizer, path: str | Path) -> Path:
    """Persist a fitted TF-IDF vectorizer with joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        joblib.dump(vectorizer, path)
    except Exception as exc:  # noqa: BLE001 - wrap into a pipeline error
        raise FeatureEngineeringError(
            f"Failed to serialize vectorizer to '{path}': {exc}"
        ) from exc
    size_mb = path.stat().st_size / 1024**2
    logger.info(
        "Saved fitted TF-IDF vectorizer (vocab=%d, %.2f MiB) to %s",
        len(vectorizer.vocabulary_),
        size_mb,
        path,
    )
    return path


def load_feature_vectorizer(path: str | Path | None = None) -> TfidfVectorizer:
    """Load a fitted TF-IDF vectorizer (joblib), independently usable.

    Module 3 can call ``load_feature_vectorizer()`` and immediately transform
    content documents without re-fitting.
    """
    path = Path(path) if path is not None else config.MOVIE_TFIDF_VECTORIZER_PATH
    if not path.is_file():
        raise MissingFileError(
            f"Vectorizer file not found: '{path}'. "
            "Run 'python -m src.feature_engineering' first."
        )
    try:
        vectorizer = joblib.load(path)
    except Exception as exc:  # noqa: BLE001 - wrap into a pipeline error
        raise FeatureEngineeringError(
            f"Failed to load vectorizer from '{path}': {exc}"
        ) from exc
    if not hasattr(vectorizer, "vocabulary_"):
        raise FeatureEngineeringError(
            f"'{path}' does not contain a fitted TfidfVectorizer (no vocabulary)."
        )
    logger.info(
        "Loaded TF-IDF vectorizer from %s (vocabulary_size=%d)",
        path,
        len(vectorizer.vocabulary_),
    )
    return vectorizer


def verify_loaded_vectorizer(
    vectorizer: TfidfVectorizer,
    documents: list[str],
    reference_matrix: sparse.spmatrix,
) -> bool:
    """Return True when ``vectorizer.transform`` reproduces the reference matrix.

    Compares shape, non-zero count and sparse values so the verification never
    densifies the feature matrix.
    """
    reproduced = vectorizer.transform(documents).tocsr()
    reference = reference_matrix.tocsr()
    if reproduced.shape != reference.shape or reproduced.nnz != reference.nnz:
        return False
    diff = reproduced - reference
    diff.eliminate_zeros()
    if diff.nnz == 0:
        return True
    return bool(abs(diff).max() < 1e-6)
# ---------------------------------------------------------------------------
# Feature report
# ---------------------------------------------------------------------------
def build_feature_report(
    movies_df: pd.DataFrame,
    tags_df: pd.DataFrame | None,
    content_df: pd.DataFrame,
    genre_df: pd.DataFrame,
    genre_vocab: list[str],
    tag_stats: dict,
    vectorizer: TfidfVectorizer,
    matrix: sparse.spmatrix,
    feature_index: pd.DataFrame,
    checks: list[dict],
    saved_files: dict[str, str],
) -> dict:
    """Assemble the Module 2 feature-engineering report as a JSON-safe dict."""
    rows, columns = int(matrix.shape[0]), int(matrix.shape[1])
    nnz = int(matrix.nnz)
    total_cells = rows * columns
    sparsity_pct = round(100.0 * (1.0 - nnz / total_cells), 6) if total_cells else 0.0

    idx = feature_index["row_index"].to_numpy()
    idx_contiguous = bool(
        len(idx) > 0
        and idx[0] == 0
        and (np.diff(idx) == 1).all()
        and idx[-1] == len(idx) - 1
    )

    report: dict = {
        "module": "Module 2 - Content Feature Engineering",
        "dataset_name": config.DATASET_NAME,
        "status": "PENDING",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": {
            "movies": {
                "path": display_path(config.PROCESSED_MOVIES_PATH),
                "rows": int(len(movies_df)),
                "columns": list(movies_df.columns),
            },
            "tags": {
                "path": display_path(config.PROCESSED_TAGS_PATH),
                "rows": int(len(tags_df)) if tags_df is not None else 0,
                "present": tags_df is not None,
            },
        },
        "content_documents": {
            "num_documents": int(len(content_df)),
            "num_movies": int(len(movies_df)),
            "movies_with_tags": int((content_df["num_tags"] > 0).sum()),
            "movies_without_tags": int((content_df["num_tags"] == 0).sum()),
            "missing_content_text": int(content_df["content_text"].isna().sum()),
            "columns": list(content_df.columns),
        },
        "genres": {
            "genre_feature_count": int(len(genre_vocab)),
            "genre_vocab": [str(g) for g in genre_vocab],
            "genre_flag_columns": int(genre_df.shape[1] - 1),
            "movies_with_no_genre": int(
                (movies_df["genres"] == config.NO_GENRES_LABEL).sum()
            ),
        },
        "tag_statistics": tag_stats,
        "tfidf": {
            "vocabulary_size": int(len(vectorizer.vocabulary_)),
            "matrix_rows": rows,
            "matrix_columns": columns,
            "non_zero_elements": nnz,
            "sparsity_percent": sparsity_pct,
            "matrix_dtype": str(matrix.dtype),
            "matrix_shape": [rows, columns],
            "vectorizer_configuration": {
                "lowercase": bool(vectorizer.lowercase),
                "token_pattern": str(vectorizer.token_pattern),
                "ngram_range": [int(v) for v in vectorizer.ngram_range],
                "min_df": vectorizer.min_df,
                "max_df": vectorizer.max_df,
                "max_features": vectorizer.max_features,
                "sublinear_tf": bool(vectorizer.sublinear_tf),
                "norm": str(vectorizer.norm),
                "dtype": str(vectorizer.dtype),
            },
        },
        "feature_index": {
            "rows": int(len(feature_index)),
            "row_index_min": int(idx.min()) if len(idx) else None,
            "row_index_max": int(idx.max()) if len(idx) else None,
            "indices_contiguous": idx_contiguous,
        },
        "quality_checks": checks,
        "quality_checks_all_passed": bool(all(c.get("passed", False) for c in checks)),
        "generated_files": saved_files,
    }
    gate = "PASS" if report["quality_checks_all_passed"] else "FAIL"
    report["feature_quality_gate"] = gate
    logger.info("Feature report assembled: quality gate = %s", gate)
    return report
def render_feature_report_text(report: dict) -> str:
    """Render the feature-engineering report as human-readable plain text."""
    width = 78
    dash = "-" * width
    lines: list[str] = [
        "=" * width,
        "MOVIELENS 25M - MODULE 2 CONTENT FEATURE ENGINEERING REPORT",
        "=" * width,
        f"Dataset                : {report.get('dataset_name')}",
        f"Generated (UTC)        : {report.get('generated_at_utc')}",
        f"Feature quality gate   : {report.get('feature_quality_gate')}",
        "",
        "Inputs",
        dash,
        f"  movies_clean rows    : {report['inputs']['movies']['rows']:,}",
        f"  tags_clean rows      : {report['inputs']['tags']['rows']:,} "
        f"(present={report['inputs']['tags']['present']})",
        "",
        "Content documents",
        dash,
        f"  documents            : {report['content_documents']['num_documents']:,}",
        f"  movies               : {report['content_documents']['num_movies']:,}",
        f"  movies with tags     : {report['content_documents']['movies_with_tags']:,}",
        f"  movies without tags  : {report['content_documents']['movies_without_tags']:,}",
        "",
        "Genres",
        dash,
        f"  distinct genres      : {report['genres']['genre_feature_count']:,}",
        f"  genre flag columns   : {report['genres']['genre_flag_columns']:,}",
        f"  movies with no genre : {report['genres']['movies_with_no_genre']:,}",
        "",
        "Tags",
        dash,
        f"  tag records          : {report['tag_statistics']['total_tag_records']:,}",
        f"  valid tag records    : {report['tag_statistics']['valid_tag_records']:,}",
        f"  unique tags          : {report['tag_statistics']['unique_tags']:,}",
        f"  movies tagged        : {report['tag_statistics']['unique_movies_tagged']:,}",
        "",
        "TF-IDF",
        dash,
        f"  vocabulary size      : {report['tfidf']['vocabulary_size']:,}",
        f"  matrix rows          : {report['tfidf']['matrix_rows']:,}",
        f"  matrix columns       : {report['tfidf']['matrix_columns']:,}",
        f"  non-zero elements    : {report['tfidf']['non_zero_elements']:,}",
        f"  sparsity             : {report['tfidf']['sparsity_percent']:.4f}%",
        f"  matrix dtype         : {report['tfidf']['matrix_dtype']}",
        "  configuration:",
    ]
    for key, value in report["tfidf"]["vectorizer_configuration"].items():
        lines.append(f"    {key:<16}: {value}")

    fi = report["feature_index"]
    lines += [
        "",
        "Feature index",
        dash,
        f"  rows                 : {fi['rows']:,}",
        f"  row range            : [{fi['row_index_min']} .. {fi['row_index_max']}]",
        f"  contiguous           : {fi['indices_contiguous']}",
        "",
        "Quality checks",
        dash,
    ]
    for chk in report.get("quality_checks", []):
        status = "PASS" if chk.get("passed", False) else "FAIL"
        lines.append(f"  [{status}] {chk['check']:<44} {chk.get('detail', '')}")

    lines += ["", "Generated files", dash]
    for name, path in report.get("generated_files", {}).items():
        lines.append(f"  {name:<22}: {path}")

    lines += ["", "=" * width, f"STATUS: {report.get('feature_quality_gate')}", "=" * width]
    return "\n".join(lines) + "\n"


def write_feature_reports(report: dict) -> tuple[Path, Path]:
    """Write the JSON and TXT feature-engineering reports."""
    json_path = write_json_report(report, config.FEATURE_ENGINEERING_REPORT_JSON_PATH)
    txt_path = config.FEATURE_ENGINEERING_REPORT_TXT_PATH
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text(render_feature_report_text(report), encoding="utf-8")
    logger.info("Wrote text feature report to %s", txt_path)
    return json_path, txt_path
# ---------------------------------------------------------------------------
# End-to-end feature engineering pipeline
# ---------------------------------------------------------------------------
_MOVIES_CLEAN_DTYPE = {"movieId": "int32", "title": "string", "genres": "string"}
_TAGS_CLEAN_DTYPE = {"userId": "int32", "movieId": "int32", "tag": "string"}


class FeatureEngineeringPipeline:
    """Build ML-ready movie content features from Module 1 processed data.

    Stages: load -> tags/genres -> content documents -> TF-IDF -> sparse
    matrix + index -> serialize -> validate -> report. Deterministic and
    idempotent given the same processed inputs.
    """

    def __init__(self) -> None:
        np.random.seed(config.RANDOM_SEED)  # defense-in-depth; pipeline is deterministic
        self.movies: pd.DataFrame | None = None
        self.tags: pd.DataFrame | None = None
        self.tag_agg: pd.DataFrame = pd.DataFrame()
        self.tag_stats: dict = {}
        self.genre_df: pd.DataFrame = pd.DataFrame()
        self.genre_vocab: list[str] = []
        self.content_df: pd.DataFrame = pd.DataFrame()
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix: sparse.csr_matrix | None = None
        self.feature_index: pd.DataFrame = pd.DataFrame()
        self.checks: list[dict] = []
        self.report: dict = {}

    # -- stages -------------------------------------------------------------
    def load_inputs(self) -> None:
        """Load processed Module 1 datasets (movies_clean, tags_clean)."""
        logger.info("=== STEP: load Module 1 processed feature source data ===")
        movies_path = config.PROCESSED_MOVIES_PATH
        if not movies_path.is_file():
            raise MissingFileError(
                f"Processed movies dataset not found: '{movies_path}'. "
                "Run 'python -m src.pipeline' before the feature pipeline."
            )
        self.movies = load_csv(
            movies_path,
            dtype=_MOVIES_CLEAN_DTYPE,
            required_columns=("movieId", "title", "genres"),
            label="movies_clean",
        )
        if self.movies["movieId"].isna().any():
            raise FeatureEngineeringError("movies_clean contains missing movieId values.")
        if not self.movies["movieId"].is_unique:
            raise FeatureEngineeringError("movies_clean contains duplicate movieId values.")
        if (self.movies["movieId"] <= 0).any():
            raise FeatureEngineeringError("movies_clean contains non-positive movieId values.")
        self.tags = self._load_tags_or_none()
        logger.info(
            "Feature inputs ready: %d movies, %s",
            len(self.movies),
            f"{len(self.tags)} tags" if self.tags is not None else "no tags available",
        )

    def _load_tags_or_none(self) -> pd.DataFrame | None:
        """Load tags_clean when present; missing files degrade to no tags."""
        tags_path = config.PROCESSED_TAGS_PATH
        if not tags_path.is_file():
            logger.warning(
                "Processed tags dataset not found (%s) - continuing WITHOUT tag "
                "features (title + genres only).",
                tags_path,
            )
            return None
        try:
            return load_csv(
                tags_path,
                dtype=_TAGS_CLEAN_DTYPE,
                required_columns=("userId", "movieId", "tag"),
                label="tags_clean",
            )
        except DataLoadError as exc:
            logger.warning(
                "Could not load processed tags (%s) - continuing WITHOUT tags.",
                exc,
            )
            return None

    def build_features(self) -> None:
        """Tag aggregation, genre features, content documents, TF-IDF, index."""
        logger.info("=== STEP: tag aggregation + genre features ===")
        self.tag_agg = aggregate_tags(self.tags)
        self.tag_stats = compute_tag_statistics(self.tags)
        self.genre_df, self.genre_vocab = build_genre_features(self.movies)
        logger.info(
            "Genre vocabulary: %d genres -> %d multi-hot columns",
            len(self.genre_vocab),
            self.genre_df.shape[1] - 1,
        )

        logger.info("=== STEP: content document construction ===")
        self.content_df = build_content_documents(self.movies, self.tag_agg)

        logger.info("=== STEP: TF-IDF feature engineering ===")
        documents = self.content_df["content_text"].astype(str).tolist()
        self.vectorizer, self.matrix = fit_tfidf(documents)

        logger.info("=== STEP: feature index ===")
        self.feature_index = build_feature_index(self.content_df["movieId"])

    def save_outputs(self) -> dict[str, str]:
        """Serialize all Module 2 artifacts; returns display-path mapping."""
        logger.info("=== STEP: artifact serialization ===")
        stored = {
            "movie_content_features.csv": (
                save_csv(self.content_df, config.PROCESSED_MOVIE_CONTENT_FEATURES_PATH)
            ),
            "movie_genre_features.csv": (
                save_csv(self.genre_df, config.PROCESSED_MOVIE_GENRE_FEATURES_PATH)
            ),
            "movie_tfidf.npz": (
                save_sparse_matrix(self.matrix, config.PROCESSED_MOVIE_TFIDF_PATH)
            ),
            "movie_tfidf_vectorizer.pkl": (
                save_feature_vectorizer(self.vectorizer, config.MOVIE_TFIDF_VECTORIZER_PATH)
            ),
            "movie_feature_index.csv": (
                save_csv(self.feature_index, config.PROCESSED_MOVIE_FEATURE_INDEX_PATH)
            ),
        }
        return {name: display_path(path) for name, path in stored.items()}

    def validate(self) -> list[dict]:
        """Run the feature-quality checks and log a summary."""
        logger.info("=== STEP: feature quality validation ===")
        self.checks = validate_feature_artifacts(
            self.movies,
            self.content_df,
            self.matrix,
            self.vectorizer,
            self.feature_index,
        )
        failed = [c for c in self.checks if not c["passed"]]
        logger.info(
            "Feature quality checks: %d/%d passed",
            len(self.checks) - len(failed),
            len(self.checks),
        )
        return self.checks

    def build_report(self, saved_files: dict[str, str]) -> dict:
        """Assemble the Module 2 feature-engineering report."""
        self.report = build_feature_report(
            movies_df=self.movies,
            tags_df=self.tags,
            content_df=self.content_df,
            genre_df=self.genre_df,
            genre_vocab=self.genre_vocab,
            tag_stats=self.tag_stats,
            vectorizer=self.vectorizer,
            matrix=self.matrix,
            feature_index=self.feature_index,
            checks=self.checks,
            saved_files=saved_files,
        )
        return self.report

    def run(self) -> dict:
        """Execute the end-to-end feature engineering pipeline."""
        self.load_inputs()
        self.build_features()
        saved_files = self.save_outputs()
        self.validate()
        report = self.build_report(saved_files)
        write_feature_reports(report)
        gate = report["feature_quality_gate"]
        logger.info("Feature engineering report written; quality gate = %s", gate)
        if gate != "PASS":
            raise FeatureEngineeringError(
                "Feature quality gate FAILED - generated artifacts do not satisfy "
                "the Module 2 validation checks."
            )
        logger.info("FEATURE ENGINEERING COMPLETE (quality gate: %s)", gate)
        return report


def main() -> int:
    try:
        report = FeatureEngineeringPipeline().run()
    except Exception:
        logger.exception("Feature engineering pipeline FAILED")
        return 1
    logger.info(
        "MODULE 2 COMPLETE - feature quality gate: %s", report["feature_quality_gate"]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())