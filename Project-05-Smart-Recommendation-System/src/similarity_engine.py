"""Similarity-based recommendation logic (Module 3).

Computes movie-to-movie and profile-to-movie similarities directly on the
sparse TF-IDF feature matrix produced by Module 2
(``data/processed/movie_tfidf.npz`` + ``movie_feature_index.csv``).

Design guarantees
-----------------
* **Sparse end-to-end** - the 62,423 x 20,000 feature matrix is never
  densified. Each query materialises a single dense score vector of length
  ``n_movies`` (one row of the similarity "matrix", ~0.5 MiB), which is the
  practical minimum for exact top-K ranking.
* **Exact cosine similarity** - cosine is computed as the dot product between
  L2-normalised vectors. Row norms are precomputed once (O(nnz)); rows are
  normalised defensively at load time in case the input matrix was produced
  without L2 normalisation. Zero-norm (empty) rows yield similarity 0.0 and
  never divide by zero.
* **Deterministic** - candidates are ranked with ``lexsort`` on
  ``(-similarity, movieId)``, so equal scores are always tie-broken by
  ascending movieId and repeated calls return identical results.
* **Self-exclusion** - the query movie itself can never be returned.
* **No refitting** - the Module 2 vectorizer/matrix are consumed as-is; this
  module never re-fits anything.

The engine stays free of data-loading concerns: callers pass artifacts they
have already loaded and validated.
"""

from __future__ import annotations

from numbers import Integral

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import linear_kernel

from src import config
from src.logging_config import get_logger

logger = get_logger(__name__)

# Tolerance used when deciding whether stored matrix rows already look
# L2-normalised (Module 2 uses norm="l2", so normally no rescaling happens).
_ROW_NORM_TOLERANCE = 1e-3


class SimilarityEngineError(Exception):
    """Base class for similarity-engine failures."""


class InvalidFeatureArtifactsError(SimilarityEngineError):
    """Raised when the matrix/feature-index pair is missing or inconsistent."""


class MovieNotFoundError(SimilarityEngineError):
    """Raised when a requested movieId has no row in the feature index."""


class InvalidTopKError(SimilarityEngineError, ValueError):
    """Raised when a top-K argument is not a positive integer.

    Inherits from :class:`ValueError` so generic validation handlers also
    catch it.
    """


def validate_top_k(top_k: object) -> int:
    """Validate a top-K request and return it as a plain ``int``.

    Raises
    ------
    InvalidTopKError
        If ``top_k`` is not an integer (bools rejected) or is < 1.
    """
    if isinstance(top_k, bool) or not isinstance(top_k, Integral):
        raise InvalidTopKError(
            f"top_k must be a positive integer, got {type(top_k).__name__}: {top_k!r}."
        )
    top_k = int(top_k)
    if top_k < 1:
        raise InvalidTopKError(f"top_k must be >= 1, got {top_k}.")
    return top_k


class SimilarityEngine:
    """Exact cosine-similarity nearest-neighbour search over TF-IDF rows.

    Parameters
    ----------
    matrix:
        Sparse CSR feature matrix (rows = movies, columns = TF-IDF terms).
    feature_index:
        DataFrame with columns ``movieId`` and ``row_index`` mapping each
        movie to its matrix row. Row indices must cover ``range(n_rows)``
        exactly once.

    Raises
    ------
    InvalidFeatureArtifactsError
        If the matrix or index is missing, malformed, or their dimensions
        disagree.
    """

    def __init__(self, matrix: sparse.spmatrix, feature_index: pd.DataFrame) -> None:
        self._matrix = self._validate_matrix(matrix)
        self._validate_index(feature_index)

        ordered = feature_index.sort_values("row_index", kind="mergesort")
        self._movie_ids: np.ndarray = ordered["movieId"].to_numpy(dtype=np.int64)
        # movieId (python int) -> matrix row for O(1) lookup.
        self._row_of_movie: dict[int, int] = {
            int(movie_id): int(row)
            for movie_id, row in zip(ordered["movieId"], ordered["row_index"])
        }

        self._inverse_norms = self._compute_inverse_row_norms()
        logger.info(
            "Similarity engine ready: %d movies x %d features (nnz=%d)",
            self.n_movies,
            self.n_features,
            self._matrix.nnz,
        )

    # ------------------------------------------------------------------ #
    # Construction helpers                                                #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate_matrix(matrix: object) -> sparse.csr_matrix:
        """Ensure the feature matrix is a finite 2-D CSR sparse matrix."""
        if not sparse.issparse(matrix):
            raise InvalidFeatureArtifactsError(
                "Feature matrix must be a scipy.sparse matrix; got "
                f"{type(matrix).__name__}. Load it with "
                "src.feature_engineering.load_sparse_matrix()."
            )
        csr = matrix.tocsr()
        if csr.ndim != 2 or csr.shape[1] == 0:
            raise InvalidFeatureArtifactsError(
                f"Feature matrix must be 2-D with non-empty rows/columns; "
                f"got shape {csr.shape}."
            )
        if not np.isfinite(csr.data).all():
            raise InvalidFeatureArtifactsError(
                "Feature matrix contains NaN or infinite values; regenerate the "
                "Module 2 artifacts."
            )
        return csr

    @staticmethod
    def _validate_index(feature_index: pd.DataFrame) -> None:
        """Ensure the movieId -> row_index map is unique and contiguous."""
        if not isinstance(feature_index, pd.DataFrame):
            raise InvalidFeatureArtifactsError(
                f"feature_index must be a pandas DataFrame; got "
                f"{type(feature_index).__name__}."
            )
        missing = [c for c in ("movieId", "row_index") if c not in feature_index.columns]
        if missing:
            raise InvalidFeatureArtifactsError(
                f"feature_index is missing required column(s): {missing}."
            )
        if len(feature_index) == 0:
            raise InvalidFeatureArtifactsError("feature_index is empty.")
        if feature_index["movieId"].isna().any() or not feature_index["movieId"].is_unique:
            raise InvalidFeatureArtifactsError(
                "feature_index movieId values must be non-null and unique."
            )
        rows = feature_index["row_index"].to_numpy()
        n_rows = len(rows)
        if not (
            rows.min() == 0
            and rows.max() == n_rows - 1
            and np.array_equal(np.sort(rows), np.arange(n_rows))
        ):
            raise InvalidFeatureArtifactsError(
                "feature_index row_index values must be a contiguous 0..N-1 range "
                "aligned with the matrix rows."
            )

    def _compute_inverse_row_norms(self) -> np.ndarray:
        """Return ``1 / ||row||`` per matrix row (zeros for empty rows).

        When every non-empty row is already unit-norm (the Module 2 default),
        the original matrix is reused; otherwise a normalised copy replaces it
        so cosine results stay mathematically exact regardless of input scale.
        """
        squared = self._matrix.multiply(self._matrix).sum(axis=1)
        norms = np.sqrt(np.asarray(squared).ravel())
        if norms.size != self.n_movies:  # defensive; scipy guarantees size
            raise InvalidFeatureArtifactsError("Could not compute row norms.")
        non_empty = norms > 0
        deviation = float(np.abs(norms[non_empty] - 1.0).max()) if non_empty.any() else 0.0
        inverse = np.zeros(self.n_movies, dtype=np.float64)
        inverse[non_empty] = 1.0 / norms[non_empty]
        if deviation > _ROW_NORM_TOLERANCE:
            logger.info(
                "Feature matrix rows are not L2-normalised (max |norm-1|=%.4f); "
                "normalising a working copy for exact cosine.",
                deviation,
            )
            scaled = sparse.diags(inverse.astype(np.float32)) @ self._matrix
            self._matrix = sparse.csr_matrix(scaled)
            # Rows are unit-norm now; keep zeros only for empty rows so query
            # scoring must not rescale again (double normalisation otherwise).
            inverse = np.where(non_empty, 1.0, 0.0)
        else:
            logger.info(
                "Feature matrix rows are L2-normalised (max |norm-1|=%.2e); "
                "using them directly.",
                deviation,
            )
        return inverse

    # ------------------------------------------------------------------ #
    # Introspection                                                       #
    # ------------------------------------------------------------------ #
    @property
    def n_movies(self) -> int:
        """Number of movies (matrix rows) known to the engine."""
        return int(self._matrix.shape[0])

    @property
    def n_features(self) -> int:
        """Number of feature columns in the matrix."""
        return int(self._matrix.shape[1])

    @property
    def movie_ids(self) -> np.ndarray:
        """All known movieIds as an ``int64`` array (ascending by row)."""
        return self._movie_ids.copy()

    def __len__(self) -> int:
        return self.n_movies

    def has_movie(self, movie_id: object) -> bool:
        """Return True when ``movie_id`` exists in the feature index."""
        try:
            key = int(movie_id)
        except (TypeError, ValueError):
            return False
        return key in self._row_of_movie

    def row_for_movie(self, movie_id: object) -> int:
        """Return the matrix row for ``movie_id``.

        Raises
        ------
        MovieNotFoundError
            If the movieId is unknown to the feature index.
        """
        try:
            key = int(movie_id)
        except (TypeError, ValueError) as exc:
            raise MovieNotFoundError(
                f"Invalid movie identifier {movie_id!r}: expected an integer movieId."
            ) from exc
        try:
            return self._row_of_movie[key]
        except KeyError as exc:
            raise MovieNotFoundError(
                f"Unknown movieId {key}: not present in the Module 2 feature "
                "index. Verify the ID against data/processed/movie_feature_index.csv."
            ) from exc

    def vector_for_movie(self, movie_id: object) -> sparse.csr_matrix:
        """Return the 1 x V sparse feature row for ``movie_id``."""
        return self._matrix[self.row_for_movie(movie_id)]

    # ------------------------------------------------------------------ #
    # Similarity queries                                                  #
    # ------------------------------------------------------------------ #
    def similar_to_movie(
        self,
        movie_id: object,
        top_k: int | None = None,
        exclude_ids: object | None = None,
        allowed_ids: np.ndarray | pd.Index | None = None,
    ) -> list[dict]:
        """Return up to ``top_k`` movies most cosine-similar to ``movie_id``.

        The seed movie itself is always excluded. Results are sorted by
        descending similarity with ascending movieId as deterministic
        tie-breaker.

        Parameters
        ----------
        movie_id:
            Seed movie present in the feature index.
        top_k:
            Maximum number of neighbours to return (>= 1). ``None`` uses
            ``config.DEFAULT_TOP_K`` at call time.
        exclude_ids:
            Optional iterable of additional movieIds to exclude (already-seen
            items, filtered-out seeds, ...).
        allowed_ids:
            Optional allow-list of candidate movieIds (e.g. popularity-filtered
            catalog). When given, only these movies can be returned; the seed
            and ``exclude_ids`` still apply on top.

        Returns
        -------
        list[dict]
            Records with keys ``movieId``, ``row_index`` and ``similarity``,
            best first. Empty when the seed has an all-zero content vector
            (no meaningful neighbours exist) or no candidate survives
            filtering.

        Raises
        ------
        MovieNotFoundError
            If the seed movieId is unknown.
        InvalidTopKError
            If ``top_k`` is invalid.
        """
        top_k = validate_top_k(config.DEFAULT_TOP_K if top_k is None else top_k)
        seed_row = self.row_for_movie(movie_id)
        excluded = self._resolve_exclusions(
            exclude_ids, extra={int(self._movie_ids[seed_row])}
        )
        allowed_mask = self._resolve_allowed_mask(allowed_ids)
        return self._rank_query(self._matrix[seed_row], top_k, excluded, allowed_mask)

    def similar_to_profile(
        self,
        profile_vector: sparse.spmatrix | None,
        top_k: int | None = None,
        exclude_ids: object | None = None,
        allowed_ids: np.ndarray | pd.Index | None = None,
    ) -> list[dict]:
        """Rank movies against a 1 x V content-profile vector.

        Used by offline evaluation to score user profiles built from rating
        history. Same contract as :meth:`similar_to_movie`; a ``None`` or
        empty profile yields an empty result.
        """
        top_k = validate_top_k(config.DEFAULT_TOP_K if top_k is None else top_k)
        if profile_vector is None:
            return []
        query = sparse.csr_matrix(profile_vector, dtype=np.float64)
        if query.shape != (1, self.n_features):
            raise InvalidFeatureArtifactsError(
                f"Profile vector must have shape (1, {self.n_features}); "
                f"got {query.shape}."
            )
        excluded = self._resolve_exclusions(exclude_ids, extra=set())
        allowed_mask = self._resolve_allowed_mask(allowed_ids)
        return self._rank_query(query, top_k, excluded, allowed_mask)

    # ------------------------------------------------------------------ #
    # Internals                                                           #
    # ------------------------------------------------------------------ #
    def _resolve_exclusions(
        self, exclude_ids: object | None, extra: set[int]
    ) -> set[int]:
        """Merge caller-supplied exclusions into one set of python ints."""
        merged = set(extra)
        if exclude_ids is not None:
            for value in exclude_ids:
                try:
                    merged.add(int(value))
                except (TypeError, ValueError):
                    continue  # ignore malformed ids rather than failing a request
        return merged

    def _resolve_allowed_mask(
        self, allowed_ids: np.ndarray | pd.Index | None
    ) -> np.ndarray | None:
        """Convert an allow-list of movieIds into a boolean mask over rows."""
        if allowed_ids is None:
            return None
        mask = np.zeros(self.n_movies, dtype=bool)
        known = pd.Index(self._movie_ids).get_indexer(pd.Index(allowed_ids))
        valid = known[known >= 0]
        if valid.size:
            mask[valid] = True
        else:
            logger.warning("Allow-list contained no movies known to the engine.")
        return mask

    def _rank_query(
        self,
        query: sparse.spmatrix,
        top_k: int,
        excluded: set[int],
        allowed_mask: np.ndarray | None,
    ) -> list[dict]:
        """Score one query against every movie and return ranked records."""
        query = sparse.csr_matrix(query, dtype=np.float64)
        if query.nnz == 0:
            logger.warning(
                "Query vector is empty (all-zero); returning no recommendations."
            )
            return []
        norm = float(np.sqrt(query.multiply(query).sum()))
        if norm <= 0.0:
            logger.warning(
                "Query vector has zero norm; returning no recommendations."
            )
            return []
        query = query / norm

        scores = linear_kernel(query, self._matrix)[0].astype(np.float64)
        scores *= self._inverse_norms  # exact cosine even for un-normalised rows

        candidate_rows = np.arange(self.n_movies)
        if allowed_mask is not None:
            candidate_rows = candidate_rows[allowed_mask]

        candidate_ids = self._movie_ids[candidate_rows]
        keep = np.fromiter(
            (int(m) not in excluded for m in candidate_ids),
            dtype=bool,
            count=candidate_rows.size,
        )
        candidate_rows = candidate_rows[keep]
        candidate_ids = candidate_ids[keep]
        candidate_scores = scores[candidate_rows]
        if candidate_rows.size == 0:
            logger.warning("No recommendation candidates survived filtering.")
            return []

        # Primary key: descending similarity; secondary: ascending movieId.
        order = np.lexsort((candidate_ids, -candidate_scores))[:top_k]
        return [
            {
                "movieId": int(candidate_ids[pos]),
                "row_index": int(candidate_rows[pos]),
                "similarity": float(candidate_scores[pos]),
            }
            for pos in order
        ]

