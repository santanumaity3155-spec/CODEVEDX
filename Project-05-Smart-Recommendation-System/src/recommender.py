"""Content-based recommendation API (Module 3).

Wraps :class:`~src.similarity_engine.SimilarityEngine` in the public
recommendation workflow used across the project:

    seed movieId  ->  candidate filtering (metadata validity, popularity,
                      optional genre/year filters)  ->  cosine similarity
                      ranking  ->  enriched, ranked DataFrame

Recommendations always contain ``rank``, ``movieId``, ``title``, ``genres``
and ``similarity``. The seed movie is never returned, results are sorted by
descending similarity with ascending ``movieId`` as deterministic tie-breaker,
and unknown seeds degrade gracefully to an empty result instead of raising.

Popularity / quality control: movies whose rating count (from the cleaned
Module 1 ratings dataset) falls below a configurable threshold are excluded
from candidates so obscure titles cannot dominate rankings. Set
``min_movie_ratings`` to ``0``/``None`` to disable.

Data-loading concerns stay out of the ranking logic: the class consumes
already-loaded artifacts; :meth:`ContentRecommender.from_artifacts` is the
convenience loader used by pipelines and scripts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.logging_config import get_logger
from src.similarity_engine import (
    MovieNotFoundError,
    SimilarityEngine,
    validate_top_k,
)

logger = get_logger(__name__)

#: Columns guaranteed on every recommendation result.
RECOMMENDATION_COLUMNS = ("rank", "movieId", "title", "genres", "similarity")

_SUPPORTED_FILTERS = ("genres", "min_year", "max_year", "min_ratings")


class RecommenderError(Exception):
    """Base class for recommender failures."""


class InvalidFilterError(RecommenderError):
    """Raised when an unsupported or malformed ``filters`` entry is passed."""


def empty_recommendations() -> pd.DataFrame:
    """Return an empty recommendation frame with the canonical columns."""
    return pd.DataFrame(
        {
            "rank": pd.Series(dtype="int32"),
            "movieId": pd.Series(dtype="int64"),
            "title": pd.Series(dtype="string"),
            "genres": pd.Series(dtype="string"),
            "similarity": pd.Series(dtype="float64"),
        }
    )


def compute_popularity_stats(ratings: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-movie rating counts and means from a ratings frame.

    Parameters
    ----------
    ratings:
        Frame with ``movieId`` and ``rating`` columns (Module 1 cleaned data).

    Returns
    -------
    pd.DataFrame
        One row per movieId with ``rating_count`` (int64) and
        ``mean_rating`` (float64), sorted ascending by movieId.
    """
    missing = [c for c in ("movieId", "rating") if c not in ratings.columns]
    if missing:
        raise RecommenderError(
            f"ratings frame is missing required column(s): {missing}."
        )
    stats = ratings.groupby("movieId")["rating"].agg(["count", "mean"])
    stats = stats.rename(columns={"count": "rating_count", "mean": "mean_rating"})
    stats.index = stats.index.astype("int64")
    stats["rating_count"] = stats["rating_count"].astype("int64")
    # Normalise to float64 regardless of the input rating dtype so chunked
    # and in-memory aggregations always produce identical frames.
    stats["mean_rating"] = stats["mean_rating"].astype("float64")
    return stats.reset_index().sort_values("movieId").reset_index(drop=True)


def load_popularity_stats(
    ratings_path: str | Path | None = None,
    *,
    chunksize: int = config.POPULARITY_CHUNKSIZE,
) -> pd.DataFrame:
    """Stream ``ratings_clean.csv`` and aggregate per-movie popularity.

    Reads in chunks to keep memory bounded on the ~678 MiB cleaned ratings
    dataset. See :func:`compute_popularity_stats` for the output shape.
    """
    path = Path(config.PROCESSED_RATINGS_PATH if ratings_path is None else ratings_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Processed ratings file not found: '{path}'. Run the Module 1 "
            "pipeline (python -m src.pipeline) first."
        )
    logger.info("Aggregating popularity statistics from %s ...", path.name)
    total = 0
    counts: dict[int, int] = {}
    sums: dict[int, float] = {}
    reader = pd.read_csv(
        path,
        dtype=config.PROCESSED_RATINGS_DTYPE,
        usecols=list(config.PROCESSED_RATINGS_DTYPE),
        chunksize=chunksize,
    )
    for chunk in reader:
        grouped = chunk.groupby("movieId")["rating"].agg(["count", "sum"])
        for movie_id, row in grouped.iterrows():
            key = int(movie_id)
            counts[key] = counts.get(key, 0) + int(row["count"])
            sums[key] = sums.get(key, 0.0) + float(row["sum"])
        total += len(chunk)
    if not counts:
        raise RecommenderError(f"Ratings file '{path}' contained no aggregatable rows.")
    stats = pd.DataFrame(
        {
            "movieId": pd.array(list(counts.keys()), dtype="int64"),
            "rating_count": pd.array(list(counts.values()), dtype="int64"),
        }
    )
    stats["mean_rating"] = [
        sums[mid] / cnt for mid, cnt in zip(stats["movieId"], stats["rating_count"])
    ]
    stats = stats.sort_values("movieId").reset_index(drop=True)
    logger.info(
        "Popularity statistics ready: %d movies from %d ratings",
        len(stats),
        total,
    )
    return stats


class ContentRecommender:
    """Content-based recommender over the Module 2 TF-IDF feature space.

    Parameters
    ----------
    engine:
        A validated :class:`SimilarityEngine` built from the Module 2
        artifacts.
    catalog:
        Movie metadata frame; requires ``movieId`` and ``title`` columns,
        typically also ``genres`` (and optionally ``release_year``) from
        ``movie_content_features.csv``.
    popularity:
        Optional per-movie frame with ``movieId``/``rating_count`` (output of
        :func:`compute_popularity_stats` / :func:`load_popularity_stats`).
    min_movie_ratings:
        Minimum rating count for a movie to be recommendable. ``None`` falls
        back to ``config.RECOMMENDATION_MIN_MOVIE_RATINGS``; ``0`` disables
        popularity filtering.
    """

    def __init__(
        self,
        engine: SimilarityEngine,
        catalog: pd.DataFrame,
        popularity: pd.DataFrame | None = None,
        min_movie_ratings: int | None = None,
    ) -> None:
        if not isinstance(catalog, pd.DataFrame):
            raise RecommenderError(
                f"catalog must be a pandas DataFrame; got {type(catalog).__name__}."
            )
        missing = [c for c in ("movieId", "title") if c not in catalog.columns]
        if missing:
            raise RecommenderError(
                f"catalog is missing required column(s): {missing}."
            )
        if catalog["movieId"].duplicated().any():
            raise RecommenderError("catalog movieId values must be unique.")

        self.engine = engine
        self.catalog = catalog.copy()
        self.catalog["movieId"] = self.catalog["movieId"].astype("int64")

        threshold = (
            config.RECOMMENDATION_MIN_MOVIE_RATINGS
            if min_movie_ratings is None
            else min_movie_ratings
        )
        self.min_movie_ratings: int = max(int(threshold or 0), 0)
        self.popularity = popularity

        # Metadata validity mask: rows lacking title/genres can never be
        # recommended regardless of filters.
        valid = (
            self.catalog["title"].notna()
            & self.catalog["title"].astype("string").str.strip().ne("")
        )
        if "genres" in self.catalog.columns:
            valid &= self.catalog["genres"].notna()
        self._valid_metadata: np.ndarray = valid.to_numpy()

        rating_count_by_movie: np.ndarray | None = None
        if popularity is not None and self.min_movie_ratings > 0:
            pop_missing = [
                c for c in ("movieId", "rating_count") if c not in popularity.columns
            ]
            if pop_missing:
                raise RecommenderError(
                    f"popularity frame is missing required column(s): {pop_missing}."
                )
            counts = popularity.set_index(popularity["movieId"].astype("int64"))[
                "rating_count"
            ]
            rating_count_by_movie = (
                pd.Series(self.catalog["movieId"])
                .map(counts)
                .fillna(0)
                .to_numpy(dtype="int64")
            )
        self._rating_counts = rating_count_by_movie

        logger.info(
            "ContentRecommender ready: %d catalog movies, %d with valid "
            "metadata, min_movie_ratings=%s",
            len(self.catalog),
            int(self._valid_metadata.sum()),
            self.min_movie_ratings,
        )

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #
    def recommend_similar_movies(
        self,
        movie_id: object,
        top_k: int | None = None,
        filters: dict | None = None,
    ) -> pd.DataFrame:
        """Return up to ``top_k`` movies similar to ``movie_id``.

        The result never contains the seed, duplicates, or movies failing the
        configured quality/filters checks, and is sorted by descending
        similarity (ties broken by ascending movieId). Unknown seeds yield an
        empty frame rather than raising.

        Parameters
        ----------
        movie_id:
            Seed movie identifier.
        top_k:
            Maximum number of recommendations (default:
            ``config.DEFAULT_TOP_K``).
        filters:
            Optional refinement dictionary:

            * ``"genres"``: list of genre names - candidates must carry at
              least one of them (case-insensitive);
            * ``"min_year"`` / ``"max_year"``: release-year bounds applied
              when the catalog has a ``release_year`` column;
            * ``"min_ratings"``: per-call override of the popularity
              threshold.

            Unknown keys raise :class:`InvalidFilterError`.

        Returns
        -------
        pd.DataFrame
            Columns ``rank, movieId, title, genres, similarity``.
        """
        top_k = validate_top_k(config.DEFAULT_TOP_K if top_k is None else top_k)
        self._validate_filters(filters)

        if not self.engine.has_movie(movie_id):
            logger.warning(
                "Recommendation request for unknown movieId %r returned no results.",
                movie_id,
            )
            return empty_recommendations()

        try:
            neighbours = self.engine.similar_to_movie(
                movie_id,
                top_k=top_k,
                allowed_ids=self._candidate_movie_ids(filters),
            )
        except MovieNotFoundError:
            logger.warning(
                "Seed movieId %r disappeared from the feature index; returning "
                "no recommendations.",
                movie_id,
            )
            return empty_recommendations()

        return self._to_frame(neighbours)

    def describe_seed(self, movie_id: object) -> dict | None:
        """Return seed metadata (movieId/title/genres/...) or ``None``."""
        try:
            key = int(movie_id)
        except (TypeError, ValueError):
            return None
        matches = self.catalog.loc[self.catalog["movieId"] == key]
        if matches.empty:
            return None
        row = matches.iloc[0]
        info: dict = {"movieId": int(row["movieId"]), "title": str(row["title"])}
        if "genres" in self.catalog.columns:
            info["genres"] = str(row["genres"])
        if "release_year" in self.catalog.columns and pd.notna(
            row.get("release_year")
        ):
            info["release_year"] = int(row["release_year"])
        return info

    # ------------------------------------------------------------------ #
    # Candidate filtering                                                 #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate_filters(filters: dict | None) -> None:
        """Raise :class:`InvalidFilterError` on unsupported filter usage."""
        if filters is None:
            return
        if not isinstance(filters, dict):
            raise InvalidFilterError(
                f"filters must be a dict; got {type(filters).__name__}."
            )
        unknown = [key for key in filters if key not in _SUPPORTED_FILTERS]
        if unknown:
            raise InvalidFilterError(
                f"Unsupported filter key(s): {unknown}. "
                f"Supported filters: {list(_SUPPORTED_FILTERS)}."
            )
        genres = filters.get("genres")
        if genres is not None and (
            not isinstance(genres, (list, tuple, set))
            or not all(isinstance(g, str) for g in genres)
        ):
            raise InvalidFilterError(
                "filters['genres'] must be a list of genre name strings."
            )
        for key in ("min_year", "max_year"):
            value = filters.get(key)
            if value is not None and not isinstance(value, (int, float)):
                raise InvalidFilterError(f"filters[{key!r}] must be a number.")
        min_ratings = filters.get("min_ratings")
        if min_ratings is not None and (
            isinstance(min_ratings, bool)
            or not isinstance(min_ratings, int)
            or min_ratings < 0
        ):
            raise InvalidFilterError(
                "filters['min_ratings'] must be a non-negative integer."
            )

    def _candidate_movie_ids(self, filters: dict | None) -> np.ndarray:
        """MovieIds of recommendable candidates after all quality filters."""
        mask = self._valid_metadata.copy()

        threshold = self.min_movie_ratings
        if filters is not None and filters.get("min_ratings") is not None:
            threshold = int(filters["min_ratings"])
        if threshold > 0:
            if self._rating_counts is None:
                logger.warning(
                    "min_ratings=%d requested but no popularity statistics are "
                    "loaded; skipping the popularity filter.",
                    threshold,
                )
            else:
                mask &= self._rating_counts >= threshold

        if filters is None:
            ids = self.catalog["movieId"].to_numpy()[mask]
            logger.info("Candidate pool: %d movies", ids.size)
            return ids

        genres = filters.get("genres")
        if genres and "genres" in self.catalog.columns:
            wanted = {g.strip().lower() for g in genres}
            catalog_genres = self.catalog["genres"].astype("string").fillna("")
            has_genre = catalog_genres.str.split("|").map(
                lambda parts: bool(wanted & {p.strip().lower() for p in parts})
            )
            mask &= has_genre.to_numpy()

        wants_year = filters.get("min_year") is not None or (
            filters.get("max_year") is not None
        )
        if wants_year and "release_year" in self.catalog.columns:
            years = pd.to_numeric(self.catalog["release_year"], errors="coerce")
            min_year = filters.get("min_year")
            max_year = filters.get("max_year")
            if min_year is not None:
                mask &= (years >= float(min_year)).to_numpy()
            if max_year is not None:
                mask &= (years <= float(max_year)).to_numpy()
        elif wants_year:
            logger.warning(
                "Year filter requested but catalog has no release_year column; "
                "skipping the year filter."
            )

        ids = self.catalog["movieId"].to_numpy()[mask]
        logger.info("Candidate pool after filtering: %d movies", ids.size)
        return ids

    def _to_frame(self, neighbours: list[dict]) -> pd.DataFrame:
        """Join engine neighbours with catalog metadata and rank them."""
        if not neighbours:
            return empty_recommendations()
        result = pd.DataFrame(neighbours)
        available = [c for c in ("title", "genres") if c in self.catalog.columns]
        metadata = self.catalog.set_index("movieId")[available]
        joined = result.join(metadata, on="movieId")
        if "genres" not in joined.columns:
            joined["genres"] = ""
        joined = joined.dropna(subset=["title"])
        joined.insert(0, "rank", np.arange(1, len(joined) + 1, dtype="int32"))
        return (
            joined.astype({"rank": "int32", "movieId": "int64"})
            .loc[:, list(RECOMMENDATION_COLUMNS)]
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------ #
    # Artifact loading                                                    #
    # ------------------------------------------------------------------ #
    @classmethod
    def from_artifacts(
        cls,
        *,
        content_features_path: str | Path | None = None,
        tfidf_path: str | Path | None = None,
        index_path: str | Path | None = None,
        vectorizer_path: str | Path | None = None,
        ratings_path: str | Path | None = None,
        min_movie_ratings: int | None = None,
        load_popularity: bool = True,
    ) -> "ContentRecommender":
        """Build a recommender from the Module 2 artifacts on disk.

        All paths default to the ``src.config`` locations. The fitted Module 2
        vectorizer is loaded only to verify compatibility with the matrix - it
        is never re-fit.
        """
        from src.data_loader import load_csv
        from src.feature_engineering import (  # local import avoids cycles
            load_feature_vectorizer,
            load_sparse_matrix,
        )

        content_path = (
            content_features_path or config.PROCESSED_MOVIE_CONTENT_FEATURES_PATH
        )
        matrix_path = tfidf_path or config.PROCESSED_MOVIE_TFIDF_PATH
        idx_path = index_path or config.PROCESSED_MOVIE_FEATURE_INDEX_PATH
        vec_path = vectorizer_path or config.MOVIE_TFIDF_VECTORIZER_PATH

        logger.info("Loading recommendation artifacts ...")
        for path in (content_path, matrix_path, idx_path, vec_path):
            if not Path(path).is_file():
                raise FileNotFoundError(
                    f"Required artifact not found: '{path}'. Run Modules 1-2 "
                    "(python -m src.pipeline && python -m src.feature_engineering) "
                    "first."
                )

        catalog = load_csv(
            content_path,
            required_columns=("movieId", "title"),
            label="movie_content_features",
        )
        matrix = load_sparse_matrix(matrix_path)
        feature_index = load_csv(
            idx_path,
            required_columns=("movieId", "row_index"),
            label="movie_feature_index",
        )
        vectorizer = load_feature_vectorizer(vec_path)
        _verify_vectorizer_compatibility(vectorizer, matrix, catalog)

        engine = SimilarityEngine(matrix, feature_index)
        popularity = None
        if load_popularity:
            try:
                popularity = load_popularity_stats(ratings_path)
            except FileNotFoundError as exc:
                logger.warning("%s", exc)
        metadata_columns = [
            c for c in ("movieId", "title", "genres", "release_year")
            if c in catalog.columns
        ]
        return cls(
            engine=engine,
            catalog=catalog[metadata_columns],
            popularity=popularity,
            min_movie_ratings=min_movie_ratings,
        )


def _verify_vectorizer_compatibility(
    vectorizer, matrix, catalog: pd.DataFrame
) -> None:
    """Raise :class:`RecommenderError` when the vectorizer/matrix disagree.

    Performs a cheap vocabulary-size check first, then - when the stored
    ``content_text`` documents are available - the exact reproduction check
    from Module 2 (:func:`verify_loaded_vectorizer`). Never re-fits anything.
    """
    vocab_size = len(getattr(vectorizer, "vocabulary_", {}))
    if vocab_size != matrix.shape[1]:
        raise RecommenderError(
            f"Vectorizer vocabulary ({vocab_size} terms) does not match the "
            f"feature matrix width ({matrix.shape[1]}); artifacts are "
            "incompatible. Re-run the Module 2 pipeline."
        )
    if "content_text" not in catalog.columns:
        return
    from src.feature_engineering import verify_loaded_vectorizer

    documents = catalog["content_text"].astype(str).tolist()
    if not verify_loaded_vectorizer(vectorizer, documents, matrix):
        raise RecommenderError(
            "The loaded TF-IDF vectorizer does not reproduce the stored "
            "feature matrix; regenerate the Module 2 artifacts."
        )
    logger.info(
        "Vectorizer/matrix compatibility verified (%d x %d).",
        matrix.shape[0],
        matrix.shape[1],
    )


def find_movie_by_title(
    catalog: pd.DataFrame,
    query: str,
    popularity: pd.DataFrame | None = None,
) -> dict | None:
    """Search the real catalog for a movie by (partial) title.

    Case-insensitive substring match over the ``title`` column. When several
    titles match and ``popularity`` statistics are supplied, the most-rated
    match wins; otherwise the lowest movieId wins. Returns
    ``{"movieId", "title", "genres"}`` or ``None`` when nothing matches.
    """
    if "title" not in catalog.columns:
        raise RecommenderError("catalog must contain a 'title' column to search.")
    titles = catalog["title"].astype("string").fillna("")
    hits = catalog.loc[
        titles.str.contains(str(query), case=False, regex=False).to_numpy()
    ]
    if hits.empty:
        return None
    if popularity is not None and {"movieId", "rating_count"}.issubset(popularity.columns):
        counts = popularity.set_index(popularity["movieId"].astype("int64"))[
            "rating_count"
        ]
        order = pd.Series(hits["movieId"].astype("int64")).map(counts).fillna(0)
        best_pos = int(order.to_numpy().argmax())
        row = hits.iloc[best_pos]
    else:
        row = hits.sort_values("movieId").iloc[0]
    return {
        "movieId": int(row["movieId"]),
        "title": str(row["title"]),
        "genres": str(row.get("genres", "")),
    }
