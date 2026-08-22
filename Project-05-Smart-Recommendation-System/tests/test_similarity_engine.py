"""Tests for Module 3 similarity engine (``src/similarity_engine.py``).

Covers cosine-similarity correctness against hand-computed values, top-K
behaviour, self-exclusion, deterministic ordering and tie-breaking, invalid
movie IDs / top-K values, sparse-matrix enforcement, movieId->row mapping,
empty feature vectors, exclusion/allow-lists, profile queries and
un-normalised matrix handling.

Engine-only tests use tiny handcrafted CSR matrices; artifact-backed tests
run inside the ``module3_workspace`` sandbox (see conftest) and never touch
``data/raw`` or the real processed files.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from src.similarity_engine import (
    InvalidFeatureArtifactsError,
    InvalidTopKError,
    MovieNotFoundError,
    SimilarityEngine,
    validate_top_k,
)


def _make_index(movie_ids: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "movieId": pd.array(movie_ids, dtype="int64"),
            "row_index": pd.array(range(len(movie_ids)), dtype="int32"),
        }
    )


@pytest.fixture
def tiny_engine() -> SimilarityEngine:
    """4 movies x 3 features with known geometry.

    row0 = e0, row1 = e1, row2 = (e0 + e1) / sqrt(2), row3 = zero vector.
    """
    data = [1.0, 1.0, np.sqrt(0.5), np.sqrt(0.5)]
    rows = [0, 1, 2, 2]
    cols = [0, 1, 0, 1]
    matrix = sparse.csr_matrix((data, (rows, cols)), shape=(4, 3))
    return SimilarityEngine(matrix, _make_index([10, 20, 30, 40]))


class TestCosineCorrectness:
    def test_known_cosine_value(self, tiny_engine):
        # cos(e0, (e0+e1)/sqrt(2)) = 1/sqrt(2)
        result = tiny_engine.similar_to_movie(10, top_k=3)
        assert result[0]["movieId"] == 30
        assert result[0]["similarity"] == pytest.approx(1 / np.sqrt(2), abs=1e-9)

    def test_orthogonal_vectors_zero_similarity(self, tiny_engine):
        result = tiny_engine.similar_to_movie(10, top_k=3)
        sim_to_20 = next(r for r in result if r["movieId"] == 20)
        assert sim_to_20["similarity"] == pytest.approx(0.0, abs=1e-12)

    def test_matches_sklearn_cosine_similarity(self):
        from sklearn.metrics.pairwise import cosine_similarity

        rng = np.random.default_rng(7)
        dense = rng.random((6, 5)).astype(np.float32)
        dense[3] = 0.0  # include an empty row
        matrix = sparse.csr_matrix(dense)
        engine = SimilarityEngine(matrix, _make_index(list(range(100, 106))))
        for seed_row in (0, 2, 3):
            expected = cosine_similarity(dense[seed_row : seed_row + 1], dense)[0]
            got = engine.similar_to_movie(100 + seed_row, top_k=5)
            for record in got:
                row = record["row_index"]
                if row == seed_row:
                    continue
                assert record["similarity"] == pytest.approx(
                    float(expected[row]), abs=1e-5
                )

    def test_unnormalised_rows_still_exact_cosine(self):
        matrix = sparse.csr_matrix(
            (np.array([3.0, 5.0, 1.0]), ([0, 1, 2], [0, 0, 1])), shape=(3, 2)
        )
        engine = SimilarityEngine(matrix, _make_index([1, 2, 3]))
        top = engine.similar_to_movie(1, top_k=2)[0]
        # cos([3], [5]) = 1.0 (parallel), cos([3], [0,1]) = 0.0
        assert top["movieId"] == 2
        assert top["similarity"] == pytest.approx(1.0, abs=1e-6)

    def test_similarity_scores_bounded(self, tiny_engine):
        for record in tiny_engine.similar_to_movie(20, top_k=3):
            assert -1e-9 <= record["similarity"] <= 1.0 + 1e-9


class TestTopKBehaviour:
    def test_returns_at_most_top_k(self, tiny_engine):
        assert len(tiny_engine.similar_to_movie(10, top_k=2)) == 2
        assert len(tiny_engine.similar_to_movie(10, top_k=1)) == 1

    def test_caps_at_candidate_pool(self, tiny_engine):
        # 4 movies, 1 seed excluded -> at most 3 neighbours.
        assert len(tiny_engine.similar_to_movie(10, top_k=99)) == 3

    def test_default_top_k_from_config(self, tiny_engine, monkeypatch):
        from src import config

        monkeypatch.setattr(config, "DEFAULT_TOP_K", 2)
        assert len(tiny_engine.similar_to_movie(10)) == 2


class TestSelfExclusion:
    def test_seed_never_returned(self, tiny_engine):
        for seed in (10, 20, 30):
            result = tiny_engine.similar_to_movie(seed, top_k=5)
            assert all(r["movieId"] != seed for r in result)

    def test_self_similarity_would_be_highest_but_is_absent(self, tiny_engine):
        result = tiny_engine.similar_to_movie(30, top_k=5)
        # Movies 10 and 20 tie at cos = 1/sqrt(2); movie 40 (zero vector)
        # trails at 0.0. The seed itself (30) is never present.
        assert [r["movieId"] for r in result] == [10, 20, 40]
        assert result[0]["similarity"] == pytest.approx(1 / np.sqrt(2), abs=1e-9)


class TestDeterministicOrdering:
    def test_repeated_calls_identical(self, tiny_engine):
        first = tiny_engine.similar_to_movie(10, top_k=5)
        for _ in range(5):
            assert tiny_engine.similar_to_movie(10, top_k=5) == first

    def test_ties_broken_by_ascending_movieid(self, tiny_engine):
        # Seed 30: movies 20 and 40 both have similarity 0.0 -> 20 before 40.
        result = tiny_engine.similar_to_movie(30, top_k=3)
        assert [r["movieId"] for r in result][:2] == [10, 20]

    def test_engine_order_independent_of_index_order(self):
        matrix = sparse.csr_matrix(
            (np.array([1.0, 1.0]), ([0, 1], [0, 0])), shape=(2, 1)
        )
        shuffled = pd.DataFrame(
            {
                "movieId": pd.array([77, 11], dtype="int64"),
                "row_index": pd.array([0, 1], dtype="int32"),
            }
        )
        engine = SimilarityEngine(matrix, shuffled)
        result = engine.similar_to_movie(77, top_k=1)
        assert result[0]["movieId"] == 11


class TestInvalidInputs:
    def test_unknown_movie_raises(self, tiny_engine):
        with pytest.raises(MovieNotFoundError):
            tiny_engine.similar_to_movie(999)

    def test_non_integer_movie_raises(self, tiny_engine):
        with pytest.raises(MovieNotFoundError):
            tiny_engine.similar_to_movie("not-a-movie")

    @pytest.mark.parametrize("bad_top_k", [0, -1, 2.5, "3", True])
    def test_invalid_top_k_raises(self, tiny_engine, bad_top_k):
        with pytest.raises(InvalidTopKError):
            tiny_engine.similar_to_movie(10, top_k=bad_top_k)

    def test_none_top_k_uses_configured_default(self, tiny_engine, monkeypatch):
        from src import config

        monkeypatch.setattr(config, "DEFAULT_TOP_K", 1)
        assert len(tiny_engine.similar_to_movie(10)) == 1

    def test_invalid_top_k_is_value_error(self):
        # InvalidTopKError doubles as ValueError for generic handlers.
        with pytest.raises(ValueError):
            validate_top_k(-3)

    def test_validate_top_k_accepts_numpy_int(self):
        assert validate_top_k(np.int64(4)) == 4


class TestSparseAndMapping:
    def test_dense_matrix_rejected(self):
        with pytest.raises(InvalidFeatureArtifactsError):
            SimilarityEngine(np.zeros((3, 2)), _make_index([1, 2, 3]))

    def test_nan_matrix_rejected(self):
        matrix = sparse.csr_matrix(
            (np.array([1.0, np.nan]), ([0, 1], [0, 0])), shape=(2, 1)
        )
        with pytest.raises(InvalidFeatureArtifactsError):
            SimilarityEngine(matrix, _make_index([1, 2]))

    def test_non_contiguous_index_rejected(self):
        matrix = sparse.csr_matrix(np.eye(3))
        bad_index = pd.DataFrame(
            {
                "movieId": pd.array([1, 2, 3], dtype="int64"),
                "row_index": pd.array([0, 1, 5], dtype="int32"),
            }
        )
        with pytest.raises(InvalidFeatureArtifactsError):
            SimilarityEngine(matrix, bad_index)

    def test_duplicate_movie_ids_rejected(self):
        matrix = sparse.csr_matrix(np.eye(2))
        bad_index = pd.DataFrame(
            {
                "movieId": pd.array([1, 1], dtype="int64"),
                "row_index": pd.array([0, 1], dtype="int32"),
            }
        )
        with pytest.raises(InvalidFeatureArtifactsError):
            SimilarityEngine(matrix, bad_index)

    def test_movie_id_row_mapping_preserved(self, tiny_engine):
        # movieId 20 must map to matrix row 1 (the e1 direction).
        vector = tiny_engine.vector_for_movie(20)
        assert vector.nnz == 1
        assert vector.indices[0] == 1
        assert tiny_engine.row_for_movie(30) == 2
        assert tiny_engine.has_movie(10) and not tiny_engine.has_movie(999)
        assert tiny_engine.n_movies == 4 and tiny_engine.n_features == 3
        assert list(tiny_engine.movie_ids) == [10, 20, 30, 40]
        assert len(tiny_engine) == 4


class TestEmptyVectors:
    def test_empty_seed_returns_no_recommendations(self, tiny_engine):
        # movieId 40 has an all-zero feature row.
        assert tiny_engine.similar_to_movie(40, top_k=3) == []

    def test_empty_row_never_divides_by_zero(self, tiny_engine):
        # Empty row as *candidate* is fine (similarity 0.0).
        result = tiny_engine.similar_to_movie(10, top_k=5)
        assert all(np.isfinite(r["similarity"]) for r in result)

    def test_empty_profile_returns_no_recommendations(self, tiny_engine):
        empty = sparse.csr_matrix((1, tiny_engine.n_features))
        assert tiny_engine.similar_to_profile(empty, top_k=3) == []

    def test_none_profile_returns_no_recommendations(self, tiny_engine):
        assert tiny_engine.similar_to_profile(None, top_k=3) == []


class TestExclusionsAndAllowLists:
    def test_exclude_ids_removed(self, tiny_engine):
        result = tiny_engine.similar_to_movie(10, top_k=5, exclude_ids=[30])
        assert all(r["movieId"] != 30 for r in result)

    def test_allowed_ids_restrict_candidates(self, tiny_engine):
        result = tiny_engine.similar_to_movie(10, top_k=5, allowed_ids=[20, 999])
        assert [r["movieId"] for r in result] == [20]

    def test_exclude_and_allow_combine(self, tiny_engine):
        result = tiny_engine.similar_to_movie(
            10, top_k=5, allowed_ids=[20, 30], exclude_ids=[30]
        )
        assert [r["movieId"] for r in result] == [20]


class TestProfileQueries:
    def test_profile_ranking(self, tiny_engine):
        profile = sparse.csr_matrix(np.array([[1.0, 1.0, 0.0]]))
        result = tiny_engine.similar_to_profile(profile, top_k=3)
        assert result[0]["movieId"] == 30
        # profile is parallel to row2 -> cosine exactly 1.
        assert result[0]["similarity"] == pytest.approx(1.0, abs=1e-9)

    def test_profile_direction_respected(self, tiny_engine):
        profile = sparse.csr_matrix(np.array([[1.0, 0.0, 0.0]]))
        result = tiny_engine.similar_to_profile(profile, top_k=3)
        assert result[0]["movieId"] == 10
        assert result[0]["similarity"] == pytest.approx(1.0, abs=1e-9)

    def test_profile_wrong_dimension_rejected(self, tiny_engine):
        with pytest.raises(InvalidFeatureArtifactsError):
            tiny_engine.similar_to_profile(sparse.csr_matrix((1, 99)), top_k=1)

    def test_profile_respects_exclusions(self, tiny_engine):
        profile = sparse.csr_matrix(np.array([[0.0, 1.0, 0.0]]))
        result = tiny_engine.similar_to_profile(profile, top_k=3, exclude_ids=[20])
        assert all(r["movieId"] != 20 for r in result)


class TestArtifactBackedEngine:
    def test_engine_loads_from_workspace(self, small_engine):
        assert small_engine.n_movies == 9
        assert small_engine.n_features > 0
        assert small_engine.has_movie(1)

    def test_similar_animation_movies_rank_first(self, small_engine):
        # Movie 1 (animation cluster) should neighbour movie 2 first.
        result = small_engine.similar_to_movie(1, top_k=8)
        assert result[0]["movieId"] == 2
        assert result[0]["similarity"] > result[1]["similarity"]

    def test_deterministic_on_workspace(self, small_engine):
        first = small_engine.similar_to_movie(4, top_k=5)
        assert first == small_engine.similar_to_movie(4, top_k=5)