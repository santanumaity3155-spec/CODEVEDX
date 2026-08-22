"""Tests for the Module 3 content-based recommender (``src/recommender.py``).

Covers the public ``recommend_similar_movies`` contract (columns, ranks,
ordering, self-exclusion, duplicates, unknown IDs, invalid top-K), popularity
and metadata quality controls, filters (genre / year / per-call rating
thresholds), popularity-statistics helpers, title search and graceful
degradation.

All artifact-backed tests run inside the ``module3_workspace`` sandbox; no
real processed files or raw data are touched.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src import config
from src.recommender import (
    RECOMMENDATION_COLUMNS,
    ContentRecommender,
    InvalidFilterError,
    RecommenderError,
    compute_popularity_stats,
    empty_recommendations,
    find_movie_by_title,
)


class TestRecommendationContract:
    def test_columns_and_ranks(self, small_recommender):
        recs = small_recommender.recommend_similar_movies(1, top_k=5)
        assert list(recs.columns) == list(RECOMMENDATION_COLUMNS)
        assert recs["rank"].tolist() == [1, 2, 3, 4, 5]

    def test_seed_excluded(self, small_recommender):
        for seed in (1, 4, 7):
            recs = small_recommender.recommend_similar_movies(seed, top_k=8)
            assert seed not in recs["movieId"].tolist()

    def test_no_duplicates(self, small_recommender):
        recs = small_recommender.recommend_similar_movies(2, top_k=8)
        assert recs["movieId"].is_unique

    def test_sorted_descending_by_similarity(self, small_recommender):
        recs = small_recommender.recommend_similar_movies(3, top_k=8)
        scores = recs["similarity"].tolist()
        assert all(a >= b for a, b in zip(scores, scores[1:]))

    def test_deterministic_results(self, small_recommender):
        first = small_recommender.recommend_similar_movies(1, top_k=6)
        for _ in range(3):
            pd.testing.assert_frame_equal(
                first, small_recommender.recommend_similar_movies(1, top_k=6)
            )

    def test_top_k_respected(self, small_recommender):
        assert len(small_recommender.recommend_similar_movies(1, top_k=3)) == 3
        # Pool smaller than K: every remaining candidate is returned.
        pool = len(small_recommender.catalog) - 1
        recs = small_recommender.recommend_similar_movies(1, top_k=100)
        assert 0 < len(recs) <= pool

    @pytest.mark.parametrize("bad_top_k", [0, -2, 1.5, "ten", True])
    def test_invalid_top_k_raises(self, small_recommender, bad_top_k):
        with pytest.raises(ValueError):
            small_recommender.recommend_similar_movies(1, top_k=bad_top_k)

    def test_unknown_movie_returns_empty_gracefully(self, small_recommender):
        recs = small_recommender.recommend_similar_movies(999_999_999)
        assert isinstance(recs, pd.DataFrame)
        assert recs.empty
        assert list(recs.columns) == list(RECOMMENDATION_COLUMNS)

    def test_metadata_enrichment(self, small_recommender):
        recs = small_recommender.recommend_similar_movies(1, top_k=3)
        row = recs.iloc[0]
        expected_title = small_recommender.catalog.set_index("movieId").loc[
            int(row["movieId"]), "title"
        ]
        assert row["title"] == expected_title
        assert isinstance(row["genres"], str)


class TestPopularityControls:
    def test_below_threshold_excluded(self, module3_workspace, small_engine):
        from src.recommender import ContentRecommender, compute_popularity_stats

        popularity = compute_popularity_stats(module3_workspace["ratings"])
        recommender = ContentRecommender(
            engine=small_engine,
            catalog=module3_workspace["catalog"],
            popularity=popularity,
            min_movie_ratings=4,  # only movies rated >= 4 times may appear
        )
        recs = recommender.recommend_similar_movies(1, top_k=8)
        counts = popularity.set_index("movieId")["rating_count"]
        assert all(counts.loc[int(mid)] >= 4 for mid in recs["movieId"])

    def test_per_call_min_ratings_override(self, small_recommender):
        strict = small_recommender.recommend_similar_movies(
            1, top_k=8, filters={"min_ratings": 4}
        )
        loose = small_recommender.recommend_similar_movies(1, top_k=8)
        assert len(strict) <= len(loose)

    def test_disabling_filter_broadens_pool(self, small_recommender):
        without = small_recommender.recommend_similar_movies(
            1, top_k=8, filters={"min_ratings": 0}
        )
        with_default = small_recommender.recommend_similar_movies(1, top_k=8)
        assert len(without) >= len(with_default)

    def test_no_popularity_stats_skips_filter(self, module3_workspace, small_engine):
        recommender = ContentRecommender(
            engine=small_engine,
            catalog=module3_workspace["catalog"],
            popularity=None,
            min_movie_ratings=5,  # would filter everything if stats existed
        )
        recs = recommender.recommend_similar_movies(1, top_k=3)
        assert not recs.empty


class TestMetadataQuality:
    def test_invalid_metadata_never_recommended(self, module3_workspace, small_engine):
        broken_catalog = module3_workspace["catalog"].copy()
        broken_catalog.loc[broken_catalog["movieId"] == 2, "title"] = None
        recommender = ContentRecommender(
            engine=small_engine,
            catalog=broken_catalog,
            popularity=None,
            min_movie_ratings=0,
        )
        recs = recommender.recommend_similar_movies(1, top_k=8)
        assert 2 not in recs["movieId"].tolist()

    def test_duplicate_catalog_ids_rejected(self, module3_workspace, small_engine):
        duplicated = pd.concat(
            [module3_workspace["catalog"], module3_workspace["catalog"].iloc[[0]]],
            ignore_index=True,
        )
        with pytest.raises(RecommenderError):
            ContentRecommender(
                engine=small_engine, catalog=duplicated, min_movie_ratings=0
            )

    def test_missing_required_columns_rejected(self, small_engine):
        with pytest.raises(RecommenderError):
            ContentRecommender(engine=small_engine, catalog=pd.DataFrame({"x": [1]}))


class TestFilters:
    def test_genre_filter(self, small_recommender):
        recs = small_recommender.recommend_similar_movies(
            1, top_k=8, filters={"genres": ["Animation"]}
        )
        assert not recs.empty
        assert all("Animation" in g for g in recs["genres"])

    def test_genre_filter_case_insensitive(self, small_recommender):
        recs = small_recommender.recommend_similar_movies(
            1, top_k=8, filters={"genres": ["aNiMaTiOn"]}
        )
        assert not recs.empty

    def test_year_filters(self, small_recommender):
        recs = small_recommender.recommend_similar_movies(
            1, top_k=8, filters={"max_year": 1995}
        )
        years = recs["title"].str.extract(r"\((\d{4})\)")[0].astype(int)
        assert all(year <= 1995 for year in years)

    def test_unknown_filter_key_raises(self, small_recommender):
        with pytest.raises(InvalidFilterError):
            small_recommender.recommend_similar_movies(1, filters={"colour": "blue"})

    def test_malformed_filter_values_raise(self, small_recommender):
        with pytest.raises(InvalidFilterError):
            small_recommender.recommend_similar_movies(1, filters={"genres": "Action"})
        with pytest.raises(InvalidFilterError):
            small_recommender.recommend_similar_movies(1, filters={"min_ratings": -1})
        with pytest.raises(InvalidFilterError):
            small_recommender.recommend_similar_movies(1, filters="genres")


class TestDescribeSeedAndTitleSearch:
    def test_describe_known_seed(self, small_recommender):
        info = small_recommender.describe_seed(1)
        assert info is not None
        assert info["movieId"] == 1
        assert "Adventure" in info["title"]
        assert "Animation" in info["genres"]

    def test_describe_unknown_seed(self, small_recommender):
        assert small_recommender.describe_seed(424242) is None

    def test_describe_non_integer_seed(self, small_recommender):
        assert small_recommender.describe_seed("junk") is None

    def test_find_movie_by_title_substring_case_insensitive(
        self, module3_workspace
    ):
        match = find_movie_by_title(module3_workspace["catalog"], "space buzz")
        assert match is not None
        assert match["movieId"] == 1
        assert match["genres"] == "Animation|Children|Comedy"

    def test_find_most_rated_match(self, module3_workspace):
        popularity = compute_popularity_stats(module3_workspace["ratings"])
        match = find_movie_by_title(
            module3_workspace["catalog"], "Adventure", popularity
        )
        # Movies 1 and 2 both contain "Adventure"; movie 1 has more ratings.
        assert match is not None and match["movieId"] == 1

    def test_find_movie_no_match_returns_none(self, module3_workspace):
        assert find_movie_by_title(module3_workspace["catalog"], "Transformers") is None


class TestPopularityStats:
    def test_compute_matches_manual_groupby(self, module3_workspace):
        ratings = module3_workspace["ratings"]
        stats = compute_popularity_stats(ratings)
        manual = ratings.groupby("movieId")["rating"].agg(["count", "mean"])
        assert set(stats["movieId"]) == set(manual.index)
        for _, row in stats.iterrows():
            key = int(row["movieId"])
            assert row["rating_count"] == int(manual.loc[key, "count"])
            assert row["mean_rating"] == pytest.approx(
                float(manual.loc[key, "mean"]), abs=1e-6
            )

    def test_compute_sorted_ascending_by_movieid(self, module3_workspace):
        stats = compute_popularity_stats(module3_workspace["ratings"])
        assert stats["movieId"].is_monotonic_increasing

    def test_compute_requires_columns(self):
        with pytest.raises(RecommenderError):
            compute_popularity_stats(pd.DataFrame({"userId": [1]}))

    def test_load_from_disk_chunked_matches_frame(self, module3_workspace):
        from src.recommender import load_popularity_stats

        loaded = load_popularity_stats(chunksize=10)
        expected = compute_popularity_stats(module3_workspace["ratings"])
        pd.testing.assert_frame_equal(loaded, expected)

    def test_load_missing_file_actionable_error(self, tmp_path, monkeypatch):
        from src.recommender import load_popularity_stats

        monkeypatch.setattr(config, "PROCESSED_RATINGS_PATH", tmp_path / "missing.csv")
        with pytest.raises(FileNotFoundError, match="Module 1"):
            load_popularity_stats()


class TestEmptyFrameContract:
    def test_empty_recommendations_shape(self):
        empty = empty_recommendations()
        assert empty.empty
        assert list(empty.columns) == list(RECOMMENDATION_COLUMNS)


class TestArtifactFactory:
    def test_from_artifacts_builds_working_recommender(self, module3_workspace):
        recommender = ContentRecommender.from_artifacts()
        recs = recommender.recommend_similar_movies(1, top_k=5)
        assert len(recs) == 5
        assert 1 not in recs["movieId"].tolist()
        # Popularity loaded by default from the workspace ratings CSV.
        assert recommender.popularity is not None

    def test_from_artifacts_missing_file_actionable(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            config,
            "PROCESSED_MOVIE_CONTENT_FEATURES_PATH",
            tmp_path / "nope.csv",
        )
        with pytest.raises(FileNotFoundError, match="Modules 1-2"):
            ContentRecommender.from_artifacts(load_popularity=False)