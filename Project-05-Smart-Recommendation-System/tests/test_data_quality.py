"""Tests for src/data_quality.py plus raw-dataset invariant checks.

Structural/invariant based only - no hardcoded dataset row counts.
"""

from __future__ import annotations

import hashlib

import pandas as pd
import pytest

from src import config
from src.data_loader import load_ratings
from src import data_quality as dq


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

class TestSchemaValidation:
    def test_valid_schema_passes(self):
        df = pd.DataFrame({"movieId": [1], "title": ["X (2000)"], "genres": ["Action"]})
        result = dq.validate_schema(df, "movies")
        assert result["valid"] is True
        assert result["missing_columns"] == []

    def test_missing_column_fails_and_reports_names(self):
        df = pd.DataFrame({"movieId": [1], "title": ["X"]})
        result = dq.validate_schema(df, "movies")
        assert result["valid"] is False
        assert "genres" in result["missing_columns"]

    @pytest.mark.parametrize("dataset", ["movies", "tags", "links",
                                         "genome_scores", "genome_tags"])
    def test_real_raw_schemas_pass(self, dataset):
        from src.data_loader import (
            load_genome_scores,
            load_genome_tags,
            load_links,
            load_movies,
            load_tags,
        )

        loaders = {
            "movies": load_movies,
            "tags": load_tags,
            "links": load_links,
            "genome_scores": load_genome_scores,
            "genome_tags": load_genome_tags,
        }
        result = dq.validate_schema(loaders[dataset](), dataset)
        assert result["valid"] is True

    def test_real_ratings_schema_passes(self, ratings_full):
        assert dq.validate_schema(ratings_full, "ratings")["valid"] is True


# ---------------------------------------------------------------------------
# Duplicate rules
# ---------------------------------------------------------------------------

def _dup_result(res: dict) -> int:
    return res["duplicate_records"]


class TestDuplicateRules:
    def test_movie_id_duplicates_detected(self, sample_movies_df):
        movies = pd.concat([sample_movies_df, sample_movies_df.iloc[[0]]],
                           ignore_index=True)
        res = dq.analyze_movie_duplicates(movies)
        assert _dup_result(res["duplicate_movieId"]) == 1

    def test_duplicate_titles_with_distinct_ids_are_reported_not_removed(self):
        movies = pd.DataFrame({
            "movieId": [1, 2],
            "title": ["Same Title (1999)", "Same Title (1999)"],
            "genres": ["Action", "Drama"],
        })
        res = dq.analyze_movie_duplicates(movies)
        assert _dup_result(res["duplicate_title"]) == 1  # reported...
        assert _dup_result(res["duplicate_movieId"]) == 0  # ...but IDs are valid

    def test_exact_rating_repeat_vs_legitimate_re_rating(self, sample_ratings_df):
        # user 10 rated movie 1 twice with different timestamps -> legitimate.
        ratings = pd.concat(
            [sample_ratings_df, sample_ratings_df.iloc[[0]]], ignore_index=True
        )  # exact repeat of row 0 (same timestamp)
        res = dq.analyze_rating_duplicates(ratings)
        assert _dup_result(res["exact_duplicates"]) == 1
        assert _dup_result(res["repeated_user_movie_interactions"]) >= 2

    def test_tag_duplicate_detection(self):
        tags = pd.DataFrame({
            "userId": [1, 1],
            "movieId": [5, 5],
            "tag": ["fun", "fun"],
            "timestamp": [100, 100],
        })
        assert _dup_result(dq.analyze_tag_duplicates(tags)["exact_duplicates"]) == 1

    def test_link_movie_id_duplicates(self):
        links = pd.DataFrame({"movieId": [1, 1], "imdbId": [10, 10], "tmdbId": [20, 20]})
        assert _dup_result(dq.analyze_link_duplicates(links)["duplicate_movieId"]) == 1

    def test_genome_pair_duplicates(self):
        scores = pd.DataFrame({
            "movieId": [1, 1],
            "tagId": [3, 3],
            "relevance": [0.1, 0.2],
        })
        res = dq.analyze_genome_score_duplicates(scores)
        assert _dup_result(res["duplicate_movie_tag_pairs"]) == 1


# ---------------------------------------------------------------------------
# Referential integrity
# ---------------------------------------------------------------------------

class TestReferentialIntegrity:
    def test_orphans_detected_with_count(self):
        parent = pd.DataFrame({"movieId": [1, 2, 3]})
        child = pd.DataFrame({"movieId": [1, 99, 98]})
        res = dq.check_orphans(child, "movieId", parent, "movieId",
                               "child -> parent")
        assert res["orphan_records"] == 2
        assert res["orphan_pct"] > 0

    def test_no_orphans_when_consistent(self, sample_movies_df, sample_ratings_df):
        res = dq.check_orphans(sample_ratings_df, "movieId", sample_movies_df,
                               "movieId", "ratings -> movies")
        assert res["orphan_records"] == 0

    def test_full_relationships_quantified(self, movies_full, ratings_full):
        res = dq.check_orphans(ratings_full, "movieId", movies_full, "movieId",
                               "ratings.movieId -> movies.movieId")
        # MovieLens 25M guarantees every rated movie exists in the catalog.
        assert res["orphan_records"] == 0


# ---------------------------------------------------------------------------
# Rating domain / ID validation
# ---------------------------------------------------------------------------

class TestRatingDomain:
    def test_valid_grid_passes(self, sample_ratings_df):
        assert dq.validate_rating_domain(sample_ratings_df)["valid"] is True

    def test_out_of_range_detected(self):
        ratings = pd.DataFrame({"rating": [-1.0, 5.5]})
        res = dq.validate_rating_domain(ratings)
        assert res["out_of_range"] == 2
        assert res["valid"] is False

    def test_off_grid_detected(self):
        ratings = pd.DataFrame({"rating": [4.7]})
        res = dq.validate_rating_domain(ratings)
        assert res["off_half_star_grid"] == 1
        assert res["valid"] is False


class TestRealRatingInvariants:
    def test_all_real_ratings_within_movielens_scale(self, ratings_full):
        r = ratings_full["rating"]
        assert float(r.min()) >= config.RATING_MIN
        assert float(r.max()) <= config.RATING_MAX

    def test_real_movie_ids_positive_and_unique(self, movies_full):
        ids = movies_full["movieId"]
        assert (ids > 0).all()
        assert ids.is_unique

    def test_real_user_movie_ids_positive(self, ratings_full):
        assert (ratings_full["userId"] > 0).all()
        assert (ratings_full["movieId"] > 0).all()


class TestIdValidation:
    def test_negative_and_missing_ids_flagged(self):
        df = pd.DataFrame({"userId": [1, -2, None], "movieId": [1, 2, 3]})
        res = dq.validate_id_columns(df, ["userId", "movieId"], "test")
        assert res["valid"] is False
        assert res["issues"]["userId"]["negative"] == 1
        assert res["issues"]["userId"]["missing"] == 1

    def test_clean_ids_pass(self, sample_ratings_df):
        res = dq.validate_id_columns(sample_ratings_df, ["userId", "movieId"], "test")
        assert res["valid"] is True


# ---------------------------------------------------------------------------
# Raw data preservation (SHA-256 vs. baseline manifest)
# ---------------------------------------------------------------------------

class TestRawDataUnchanged:
    @staticmethod
    def _sha256(path) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @pytest.mark.parametrize("file_name", [
        "movies.csv",
        "ratings.csv",
        "tags.csv",
        "links.csv",
        "genome-scores.csv",
        "genome-tags.csv",
    ])
    def test_raw_file_hash_matches_baseline(self, baseline_raw_manifest, file_name):
        expected = baseline_raw_manifest[file_name]["sha256"]
        path = config.RAW_DATA_DIR / "ml-25m" / file_name
        assert self._sha256(path) == expected, f"{file_name} was modified since baseline!"
