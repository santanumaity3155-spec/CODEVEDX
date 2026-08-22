"""Tests for Module 3 offline evaluation (``src/evaluation.py``).

Covers the pure ranking metrics against hand-computed values, the temporal
per-user split (determinism, disjointness, temporal ordering), deterministic
user selection, profile building, and end-to-end evaluator runs inside the
``module3_workspace`` sandbox - including explicit no-data-leakage checks.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src import config
from src.evaluation import (
    EvaluationError,
    average_precision_at_k,
    build_user_profile,
    evaluate_content_based,
    evaluate_rankings,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    select_evaluation_users,
    split_user_interactions_temporal,
)


class TestPrecisionRecall:
    def test_precision_counts_hits_over_returned(self):
        assert precision_at_k([1, 2, 3, 4], {2, 4}, 4) == 0.5
        assert precision_at_k([2, 9], {2}, 1) == 1.0
        assert precision_at_k([9, 8], {2}, 2) == 0.0

    def test_precision_empty_recommendations(self):
        assert precision_at_k([], {1}, 5) == 0.0

    def test_recall_normalised_by_relevant_size(self):
        assert recall_at_k([1, 2, 3, 4], {2, 4, 5}, 4) == pytest.approx(2 / 3)
        assert recall_at_k([2], {2}, 1) == 1.0
        assert recall_at_k([9], {2}, 5) == 0.0

    def test_recall_no_relevant_items(self):
        assert recall_at_k([1, 2], set(), 2) == 0.0

    def test_duplicates_in_ranking_counted_once(self):
        # [2, 2] truncated to K=1 -> one hit out of one returned item.
        assert precision_at_k([2, 2], {2}, 1) == 1.0


class TestHitRate:
    def test_hit_within_top_k(self):
        assert hit_rate_at_k([7, 3, 5], {3}, 2) == 1.0

    def test_hit_beyond_top_k_ignored(self):
        assert hit_rate_at_k([7, 3, 5], {5}, 2) == 0.0

    def test_no_hit(self):
        assert hit_rate_at_k([1], {2}, 1) == 0.0


class TestAveragePrecision:
    def test_hand_computed_ap(self):
        # rec=[2,1,3], rel={2,3}: P(1)=1/1, P(3)=2/3; normaliser min(2, 3)=2.
        expected = (1.0 + (2 / 3)) / 2
        assert average_precision_at_k([2, 1, 3], {2, 3}, 3) == pytest.approx(expected)

    def test_perfect_and_zero(self):
        assert average_precision_at_k([1, 2, 3], {1, 2}, 3) == pytest.approx(1.0)
        assert average_precision_at_k([7, 8], {1}, 2) == 0.0

    def test_no_relevant_items(self):
        assert average_precision_at_k([1], set(), 1) == 0.0


class TestNDCG:
    def test_single_hit_at_rank_two(self):
        dcg = 1 / math.log2(3)
        idcg = 1 / math.log2(2) + 1 / math.log2(3)
        assert ndcg_at_k([1, 2, 9], {2, 3}, 3) == pytest.approx(dcg / idcg)

    def test_perfect_ranking_scores_one(self):
        assert ndcg_at_k([2, 3, 1], {2, 3}, 3) == pytest.approx(1.0)

    def test_no_hits_scores_zero(self):
        assert ndcg_at_k([5, 6, 7], {2, 3}, 3) == 0.0

    def test_no_relevant_items(self):
        assert ndcg_at_k([1], set(), 1) == 0.0


class TestEvaluateRankings:
    def test_all_metrics_present_per_k(self):
        result = evaluate_rankings([2, 1, 3], {2}, (1, 3))
        assert set(result.keys()) == {1, 3}
        for metrics in result.values():
            assert set(metrics) == {"precision", "recall", "hit_rate", "map", "ndcg"}
            assert all(0.0 <= v <= 1.0 for v in metrics.values())

    def test_invalid_k_rejected(self):
        with pytest.raises(EvaluationError):
            evaluate_rankings([1], {1}, (0,))
        with pytest.raises(EvaluationError):
            precision_at_k([1], {1}, True)


# ---------------------------------------------------------------------------
# Temporal split & user selection
# ---------------------------------------------------------------------------
def _ratings_frame(rows: list[tuple[int, int, float, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "userId": pd.array([r[0] for r in rows], dtype="int32"),
            "movieId": pd.array([r[1] for r in rows], dtype="int32"),
            "rating": pd.array([r[2] for r in rows], dtype="float32"),
            "timestamp": pd.array([r[3] for r in rows], dtype="int64"),
        }
    )


class TestTemporalSplit:
    def test_split_disjoint_and_ordered_per_user(self):
        ratings = _ratings_frame(
            [
                (1, 10, 4.0, 100),
                (1, 11, 5.0, 200),
                (1, 12, 3.0, 300),
                (1, 13, 4.0, 400),
                (2, 20, 5.0, 150),
                (2, 21, 4.0, 250),
            ]
        )
        split = split_user_interactions_temporal(ratings, test_fraction=0.25)
        train = split[split["split"] == "train"]
        test = split[split["split"] == "test"]

        # Disjoint per user.
        for user_id, group in train.groupby("userId"):
            test_ids = set(test.loc[test["userId"] == user_id, "movieId"])
            assert not (set(group["movieId"]) & test_ids)

        # Newest interactions became the test part.
        u1_test = test.loc[test["userId"] == 1, "timestamp"].tolist()
        u1_train_max = train.loc[train["userId"] == 1, "timestamp"].max()
        assert all(ts > u1_train_max for ts in u1_test)

        # At least one training interaction always remains.
        sizes = split.groupby("userId")["split"].apply(
            lambda s: (s == "train").sum()
        )
        assert (sizes >= 1).all()

    def test_deterministic_split(self):
        rng = np.random.default_rng(config.RANDOM_SEED)
        rows = []
        counter = 0
        for user in range(1, 6):
            for _ in range(6):
                counter += 1
                rows.append(
                    (user, counter % 9 + 1, float(rng.choice([3.0, 4.0, 5.0])), counter)
                )
        ratings = _ratings_frame(rows)
        first = split_user_interactions_temporal(ratings)
        second = split_user_interactions_temporal(ratings)
        pd.testing.assert_frame_equal(first, second)

    def test_tie_break_on_movieid_when_timestamps_equal(self):
        ratings = _ratings_frame(
            [(1, 30, 5.0, 500), (1, 20, 4.0, 500), (1, 10, 3.0, 500)]
        )
        split = split_user_interactions_temporal(ratings, test_fraction=1 / 3)
        test_ids = set(split.loc[split["split"] == "test", "movieId"])
        assert test_ids == {30}  # largest movieId wins the tie deterministically

    def test_invalid_inputs_rejected(self):
        ratings = _ratings_frame([(1, 2, 4.0, 10)])
        with pytest.raises(EvaluationError):
            split_user_interactions_temporal(ratings, test_fraction=1.5)
        with pytest.raises(EvaluationError):
            split_user_interactions_temporal(ratings.drop(columns=["rating"]))
        with pytest.raises(EvaluationError):
            split_user_interactions_temporal(ratings, min_test_items=-1)


class TestUserSelection:
    def test_eligibility_decided_on_train_only(self, module3_workspace):
        split = split_user_interactions_temporal(module3_workspace["ratings"])
        train = split[split["split"] == "train"]
        users = select_evaluation_users(train, min_user_ratings=5)
        # u12 has only 2 interactions total -> never eligible.
        assert 12 not in users
        assert users == sorted(users)

    def test_sampling_is_deterministic(self, module3_workspace):
        split = split_user_interactions_temporal(module3_workspace["ratings"])
        train = split[split["split"] == "train"]
        kwargs = dict(min_user_ratings=1, max_users=2)
        assert (
            select_evaluation_users(train, **kwargs)
            == select_evaluation_users(train, **kwargs)
        )

    def test_min_user_ratings_boundary(self, module3_workspace):
        split = split_user_interactions_temporal(module3_workspace["ratings"])
        train = split[split["split"] == "train"]
        users = select_evaluation_users(train, min_user_ratings=99)
        assert users == []


class TestProfileBuilding:
    def test_profile_is_mean_of_rows(self, small_engine):
        row_of_movie = dict(small_engine._row_of_movie)
        profile = build_user_profile([1], small_engine._matrix, row_of_movie)
        expected_row = small_engine.row_for_movie(1)
        expected = np.asarray(small_engine._matrix[expected_row].todense())
        assert np.allclose(np.asarray(profile.todense()), expected)

    def test_unknown_movies_skipped(self, small_engine):
        row_of_movie = dict(small_engine._row_of_movie)
        profile = build_user_profile([1, 999_999], small_engine._matrix, row_of_movie)
        assert profile is not None
        expected_row = small_engine.row_for_movie(1)
        expected = np.asarray(small_engine._matrix[expected_row].todense())
        assert np.allclose(np.asarray(profile.todense()), expected)

    def test_all_unknown_movies_give_none(self, small_engine):
        row_of_movie = dict(small_engine._row_of_movie)
        assert (
            build_user_profile([999_999], small_engine._matrix, row_of_movie) is None
        )


class TestEndToEndEvaluation:
    def test_workspace_evaluation_runs_and_is_deterministic(
        self, small_engine, module3_workspace
    ):
        report = evaluate_content_based(
            small_engine,
            module3_workspace["ratings"],
            k_values=(5,),
        )
        users = report["users"]
        assert users["evaluated"] >= 1
        assert users["skipped_no_relevant_test_items"] >= 1
        metrics = report["metrics_at_k"]["5"]
        assert set(metrics) == {"precision", "recall", "hit_rate", "map", "ndcg"}
        assert all(0.0 <= v <= 1.0 for v in metrics.values())

        again = evaluate_content_based(
            small_engine,
            module3_workspace["ratings"],
            k_values=(5,),
        )
        assert again["metrics_at_k"] == report["metrics_at_k"]
        assert again["users"] == report["users"]

    def test_protocol_documented_in_report(self, small_engine, module3_workspace):
        report = evaluate_content_based(small_engine, module3_workspace["ratings"])
        assert "temporal" in report["protocol"].lower()
        params = report["params"]
        assert params["like_threshold"] == config.EVALUATION_LIKE_THRESHOLD
        assert params["test_fraction"] == config.EVALUATION_TEST_FRACTION

    def test_missing_rating_columns_rejected(self, small_engine):
        with pytest.raises(EvaluationError):
            evaluate_content_based(small_engine, pd.DataFrame({"userId": [1]}))


class TestNoDataLeakage:
    def test_ground_truth_never_in_history_split(self, module3_workspace):
        split = split_user_interactions_temporal(module3_workspace["ratings"])
        for user_id, group in split.groupby("userId"):
            history = set(group.loc[group["split"] == "train", "movieId"])
            holdout = set(group.loc[group["split"] == "test", "movieId"])
            assert not (history & holdout)

    def test_history_movies_never_appear_in_evaluated_lists(
        self, small_engine, module3_workspace
    ):
        """Recompute the evaluator's inputs and confirm that every scored
        user's recommendation list excludes their entire history."""
        from src.evaluation import (
            ContentBasedEvaluator,
            build_user_profile,
        )

        evaluator = ContentBasedEvaluator(small_engine, module3_workspace["ratings"])
        split = split_user_interactions_temporal(evaluator.ratings)
        train = split[split["split"] == "train"]
        test = split[split["split"] == "test"]
        users = select_evaluation_users(
            train, min_user_ratings=evaluator.min_user_ratings
        )

        row_of_movie = dict(small_engine._row_of_movie)
        max_k = max(evaluator.k_values)
        scored_any = False
        for user_id in users:
            history = train.loc[train["userId"] == user_id]
            held_out = test.loc[test["userId"] == user_id]
            liked_history = history.loc[
                history["rating"].astype("float64") >= evaluator.like_threshold,
                "movieId",
            ]
            relevant = set(
                held_out.loc[
                    held_out["rating"].astype("float64") >= evaluator.like_threshold,
                    "movieId",
                ].astype("int64")
            )
            if liked_history.empty or not relevant:
                continue
            profile = build_user_profile(
                liked_history, small_engine._matrix, row_of_movie
            )
            exclude_ids = {
                int(mid) for mid in history["movieId"] if int(mid) in row_of_movie
            }
            neighbours = small_engine.similar_to_profile(
                profile, top_k=max_k, exclude_ids=exclude_ids
            )
            recommended = [entry["movieId"] for entry in neighbours]
            history_ids = {int(m) for m in history["movieId"]}
            assert not (set(recommended) & history_ids)
            scored_any = True
        assert scored_any  # the workspace must actually exercise this path

    def test_mutating_test_ratings_leaves_train_untouched(self, module3_workspace):
        """Destroying a ground-truth like cannot change what the model learns
        from: profiles derive solely from the earlier (train) interactions."""
        original = module3_workspace["ratings"]
        mutated = original.copy()
        u10_ts = mutated.loc[mutated["userId"] == 10, "timestamp"]
        newest_mask = (mutated["userId"] == 10) & (
            mutated["timestamp"] == u10_ts.max()
        )
        mutated.loc[newest_mask, "rating"] = 1.0

        split_before = split_user_interactions_temporal(original)
        split_after = split_user_interactions_temporal(mutated)

        # The train part - and therefore the profile input - is identical.
        train_before = split_before.loc[
            (split_before["userId"] == 10) & (split_before["split"] == "train")
        ].reset_index(drop=True)
        train_after = split_after.loc[
            (split_after["userId"] == 10) & (split_after["split"] == "train")
        ].reset_index(drop=True)
        pd.testing.assert_frame_equal(train_before, train_after)

        # The mutated rating sits in the test part only.
        assert (split_after.loc[newest_mask.repeat(1), "split"] == "test").all()