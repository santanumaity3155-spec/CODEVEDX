"""Offline evaluation for content-based recommendations (Module 3).

Metrics
-------
Pure ranking metrics over one user's recommendation list ``R`` (length <= K)
against a set of *relevant* items ``G``:

* ``precision@K`` = |R[:K] ∩ G| / len(R[:K])
* ``recall@K``    = |R[:K] ∩ G| / |G|
* ``hit_rate@K``  = 1 if R[:K] ∩ G is non-empty else 0
* ``AP@K``        = mean of P(i) over hits at ranks i <= K, normalised by
                    min(|G|, K)   (MAP@K = mean AP@K over users)
* ``NDCG@K``      = DCG / IDCG with binary relevance gains

Evaluation protocol (leakage-free)
----------------------------------
For every evaluated user the MovieLens interactions are split **by timestamp**:

1. The most recent share ``EVALUATION_TEST_FRACTION`` (at least
   ``EVALUATION_MIN_TEST_ITEMS``, always leaving at least one older
   interaction) of that user's ratings forms the held-out **test** part.
2. All earlier interactions form the **train/history** part.
3. The content profile vector is the mean TF-IDF vector of the history movies
   rated >= ``EVALUATION_LIKE_THRESHOLD`` ("liked"). It therefore uses *only*
   information available before the hold-out period.
4. Every history movie is excluded as a recommendation candidate, so the model
   cannot score already-seen titles.
5. Ground-truth relevant items are exactly the *liked* (rating >= threshold)
   movies inside the user's test part. Disliked test items are neither
   rewarded nor punished.

Leakage guarantees: profiles/exclusions derive solely from the earlier
interactions; ground truth derives solely from later ones; the TF-IDF
vectorizer was fit on content text only (Module 2), never on ratings.
User eligibility is decided on the train part alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from src import config
from src.logging_config import get_logger
from src.similarity_engine import SimilarityEngine

logger = get_logger(__name__)

#: Metric names produced per K value.
METRIC_NAMES = ("precision", "recall", "hit_rate", "map", "ndcg")


class EvaluationError(Exception):
    """Raised for malformed inputs to evaluation routines."""


def _validate_k(k: object) -> int:
    """Validate a cut-off value K (positive integer, bools rejected)."""
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise EvaluationError(f"K must be a positive integer; got {k!r}.")
    k = int(k)
    if k < 1:
        raise EvaluationError(f"K must be >= 1; got {k}.")
    return k


# --------------------------------------------------------------------------- #
# Pure ranking metrics                                                        #
# --------------------------------------------------------------------------- #
def precision_at_k(
    recommended: object, relevant: object, k: int
) -> float:
    """Precision within the first K recommendations."""
    k = _validate_k(k)
    recs = list(dict.fromkeys(recommended))[:k]
    rel = set(relevant)
    if not recs:
        return 0.0
    return sum(1 for mid in recs if mid in rel) / len(recs)


def recall_at_k(recommended: object, relevant: object, k: int) -> float:
    """Recall within the first K recommendations."""
    k = _validate_k(k)
    rel = set(relevant)
    if not rel:
        return 0.0
    recs = list(dict.fromkeys(recommended))[:k]
    return sum(1 for mid in recs if mid in rel) / len(rel)


def hit_rate_at_k(recommended: object, relevant: object, k: int) -> float:
    """1.0 when any of the first K recommendations is relevant, else 0.0."""
    k = _validate_k(k)
    recs = list(dict.fromkeys(recommended))[:k]
    rel = set(relevant)
    return 1.0 if any(mid in rel for mid in recs) else 0.0


def average_precision_at_k(
    recommended: object, relevant: object, k: int
) -> float:
    """Average Precision at K (binary relevance), normalised by min(|G|, K)."""
    k = _validate_k(k)
    rel = set(relevant)
    if not rel:
        return 0.0
    score, hits = 0.0, 0
    for pos, mid in enumerate(dict.fromkeys(recommended), start=1):
        if pos > k:
            break
        if mid in rel:
            hits += 1
            score += hits / pos
    return score / min(len(rel), k)


def ndcg_at_k(recommended: object, relevant: object, k: int) -> float:
    """Normalised Discounted Cumulative Gain at K (binary gains)."""
    k = _validate_k(k)
    rel = set(relevant)
    if not rel:
        return 0.0
    dcg = sum(
        1.0 / np.log2(pos + 1)
        for pos, mid in enumerate(list(dict.fromkeys(recommended))[:k], start=1)
        if mid in rel
    )
    ideal_hits = min(len(rel), k)
    idcg = sum(1.0 / np.log2(pos + 1) for pos in range(1, ideal_hits + 1))
    return float(dcg / idcg)


def evaluate_rankings(
    recommended: object, relevant: object, k_values: tuple[int, ...]
) -> dict[int, dict[str, float]]:
    """Compute all metrics for one ranking across several K values."""
    ks = tuple(sorted({_validate_k(k) for k in k_values}))
    if not ks:
        raise EvaluationError("k_values must contain at least one cut-off.")
    result: dict[int, dict[str, float]] = {}
    for k in ks:
        result[k] = {
            "precision": precision_at_k(recommended, relevant, k),
            "recall": recall_at_k(recommended, relevant, k),
            "hit_rate": hit_rate_at_k(recommended, relevant, k),
            "map": average_precision_at_k(recommended, relevant, k),
            "ndcg": ndcg_at_k(recommended, relevant, k),
        }
    return result


# --------------------------------------------------------------------------- #
# Temporal split & user selection                                             #
# --------------------------------------------------------------------------- #
def split_user_interactions_temporal(
    ratings: pd.DataFrame,
    *,
    test_fraction: float = config.EVALUATION_TEST_FRACTION,
    min_test_items: int = config.EVALUATION_MIN_TEST_ITEMS,
) -> pd.DataFrame:
    """Assign every interaction of every user to ``train`` or ``test``.

    Per user, interactions are ordered by ``(timestamp, movieId)`` and the
    newest ``max(min_test_items, round(n * test_fraction))`` become the test
    part (at least one interaction always remains in train). The split is
    deterministic and uses only per-user information - no global leakage.

    Returns a copy of ``ratings`` with an extra ``split`` column.
    """
    missing = [
        c
        for c in ("userId", "movieId", "rating", "timestamp")
        if c not in ratings.columns
    ]
    if missing:
        raise EvaluationError(f"ratings frame is missing column(s): {missing}.")
    if not 0.0 < test_fraction < 1.0:
        raise EvaluationError(
            f"test_fraction must be within (0, 1); got {test_fraction}."
        )
    if min_test_items < 0:
        raise EvaluationError("min_test_items must be >= 0.")

    ordered = ratings.sort_values(
        ["userId", "timestamp", "movieId"], kind="mergesort"
    ).reset_index(drop=True)
    sizes = ordered.groupby("userId")["movieId"].transform("size")
    rank_from_end = ordered.groupby("userId").cumcount(ascending=False)
    n_test = np.maximum(
        int(min_test_items), (sizes * float(test_fraction)).round().astype("int64")
    )
    # Never move everything into test; keep at least one training item.
    n_test = np.minimum(n_test, sizes - 1)
    n_test = np.maximum(n_test, 0)
    is_test = rank_from_end.to_numpy() < n_test.to_numpy()

    split_df = ordered.copy()
    split_df["split"] = np.where(is_test, "test", "train")
    return split_df


def select_evaluation_users(
    train_ratings: pd.DataFrame,
    *,
    min_user_ratings: int = config.EVALUATION_MIN_USER_RATINGS,
    max_users: int = config.EVALUATION_MAX_USERS,
    seed: int = config.RANDOM_SEED,
) -> list[int]:
    """Deterministically pick which users to evaluate.

    Eligibility counts **train** interactions only (never the hold-out), so
    the selection itself cannot leak test information. When more eligible
    users exist than ``max_users``, a fixed-seed sample without replacement
    keeps runtime bounded while staying reproducible.
    """
    counts = train_ratings.groupby("userId")["movieId"].size()
    eligible = sorted(int(u) for u, c in counts.items() if c >= min_user_ratings)
    rng = np.random.default_rng(seed)
    if len(eligible) > max_users:
        chosen = rng.choice(np.array(eligible), size=max_users, replace=False)
        selected = sorted(int(u) for u in chosen)
        logger.info(
            "Sampled %d evaluation users out of %d eligible (seed=%d).",
            len(selected),
            len(eligible),
            seed,
        )
    else:
        selected = eligible
    logger.info("%d users eligible for evaluation.", len(selected))
    return selected


def build_user_profile(
    movie_ids: object,
    matrix: sparse.spmatrix,
    row_of_movie: dict[int, int],
) -> sparse.csr_matrix | None:
    """Mean TF-IDF vector over ``movie_ids`` present in the feature index.

    Returns ``None`` when none of the movies map to matrix rows.
    """
    rows = [row_of_movie[int(m)] for m in movie_ids if int(m) in row_of_movie]
    if not rows:
        return None
    profile = np.asarray(matrix[rows].mean(axis=0))
    return sparse.csr_matrix(profile)


# --------------------------------------------------------------------------- #
# Evaluator                                                                   #
# --------------------------------------------------------------------------- #
class ContentBasedEvaluator:
    """Leakage-free offline evaluation of a :class:`SimilarityEngine`.

    Parameters mirror the module-level configuration defaults so experiments
    can override any knob without touching ``src.config``.
    """

    def __init__(
        self,
        engine: SimilarityEngine,
        ratings: pd.DataFrame,
        *,
        k_values: tuple[int, ...] | None = None,
        like_threshold: float = config.EVALUATION_LIKE_THRESHOLD,
        test_fraction: float = config.EVALUATION_TEST_FRACTION,
        min_test_items: int = config.EVALUATION_MIN_TEST_ITEMS,
        min_user_ratings: int = config.EVALUATION_MIN_USER_RATINGS,
        max_users: int = config.EVALUATION_MAX_USERS,
        progress_every: int = config.EVALUATION_PROGRESS_LOG_EVERY,
        allowed_ids: np.ndarray | pd.Index | None = None,
        seed: int = config.RANDOM_SEED,
    ) -> None:
        self.engine = engine
        self.ratings = ratings
        self.k_values = tuple(k_values or tuple(config.EVALUATION_K_VALUES))
        self.like_threshold = float(like_threshold)
        self.test_fraction = float(test_fraction)
        self.min_test_items = int(min_test_items)
        self.min_user_ratings = int(min_user_ratings)
        self.max_users = int(max_users)
        self.progress_every = max(int(progress_every), 1)
        self.allowed_ids = allowed_ids
        self.seed = int(seed)

    def evaluate(self) -> dict:
        """Run the full protocol and return an aggregated report dict."""
        missing = [
            c
            for c in ("userId", "movieId", "rating", "timestamp")
            if c not in self.ratings.columns
        ]
        if missing:
            raise EvaluationError(
                f"ratings frame is missing required column(s): {missing}."
            )

        logger.info(
            "Evaluation starting: %d interactions, K=%s, like_threshold=%.1f",
            len(self.ratings),
            list(self.k_values),
            self.like_threshold,
        )
        split = split_user_interactions_temporal(
            self.ratings,
            test_fraction=self.test_fraction,
            min_test_items=self.min_test_items,
        )
        train = split[split["split"] == "train"]
        test = split[split["split"] == "test"]

        users = select_evaluation_users(
            train,
            min_user_ratings=self.min_user_ratings,
            max_users=self.max_users,
            seed=self.seed,
        )

        row_of_movie = dict(self.engine._row_of_movie)
        max_k = max(self.k_values)

        totals = {k: {name: 0.0 for name in METRIC_NAMES} for k in self.k_values}
        evaluated = 0
        skipped_no_liked_history = 0
        skipped_no_relevant_test = 0

        for user_id in users:
            history = train.loc[train["userId"] == user_id]
            held_out = test.loc[test["userId"] == user_id]

            liked_history = history.loc[
                history["rating"].astype("float64") >= self.like_threshold, "movieId"
            ]
            relevant = set(
                held_out.loc[
                    held_out["rating"].astype("float64") >= self.like_threshold,
                    "movieId",
                ].astype("int64")
            )
            if liked_history.empty or not relevant:
                if liked_history.empty:
                    skipped_no_liked_history += 1
                else:
                    skipped_no_relevant_test += 1
                continue

            profile = build_user_profile(
                liked_history, self.engine._matrix, row_of_movie
            )
            if profile is None:
                skipped_no_liked_history += 1
                continue

            # Leakage guard: exclude every history movie from candidates.
            exclude_ids = {
                int(mid) for mid in history["movieId"] if int(mid) in row_of_movie
            }
            neighbours = self.engine.similar_to_profile(
                profile,
                top_k=max_k,
                exclude_ids=exclude_ids,
                allowed_ids=self.allowed_ids,
            )
            recommended = [entry["movieId"] for entry in neighbours]
            per_k = evaluate_rankings(recommended, relevant, self.k_values)
            for k, metrics in per_k.items():
                for name, value in metrics.items():
                    totals[k][name] += value
            evaluated += 1
            if evaluated % self.progress_every == 0:
                logger.info(
                    "Evaluation progress: %d/%d users scored", evaluated, len(users)
                )

        metrics_at_k = {
            str(k): {
                name: (value / evaluated if evaluated else 0.0)
                for name, value in per_k_totals.items()
            }
            for k, per_k_totals in totals.items()
        }

        report = {
            "protocol": (
                "Per-user temporal split: newest "
                f"{self.test_fraction:.0%} of each user's interactions (min "
                f"{self.min_test_items}) held out as ground truth; profiles "
                f"built from earlier likes only (rating >= "
                f"{self.like_threshold}); all history movies excluded from "
                "candidates."
            ),
            "params": {
                "k_values": [int(k) for k in self.k_values],
                "like_threshold": self.like_threshold,
                "test_fraction": self.test_fraction,
                "min_test_items": self.min_test_items,
                "min_user_ratings": self.min_user_ratings,
                "max_users": self.max_users,
                "seed": self.seed,
                "popularity_filter_applied": self.allowed_ids is not None,
            },
            "users": {
                "eligible": len(users),
                "evaluated": evaluated,
                "skipped_no_liked_history": skipped_no_liked_history,
                "skipped_no_relevant_test_items": skipped_no_relevant_test,
            },
            "metrics_at_k": metrics_at_k,
        }
        logger.info(
            "Evaluation finished: %d users scored (%d skipped).",
            evaluated,
            skipped_no_liked_history + skipped_no_relevant_test,
        )
        return report


def evaluate_content_based(
    engine: SimilarityEngine, ratings: pd.DataFrame, **overrides: object
) -> dict:
    """Convenience wrapper around :class:`ContentBasedEvaluator`."""
    return ContentBasedEvaluator(engine, ratings, **overrides).evaluate()
