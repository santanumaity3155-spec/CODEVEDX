"""Module 3 smoke test: real recommendations from the full processed dataset.

Loads the Module 2 TF-IDF artifacts, resolves several well-known seed movies
by searching the actual catalog titles (no assumptions about ID ordering),
prints ranked recommendations with similarity scores, and verifies the core
guarantees (no self-recommendation, no duplicates, descending scores,
top-K respected). Unknown movie IDs are demonstrated to fail gracefully.

Run from the project root::

    python scripts/smoke_test_recommendations.py

Exit code 0 = all smoke checks passed; 1 = failure.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src.logging_config import get_logger  # noqa: E402
from src.recommender import ContentRecommender, find_movie_by_title  # noqa: E402

logger = get_logger("scripts.smoke_test")

WIDTH = 78


def _print_header(title: str) -> None:
    print()
    print("=" * WIDTH)
    print(title)
    print("=" * WIDTH)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:  # older interpreters
        pass

    _print_header("MODULE 3 SMOKE TEST - CONTENT-BASED RECOMMENDATIONS")
    print(f"Dataset: {config.DATASET_NAME}")
    print(f"Top-K: {config.DEFAULT_TOP_K} | "
          f"min ratings per recommended movie: {config.RECOMMENDATION_MIN_MOVIE_RATINGS}")

    logger.info("Loading recommendation artifacts (this may take a moment) ...")
    recommender = ContentRecommender.from_artifacts()
    catalog = recommender.catalog
    popularity = recommender.popularity
    print(f"Feature matrix: {recommender.engine.n_movies} movies x "
          f"{recommender.engine.n_features} TF-IDF features")
    print(f"Candidate pool after quality filters: {len(catalog)} movies in catalog; "
          f"{int(recommender._valid_metadata.sum())} with valid metadata")

    failures: list[str] = []
    shown = 0
    for hint in config.RECOMMENDATION_SEED_TITLE_HINTS:
        match = find_movie_by_title(catalog, hint, popularity)
        if match is None:
            print(f"\n[hint '{hint}'] no matching title found - skipped")
            continue
        if shown >= config.RECOMMENDATION_REPORT_NUM_SEEDS:
            break
        shown += 1
        recs = recommender.recommend_similar_movies(
            match["movieId"], top_k=config.DEFAULT_TOP_K
        )

        # --- inline guarantee checks ------------------------------------- #
        seed_id = match["movieId"]
        ids = recs["movieId"].tolist() if not recs.empty else []
        if seed_id in ids:
            failures.append(f"seed {seed_id} ({match['title']}) was recommended")
        if len(set(ids)) != len(ids):
            failures.append(f"duplicate movieIds for seed {seed_id}")
        scores = recs["similarity"].tolist() if not recs.empty else []
        if any(a < b for a, b in zip(scores, scores[1:])):
            failures.append(f"scores not descending for seed {seed_id}")
        if len(ids) > config.DEFAULT_TOP_K:
            failures.append(f"more than top_k results for seed {seed_id}")

        print()
        print(f"Seed: #{seed_id}  {match['title']}  [{match['genres']}]")
        print("-" * WIDTH)
        if recs.empty:
            print("  (no recommendations)")
            continue
        for _, row in recs.iterrows():
            year = ""
            info = recommender.describe_seed(row["movieId"])
            if info and "release_year" in info:
                year = f" [{info['release_year']}]"
            print(
                f"  {int(row['rank']):>2}. {str(row['title'])[:56]:<56} "
                f"score={row['similarity']:.4f}  (movieId={row['movieId']}){year}"
            )
    if shown == 0:
        failures.append("no seed hints matched the catalog at all")

    # --- graceful unknown-ID handling ------------------------------------ #
    _print_header("UNKNOWN MOVIE HANDLING (graceful degradation expected)")
    unknown = recommender.recommend_similar_movies(999_999_999)
    ok_unknown = isinstance(unknown, pd.DataFrame) and unknown.empty
    print(f"recommend_similar_movies(999999999) -> {'empty DataFrame' if ok_unknown else 'UNEXPECTED'}")
    if not ok_unknown:
        failures.append("unknown movieId did not return an empty result")

    _print_header("RESULT")
    if failures:
        for message in failures:
            print(f"FAIL: {message}")
        print("SMOKE TEST FAILED")
        return 1
    print("All smoke checks passed:")
    print("  * seeds resolved by searching real catalog titles")
    print("  * ranked recommendations with cosine similarity scores")
    print("  * no self-recommendation, no duplicates, descending order, top-K respected")
    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())