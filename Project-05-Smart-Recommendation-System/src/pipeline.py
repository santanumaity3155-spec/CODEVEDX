"""Module 1 processing pipeline: load -> validate -> profile -> clean ->
analyze -> visualize -> report.

Run from the project root with:

    python -m src.pipeline

The pipeline is deterministic and idempotent: re-running it on the same
raw data reproduces equivalent outputs. Raw data is never modified.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src import analysis, config, data_quality as dq, reporting, visualization
from src.data_loader import (
    DataLoadError,
    MissingFileError,
    load_genome_scores,
    load_genome_tags,
    load_links,
    load_movies,
    load_ratings,
    load_tags,
)
from src.data_preprocessor import DataPreprocessor
from src.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def save_csv(df: pd.DataFrame, path: Path) -> Path:
    """Write a processed dataset to CSV (UTF-8, no index)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    size_mb = path.stat().st_size / 1024**2
    logger.info("Saved %s (%d rows, %.1f MiB)", path.name, len(df), size_mb)
    return path


def display_path(path: Path) -> str:
    """Path relative to the project root when possible, else absolute."""
    try:
        return str(path.resolve().relative_to(config.PROJECT_ROOT))
    except ValueError:
        return str(path)


def file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

class ModuleOnePipeline:
    """End-to-end Module 1 data-foundation pipeline."""

    DATASETS = ("movies", "ratings", "tags", "links", "genome_scores", "genome_tags")

    def __init__(self) -> None:
        np.random.seed(config.RANDOM_SEED)
        self.preprocessor = DataPreprocessor()
        self.raw: dict[str, pd.DataFrame] = {}
        self.clean: dict[str, pd.DataFrame] = {}
        self.report: dict = {}

    # -- loading ----------------------------------------------------------
    def load_raw(self) -> None:
        logger.info("=== STEP: raw data loading ===")
        self.raw["movies"] = load_movies()
        self.raw["ratings"] = load_ratings()
        self.raw["tags"] = load_tags()
        self.raw["links"] = load_links()
        self.raw["genome_scores"] = load_genome_scores()
        self.raw["genome_tags"] = load_genome_tags()

    # -- validation & profiling ---------------------------------------------
    def validate_and_profile(self) -> None:
        logger.info("=== STEP: schema validation, profiling and integrity ===")
        profiles = {}
        schema_checks = {}
        missing_reports = {}
        id_checks = {}

        id_columns = {
            "movies": ["movieId"],
            "ratings": ["userId", "movieId"],
            "tags": ["userId", "movieId"],
            "links": ["movieId"],
            "genome_scores": ["movieId", "tagId"],
            "genome_tags": ["tagId"],
        }

        for name in self.DATASETS:
            df = self.raw[name]
            profiles[name] = dq.profile_dataset(df, name)
            schema_checks[name] = dq.validate_schema(df, name)
            missing_reports[name] = dq.analyze_missing_values(df, name)
            id_checks[name] = dq.validate_id_columns(df, id_columns[name], name)

        duplicates = {
            "movies": dq.analyze_movie_duplicates(self.raw["movies"]),
            "ratings": dq.analyze_rating_duplicates(self.raw["ratings"]),
            "tags": dq.analyze_tag_duplicates(self.raw["tags"]),
            "links": dq.analyze_link_duplicates(self.raw["links"]),
            "genome_scores": dq.analyze_genome_score_duplicates(self.raw["genome_scores"]),
        }

        integrity = dq.analyze_referential_integrity(
            movies=self.raw["movies"],
            ratings=self.raw["ratings"],
            tags=self.raw["tags"],
            links=self.raw["links"],
            genome_scores=self.raw["genome_scores"],
            genome_tags=self.raw["genome_tags"],
        )

        rating_domain = dq.validate_rating_domain(self.raw["ratings"])

        self.profiles = profiles
        self.quality_checks = {
            "schema_validation": schema_checks,
            "missing_values": missing_reports,
            "id_validation": id_checks,
            "duplicates": duplicates,
            "referential_integrity": integrity,
            "rating_domain": rating_domain,
        }

    # -- cleaning ------------------------------------------------------------
    def clean_data(self) -> None:
        logger.info("=== STEP: cleaning (documented rules only) ===")
        movie_ids = pd.Index(self.raw["movies"]["movieId"].unique())

        self.clean["movies"] = self.preprocessor.clean_movies(self.raw["movies"])
        catalog_ids = pd.Index(self.clean["movies"]["movieId"].unique())
        self.clean["ratings"] = self.preprocessor.clean_ratings(
            self.raw["ratings"], valid_movie_ids=catalog_ids
        )
        self.clean["tags"] = self.preprocessor.clean_tags(self.raw["tags"])
        self.clean["links"] = self.preprocessor.clean_links(
            self.raw["links"], valid_movie_ids=catalog_ids
        )

    # -- analyses ----------------------------------------------------------------
    def run_analyses(self) -> None:
        logger.info("=== STEP: statistical analyses ===")
        ratings = self.clean["ratings"]

        self.rating_stats = analysis.analyze_ratings(ratings)
        self.rating_dist = analysis.rating_distribution(ratings)
        self.user_stats = analysis.analyze_users(ratings)
        self.movie_stats = analysis.analyze_movie_popularity(ratings, self.clean["movies"])
        self.genre_stats = analysis.analyze_genres(self.clean["movies"])
        self.tag_stats = analysis.analyze_tags(self.clean["tags"])
        self.tag_freq = analysis.tag_frequency_series(self.clean["tags"])

    # -- outputs ---------------------------------------------------------------
    def write_processed_datasets(self) -> list[str]:
        logger.info("=== STEP: writing processed datasets ===")
        saved = [
            save_csv(self.clean["movies"], config.PROCESSED_MOVIES_PATH),
            save_csv(self.clean["ratings"], config.PROCESSED_RATINGS_PATH),
            save_csv(self.clean["tags"], config.PROCESSED_TAGS_PATH),
            save_csv(self.clean["links"], config.PROCESSED_LINKS_PATH),
        ]
        features = self.preprocessor.build_movies_features_base(self.clean["movies"])
        saved.append(save_csv(features, config.PROCESSED_MOVIES_FEATURES_PATH))

        note_path = config.PROCESSED_DATA_DIR / "README.txt"
        note_path.parent.mkdir(parents=True, exist_ok=True)
        note_path.write_text(
            "Processed datasets generated by the Module 1 pipeline "
            "(python -m src.pipeline). Sources are immutable files in data/raw.\n\n"
            "Files:\n"
            "- movies_clean.csv          cleaned movie catalog\n"
            "- ratings_clean.csv         cleaned ratings (valid scale, no exact dupes)\n"
            "- tags_clean.csv            cleaned tags (trimmed, deduplicated)\n"
            "- links_clean.csv           cleaned external-ID links (<NA> kept)\n"
            "- movies_features_base.csv  recommendation-ready base table "
            "(id/title/year/genres + binary genre flags; no ML features yet)\n\n"
            "Genome files are validated but intentionally NOT duplicated into "
            "data/processed at Module 1 (~435 MiB); later modules consume them "
            "directly from data/raw when required.\n",
            encoding="utf-8",
        )
        return [display_path(p) for p in saved]

    def make_charts(self) -> list[str]:
        logger.info("=== STEP: chart generation ===")
        paths = [
            visualization.plot_rating_distribution(self.clean["ratings"]),
            visualization.plot_user_activity_distribution(self.clean["ratings"]),
            visualization.plot_movie_popularity_distribution(self.clean["ratings"]),
            visualization.plot_genre_distribution(self.genre_stats),
            visualization.plot_top_tags(self.tag_freq),
        ]
        return [display_path(p) for p in paths]

    def build_report(self, processed_files: list[str], charts: list[str]) -> dict:
        report = {}
        for name in self.DATASETS:
            entry = dict(self.profiles[name])
            entry["missing_values_total"] = sum(entry["missing_values"].values())
            report[name] = entry

        report.update({
            "status": "PENDING",
            "dataset_name": config.DATASET_NAME,
            "rating_statistics": self.rating_stats,
            "rating_distribution": self.rating_dist,
            "user_statistics": self.user_stats,
            "movie_statistics": self.movie_stats,
            "genre_statistics": self.genre_stats,
            "tag_statistics": self.tag_stats,
            "quality_checks": self.quality_checks,
            "cleaning": self.preprocessor.cleaning_summary(),
            "processed_files": processed_files,
            "charts": charts,
            "processed_dataset_dtypes": {
                name: {c: str(t) for c, t in df.dtypes.items()}
                for name, df in self.clean.items()
            },
        })
        report["status"] = reporting.determine_status(report)
        self.report = report
        logger.info("Quality gate status determined: %s", report["status"])
        if report["status"] != "PASS":
            raise DataLoadError(
                "Critical data-integrity failure detected; refusing to finalize "
                "Module 1 outputs with a FAIL status."
            )
        return report

    # -- orchestration ---------------------------------------------------------
    def run(self) -> dict:
        self.load_raw()
        self.validate_and_profile()
        self.clean_data()
        self.run_analyses()
        processed_files = self.write_processed_datasets()
        charts = self.make_charts()
        report = self.build_report(processed_files, charts)

        json_path, txt_path = reporting.write_quality_reports(report)

        summary = {
            "status": report["status"],
            "records_removed_by_cleaning": self.preprocessor.total_removed,
            "processed_files": processed_files + [str(json_path), str(txt_path)],
            "charts": charts,
        }
        logger.info("Pipeline finished: %s", json.dumps(summary, indent=2))
        return report


def main() -> int:
    try:
        pipeline = ModuleOnePipeline()
        report = pipeline.run()
    except Exception:
        logger.exception("Module 1 pipeline FAILED")
        return 1
    logger.info("MODULE 1 COMPLETE - final quality gate: %s", report["status"])
    return 0


# ===========================================================================
# Module 3 - Recommendation pipeline
# ===========================================================================
class RecommendationPipelineError(Exception):
    """Raised when the Module 3 quality gate fails."""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_regression_tests(timeout_sec: int = 3600) -> dict:
    """Run ``pytest tests -q`` in a subprocess and summarise the outcome."""
    logger.info("Running full regression test suite (python -m pytest tests -q) ...")
    started = time.time()
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q", "--no-header"],
        cwd=str(config.PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )
    duration = time.time() - started
    tail = "\n".join((proc.stdout or "").strip().splitlines()[-3:])
    status = "PASS" if proc.returncode == 0 else "FAIL"
    logger.info("Regression tests finished in %.1fs: %s", duration, status)
    return {
        "status": status,
        "duration_sec": round(duration, 1),
        "summary_tail": tail,
    }


def _wrap_paragraph(text: str, width: int = 72) -> list[str]:
    """Simple deterministic word-wrap used by the TXT renderers."""
    words = text.split()
    wrapped: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > width and current:
            wrapped.append(current)
            current = word
        else:
            current = candidate
    if current:
        wrapped.append(current)
    return wrapped


def render_recommendation_report_text(report: dict) -> str:
    """Human-readable rendering of the recommendation-quality report."""
    width = 78
    lines = [
        "=" * width,
        "MODULE 3 - RECOMMENDATION QUALITY REPORT",
        "=" * width,
        f"Generated (UTC): {report.get('generated_at_utc')}",
        f"Dataset: {config.DATASET_NAME}",
        "",
        "Artifacts loaded:",
        f"  movies in feature index : {report['engine']['n_movies']}",
        f"  TF-IDF features         : {report['engine']['n_features']}",
        f"  matrix non-zeros        : {report['engine']['nnz']}",
        f"  catalog movies          : {report['catalog']['movies']}",
        f"  candidate pool (quality): {report['catalog']['candidates']}",
        f"  min movie ratings       : {report['config']['min_movie_ratings']}",
        f"  top_k                   : {report['config']['top_k']}",
        "",
        "Sample recommendations",
        "-" * width,
    ]
    for sample in report.get("samples", []):
        seed = sample["seed"]
        lines.append(
            f"Seed #{seed['movieId']}: {seed['title']} [{seed.get('genres', '')}]"
        )
        for rec in sample["recommendations"]:
            lines.append(
                f"  {rec['rank']:>2}. {rec['title'][:52]:<52} "
                f"(id={rec['movieId']}, score={rec['similarity']:.4f})"
            )
        if not sample["recommendations"]:
            lines.append("  (no recommendations returned)")
    lines += ["", "Quality checks", "-" * width]
    for check in report.get("quality_checks", []):
        mark = "PASS" if check["passed"] else "FAIL"
        detail = f" - {check['detail']}" if check.get("detail") else ""
        lines.append(f"  [{mark}] {check['check']}{detail}")
    checks = report.get("quality_checks", [])
    failed = [c for c in checks if not c["passed"]]
    lines += [
        "",
        f"Checks passed: {len(checks) - len(failed)}/{len(checks)}",
        f"STATUS: {report.get('status')}",
        "=" * width,
    ]
    return "\n".join(lines) + "\n"


def render_evaluation_report_text(report: dict) -> str:
    """Human-readable rendering of the offline-evaluation report."""
    width = 78
    users = report["users"]
    lines = [
        "=" * width,
        "MODULE 3 - OFFLINE EVALUATION REPORT",
        "=" * width,
        f"Generated (UTC): {report.get('generated_at_utc')}",
        "",
        "Protocol:",
    ]
    for chunk in _wrap_paragraph(report.get("protocol", "")):
        lines.append(f"  {chunk}")
    params = report["params"]
    lines += [
        "",
        "Configuration:",
        f"  K values                : {params['k_values']}",
        f"  like threshold          : {params['like_threshold']}",
        f"  test fraction           : {params['test_fraction']}",
        f"  min test items per user : {params['min_test_items']}",
        f"  min user history        : {params['min_user_ratings']}",
        f"  max evaluated users     : {params['max_users']}",
        f"  random seed             : {params['seed']}",
        f"  popularity filter       : {params['popularity_filter_applied']}",
        "",
        "Users:",
        f"  eligible                : {users['eligible']}",
        f"  evaluated               : {users['evaluated']}",
        f"  skipped (no liked hist) : {users['skipped_no_liked_history']}",
        f"  skipped (no liked test) : {users['skipped_no_relevant_test_items']}",
        "",
        "Metrics (means over evaluated users)",
        "-" * width,
        f"  {'K':>4} {'Precision':>10} {'Recall':>10} {'HitRate':>10} "
        f"{'MAP':>10} {'NDCG':>10}",
    ]
    for k_str, metrics in report.get("metrics_at_k", {}).items():
        lines.append(
            f"  {k_str:>4} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} "
            f"{metrics['hit_rate']:>10.4f} {metrics['map']:>10.4f} "
            f"{metrics['ndcg']:>10.4f}"
        )
    lines += ["", f"STATUS: {report.get('status')}", "=" * width]
    return "\n".join(lines) + "\n"


def render_quality_gate_text(report: dict) -> str:
    """Human-readable rendering of the Module 3 quality-gate report."""
    width = 78
    lines = [
        "=" * width,
        "MODULE 3 - QUALITY GATE REPORT",
        "=" * width,
        f"Generated (UTC): {report.get('generated_at_utc')}",
        f"Regression tests executed: {report.get('regression_tests_executed')}",
    ]
    if report.get("regression_tests"):
        tests = report["regression_tests"]
        lines.append(f"Regression result: {tests['status']} ({tests['duration_sec']}s)")
    lines += ["", "Checks", "-" * width]
    for check in report.get("quality_checks", []):
        mark = "PASS" if check["passed"] else "FAIL"
        detail = f" - {check['detail']}" if check.get("detail") else ""
        lines.append(f"  [{mark}] {check['check']}{detail}")
    lines += ["", f"GATE: {report.get('gate')}", "=" * width]
    return "\n".join(lines) + "\n"


class RecommendationPipeline:
    """End-to-end Module 3 pipeline: artifacts -> engine -> recommendations
    -> evaluation -> quality gate -> reports.

    Consumes the Module 1 processed datasets and the Module 2 feature
    artifacts; never re-fits the TF-IDF vectorizer and never touches
    ``data/raw``. Run from the project root with::

        python -m src.pipeline --module 3 [--skip-regression-tests]
    """

    def __init__(self, *, run_regression_tests: bool = True) -> None:
        self.run_regression_tests_enabled = bool(run_regression_tests)
        self.recommender = None
        self.ratings: pd.DataFrame | None = None
        self.popularity: pd.DataFrame | None = None
        self.samples: list[dict] = []
        self.quality_checks: list[dict] = []
        self.evaluation_report: dict | None = None

    # -- loading ---------------------------------------------------------- #
    def load_inputs(self) -> None:
        """Load Module 1 processed data + Module 2 feature artifacts."""
        from src.feature_engineering import (  # lazy: avoids import cycle
            load_feature_vectorizer,
            load_sparse_matrix,
        )
        from src.recommender import ContentRecommender, compute_popularity_stats
        from src.similarity_engine import SimilarityEngine

        logger.info("=== STEP: artifact loading ===")
        for label, path in (
            ("movie content features", config.PROCESSED_MOVIE_CONTENT_FEATURES_PATH),
            ("TF-IDF matrix", config.PROCESSED_MOVIE_TFIDF_PATH),
            ("feature index", config.PROCESSED_MOVIE_FEATURE_INDEX_PATH),
            ("TF-IDF vectorizer", config.MOVIE_TFIDF_VECTORIZER_PATH),
            ("processed ratings", config.PROCESSED_RATINGS_PATH),
        ):
            if not Path(path).is_file():
                raise MissingFileError(
                    f"Required {label} artifact not found: '{path}'. Run the "
                    "Module 1 and Module 2 pipelines first."
                )

        catalog = pd.read_csv(config.PROCESSED_MOVIE_CONTENT_FEATURES_PATH)
        matrix = load_sparse_matrix(config.PROCESSED_MOVIE_TFIDF_PATH)
        feature_index = pd.read_csv(config.PROCESSED_MOVIE_FEATURE_INDEX_PATH)
        vectorizer = load_feature_vectorizer(config.MOVIE_TFIDF_VECTORIZER_PATH)

        # Vectorizer/matrix compatibility: cheap dimension gate plus the
        # exact Module 2 reproduction check on the stored documents.
        if len(getattr(vectorizer, "vocabulary_", {})) != matrix.shape[1]:
            raise RecommendationPipelineError(
                f"Vectorizer vocabulary ({len(getattr(vectorizer, 'vocabulary_', {}))} "
                f"terms) does not match the feature matrix width "
                f"({matrix.shape[1]}); regenerate the Module 2 artifacts."
            )
        if "content_text" in catalog.columns:
            from src.feature_engineering import verify_loaded_vectorizer

            if not verify_loaded_vectorizer(
                vectorizer, catalog["content_text"].astype(str).tolist(), matrix
            ):
                raise RecommendationPipelineError(
                    "The loaded TF-IDF vectorizer does not reproduce the "
                    "stored feature matrix; regenerate the Module 2 artifacts."
                )
        logger.info(
            "Vectorizer/matrix compatibility verified (%d x %d).",
            matrix.shape[0],
            matrix.shape[1],
        )

        if len(feature_index) != matrix.shape[0]:
            raise RecommendationPipelineError(
                f"Feature index has {len(feature_index)} rows but the TF-IDF "
                f"matrix has {matrix.shape[0]}; artifacts are inconsistent."
            )

        logger.info("Loading processed ratings for popularity/evaluation ...")
        self.ratings = pd.read_csv(
            config.PROCESSED_RATINGS_PATH, dtype=config.PROCESSED_RATINGS_DTYPE
        )
        logger.info(
            "Loaded ratings_clean: %d interactions for %d users",
            len(self.ratings),
            self.ratings["userId"].nunique(),
        )
        self.popularity = compute_popularity_stats(self.ratings)

        engine = SimilarityEngine(matrix, feature_index)
        self.recommender = ContentRecommender(
            engine=engine,
            catalog=catalog[["movieId", "title", "genres", "release_year"]],
            popularity=self.popularity,
            min_movie_ratings=config.RECOMMENDATION_MIN_MOVIE_RATINGS,
        )
        logger.info(
            "Recommendation engine initialised: %d movies in index",
            engine.n_movies,
        )

    # -- recommendation samples -------------------------------------------- #
    def generate_samples(self) -> list[dict]:
        """Produce showcase recommendations for real, well-known seed movies."""
        from src.recommender import find_movie_by_title

        logger.info("=== STEP: sample recommendation generation ===")
        catalog = self.recommender.catalog
        seeds: list[dict] = []
        for hint in config.RECOMMENDATION_SEED_TITLE_HINTS:
            match = find_movie_by_title(catalog, hint, self.popularity)
            if match is None:
                logger.warning("No catalog title matched hint %r; skipped.", hint)
                continue
            if any(s["movieId"] == match["movieId"] for s in seeds):
                continue
            seeds.append(match)
            if len(seeds) >= config.RECOMMENDATION_REPORT_NUM_SEEDS:
                break
        if not seeds:
            raise RecommendationPipelineError(
                "None of the configured seed title hints matched the catalog; "
                "check RECOMMENDATION_SEED_TITLE_HINTS in src/config.py."
            )

        self.samples = []
        for seed in seeds:
            recs = self.recommender.recommend_similar_movies(
                seed["movieId"], top_k=config.DEFAULT_TOP_K
            )
            self.samples.append(
                {"seed": seed, "recommendations": recs.to_dict(orient="records")}
            )
            logger.info("Seed '%s' -> %d recommendations", seed["title"], len(recs))
        return self.samples

    def run_evaluation(self) -> dict:
        """Run the leakage-free offline evaluation on the ratings data."""
        from src.evaluation import ContentBasedEvaluator

        logger.info("=== STEP: offline evaluation ===")
        evaluator = ContentBasedEvaluator(self.recommender.engine, self.ratings)
        self.evaluation_report = evaluator.evaluate()
        return self.evaluation_report

    # -- quality checks ----------------------------------------------------- #
    def run_quality_checks(self) -> list[dict]:
        """Verify every Module 3 guarantee; each entry records pass/fail."""
        from src.evaluation import METRIC_NAMES
        from src.similarity_engine import MovieNotFoundError

        logger.info("=== STEP: Module 3 quality checks ===")
        checks: list[dict] = []

        def record(name: str, passed: bool, detail: str = "") -> None:
            checks.append({"check": name, "passed": bool(passed), "detail": detail})

        engine = self.recommender.engine

        # Artifacts / dimensions / vectorizer compatibility.
        dims_ok = engine.n_movies > 0 and engine.n_features > 0
        record(
            "engine_loaded_and_dimensions_agree",
            dims_ok,
            f"matrix={engine.n_movies}x{engine.n_features}",
        )
        record(
            "candidate_pool_non_empty",
            int(self.recommender._valid_metadata.sum()) > 0,
            f"candidates={int(self.recommender._valid_metadata.sum())}",
        )

        # Recommendation behaviour on the generated samples.
        top_k = config.DEFAULT_TOP_K
        all_within_top_k = bool(self.samples) and all(
            0 < len(s["recommendations"]) <= top_k for s in self.samples
        )
        record(
            "top_k_respected",
            all_within_top_k,
            f"top_k={top_k}, samples={len(self.samples)}",
        )

        no_self = all(
            s["seed"]["movieId"] not in {r["movieId"] for r in s["recommendations"]}
            for s in self.samples
        )
        record("self_recommendation_excluded", no_self)

        no_dupes = all(
            len({r["movieId"] for r in s["recommendations"]})
            == len(s["recommendations"])
            for s in self.samples
        )
        record("duplicate_recommendations_absent", no_dupes)

        sorted_ok = True
        for s in self.samples:
            scores = [r["similarity"] for r in s["recommendations"]]
            sorted_ok &= all(a >= b for a, b in zip(scores, scores[1:]))
        record("recommendations_sorted_descending", sorted_ok)

        metadata_ok = all(
            isinstance(r.get("title"), str) and r["title"].strip()
            for s in self.samples
            for r in s["recommendations"]
        )
        record("recommended_metadata_valid", metadata_ok)

        unknown_id = 10_000_000
        empty_result = self.recommender.recommend_similar_movies(unknown_id)
        record(
            "unknown_movie_handled_gracefully",
            isinstance(empty_result, pd.DataFrame) and empty_result.empty,
            "unknown movieId returned an empty frame instead of raising",
        )

        try:
            engine.row_for_movie(unknown_id)
            raised = False
        except MovieNotFoundError:
            raised = True
        record(
            "strict_engine_raises_movie_not_found",
            raised,
            "SimilarityEngine raises MovieNotFoundError for unknown IDs",
        )

        # Evaluation metrics executed.
        metrics = (self.evaluation_report or {}).get("metrics_at_k", {})
        eval_ok = bool(metrics) and all(
            set(m.keys()) == set(METRIC_NAMES)
            and all(0.0 <= v <= 1.0 for v in m.values())
            for m in metrics.values()
        )
        record(
            "evaluation_metrics_executed",
            eval_ok,
            f"k_values={sorted(metrics)}" if metrics else "evaluation missing",
        )
        self.quality_checks = checks
        failed = [c for c in checks if not c["passed"]]
        logger.info(
            "Module 3 quality checks: %d/%d passed",
            len(checks) - len(failed),
            len(checks),
        )
        return checks

    # -- reports ------------------------------------------------------------ #
    def build_recommendation_report(self) -> dict:
        """Assemble the recommendation-quality report dictionary."""
        engine = self.recommender.engine
        candidates = int(self.recommender._valid_metadata.sum())
        if self.popularity is not None and self.recommender.min_movie_ratings > 0:
            candidates = int(
                (
                    self.recommender._rating_counts
                    >= self.recommender.min_movie_ratings
                ).sum()
            )
        return {
            "module": "Module 3 - Content-Based Recommendation Engine",
            "generated_at_utc": _utc_now_iso(),
            "dataset_name": config.DATASET_NAME,
            "engine": {
                "n_movies": engine.n_movies,
                "n_features": engine.n_features,
                "nnz": int(engine._matrix.nnz),
            },
            "catalog": {
                "movies": len(self.recommender.catalog),
                "candidates": candidates,
            },
            "config": {
                "top_k": config.DEFAULT_TOP_K,
                "min_movie_ratings": self.recommender.min_movie_ratings,
                "seed_title_hints": list(config.RECOMMENDATION_SEED_TITLE_HINTS),
                "num_seeds": config.RECOMMENDATION_REPORT_NUM_SEEDS,
            },
            "samples": self.samples,
            "quality_checks": self.quality_checks,
            "status": "PENDING",
        }

    def run(self) -> dict:
        """Execute the full Module 3 pipeline; raises on gate failure."""
        logger.info("=== MODULE 3 PIPELINE START ===")
        self.load_inputs()
        self.generate_samples()
        evaluation = self.run_evaluation()
        self.run_quality_checks()

        recommendation_report = self.build_recommendation_report()

        regression: dict | None = None
        if self.run_regression_tests_enabled:
            regression = run_regression_tests()
            recommendation_report["quality_checks"].append({
                "check": "module1_module2_regression_tests",
                "passed": regression["status"] == "PASS",
                "detail": (
                    regression["summary_tail"].splitlines()[-1]
                    if regression["summary_tail"]
                    else ""
                ),
            })

        # Serialize reports (JSON via shared writer, TXT via renderers).
        rec_json = reporting.write_json_report(
            recommendation_report, config.RECOMMENDATION_REPORT_JSON_PATH
        )
        eval_payload = dict(evaluation)
        eval_payload["generated_at_utc"] = _utc_now_iso()
        eval_payload["status"] = "PENDING"
        eval_json = reporting.write_json_report(
            eval_payload, config.EVALUATION_REPORT_JSON_PATH
        )

        all_checks_pass = all(
            c["passed"] for c in recommendation_report["quality_checks"]
        )
        recommendation_report["status"] = "PASS" if all_checks_pass else "FAIL"
        eval_payload["status"] = (
            "PASS" if evaluation["users"]["evaluated"] > 0 else "FAIL"
        )
        gate = (
            "PASS"
            if (all_checks_pass and eval_payload["status"] == "PASS")
            else "FAIL"
        )
        # Re-serialise with final statuses.
        with open(rec_json, "w", encoding="utf-8") as fh:
            json.dump(recommendation_report, fh, indent=2, ensure_ascii=False)
        with open(eval_json, "w", encoding="utf-8") as fh:
            json.dump(eval_payload, fh, indent=2, ensure_ascii=False)

        gate_report = {
            "module": "Module 3 - Quality Gate",
            "generated_at_utc": _utc_now_iso(),
            "regression_tests_executed": regression is not None,
            "regression_tests": regression,
            "quality_checks": recommendation_report["quality_checks"],
            "evaluation_status": eval_payload["status"],
            "gate": gate,
        }
        reporting.write_json_report(gate_report, config.MODULE3_QUALITY_GATE_JSON_PATH)
        config.RECOMMENDATION_REPORT_TXT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(config.RECOMMENDATION_REPORT_TXT_PATH, "w", encoding="utf-8") as fh:
            fh.write(render_recommendation_report_text(recommendation_report))
        with open(config.EVALUATION_REPORT_TXT_PATH, "w", encoding="utf-8") as fh:
            fh.write(render_evaluation_report_text(eval_payload))
        with open(config.MODULE3_QUALITY_GATE_TXT_PATH, "w", encoding="utf-8") as fh:
            fh.write(render_quality_gate_text(gate_report))

        logger.info(
            "Reports written: %s, %s, %s (+ .txt variants)",
            display_path(rec_json),
            display_path(eval_json),
            display_path(config.MODULE3_QUALITY_GATE_JSON_PATH),
        )

        if gate != "PASS":
            raise RecommendationPipelineError(
                "Module 3 quality gate FAILED - see "
                f"{display_path(config.MODULE3_QUALITY_GATE_TXT_PATH)} for details."
            )
        logger.info("MODULE 3 COMPLETE - quality gate: %s", gate)
        return gate_report


def main_module_three(run_regression_tests_flag: bool = True) -> int:
    """CLI entry point for the Module 3 recommendation pipeline."""
    try:
        pipeline = RecommendationPipeline(run_regression_tests=run_regression_tests_flag)
        pipeline.run()
    except Exception:
        logger.exception("Module 3 pipeline FAILED")
        return 1
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Project 5 processing pipelines")
    parser.add_argument(
        "--module",
        type=int,
        choices=(1, 3),
        default=1,
        help="Pipeline to run: 1 = data foundation (default), 3 = recommendations.",
    )
    parser.add_argument(
        "--skip-regression-tests",
        action="store_true",
        help="Module 3 only: skip the full pytest regression stage.",
    )
    args = parser.parse_args()
    if args.module == 1:
        raise SystemExit(main())
    raise SystemExit(main_module_three(not args.skip_regression_tests))
