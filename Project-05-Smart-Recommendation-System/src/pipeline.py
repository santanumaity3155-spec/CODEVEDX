"""Module 1 processing pipeline: load -> validate -> profile -> clean ->
analyze -> visualize -> report.

Run from the project root with:

    python -m src.pipeline

The pipeline is deterministic and idempotent: re-running it on the same
raw data reproduces equivalent outputs. Raw data is never modified.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import analysis, config, data_quality as dq, reporting, visualization
from src.data_loader import (
    DataLoadError,
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


if __name__ == "__main__":
    raise SystemExit(main())
