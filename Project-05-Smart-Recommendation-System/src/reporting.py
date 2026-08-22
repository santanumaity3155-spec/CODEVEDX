"""Quality-report generation (human-readable TXT + machine-readable JSON).

The pipeline assembles a single ``report`` dictionary; this module
serializes it into ``outputs/reports/dataset_quality_report.json`` and
renders ``dataset_quality_report.txt`` from the exact same values so the
two artifacts can never disagree.
"""

from __future__ import annotations

import json
from pathlib import Path

from src import config
from src.logging_config import get_logger

logger = get_logger(__name__)

_WIDTH = 72


def _fmt_int(value) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_float(value, digits: int = 4) -> str:
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _line(char: str = "-") -> str:
    return char * _WIDTH


def _section(title: str) -> list[str]:
    return ["", title, _line()]


def determine_status(report: dict) -> str:
    """Final PASS/FAIL gate based on critical integrity checks only.

    FAIL when any of the following hold (everything else - duplicates,
    orphans, missing external IDs - is reported and either cleaned or
    documented):
      * any raw schema validation failed
      * rating-domain violations exist
      * ID-column violations exist
      * any dataset is empty after loading
    """
    quality = report.get("quality_checks", {})

    schema = quality.get("schema_validation", {})
    if isinstance(schema, dict):
        for result in schema.values():
            if isinstance(result, dict) and not result.get("valid", True):
                return "FAIL"

    rating_domain = quality.get("rating_domain", {})
    if rating_domain and not rating_domain.get("valid", True):
        return "FAIL"

    id_validation = quality.get("id_validation", {})
    if isinstance(id_validation, dict):
        for result in id_validation.values():
            if isinstance(result, dict) and not result.get("valid", True):
                return "FAIL"

    datasets = {
        k: v for k, v in report.items()
        if isinstance(v, dict) and "rows" in v
    }
    if any(info.get("rows", 0) == 0 for info in datasets.values()):
        return "FAIL"

    return "PASS"


# ---------------------------------------------------------------------------
# Text rendering
# ---------------------------------------------------------------------------

def render_text_report(report: dict) -> str:
    """Render the full quality report as formatted plain text."""
    lines: list[str] = [
        "=" * _WIDTH,
        "MOVIELENS 25M - MODULE 1 DATASET QUALITY REPORT",
        "=" * _WIDTH,
        f"Dataset: {config.DATASET_NAME}",
        f"Raw directory: {config.RAW_DATA_DIR}",
        f"Random seed: {config.RANDOM_SEED}",
    ]

    # --- Dataset overview -------------------------------------------------
    lines += _section("Dataset overview")
    for name in ("movies", "ratings", "tags", "links", "genome_scores", "genome_tags"):
        info = report.get(name)
        if not info:
            continue
        lines.append(f"{name}:")
        lines.append(f"  rows:             {_fmt_int(info.get('rows'))}")
        lines.append(f"  columns:          {_fmt_int(info.get('columns'))} "
                     f"({', '.join(info.get('column_names', []))})")
        lines.append(f"  memory (MiB):     {_fmt_float(info.get('memory_mb'), 2)}")

    # --- Data quality -----------------------------------------------------
    lines += _section("Data quality")
    quality = report.get("quality_checks", {})

    lines.append("Schema validation:")
    for ds_name, result in quality.get("schema_validation", {}).items():
        status = "PASS" if result.get("valid") else "FAIL"
        extra = "" if result.get("valid") else f" missing={result.get('missing_columns')}"
        lines.append(f"  {ds_name}: {status}{extra}")

    total_missing = 0
    for info in report.values():
        if isinstance(info, dict) and "rows" in info and isinstance(info.get("missing_values"), dict):
            total_missing += sum(int(v) for v in info["missing_values"].values())
    lines.append(f"Missing cells across raw datasets: {_fmt_int(total_missing)}")

    dupes = quality.get("duplicates", {})
    lines.append("Duplicates (by documented rule):")
    for ds_name, group in dupes.items():
        if isinstance(group, dict):
            for rule_name, res in group.items():
                if isinstance(res, dict) and "duplicate_records" in res:
                    lines.append(f"  {ds_name}.{rule_name}: "
                                 f"{_fmt_int(res['duplicate_records'])}")
    ri = quality.get("referential_integrity", {})
    lines.append(f"Orphan records (all relationships): {_fmt_int(ri.get('total_orphan_records', 0))}")
    for check in ri.get("checks", []):
        lines.append(f"  {check['relationship']}: {_fmt_int(check['orphan_records'])} "
                     f"({check['orphan_pct']}%)")

    rd = quality.get("rating_domain", {})
    if rd:
        lines.append("Rating domain validation:")
        lines.append(f"  out-of-range: {_fmt_int(rd.get('out_of_range', 0))}")
        lines.append(f"  off half-star grid: {_fmt_int(rd.get('off_half_star_grid', 0))}")

    idv = quality.get("id_validation", {})
    lines.append("ID validation:")
    for ds_name, res in idv.items():
        suffix = "" if res.get("valid") else f" -> {res.get('issues')}"
        lines.append(f"  {ds_name}: {'PASS' if res.get('valid') else 'ISSUES'}{suffix}")

    # --- Rating statistics --------------------------------------------------
    rs = report.get("rating_statistics", {})
    lines += _section("Rating statistics")
    lines.append(f"total ratings: {_fmt_int(rs.get('total_ratings'))}")
    lines.append(f"mean:   {_fmt_float(rs.get('mean'))}")
    lines.append(f"median: {_fmt_float(rs.get('median'))}")
    lines.append(f"min:    {_fmt_float(rs.get('min'), 1)}")
    lines.append(f"max:    {_fmt_float(rs.get('max'), 1)}")
    lines.append(f"std:    {_fmt_float(rs.get('std'))}")
    dist = report.get("rating_distribution", {})
    if dist:
        lines.append("counts by value:")
        for value, count in dist.get("counts_by_value", {}).items():
            lines.append(f"  {value}-star: {_fmt_int(count)}")

    # --- User statistics ----------------------------------------------------
    us = report.get("user_statistics", {})
    lines += _section("User statistics")
    lines.append(f"unique users: {_fmt_int(us.get('unique_users'))}")
    lines.append(f"ratings/user min:    {_fmt_int(us.get('ratings_per_user_min'))}")
    lines.append(f"ratings/user max:    {_fmt_int(us.get('ratings_per_user_max'))}")
    lines.append(f"ratings/user mean:   {_fmt_float(us.get('ratings_per_user_mean'), 2)}")
    lines.append(f"ratings/user median: {_fmt_float(us.get('ratings_per_user_median'), 1)}")
    lines.append(f"highly active users (>= {us.get('high_activity_threshold_99th_pct')}): "
                 f"{_fmt_int(us.get('highly_active_users_ge_99th_pct'))}")
    lines.append(f"low activity users (<= {us.get('low_activity_threshold_1st_pct')}): "
                 f"{_fmt_int(us.get('low_activity_users_le_1st_pct'))}")

    # --- Movie statistics ---------------------------------------------------
    ms = report.get("movie_statistics", {})
    lines += _section("Movie statistics")
    catalog = report.get("movies", {})
    lines.append(f"movies in catalog: {_fmt_int(catalog.get('rows'))}")
    lines.append(f"unique movies rated: {_fmt_int(ms.get('unique_movies_rated'))}")
    lines.append(f"ratings/movie min:    {_fmt_int(ms.get('ratings_per_movie_min'))}")
    lines.append(f"ratings/movie max:    {_fmt_int(ms.get('ratings_per_movie_max'))}")
    lines.append(f"ratings/movie mean:   {_fmt_float(ms.get('ratings_per_movie_mean'), 2)}")
    lines.append(f"ratings/movie median: {_fmt_float(ms.get('ratings_per_movie_median'), 1)}")
    top = ms.get("top_movies_by_rating_count", [])
    if top:
        lines.append(f"top movies by rating count (first 10 of {len(top)}):")
        for entry in top[:10]:
            lines.append(f"  #{entry['rank']:>3} {_fmt_int(entry['rating_count']):>9} ratings | "
                         f"avg {entry['mean_rating']:.2f} | {entry['title']}")

    # --- Genre statistics -----------------------------------------------------
    gs = report.get("genre_statistics", {})
    lines += _section("Genre statistics")
    lines.append(f"distinct genres: {gs.get('num_distinct_genres')}")
    lines.append(f"movies with '{config.NO_GENRES_LABEL}': {_fmt_int(gs.get('movies_with_no_genre'))}")
    lines.append(f"average genres per movie: {_fmt_float(gs.get('avg_genres_per_movie'), 2)}")
    lines.append("movies per genre:")
    for g, info in gs.get("movies_by_genre", {}).items():
        lines.append(f"  {g:<28} {_fmt_int(info['count']):>8} ({info['pct_of_movies']:.2f}% of movies)")

    # --- Tag statistics -------------------------------------------------------
    ts = report.get("tag_statistics", {})
    if ts:
        lines += _section("Tag statistics")
        lines.append(f"tag records: {_fmt_int(ts.get('total_tag_records'))}")
        lines.append(f"unique users tagging: {_fmt_int(ts.get('unique_users_tagging'))}")
        lines.append(f"unique movies tagged: {_fmt_int(ts.get('unique_movies_tagged'))}")
        lines.append(f"unique tags (case-insensitive): {_fmt_int(ts.get('unique_tags_case_insensitive'))}")
        lines.append("most frequent tags (top 10):")
        for t, c in list(ts.get("most_frequent_tags", {}).items())[:10]:
            lines.append(f"  {t:<30} {_fmt_int(c)}")

    # --- Final cleaning ---------------------------------------------------------
    cl = report.get("cleaning", {})
    lines += _section("Final cleaning")
    lines.append(f"total records removed: {_fmt_int(cl.get('total_removed'))}")
    for entry in cl.get("rules", []):
        lines.append(f"  [{entry['dataset']}] {entry['rule']}: removed "
                     f"{_fmt_int(entry['records_removed'])}")
    lines.append("processed files generated:")
    for path in report.get("processed_files", []):
        lines.append(f"  {path}")

    # --- Final status -------------------------------------------------------------
    lines += _section("Final status")
    lines.append(f"STATUS: {report.get('status')}")
    lines.append("=" * _WIDTH)

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def write_json_report(report: dict, path: Path | None = None) -> Path:
    """Serialize the full report dictionary to JSON."""
    path = path or config.QUALITY_REPORT_JSON_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    logger.info("Wrote JSON quality report to %s", path)
    return path


def write_text_report(report: dict, path: Path | None = None) -> Path:
    """Render and write the plain-text quality report."""
    path = path or config.QUALITY_REPORT_TXT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    text = render_text_report(report)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    logger.info("Wrote text quality report to %s", path)
    return path


def write_quality_reports(report: dict) -> tuple[Path, Path]:
    """Write both report variants; returns (json_path, txt_path)."""
    return write_json_report(report), write_text_report(report)
