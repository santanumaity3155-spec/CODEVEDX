"""Build and execute notebooks/module_1_data_exploration.ipynb.

Creates the notebook programmatically (so it is reproducible) and runs
it with nbclient so the committed file contains real executed outputs.
"""

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NB_PATH = PROJECT_ROOT / "notebooks" / "module_1_data_exploration.ipynb"

nb = nbf.v4.new_notebook()
cells = []

md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# Module 1 — Data Exploration: MovieLens 25M
**Smart Recommendation System · Data foundation, exploration & preprocessing**

This notebook demonstrates the Module 1 pipeline on the raw MovieLens 25M dataset:
loading, schema inspection, missing values, duplicates, rating/user/movie/genre/tag
analysis, and the final data-quality summary.

Raw files in `data/raw/ml-25m/` are read-only; all outputs live under `data/processed/`
and `outputs/`.""")

code("""import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
for candidate in (PROJECT_ROOT, Path.cwd().resolve()):
    if (candidate / "src").is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from src import config
from src.data_loader import (load_movies, load_ratings, load_tags, load_links,
                             load_genome_scores, load_genome_tags)
from src import analysis, data_quality as dq
from src.data_preprocessor import DataPreprocessor, extract_release_year

import pandas as pd

movies = load_movies()
ratings = load_ratings()
tags = load_tags()
links = load_links()
genome_scores = load_genome_scores()
genome_tags = load_genome_tags()

DATASETS = {"movies": movies, "ratings": ratings, "tags": tags,
            "links": links, "genome-scores": genome_scores, "genome-tags": genome_tags}
print({name: df.shape for name, df in DATASETS.items()})""")

md("## 1–2. Dataset loading & schema inspection")
code("""schema_summary = pd.DataFrame(
    {name: dq.validate_schema(df, name.replace("-", "_"))["valid"] for name, df in {
        "movies": movies, "ratings": ratings, "tags": tags,
        "links": links, "genome_scores": genome_scores, "genome_tags": genome_tags,
    }.items()},
    index=["schema_valid"],
).T
display(schema_summary)

profiles = pd.DataFrame(
    {name: dq.profile_dataset(df, name) for name, df in {
        "movies": movies, "ratings": ratings, "tags": tags,
        "links": links, "genome-scores": genome_scores, "genome-tags": genome_tags,
    }.items()}
).T[["rows", "columns", "memory_mb", "duplicate_rows"]]
profiles""")

md("## 3. Missing-value analysis")
code("""missing_rows = []
for name, df in DATASETS.items():
    for col, info in dq.analyze_missing_values(df, name)["by_column"].items():
        if info["missing"] > 0:
            missing_rows.append({"dataset": name, "column": col,
                                 "missing": info["missing"], "pct": info["pct"]})
missing_df = pd.DataFrame(missing_rows).set_index(["dataset", "column"])
print(f"Total missing cells across all raw datasets: "
      f"{sum(int(df.isna().sum().sum()) for df in DATASETS.values()):,}")
missing_df""")

md("""Missing-value handling rules (applied during cleaning):
- `tags.tag` null/empty → record dropped (a tag without text has no value)
- `tags.timestamp` missing → **retained** as `<NA>` (tag text still valuable)
- `links.imdbId/tmdbId` missing → legitimate, retained as `<NA>`
- `ratings` required fields / `movies.title` → invalid records dropped""")

md("## 4. Duplicate analysis")
code("""dup_movies = dq.analyze_movie_duplicates(movies)
dup_ratings = dq.analyze_rating_duplicates(ratings)
dup_tags = dq.analyze_tag_duplicates(tags)

pd.DataFrame({
    "check": ["movies duplicate movieId", "movies duplicate title (distinct ids)",
              "ratings exact repeats (user+movie+timestamp)",
              "ratings repeated user+movie re-ratings",
              "tags exact repeats",
              "links duplicate movieId"],
    "count": [dup_movies["duplicate_movieId"]["duplicate_records"],
              dup_movies["duplicate_title"]["duplicate_records"],
              dup_ratings["exact_duplicates"]["duplicate_records"],
              dup_ratings["repeated_user_movie_interactions"]["duplicate_records"],
              dup_tags["exact_duplicates"]["duplicate_records"],
              dq.analyze_link_duplicates(links)["duplicate_movieId"]["duplicate_records"]],
}).set_index("check")""")

code("""# Duplicate titles with distinct movieIds are legitimate re-issues - retained.
dup_title_counts = movies["title"].value_counts()
movies[movies["title"].isin(dup_title_counts[dup_title_counts > 1].index)].head(6)""")

md("## 5. Rating analysis")
code("""pd.concat([
    pd.Series(analysis.analyze_ratings(ratings), name="value"),
    pd.Series(analysis.rating_distribution(ratings)["counts_by_value"], name="rating_value"),
]).to_frame().T""")

code("""from src.visualization import plot_rating_distribution
plot_rating_distribution(ratings);""")

md("## 6. User activity analysis")
code("""user_stats = analysis.analyze_users(ratings)
pd.Series(user_stats, name="user statistics").to_frame()""")

code("""from src.visualization import plot_user_activity_distribution
plot_user_activity_distribution(ratings);""")

md("## 7. Movie popularity analysis")
code("""movie_stats = analysis.analyze_movie_popularity(ratings, movies)
{k: v for k, v in movie_stats.items() if k != "top_movies_by_rating_count"}""")

code("""top10 = pd.DataFrame(movie_stats["top_movies_by_rating_count"]).head(10)
top10.set_index("rank")[["title", "rating_count", "mean_rating"]]""")

code("""from src.visualization import plot_movie_popularity_distribution
plot_movie_popularity_distribution(ratings);""")

md("## 8. Genre analysis")
code("""genre_stats = analysis.analyze_genres(movies)
genre_table = pd.DataFrame(genre_stats["movies_by_genre"]).T
genre_table.index.name = "genre"
print(f"distinct genres: {genre_stats['num_distinct_genres']} | "
      f"no-genre movies: {genre_stats['movies_with_no_genre']:,} | "
      f"avg genres/movie: {genre_stats['avg_genres_per_movie']}")
genre_table""")

code("""from src.visualization import plot_genre_distribution
plot_genre_distribution(genre_stats);""")

md("## 9. Tag analysis")
code("""tag_stats = analysis.analyze_tags(tags)
{ k: v for k, v in tag_stats.items() if k != "most_frequent_tags" }""")

code("""from src.visualization import plot_top_tags
plot_top_tags(analysis.tag_frequency_series(tags));""")

md("""## 10. Cleaning & final data-quality summary

Cleaning applies documented rules only; removal counts are reported below.
Genome datasets are validated here but not duplicated into `data/processed`
(~435 MiB) until a later module needs them.""")

code("""pre = DataPreprocessor()
movies_clean = pre.clean_movies(movies)
catalog_ids = pd.Index(movies_clean["movieId"].unique())
ratings_clean = pre.clean_ratings(ratings, valid_movie_ids=catalog_ids)
tags_clean = pre.clean_tags(tags)
links_clean = pre.clean_links(links, valid_movie_ids=catalog_ids)
features_base = pre.build_movies_features_base(movies_clean)
print(f"removed {pre.total_removed} rows total; "
      f"features_base shape: {features_base.shape}")""")

code("""cleaning_df = pd.DataFrame(pre.cleaning_log)
cleaning_df[cleaning_df["records_removed"] > 0].set_index(["dataset", "rule"])""")

md("""The authoritative quality gate lives in the generated reports:
`outputs/reports/dataset_quality_report.txt` and `.json`. The pipeline
(`python -m src.pipeline`) recomputes them and refuses to finish unless every
critical check passes. Final status of the last run:""")

code("""import json
report = json.loads(config.QUALITY_REPORT_JSON_PATH.read_text(encoding="utf-8"))
print(f"Final status: {report['status']}")
pd.DataFrame({
    name: {"rows": report[name]["rows"], "columns": report[name]["columns"]}
    for name in ("movies", "ratings", "tags", "links", "genome_scores", "genome_tags")
})""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

NB_PATH.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, NB_PATH)

client = NotebookClient(nb, timeout=1200, kernel_name="python3",
                        resources={"metadata": {"path": str(NB_PATH.parent)}})
client.execute()
nbf.write(nb, NB_PATH)
print(f"Notebook written & executed: {NB_PATH}")
