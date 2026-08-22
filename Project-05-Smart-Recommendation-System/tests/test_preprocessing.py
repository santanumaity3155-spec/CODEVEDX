"""Tests for src/data_preprocessor.py, the end-to-end pipeline, and the
generated processed artifacts.

Isolation: the end-to-end test runs entirely inside ``tmp_path`` with
monkeypatched ``src.config`` paths - it never touches ``data/raw`` or
the real processed outputs. Tests for real processed artifacts are
read-only and assume ``python -m src.pipeline`` has been executed
beforehand (documented run order).
"""

from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from src import config
from src.data_preprocessor import DataPreprocessor, extract_release_year


# ---------------------------------------------------------------------------
# Movie cleaning rules
# ---------------------------------------------------------------------------

class TestMovieCleaning:
    def test_blank_or_missing_title_dropped(self, sample_movies_df):
        bad = pd.DataFrame({
            "movieId": [5, 6],
            "title": pd.array(["   ", None], dtype="string"),
            "genres": ["Action", "Drama"],
        })
        movies = pd.concat([sample_movies_df, bad], ignore_index=True)
        clean = DataPreprocessor().clean_movies(movies)
        assert 5 not in set(clean["movieId"])
        assert 6 not in set(clean["movieId"])
        assert len(clean) == len(sample_movies_df)

    def test_missing_genres_becomes_no_genres_label(self):
        movies = pd.DataFrame({
            "movieId": [1],
            "title": ["X (2000)"],
            "genres": pd.array([None], dtype="string"),
        })
        clean = DataPreprocessor().clean_movies(movies)
        assert clean.loc[0, "genres"] == config.NO_GENRES_LABEL

    def test_whitespace_stripped(self):
        movies = pd.DataFrame({
            "movieId": [1],
            "title": ["  Spaced Out (1990)  "],
            "genres": [" Comedy | Drama  "],
        })
        clean = DataPreprocessor().clean_movies(movies)
        assert clean.loc[0, "title"] == "Spaced Out (1990)"
        assert clean.loc[0, "genres"] == "Comedy | Drama"

    def test_duplicate_movie_id_removed_keep_first(self, sample_movies_df):
        dup = pd.concat([sample_movies_df, sample_movies_df.iloc[[1]]],
                        ignore_index=True)
        clean = DataPreprocessor().clean_movies(dup)
        assert len(clean) == len(sample_movies_df)
        assert clean["movieId"].is_unique

    def test_duplicate_titles_with_distinct_ids_retained(self):
        movies = pd.DataFrame({
            "movieId": [1, 2],
            "title": ["Same (2000)", "Same (2000)"],
            "genres": ["Action", "Drama"],
        })
        clean = DataPreprocessor().clean_movies(movies)
        assert len(clean) == 2

    def test_dtypes_preserved(self, sample_movies_df):
        clean = DataPreprocessor().clean_movies(sample_movies_df)
        assert str(clean["movieId"].dtype) == "int32"


# ---------------------------------------------------------------------------
# Rating cleaning rules
# ---------------------------------------------------------------------------

def _ratings_frame(rows: list[tuple]) -> pd.DataFrame:
    """Build a ratings frame that may contain nulls (nullable dtypes)."""
    return pd.DataFrame(
        rows,
        columns=["userId", "movieId", "rating", "timestamp"],
    ).astype({"userId": "Int64", "movieId": "Int64",
              "rating": "Float64", "timestamp": "Int64"})


class TestRatingCleaning:
    def test_invalid_rows_removed_legitimate_re_rating_kept(self):
        ratings = _ratings_frame([
            (10, 1, 4.5, 100),
            (10, 1, 4.5, 100),   # exact duplicate -> removed
            (10, 1, 3.5, 200),   # legitimate re-rating -> kept
            (11, 1, 4.7, 300),   # off-grid -> removed
            (11, 1, 6.0, 400),   # out of range -> removed
            (12, 1, None, 500),  # missing rating -> removed
        ])
        clean = DataPreprocessor().clean_ratings(ratings)
        assert sorted(clean["timestamp"].tolist()) == [100, 200]

    def test_null_required_fields_removed(self):
        ratings = _ratings_frame([
            (10, 1, 4.0, 100),
            (None, 1, 4.0, 110),
            (10, None, 4.0, 120),
            (10, 1, 4.0, None),
        ])
        clean = DataPreprocessor().clean_ratings(ratings)
        assert len(clean) == 1

    def test_orphan_movie_ids_removed_against_catalog(self):
        ratings = _ratings_frame([
            (10, 1, 4.0, 100),
            (11, 999, 5.0, 200),
        ])
        clean = DataPreprocessor().clean_ratings(ratings, valid_movie_ids={1})
        assert clean["movieId"].tolist() == [1]

    def test_clean_output_dtypes(self, sample_ratings_df):
        clean = DataPreprocessor().clean_ratings(sample_ratings_df)
        assert str(clean["userId"].dtype) == "int32"
        assert str(clean["movieId"].dtype) == "int32"
        assert str(clean["rating"].dtype) == "float32"
        assert str(clean["timestamp"].dtype) == "int64"


# ---------------------------------------------------------------------------
# Tag cleaning rules
# ---------------------------------------------------------------------------

class TestTagCleaning:
    def test_trim_and_drop_empty_and_null_tags(self, sample_tags_df):
        extra = pd.DataFrame({
            "userId": [12, 13],
            "movieId": [3, 4],
            "tag": pd.array(["  ", None], dtype="string"),
            "timestamp": [300, 400],
        })
        tags = pd.concat([sample_tags_df, extra], ignore_index=True)
        clean = DataPreprocessor().clean_tags(tags)
        assert clean["tag"].tolist() == ["sci-fi", "classic", "thought-provoking"]
        assert clean.loc[clean["tag"] == "thought-provoking", "timestamp"].isna().all()

    def test_exact_duplicates_removed(self):
        tags = pd.DataFrame({
            "userId": [1, 1],
            "movieId": [5, 5],
            "tag": ["fun", " fun"],
            "timestamp": [100, 100],
        })
        # after trimming both rows become identical -> one removed
        clean = DataPreprocessor().clean_tags(tags)
        assert len(clean) == 1

    def test_missing_timestamps_reported_and_retained(self, sample_tags_df):
        pre = DataPreprocessor()
        pre.clean_tags(sample_tags_df)
        rule = next(r for r in pre.cleaning_log
                    if r["rule"].startswith("retain missing timestamps"))
        assert rule is not None


# ---------------------------------------------------------------------------
# Link cleaning rules
# ---------------------------------------------------------------------------

class TestLinkCleaning:
    def test_missing_external_ids_retained_as_na(self, sample_links_df):
        clean = DataPreprocessor().clean_links(sample_links_df)
        row = clean.loc[clean["movieId"] == 3]
        assert len(row) == 1
        assert pd.isna(row["imdbId"].iloc[0])

    def test_duplicate_movie_id_removed(self, sample_links_df):
        links = pd.concat([sample_links_df, sample_links_df.iloc[[0]]],
                          ignore_index=True)
        clean = DataPreprocessor().clean_links(links)
        assert clean["movieId"].is_unique


# ---------------------------------------------------------------------------
# Recommendation-ready features base
# ---------------------------------------------------------------------------

class TestFeaturesBase:
    def test_year_extraction(self):
        assert extract_release_year("Toy Story (1995)") == 1995
        assert extract_release_year("No Year Here") is None
        assert extract_release_year(None) is None

    def test_genre_flags_and_counts(self, sample_movies_df):
        pre = DataPreprocessor()
        feats = pre.build_movies_features_base(pre.clean_movies(sample_movies_df))
        alpha = feats.loc[feats["movieId"] == 1].iloc[0]
        gamma = feats.loc[feats["movieId"] == 3].iloc[0]
        assert alpha["release_year"] == 2001
        assert alpha["num_genres"] == 2
        assert alpha["genre_Action"] == 1 and alpha["genre_Comedy"] == 1
        assert gamma["num_genres"] == 0
        assert gamma["genre_(no_genres_listed)"] == 1
        assert len(feats) == len(sample_movies_df)

    def test_no_row_count_changes(self, sample_movies_df):
        pre = DataPreprocessor()
        feats = pre.build_movies_features_base(pre.clean_movies(sample_movies_df))
        entry = next(e for e in pre.cleaning_log
                     if e["dataset"] == "movies_features_base")
        assert entry["records_removed"] == 0
        assert entry["records_retained"] == len(feats)


# ---------------------------------------------------------------------------
# Determinism / reproducibility of the cleaning rules
# ---------------------------------------------------------------------------

class TestReproducibilityOfCleaning:
    def test_repeated_cleaning_is_identical(self, sample_movies_df, sample_ratings_df):
        p1, p2 = DataPreprocessor(), DataPreprocessor()
        c1 = p1.clean_ratings(p1.clean_movies(sample_movies_df).pipe(
            lambda m: sample_ratings_df[sample_ratings_df["movieId"].isin(m["movieId"])])
        )
        c2 = p2.clean_ratings(p2.clean_movies(sample_movies_df).pipe(
            lambda m: sample_ratings_df[sample_ratings_df["movieId"].isin(m["movieId"])])
        )
        assert c1.equals(c2)


# ---------------------------------------------------------------------------
# End-to-end pipeline on synthetic data inside tmp_path only
# ---------------------------------------------------------------------------

@pytest.fixture
def pipeline_env(tmp_path, monkeypatch, sample_movies_df, sample_ratings_df,
                 sample_tags_df, sample_links_df, sample_genome_scores_df,
                 sample_genome_tags_df):
    """Full isolated environment: synthetic raw files + patched config paths."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    frames = {
        config.MOVIES_PATH: sample_movies_df,
        config.RATINGS_PATH: sample_ratings_df,
        config.TAGS_PATH: sample_tags_df,
        config.LINKS_PATH: sample_links_df,
        config.GENOME_SCORES_PATH: sample_genome_scores_df,
        config.GENOME_TAGS_PATH: sample_genome_tags_df,
    }
    raw_paths = {}
    for default_path, frame in frames.items():
        target = raw_dir / default_path.name
        frame.to_csv(target, index=False)
        raw_paths[default_path.name] = target

    proc_dir = tmp_path / "processed"
    charts_dir = tmp_path / "charts"
    reports_dir = tmp_path / "reports"

    monkeypatch.setattr(config, "MOVIES_PATH", raw_paths["movies.csv"])
    monkeypatch.setattr(config, "RATINGS_PATH", raw_paths["ratings.csv"])
    monkeypatch.setattr(config, "TAGS_PATH", raw_paths["tags.csv"])
    monkeypatch.setattr(config, "LINKS_PATH", raw_paths["links.csv"])
    monkeypatch.setattr(config, "GENOME_SCORES_PATH", raw_paths["genome-scores.csv"])
    monkeypatch.setattr(config, "GENOME_TAGS_PATH", raw_paths["genome-tags.csv"])

    monkeypatch.setattr(config, "PROCESSED_DATA_DIR", proc_dir)
    monkeypatch.setattr(config, "PROCESSED_MOVIES_PATH", proc_dir / "movies_clean.csv")
    monkeypatch.setattr(config, "PROCESSED_RATINGS_PATH", proc_dir / "ratings_clean.csv")
    monkeypatch.setattr(config, "PROCESSED_TAGS_PATH", proc_dir / "tags_clean.csv")
    monkeypatch.setattr(config, "PROCESSED_LINKS_PATH", proc_dir / "links_clean.csv")
    monkeypatch.setattr(config, "PROCESSED_MOVIES_FEATURES_PATH",
                        proc_dir / "movies_features_base.csv")
    monkeypatch.setattr(config, "CHARTS_DIR", charts_dir)
    monkeypatch.setattr(config, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(config, "QUALITY_REPORT_TXT_PATH",
                        reports_dir / "dataset_quality_report.txt")
    monkeypatch.setattr(config, "QUALITY_REPORT_JSON_PATH",
                        reports_dir / "dataset_quality_report.json")

    return {
        "raw_paths": list(raw_paths.values()),
        "proc_dir": proc_dir,
        "charts_dir": charts_dir,
        "reports_dir": reports_dir,
    }


def _hash(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TestEndToEndPipelineSynthetic:
    def test_pipeline_generates_all_outputs_without_touching_raw(
        self, pipeline_env
    ):
        from src.pipeline import ModuleOnePipeline

        env = pipeline_env
        hashes_before = {p.name: _hash(p) for p in env["raw_paths"]}

        report = ModuleOnePipeline().run()

        # Raw inputs untouched by processing.
        for path in env["raw_paths"]:
            assert _hash(path) == hashes_before[path.name]

        # Processed datasets generated.
        expected_files = [
            "movies_clean.csv",
            "ratings_clean.csv",
            "tags_clean.csv",
            "links_clean.csv",
            "movies_features_base.csv",
        ]
        for name in expected_files:
            path = env["proc_dir"] / name
            assert path.is_file() and path.stat().st_size > 0, name

        # Reports generated; status gate passed.
        json_path = env["reports_dir"] / "dataset_quality_report.json"
        txt_path = env["reports_dir"] / "dataset_quality_report.txt"
        assert json_path.is_file() and txt_path.is_file()
        parsed = json.loads(json_path.read_text(encoding="utf-8"))
        assert parsed["status"] == "PASS"
        assert "STATUS: PASS" in txt_path.read_text(encoding="utf-8")

        # Charts generated.
        chart_names = [
            "rating_distribution.png",
            "user_activity_distribution.png",
            "movie_popularity_distribution.png",
            "genre_distribution.png",
            "top_tags.png",
        ]
        for name in chart_names:
            path = env["charts_dir"] / name
            assert path.is_file() and path.stat().st_size > 0, name

        # Cleaned ratings keep all valid synthetic rows.
        cleaned = pd.read_csv(env["proc_dir"] / "ratings_clean.csv")
        assert len(cleaned) == 6

    def test_pipeline_is_reproducible(self, pipeline_env):
        from src.pipeline import ModuleOnePipeline

        env = pipeline_env
        ModuleOnePipeline().run()
        first = {
            name: _hash(env["proc_dir"] / name)
            for name in ("movies_clean.csv", "ratings_clean.csv", "tags_clean.csv",
                         "links_clean.csv", "movies_features_base.csv")
        }
        first_json = _hash(env["reports_dir"] / "dataset_quality_report.json")

        ModuleOnePipeline().run()

        for name, digest in first.items():
            assert _hash(env["proc_dir"] / name) == digest, f"{name} not reproducible"
        assert _hash(env["reports_dir"] / "dataset_quality_report.json") == first_json


# ---------------------------------------------------------------------------
# Real processed artifacts (read-only; require a prior pipeline run)
# ---------------------------------------------------------------------------

EXPECTED_PROCESSED = [
    config.PROCESSED_MOVIES_PATH,
    config.PROCESSED_RATINGS_PATH,
    config.PROCESSED_TAGS_PATH,
    config.PROCESSED_LINKS_PATH,
    config.PROCESSED_MOVIES_FEATURES_PATH,
]


class TestRealProcessedArtifacts:
    @pytest.mark.parametrize("path", EXPECTED_PROCESSED,
                             ids=[p.name for p in EXPECTED_PROCESSED])
    def test_processed_file_exists_and_readable(self, path):
        assert path.is_file(), f"{path} missing - run 'python -m src.pipeline' first"
        df = pd.read_csv(path, nrows=50)
        assert len(df) > 0

    def test_real_quality_reports_exist_and_pass(self):
        assert config.QUALITY_REPORT_JSON_PATH.is_file()
        assert config.QUALITY_REPORT_TXT_PATH.is_file()
        report = json.loads(
            config.QUALITY_REPORT_JSON_PATH.read_text(encoding="utf-8")
        )
        assert report["status"] == "PASS"
        assert "STATUS: PASS" in config.QUALITY_REPORT_TXT_PATH.read_text(
            encoding="utf-8"
        )

    def test_real_charts_exist(self):
        charts = [
            "rating_distribution.png",
            "user_activity_distribution.png",
            "movie_popularity_distribution.png",
            "genre_distribution.png",
            "top_tags.png",
        ]
        for name in charts:
            path = config.CHARTS_DIR / name
            assert path.is_file() and path.stat().st_size > 0

    def test_real_movies_clean_invariants(self):
        movies = pd.read_csv(
            config.PROCESSED_MOVIES_PATH,
            usecols=["movieId", "genres"],
        )
        assert movies["movieId"].is_unique
        assert (movies["movieId"] > 0).all()
        assert movies["genres"].notna().all()

    def test_real_ratings_clean_within_scale(self):
        chunk_iter = pd.read_csv(
            config.PROCESSED_RATINGS_PATH,
            usecols=["rating"],
            chunksize=1_000_000,
        )
        for chunk in chunk_iter:
            r = chunk["rating"]
            assert float(r.min()) >= config.RATING_MIN - 1e-9
            assert float(r.max()) <= config.RATING_MAX + 1e-9

    def test_raw_data_unchanged_by_processing(self, baseline_raw_manifest):
        """Processed artifacts existing must not correlate with raw edits."""
        manifest_name = {
            config.PROCESSED_MOVIES_PATH: "movies.csv",
            config.PROCESSED_RATINGS_PATH: "ratings.csv",
        }
        for proc_path, raw_name in manifest_name.items():
            if not proc_path.is_file():
                continue
            raw_path = config.RAW_DATA_DIR / "ml-25m" / raw_name
            digest = hashlib.sha256()
            with open(raw_path, "rb") as fh:
                for chunk in iter(lambda: fh.read(4 * 1024 * 1024), b""):
                    digest.update(chunk)
            assert digest.hexdigest() == baseline_raw_manifest[raw_name]["sha256"]
