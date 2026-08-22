"""Tests for Module 2 content feature engineering (``src/feature_engineering.py``).

Covers text normalization, genre feature construction, tag aggregation,
content-document building, TF-IDF vectorization, sparse matrix generation,
feature index, serialization/loading, pipeline determinism, error handling,
reports and raw-data immutability.

Isolation rules (enforced):

* Every synthetic run redirects ``src.config`` paths into ``tmp_path`` and
  relaxes ``min_df`` to 1 so tiny corpora still produce a vocabulary.
* Nothing in this module writes to ``data/raw``, to the real
  ``data/processed`` files, ``models/`` or ``outputs/reports/``.
* Real-data sanity tests are read-only (they assume the pipelines from
  Modules 1 and 2 were executed, matching the documented run order).

Module 1 regression is guarded by the whole suite: existing Module 1 test
files run alongside these and must stay green.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from src import config
from src.data_loader import DataLoadError, MissingFileError
from src.feature_engineering import (
    FeatureEngineeringError,
    FeatureEngineeringPipeline,
    aggregate_tags,
    build_content_documents,
    build_feature_index,
    build_feature_report,
    build_genre_features,
    compute_tag_statistics,
    fit_tfidf,
    load_feature_vectorizer,
    load_sparse_matrix,
    normalize_text,
    render_feature_report_text,
    save_feature_vectorizer,
    save_sparse_matrix,
    split_normalized_genres,
    strip_release_year,
    validate_feature_artifacts,
    verify_loaded_vectorizer,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _no_genre_flag_column() -> str:
    return "genre_" + config.NO_GENRES_LABEL.replace(" ", "_").replace("-", "_")


def _run_pipeline(rewrite_dir: Path, movies, tags, monkeypatch) -> dict:
    """Wire config paths to `rewrite_dir`, write inputs, run the pipeline."""
    from src.feature_engineering import FeatureEngineeringPipeline

    processed = rewrite_dir / "processed"
    models = rewrite_dir / "models"
    reports = rewrite_dir / "reports"
    for folder in (processed, models, reports):
        folder.mkdir(parents=True, exist_ok=True)
    movies.to_csv(processed / "movies_clean.csv", index=False)
    tags.to_csv(processed / "tags_clean.csv", index=False)

    monkeypatch.setattr(config, "PROCESSED_MOVIES_PATH", processed / "movies_clean.csv")
    monkeypatch.setattr(config, "PROCESSED_TAGS_PATH", processed / "tags_clean.csv")
    monkeypatch.setattr(
        config, "PROCESSED_MOVIE_CONTENT_FEATURES_PATH",
        processed / "movie_content_features.csv",
    )
    monkeypatch.setattr(
        config, "PROCESSED_MOVIE_GENRE_FEATURES_PATH",
        processed / "movie_genre_features.csv",
    )
    monkeypatch.setattr(config, "PROCESSED_MOVIE_TFIDF_PATH", processed / "movie_tfidf.npz")
    monkeypatch.setattr(
        config, "PROCESSED_MOVIE_FEATURE_INDEX_PATH",
        processed / "movie_feature_index.csv",
    )
    monkeypatch.setattr(config, "MOVIE_TFIDF_VECTORIZER_PATH", models / "movie_tfidf_vectorizer.pkl")
    monkeypatch.setattr(
        config, "FEATURE_ENGINEERING_REPORT_JSON_PATH",
        reports / "feature_engineering_report.json",
    )
    monkeypatch.setattr(
        config, "FEATURE_ENGINEERING_REPORT_TXT_PATH",
        reports / "feature_engineering_report.txt",
    )
    monkeypatch.setattr(config, "TFIDF_MIN_DF", 1)
    monkeypatch.setattr(config, "TFIDF_MAX_DF", 1.0)
    return FeatureEngineeringPipeline().run()


# ---------------------------------------------------------------------------
# Text normalization (Step 4)
# ---------------------------------------------------------------------------

class TestTextNormalization:
    def test_lowercase_punctuation_and_whitespace(self):
        assert normalize_text("  The Matrix  (1999)!  ") == "the matrix 1999"
        assert normalize_text("Action|Comedy") == "action comedy"
        assert normalize_text("sci-fi") == "sci-fi"  # hyphenated token preserved

    def test_missing_values_become_empty(self):
        assert normalize_text(None) == ""
        assert normalize_text(np.nan) == ""
        assert normalize_text(pd.NA) == ""
        assert normalize_text("") == ""
        assert normalize_text(12345) == "12345"

    def test_normalization_is_idempotent(self):
        for raw in ("The Matrix (1999)!!", "  Sci-Fi   ", "A:a B[b] C{c}"):
            once = normalize_text(raw)
            assert normalize_text(once) == once

    def test_strip_release_year(self):
        assert strip_release_year("The Matrix (1999)") == "The Matrix"
        assert strip_release_year("Toy Story (1995)") == "Toy Story"
        assert strip_release_year("No Year Here") == "No Year Here"
        assert strip_release_year(None) == ""


# ---------------------------------------------------------------------------
# Genre feature processing (Step 5)
# ---------------------------------------------------------------------------

class TestGenreFeatures:
    def test_pipe_separated_genres_split(self, sample_movies_df):
        normalized = split_normalized_genres(sample_movies_df["genres"])
        assert normalized.tolist()[0] == ["action", "comedy"]
        assert normalized.tolist()[1] == ["drama"]
        assert normalized.tolist()[2] == []  # "(no genres listed)" -> no real genres

    def test_multi_hot_flags(self, sample_movies_df):
        genre_df, vocab = build_genre_features(sample_movies_df)
        assert "Action" in vocab and "Comedy" in vocab and "Drama" in vocab
        assert genre_df["genre_Action"].tolist() == [1, 0, 0, 1]
        assert genre_df["genre_Comedy"].tolist() == [1, 0, 0, 0]
        assert genre_df["genre_Drama"].tolist() == [0, 1, 0, 0]
        assert genre_df[_no_genre_flag_column()].tolist() == [0, 0, 1, 0]

    def test_values_are_int8_zero_one(self, sample_movies_df):
        genre_df, _ = build_genre_features(sample_movies_df)
        flags = genre_df.drop(columns=["movieId"])
        assert (flags.dtypes == np.dtype("int8")).all()
        assert flags.to_numpy().min() >= 0 and flags.to_numpy().max() <= 1

    def test_vocab_derived_from_data_not_hardcoded(self):
        movies = pd.DataFrame(
            {
                "movieId": [1, 2],
                "title": ["Odd (2001)", "Neo-Noir (2002)"],
                "genres": ["Weird-Noir|Mumblecore", "Weird-Noir|IMAX"],
            }
        )
        genre_df, vocab = build_genre_features(movies)
        for token in ("Weird-Noir", "Mumblecore", "IMAX"):
            assert token in vocab
        assert "genre_Weird_Noir" in genre_df.columns
        assert genre_df["genre_Mumblecore"].tolist() == [1, 0]
        assert genre_df["genre_IMAX"].tolist() == [0, 1]

    def test_no_genre_movie_flagged_but_no_real_genre(self, sample_movies_df):
        genre_df, _ = build_genre_features(sample_movies_df)
        no_genre_row = genre_df.iloc[2]
        assert no_genre_row[_no_genre_flag_column()] == 1
        real = [g for g in genre_df.columns if g.startswith("genre_")
                and g != _no_genre_flag_column()]
        assert (no_genre_row[real].to_numpy() == 0).all()

    def test_one_row_per_movie(self, feature_movies_df):
        genre_df, vocab = build_genre_features(feature_movies_df)
        assert len(genre_df) == len(feature_movies_df)
        assert genre_df["movieId"].is_unique
        assert len(vocab) == genre_df.shape[1] - 1
# ---------------------------------------------------------------------------
# Tag aggregation (Step 6)
# ---------------------------------------------------------------------------

class TestTagAggregation:
    def test_groups_by_movie_and_dedupes(self, feature_tags_df):
        agg = aggregate_tags(feature_tags_df)
        agg = agg.set_index("movieId")
        assert agg.loc[1, "aggregated_tags"] == "sci-fi futuristic"
        assert agg.loc[3, "aggregated_tags"] == "sci-fi dinosaurs"
        assert agg.loc[4, "aggregated_tags"] == "classic mafia"
        assert agg.loc[5, "aggregated_tags"] == "godfather"
        assert agg.loc[1, "num_tags"] == 2
        assert agg.loc[5, "num_tags"] == 1

    def test_no_duplicate_tag_within_movie(self, feature_tags_df):
        agg = aggregate_tags(feature_tags_df)
        movie3 = agg.set_index("movieId").loc[3, "aggregated_tags"]
        tokens = movie3.split()
        # Deterministic ordering is global-frequency desc then alphabetical;
        # the core contract checked here is "no tag is repeated in one movie".
        assert len(tokens) == len(set(tokens))
        assert tokens.count("sci-fi") == 1
        assert "sci-fi" in tokens
        # re-running yields the exact same string (deterministic aggregation)
        movie3_again = aggregate_tags(feature_tags_df).set_index("movieId").loc[3, "aggregated_tags"]
        assert movie3_again == movie3

    def test_empty_input_returns_empty_frame(self):
        empty = aggregate_tags(None)
        assert len(empty) == 0
        assert list(empty.columns) == ["movieId", "aggregated_tags", "num_tags"]
        empty2 = aggregate_tags(pd.DataFrame(columns=["userId", "movieId", "tag"]))
        assert len(empty2) == 0

    def test_blank_tags_dropped(self, feature_tags_df):
        agg = aggregate_tags(feature_tags_df)
        for value in agg["aggregated_tags"]:
            assert all(tok.strip() != "" for tok in str(value).split())

    def test_tag_statistics_shape(self, feature_tags_df):
        stats = compute_tag_statistics(feature_tags_df)
        assert stats["total_tag_records"] == 10
        assert stats["valid_tag_records"] == 9  # blank tag removed
        assert stats["unique_movies_tagged"] == 5  # movie 6 has no tags
        assert stats["unique_tags"] == 8

    def test_tag_statistics_empty_input(self):
        stats = compute_tag_statistics(None)
        assert stats["total_tag_records"] == 0
        assert stats["unique_tags"] == 0


# ---------------------------------------------------------------------------
# Content documents (Steps 3, 7)
# ---------------------------------------------------------------------------

class TestContentDocuments:
    def _build(self, feature_movies_df, feature_tags_df):
        return build_content_documents(
            feature_movies_df, aggregate_tags(feature_tags_df)
        )

    def test_one_document_per_movie(self, feature_movies_df, feature_tags_df):
        content = self._build(feature_movies_df, feature_tags_df)
        assert len(content) == len(feature_movies_df)
        assert content["movieId"].is_unique
        assert set(content["movieId"]) == set(feature_movies_df["movieId"])
        assert list(content["movieId"]) == sorted(content["movieId"])

    def test_movies_without_tags_still_have_content(
        self, feature_movies_df, feature_tags_df
    ):
        content = self._build(feature_movies_df, feature_tags_df)
        no_tag = content[content["num_tags"] == 0]
        assert no_tag["movieId"].tolist() == [6]
        assert (no_tag["content_text"].astype(str).str.len() > 0).all()

    def test_content_text_contains_title_genres_tags(self):
        content = self._build(
            pd.DataFrame(
                {
                    "movieId": [1],
                    "title": ["The Matrix (1999)"],
                    "genres": ["Action|Sci-Fi|Thriller"],
                }
            ),
            pd.DataFrame(
                {"userId": [10], "movieId": [1], "tag": ["futuristic"]}
            ),
        )
        text = content.loc[0, "content_text"]
        assert "the matrix" in text
        assert "action" in text and "sci-fi" in text and "thriller" in text
        assert "futuristic" in text
        assert "1999" not in text  # release year stripped from the document

    def test_normalized_genres_set(self):
        content = self._build(
            pd.DataFrame(
                {"movieId": [1], "title": ["X (2000)"], "genres": ["Drama|Comedy"]}
            ),
            aggregate_tags(None),
        )
        assert content.loc[0, "normalized_genres"] == "drama|comedy"
        assert content.loc[0, "num_genres"] == 2

    def test_documents_deterministic(self, feature_movies_df, feature_tags_df):
        first = self._build(feature_movies_df, feature_tags_df)
        second = self._build(feature_movies_df, feature_tags_df)
        pd.testing.assert_frame_equal(first, second)
# ---------------------------------------------------------------------------
# TF-IDF feature engineering (Steps 8-10)
# ---------------------------------------------------------------------------

class TestTfidf:
    def _documents(self, feature_movies_df, feature_tags_df):
        content = build_content_documents(feature_movies_df, aggregate_tags(feature_tags_df))
        return content["content_text"].astype(str).tolist()

    def test_fit_succeeds_sparse_matrix(self, feature_paths, feature_movies_df, feature_tags_df):
        vectorizer, matrix = fit_tfidf(self._documents(feature_movies_df, feature_tags_df))
        assert len(vectorizer.vocabulary_) > 0
        assert isinstance(matrix, sparse.csr_matrix)
        assert not isinstance(matrix, np.ndarray)

    def test_vocabulary_non_empty(self, feature_paths, feature_movies_df, feature_tags_df):
        vectorizer, matrix = fit_tfidf(self._documents(feature_movies_df, feature_tags_df))
        assert len(vectorizer.vocabulary_) > 0
        assert len(vectorizer.vocabulary_) == matrix.shape[1]
        assert all(isinstance(term, str) for term in vectorizer.vocabulary_)

    def test_matrix_rows_equal_documents(self, feature_paths, feature_movies_df, feature_tags_df):
        documents = self._documents(feature_movies_df, feature_tags_df)
        _, matrix = fit_tfidf(documents)
        assert matrix.shape[0] == len(documents)

    def test_matrix_no_nan_no_inf(self, feature_paths, feature_movies_df, feature_tags_df):
        _, matrix = fit_tfidf(self._documents(feature_movies_df, feature_tags_df))
        assert not np.isnan(matrix.data).any()
        assert not np.isinf(matrix.data).any()

    def test_empty_corpus_raises(self, feature_paths):
        with pytest.raises(FeatureEngineeringError):
            fit_tfidf([])

    def test_different_documents_different_vectors(
        self, feature_paths, feature_movies_df, feature_tags_df
    ):
        documents = self._documents(feature_movies_df, feature_tags_df)
        _, matrix = fit_tfidf(documents)
        row0 = matrix.getrow(0).toarray().ravel()
        row1 = matrix.getrow(1).toarray().ravel()
        assert row0.shape == row1.shape
        if documents[0] != documents[1]:
            assert not np.allclose(row0, row1)


# ---------------------------------------------------------------------------
# Feature index (Step 11)
# ---------------------------------------------------------------------------

class TestFeatureIndex:
    def test_build_index_contiguous_and_sorted(self, feature_movies_df):
        index = build_feature_index(feature_movies_df["movieId"])
        assert index["movieId"].tolist() == sorted(feature_movies_df["movieId"].tolist())
        assert index["row_index"].tolist() == list(range(len(feature_movies_df)))
        assert pd.api.types.is_integer_dtype(index["row_index"])
        assert index["movieId"].is_unique

    def test_build_index_dedupes_and_orders(self):
        index = build_feature_index([5, 3, 5, 1])
        assert index["movieId"].tolist() == [1, 3, 5]
        assert index["row_index"].tolist() == [0, 1, 2]
# ---------------------------------------------------------------------------
# End-to-end pipeline (Steps 1-3, 13)
# ---------------------------------------------------------------------------

class TestFeaturePipeline:
    def test_loads_processed_movies(self, feature_pipeline):
        feature_pipeline.load_inputs()
        assert feature_pipeline.movies is not None
        assert len(feature_pipeline.movies) == 6
        assert list(feature_pipeline.movies.columns) == ["movieId", "title", "genres"]
        assert feature_pipeline.movies["movieId"].is_unique

    def test_loads_processed_tags(self, feature_pipeline, feature_tags_df):
        feature_pipeline.load_inputs()
        assert feature_pipeline.tags is not None
        assert len(feature_pipeline.tags) == len(feature_tags_df)

    def test_imports_feature_globals_are_available(self):
        # the module exposes a ready-made pipeline entry point
        from src.feature_engineering import main
        assert callable(main)

    def test_full_run_generates_all_artifacts(self, feature_pipeline):
        report = feature_pipeline.run()
        expected = [
            config.PROCESSED_MOVIE_CONTENT_FEATURES_PATH,
            config.PROCESSED_MOVIE_GENRE_FEATURES_PATH,
            config.PROCESSED_MOVIE_TFIDF_PATH,
            config.PROCESSED_MOVIE_FEATURE_INDEX_PATH,
            config.MOVIE_TFIDF_VECTORIZER_PATH,
            config.FEATURE_ENGINEERING_REPORT_JSON_PATH,
            config.FEATURE_ENGINEERING_REPORT_TXT_PATH,
        ]
        for path in expected:
            assert path.is_file(), f"missing artifact {path}"
        assert report["feature_quality_gate"] == "PASS"

    def test_quality_gate_all_checks_pass(self, feature_pipeline):
        report = feature_pipeline.run()
        assert report["quality_checks_all_passed"] is True
        assert all(chk["passed"] for chk in report["quality_checks"])

    def test_movie_ids_unique_across_pipeline(self, feature_pipeline):
        report = feature_pipeline.run()
        assert report["content_documents"]["num_documents"] == 6
        assert report["content_documents"]["movies_without_tags"] == 1

    def test_matrix_equals_movie_count_and_index(self, feature_pipeline):
        report = feature_pipeline.run()
        n = report["inputs"]["movies"]["rows"]
        assert report["tfidf"]["matrix_rows"] == n
        assert report["feature_index"]["rows"] == n
        assert report["feature_index"]["indices_contiguous"] is True

    def test_matrix_no_nan_inf_from_pipeline(self, feature_pipeline):
        report = feature_pipeline.run()
        checks = {c["check"]: c["passed"] for c in report["quality_checks"]}
        assert checks["matrix_no_nan"] and checks["matrix_no_inf"]

    def test_movies_without_tags_get_valid_vectors(self, feature_pipeline):
        feature_pipeline.run()
        index = pd.read_csv(config.PROCESSED_MOVIE_FEATURE_INDEX_PATH)
        matrix = load_sparse_matrix(config.PROCESSED_MOVIE_TFIDF_PATH)
        row = index[index["movieId"] == 6]["row_index"].iloc[0]
        vector = matrix.getrow(row).toarray().ravel()
        assert (vector > 0).sum() > 0

    def test_pipeline_missing_tags_file_tolerated(self, feature_paths, feature_movies_df):
        feature_movies_df.to_csv(config.PROCESSED_MOVIES_PATH, index=False)
        report = FeatureEngineeringPipeline().run()
        content = pd.read_csv(config.PROCESSED_MOVIE_CONTENT_FEATURES_PATH)
        assert (content["aggregated_tags"].fillna("").astype(str).str.len() == 0).all()
        assert report["inputs"]["tags"]["present"] is False
        assert report["feature_quality_gate"] == "PASS"

    def test_pipeline_missing_movies_file_raises(self, feature_paths):
        with pytest.raises(MissingFileError):
            FeatureEngineeringPipeline().run()

    def test_pipeline_empty_movies_raises(self, feature_paths):
        pd.DataFrame(columns=["movieId", "title", "genres"]).to_csv(
            config.PROCESSED_MOVIES_PATH, index=False
        )
        with pytest.raises(DataLoadError):
            FeatureEngineeringPipeline().run()

    def test_pipeline_duplicate_movie_ids_raise(self, feature_paths, feature_movies_df):
        dup = pd.concat([feature_movies_df, feature_movies_df.iloc[[0]]], ignore_index=True)
        dup.to_csv(config.PROCESSED_MOVIES_PATH, index=False)
        with pytest.raises(FeatureEngineeringError):
            FeatureEngineeringPipeline().run()
# ---------------------------------------------------------------------------
# Serialization / loading (Steps 10, 17-19)
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_sparse_matrix_roundtrip(self, feature_pipeline):
        feature_pipeline.run()
        original = feature_pipeline.matrix.tocsr()
        path = config.PROCESSED_MOVIE_TFIDF_PATH
        saved = save_sparse_matrix(original, path)
        assert saved.name == "movie_tfidf.npz"
        loaded = load_sparse_matrix(path)
        assert loaded.shape == original.shape
        assert loaded.nnz == original.nnz
        assert (abs(loaded - original)).nnz == 0

    def test_vectorizer_serializable_and_loadable(self, feature_pipeline):
        feature_pipeline.run()
        save_feature_vectorizer(feature_pipeline.vectorizer, config.MOVIE_TFIDF_VECTORIZER_PATH)
        loaded = load_feature_vectorizer(config.MOVIE_TFIDF_VECTORIZER_PATH)
        assert loaded.vocabulary_ == feature_pipeline.vectorizer.vocabulary_

    def test_loaded_vectorizer_produces_compatible_output(self, feature_pipeline):
        feature_pipeline.run()
        loaded = load_feature_vectorizer(config.MOVIE_TFIDF_VECTORIZER_PATH)
        documents = feature_pipeline.content_df["content_text"].astype(str).tolist()
        assert verify_loaded_vectorizer(loaded, documents, feature_pipeline.matrix)

    def test_default_load_path_matches_config(self, feature_pipeline):
        feature_pipeline.run()
        loaded = load_feature_vectorizer()  # defaults to config path
        assert len(loaded.vocabulary_) == feature_pipeline.report["tfidf"]["vocabulary_size"]

    def test_load_missing_vectorizer_raises(self, tmp_path):
        with pytest.raises(MissingFileError):
            load_feature_vectorizer(tmp_path / "nope.pkl")

    def test_load_missing_matrix_raises(self, tmp_path):
        with pytest.raises(MissingFileError):
            load_sparse_matrix(tmp_path / "nope.npz")
# ---------------------------------------------------------------------------
# Determinism & immutability (Steps 16, 21)
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_pipeline_deterministic(
        self, tmp_path, monkeypatch, feature_movies_df, feature_tags_df
    ):
        report_a = _run_pipeline(tmp_path / "run_a", feature_movies_df, feature_tags_df, monkeypatch)
        report_b = _run_pipeline(tmp_path / "run_b", feature_movies_df, feature_tags_df, monkeypatch)

        content_a = pd.read_csv(tmp_path / "run_a" / "processed" / "movie_content_features.csv")
        content_b = pd.read_csv(tmp_path / "run_b" / "processed" / "movie_content_features.csv")
        pd.testing.assert_frame_equal(content_a, content_b)

        index_a = pd.read_csv(tmp_path / "run_a" / "processed" / "movie_feature_index.csv")
        index_b = pd.read_csv(tmp_path / "run_b" / "processed" / "movie_feature_index.csv")
        pd.testing.assert_frame_equal(index_a, index_b)

        matrix_a = sparse.load_npz(tmp_path / "run_a" / "processed" / "movie_tfidf.npz").tocsr()
        matrix_b = sparse.load_npz(tmp_path / "run_b" / "processed" / "movie_tfidf.npz").tocsr()
        assert matrix_a.shape == matrix_b.shape
        assert matrix_a.nnz == matrix_b.nnz
        assert (abs(matrix_a - matrix_b)).nnz == 0

        assert report_a["tfidf"]["vocabulary_size"] == report_b["tfidf"]["vocabulary_size"]
        assert report_a["feature_index"]["rows"] == report_b["feature_index"]["rows"]

    def test_raw_data_not_modified(self, feature_pipeline, baseline_raw_manifest):
        raw_movies = config.RAW_DATA_DIR / "ml-25m" / "movies.csv"
        raw_tags = config.RAW_DATA_DIR / "ml-25m" / "tags.csv"
        before = {raw_movies.name: _sha256(raw_movies), raw_tags.name: _sha256(raw_tags)}
        feature_pipeline.run()
        after = {raw_movies.name: _sha256(raw_movies), raw_tags.name: _sha256(raw_tags)}
        assert before == after
        assert before["movies.csv"] == baseline_raw_manifest["movies.csv"]["sha256"]

    def test_rerun_overwrites_with_equivalent_outputs(
        self, tmp_path, monkeypatch, feature_movies_df, feature_tags_df
    ):
        _run_pipeline(tmp_path / "run", feature_movies_df, feature_tags_df, monkeypatch)
        first = pd.read_csv(tmp_path / "run" / "processed" / "movie_content_features.csv")
        _run_pipeline(tmp_path / "run", feature_movies_df, feature_tags_df, monkeypatch)
        second = pd.read_csv(tmp_path / "run" / "processed" / "movie_content_features.csv")
        pd.testing.assert_frame_equal(first, second)


# ---------------------------------------------------------------------------
# Reports (Step 12)
# ---------------------------------------------------------------------------

class TestReports:
    def test_report_files_generated(self, feature_pipeline):
        feature_pipeline.run()
        assert config.FEATURE_ENGINEERING_REPORT_JSON_PATH.is_file()
        assert config.FEATURE_ENGINEERING_REPORT_TXT_PATH.is_file()
        report = json.loads(config.FEATURE_ENGINEERING_REPORT_JSON_PATH.read_text(encoding="utf-8"))
        assert report["feature_quality_gate"] == "PASS"

    def test_report_contains_required_metrics(self, feature_pipeline):
        report = feature_pipeline.run()
        assert report["content_documents"]["num_documents"] == 6
        assert report["genres"]["genre_feature_count"] >= 7
        assert report["tfidf"]["vocabulary_size"] > 0
        assert report["tfidf"]["matrix_rows"] == 6
        assert report["tfidf"]["matrix_columns"] == report["tfidf"]["vocabulary_size"]
        assert report["tfidf"]["non_zero_elements"] > 0
        assert 0.0 <= report["tfidf"]["sparsity_percent"] <= 100.0
        assert report["feature_index"]["rows"] == 6
        assert report["feature_index"]["indices_contiguous"] is True

    def test_report_text_renders_gate(self, feature_pipeline):
        report = feature_pipeline.run()
        text = render_feature_report_text(report)
        assert "STATUS: PASS" in text
        assert "vocabulary size" in text


# ---------------------------------------------------------------------------
# Semantic sanity on known movies (Step 14)
# ---------------------------------------------------------------------------

class TestKnownMoviesSanity:
    def test_matrix_and_toy_story_documents(self, feature_movies_df, feature_tags_df):
        content = build_content_documents(feature_movies_df, aggregate_tags(feature_tags_df))
        matrix_row = content[content["title"] == "The Matrix (1999)"].iloc[0]
        toy_row = content[content["title"] == "Toy Story (1995)"].iloc[0]
        assert "matrix" in matrix_row["content_text"]
        assert "sci-fi" in matrix_row["content_text"]
        assert "thriller" in matrix_row["content_text"]
        assert "toy story" in toy_row["content_text"]
        assert "pixar" in toy_row["content_text"]
        assert matrix_row["num_genres"] == 3
        assert toy_row["num_tags"] == 2

    def test_validation_utility_full_pass(self, feature_pipeline):
        report = feature_pipeline.run()
        checks = feature_pipeline.checks
        assert len(checks) > 0
        assert all(c["passed"] for c in checks)
# ---------------------------------------------------------------------------
# Real production artifacts (read-only; require a Module 2 pipeline run)
# ---------------------------------------------------------------------------

class TestRealFeatureArtifacts:
    """Structural checks on the actual production artifacts.

    These mirror the Module 1 convention: the tests assume
    ``python -m src.feature_engineering`` has been executed from the project
    root (documented run order). Everything is read-only.
    """

    def test_missing_artifact_fails_with_clear_message(self):
        if not config.PROCESSED_MOVIE_FEATURE_INDEX_PATH.is_file():
            pytest.fail("movie_feature_index.csv missing - run 'python -m "
                        "src.feature_engineering' first")

    def test_real_matrix_matches_movies_and_index(self):
        movies = pd.read_csv(config.PROCESSED_MOVIES_PATH, usecols=["movieId"])
        index = pd.read_csv(config.PROCESSED_MOVIE_FEATURE_INDEX_PATH)
        content = pd.read_csv(config.PROCESSED_MOVIE_CONTENT_FEATURES_PATH)
        matrix = load_sparse_matrix(config.PROCESSED_MOVIE_TFIDF_PATH)

        n = len(movies)
        assert n == matrix.shape[0]
        assert len(index) == n
        assert len(content) == n
        assert index["movieId"].is_unique
        assert index["row_index"].tolist() == list(range(n))
        assert set(index["movieId"]) == set(movies["movieId"])
        assert not np.isnan(matrix.data).any()
        assert not np.isinf(matrix.data).any()

    def test_real_vectorizer_loads_and_matches_matrix(self):
        vectorizer = load_feature_vectorizer()
        assert len(vectorizer.vocabulary_) > 0
        matrix = load_sparse_matrix(config.PROCESSED_MOVIE_TFIDF_PATH)
        assert matrix.shape[1] == len(vectorizer.vocabulary_)

    def test_real_content_documents_have_text(self):
        content = pd.read_csv(
            config.PROCESSED_MOVIE_CONTENT_FEATURES_PATH,
            usecols=["movieId", "content_text", "num_tags"],
        )
        assert content["movieId"].is_unique
        non_empty = content["content_text"].fillna("").astype(str).str.len() > 0
        assert non_empty.all()
        assert int((content["num_tags"] > 0).sum()) > 0

    def test_real_vectorizer_reproduces_matrix(self):
        documents = pd.read_csv(
            config.PROCESSED_MOVIE_CONTENT_FEATURES_PATH,
            usecols=["content_text"],
        )["content_text"].fillna("")
        vectorizer = load_feature_vectorizer()
        matrix = load_sparse_matrix(config.PROCESSED_MOVIE_TFIDF_PATH)
        assert verify_loaded_vectorizer(vectorizer, documents.tolist(), matrix)

    def test_real_feature_report_status_pass(self):
        report_path = config.FEATURE_ENGINEERING_REPORT_JSON_PATH
        assert report_path.is_file()
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["feature_quality_gate"] == "PASS"
        assert report["tfidf"]["matrix_rows"] == report["inputs"]["movies"]["rows"]
        assert report["feature_index"]["rows"] == report["tfidf"]["matrix_rows"]
        txt_path = config.FEATURE_ENGINEERING_REPORT_TXT_PATH
        assert "STATUS: PASS" in txt_path.read_text(encoding="utf-8")