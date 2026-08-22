"""Tests for src/data_loader.py.

Covers: raw-file discovery, successful loading, schema presence, dtype
validity, and every documented failure mode (missing file, malformed
CSV, missing columns, empty file, encoding failure).
"""

from __future__ import annotations

import pandas as pd
import pytest

from src import config
from src.data_loader import (
    DataLoadError,
    MalformedCSVError,
    MissingFileError,
    SchemaError,
    load_csv,
    load_genome_scores,
    load_genome_tags,
    load_links,
    load_movies,
)


class TestRawDatasetDiscovery:
    def test_raw_directory_exists(self):
        assert config.RAW_DATA_DIR.is_dir()

    @pytest.mark.parametrize(
        "path",
        [
            config.MOVIES_PATH,
            config.RATINGS_PATH,
            config.TAGS_PATH,
            config.LINKS_PATH,
            config.GENOME_SCORES_PATH,
            config.GENOME_TAGS_PATH,
        ],
        ids=["movies", "ratings", "tags", "links", "genome-scores", "genome-tags"],
    )
    def test_required_csv_exists_and_non_empty(self, path):
        assert path.is_file(), f"missing raw file: {path}"
        assert path.stat().st_size > 0


class TestLoaderOnRealFiles:
    def test_load_movies_works(self, movies_full):
        assert len(movies_full) > 0
        assert list(movies_full.columns) == ["movieId", "title", "genres"]

    def test_load_links_works(self):
        links = load_links()
        assert len(links) > 0
        assert set(["movieId", "imdbId", "tmdbId"]).issubset(links.columns)

    def test_load_genome_tags_works(self):
        tags = load_genome_tags()
        assert len(tags) > 0
        assert list(tags.columns) == ["tagId", "tag"]

    def test_load_genome_scores_dtypes(self):
        scores = load_genome_scores()
        assert str(scores["movieId"].dtype) == "int32"
        assert str(scores["tagId"].dtype) == "int16"
        assert pd.api.types.is_float_dtype(scores["relevance"])

    def test_load_ratings_full_dtypes(self, ratings_full):
        assert str(ratings_full["userId"].dtype) == "int32"
        assert str(ratings_full["movieId"].dtype) == "int32"
        assert pd.api.types.is_float_dtype(ratings_full["rating"])
        assert str(ratings_full["timestamp"].dtype) == "int64"


class TestSchemaPresenceOnRawFiles:
    # 'ratings' schema is covered by tests/test_data_quality.py via the
    # session-scoped ratings fixture to avoid a second full 25M-row load.
    @pytest.mark.parametrize(
        ("loader", "dataset"),
        [
            (load_movies, "movies"),
            (load_links, "links"),
            (load_genome_scores, "genome_scores"),
            (load_genome_tags, "genome_tags"),
        ],
    )
    def test_required_columns_present(self, loader, dataset):
        df = loader()
        for column in config.EXPECTED_SCHEMAS[dataset]:
            assert column in df.columns, f"{dataset} lacks column {column}"


class TestLoaderFailureModes:
    def _write(self, tmp_path, name: str, content: str) -> object:
        path = tmp_path / name
        path.write_text(content, encoding="utf-8")
        return path

    def test_missing_file_raises_missing_file_error(self, tmp_path):
        with pytest.raises(MissingFileError, match="not found"):
            load_csv(tmp_path / "does_not_exist.csv")

    def test_missing_required_column_raises_schema_error(self, tmp_path):
        path = self._write(tmp_path, "bad_schema.csv", "a,b\n1,2\n")
        with pytest.raises(SchemaError, match="required column"):
            load_csv(path, required_columns=("movieId", "title"))

    def test_malformed_csv_raises_malformed_error(self, tmp_path):
        path = self._write(tmp_path, "broken.csv", 'a,b\n"unclosed,1\n2,3\n')
        with pytest.raises(MalformedCSVError):
            load_csv(path)

    def test_empty_file_raises_malformed_error(self, tmp_path):
        path = self._write(tmp_path, "empty.csv", "")
        with pytest.raises(MalformedCSVError, match="no data"):
            load_csv(path)

    def test_invalid_encoding_raises_data_load_error(self, tmp_path):
        path = tmp_path / "latin.csv"
        path.write_bytes(b"id,name\n1,\xff\xfe invalid utf8\n")
        with pytest.raises(DataLoadError, match="decode"):
            load_csv(path, required_columns=("id",))

    def test_dtype_conflict_raises_data_load_error(self, tmp_path):
        path = self._write(tmp_path, "text_ids.csv", "id,val\nabc,1\n")
        with pytest.raises(DataLoadError, match="dtypes"):
            load_csv(path, dtype={"id": "int32"}, required_columns=("id",))

    def test_valid_custom_file_loads_fine(self, tmp_path):
        path = self._write(tmp_path, "ok.csv", "movieId,title\n1,Test (2000)\n")
        df = load_csv(path, dtype={"movieId": "int32"},
                      required_columns=("movieId", "title"))
        assert len(df) == 1
