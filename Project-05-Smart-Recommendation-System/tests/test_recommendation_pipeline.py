"""Integration tests for the Module 3 recommendation pipeline.

Runs :class:`~src.pipeline.RecommendationPipeline` end-to-end inside the
``module3_workspace`` sandbox (synthetic Module 1 + Module 2 artifacts) and
verifies:

* artifact loading, sample generation, evaluation and quality checks;
* every report file is written (JSON + TXT pairs);
* the quality gate passes without needing the heavyweight regression stage;
* missing artifacts fail fast with actionable errors;
* raw MovieLens data is never modified by Module 3 code.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src import config


@pytest.fixture
def workspace_hints(module3_workspace, monkeypatch):
    """Point the pipeline at titles that exist in the synthetic catalog."""
    monkeypatch.setattr(
        config,
        "RECOMMENDATION_SEED_TITLE_HINTS",
        ("Space Buzz", "Hard Target", "Cosmos"),
    )
    monkeypatch.setattr(config, "RECOMMENDATION_REPORT_NUM_SEEDS", 3)
    return module3_workspace


class TestPipelineEndToEnd:
    def test_full_run_passes_gate_and_writes_reports(
        self, module3_workspace, workspace_hints
    ):
        from src.pipeline import RecommendationPipeline

        pipeline = RecommendationPipeline(run_regression_tests=False)
        gate_report = pipeline.run()

        reports = module3_workspace["reports"]
        expected_files = (
            "recommendation_quality_report.json",
            "recommendation_quality_report.txt",
            "evaluation_report.json",
            "evaluation_report.txt",
            "module3_quality_gate_report.json",
            "module3_quality_gate_report.txt",
        )
        for name in expected_files:
            path = reports / name
            assert path.is_file(), f"missing report artifact: {name}"
            assert path.stat().st_size > 0

        # Gate verdict.
        assert gate_report["gate"] == "PASS"
        failed = [c for c in gate_report["quality_checks"] if not c["passed"]]
        assert failed == []
        assert gate_report["evaluation_status"] == "PASS"
        assert gate_report["regression_tests_executed"] is False

        # JSON payloads are valid and consistent.
        rec_payload = json.loads(
            (reports / "recommendation_quality_report.json").read_text(encoding="utf-8")
        )
        eval_payload = json.loads(
            (reports / "evaluation_report.json").read_text(encoding="utf-8")
        )
        assert rec_payload["status"] == "PASS"
        assert eval_payload["status"] == "PASS"
        assert len(rec_payload["samples"]) == 3
        for sample in rec_payload["samples"]:
            ids = [r["movieId"] for r in sample["recommendations"]]
            assert 0 < len(ids) <= config.DEFAULT_TOP_K
            assert sample["seed"]["movieId"] not in ids
            assert len(set(ids)) == len(ids)

        # TXT renderers mirror their JSON sources.
        txt = (reports / "recommendation_quality_report.txt").read_text(
            encoding="utf-8"
        )
        assert "MODULE 3 - RECOMMENDATION QUALITY REPORT" in txt
        assert "STATUS: PASS" in txt

    def test_evaluation_metrics_present_in_report(
        self, module3_workspace, workspace_hints
    ):
        from src.pipeline import RecommendationPipeline

        RecommendationPipeline(run_regression_tests=False).run()
        payload = json.loads(
            (module3_workspace["reports"] / "evaluation_report.json").read_text(
                encoding="utf-8"
            )
        )
        assert set(payload["metrics_at_k"].keys()) == {
            str(k) for k in config.EVALUATION_K_VALUES
        }
        assert payload["users"]["evaluated"] >= 1
        assert "temporal" in payload["protocol"].lower()


class TestPipelineFailureModes:
    def test_missing_artifacts_fail_fast(self, module3_workspace, monkeypatch):
        from src.pipeline import RecommendationPipeline

        monkeypatch.setattr(
            config,
            "PROCESSED_MOVIE_TFIDF_PATH",
            module3_workspace["processed"] / "missing.npz",
        )
        pipeline = RecommendationPipeline(run_regression_tests=False)
        with pytest.raises(Exception, match="not found"):
            pipeline.load_inputs()

    def test_no_matching_seeds_raises_actionable_error(
        self, module3_workspace, monkeypatch
    ):
        from src.pipeline import RecommendationPipeline, RecommendationPipelineError

        monkeypatch.setattr(
            config,
            "RECOMMENDATION_SEED_TITLE_HINTS",
            ("Definitely Not A Real Title 12345",),
        )
        pipeline = RecommendationPipeline(run_regression_tests=False)
        pipeline.load_inputs()
        with pytest.raises(RecommendationPipelineError, match="seed title hints"):
            pipeline.generate_samples()

    def test_gate_fails_when_checks_fail(
        self, module3_workspace, workspace_hints, monkeypatch
    ):
        from src.pipeline import RecommendationPipeline, RecommendationPipelineError
        from src.similarity_engine import SimilarityEngine

        # Force an unknown-ID regression at class level so every engine
        # instance built during run() treats all movies as unknown.
        monkeypatch.setattr(
            SimilarityEngine, "has_movie", lambda self, movie_id: False
        )
        pipeline = RecommendationPipeline(run_regression_tests=False)
        with pytest.raises(RecommendationPipelineError, match="quality gate FAILED"):
            pipeline.run()
        gate_payload = json.loads(
            (
                module3_workspace["reports"] / "module3_quality_gate_report.json"
            ).read_text(encoding="utf-8")
        )
        assert gate_payload["gate"] == "FAIL"


class TestRenderers:
    def test_renderers_contain_key_sections(self, workspace_hints):
        from src.pipeline import (
            RecommendationPipeline,
            render_recommendation_report_text,
        )

        pipeline = RecommendationPipeline(run_regression_tests=False)
        pipeline.load_inputs()
        pipeline.generate_samples()
        pipeline.run_quality_checks()
        report = pipeline.build_recommendation_report()
        report["status"] = "PASS"
        text = render_recommendation_report_text(report)
        assert "RECOMMENDATION QUALITY REPORT" in text
        assert "Seed #" in text
        assert "[PASS]" in text
        assert "STATUS: PASS" in text

    def test_wrap_paragraph_deterministic(self):
        from src.pipeline import _wrap_paragraph

        words = " ".join(f"word{i}" for i in range(40))
        assert _wrap_paragraph(words) == _wrap_paragraph(words)
        assert all(len(line) <= 72 for line in _wrap_paragraph(words))


class TestRawDataSafety:
    def test_pipeline_never_touches_raw_or_real_processed(
        self, module3_workspace, workspace_hints, baseline_raw_manifest
    ):
        import hashlib

        from src.pipeline import RecommendationPipeline

        raw_movies = config.RAW_DATA_DIR / "ml-25m" / "movies.csv"
        digest_before = hashlib.sha256(raw_movies.read_bytes()).hexdigest()

        RecommendationPipeline(run_regression_tests=False).run()

        assert hashlib.sha256(raw_movies.read_bytes()).hexdigest() == digest_before
        assert digest_before == baseline_raw_manifest["movies.csv"]["sha256"]

        # The real processed ratings file was not rewritten either (the
        # workspace redirected all paths into tmp_path).
        real_ratings = pd.read_csv(config.PROCESSED_RATINGS_PATH, nrows=5)
        assert len(real_ratings) == 5