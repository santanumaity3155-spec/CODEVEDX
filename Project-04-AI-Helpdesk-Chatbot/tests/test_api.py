"""
Test suite for Module 5: AI Helpdesk Chatbot API.

Covers:
- application imports
- health endpoint
- chat endpoint
- valid / invalid requests
- intent and confidence responses
- fallback handling
- error handling
- model-loaded state
- sequential requests
- error response format
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import APIConfig, get_api_config  # noqa: E402
from chatbot import HelpdeskChatbot  # noqa: E402


# ---------------------------------------------------------------------------
# Flask app fixture
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def api_config() -> APIConfig:
    cfg = get_api_config()
    cfg.debug = False
    cfg.allowed_origins = ["http://localhost:3000"]
    return cfg


@pytest.fixture(scope="session")
def flask_app(api_config: APIConfig):
    from app import create_app

    problems = api_config.chatbot.validate()
    if problems:
        pytest.skip(f"Chatbot configuration invalid: {'; '.join(problems)}")
    return create_app(api_config)


@pytest.fixture(scope="session")
def client(flask_app):
    return flask_app.test_client()


@pytest.fixture(scope="session")
def chatbot(api_config: APIConfig) -> HelpdeskChatbot:
    return HelpdeskChatbot(config=api_config.chatbot)


# ---------------------------------------------------------------------------
# 1. Application imports
# ---------------------------------------------------------------------------
class TestApplicationImports:
    def test_config_imports(self):
        from config import APIConfig, get_api_config, setup_logging  # noqa: F401

        assert APIConfig is not None

    def test_app_imports(self):
        from app import create_app, main  # noqa: F401

        assert create_app is not None
        assert callable(create_app)

    def test_routes_imports(self):
        from api.routes import api_bp, health_check, chat, predict  # noqa: F401

        assert api_bp is not None


# ---------------------------------------------------------------------------
# 2. Health endpoint
# ---------------------------------------------------------------------------
class TestHealthEndpoint:
    def test_health_status_code(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_json(self, client):
        resp = client.get("/health")
        data = resp.get_json()
        assert isinstance(data, dict)

    def test_health_fields(self, client):
        resp = client.get("/health")
        data = resp.get_json()
        assert "status" in data
        assert "service" in data
        assert "version" in data
        assert "model_loaded" in data
        assert "faq_loaded" in data

    def test_health_status_healthy(self, client):
        resp = client.get("/health")
        data = resp.get_json()
        assert data["status"] == "healthy"

    def test_health_model_loaded(self, client):
        resp = client.get("/health")
        data = resp.get_json()
        assert data["model_loaded"] is True

    def test_health_faq_loaded(self, client):
        resp = client.get("/health")
        data = resp.get_json()
        assert data["faq_loaded"] is True

    def test_health_service_name(self, client):
        resp = client.get("/health")
        data = resp.get_json()
        assert data["service"] == "AI Helpdesk Chatbot"


# ---------------------------------------------------------------------------
# 3. Chat endpoint
# ---------------------------------------------------------------------------
class TestChatEndpoint:
    def test_chat_requires_post(self, client):
        resp = client.get("/api/chat")
        assert resp.status_code == 405

    def test_chat_requires_json(self, client):
        resp = client.post("/api/chat", data="not json", content_type="text/plain")
        assert resp.status_code == 415

    def test_chat_valid_request(self, client):
        resp = client.post(
            "/api/chat",
            json={"message": "How do I reset my password?"},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True

    def test_chat_password_reset_question(self, client, chatbot):
        resp = client.post(
            "/api/chat",
            json={"message": "How do I reset my password?"},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["intent"] == "password_reset"
        assert data["confidence"] > 0.0

    def test_chat_working_hours_question(self, client, chatbot):
        resp = client.post(
            "/api/chat",
            json={"message": "What are the working hours?"},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["intent"] == "working_hours"

    def test_chat_faq_response(self, client, chatbot):
        resp = client.post(
            "/api/chat",
            json={"message": "How do I reset my password?"},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["response"]  # non-empty
        assert not data["fallback"]

    def test_chat_intent_returned(self, client, chatbot):
        resp = client.post(
            "/api/chat",
            json={"message": "My laptop is not working."},
        )
        data = resp.get_json()
        assert "intent" in data
        assert data["intent"] == "laptop_problems"

    def test_chat_confidence_returned(self, client, chatbot):
        resp = client.post(
            "/api/chat",
            json={"message": "How do I reset my password?"},
        )
        data = resp.get_json()
        assert "confidence" in data
        assert 0.0 <= data["confidence"] <= 1.0

    def test_chat_fallback_gibberish(self, client, chatbot):
        resp = client.post(
            "/api/chat",
            json={"message": "asdfghjkl random text"},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["fallback"] is True
        assert "rephrase" in data["response"].lower() or "not confident" in data["response"].lower()

    def test_chat_empty_message(self, client, chatbot):
        resp = client.post(
            "/api/chat",
            json={"message": ""},
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["success"] is False

    def test_chat_whitespace_message(self, client, chatbot):
        resp = client.post(
            "/api/chat",
            json={"message": "   "},
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["success"] is False

    def test_chat_missing_message(self, client, chatbot):
        resp = client.post(
            "/api/chat",
            json={},
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["success"] is False

    def test_chat_null_message(self, client, chatbot):
        resp = client.post(
            "/api/chat",
            json={"message": None},
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["success"] is False

    def test_chat_excessively_long_message(self, client, chatbot):
        long_msg = "a" * 5001
        resp = client.post(
            "/api/chat",
            json={"message": long_msg},
        )
        assert resp.status_code == 413
        data = resp.get_json()
        assert data["success"] is False

    def test_chat_invalid_json(self, client, chatbot):
        resp = client.post(
            "/api/chat",
            data="not valid json{{{",
            content_type="application/json",
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["success"] is False

    def test_chat_multiple_sequential_requests(self, client, chatbot):
        questions = [
            "How do I reset my password?",
            "My laptop is not working.",
            "What are the working hours?",
        ]
        for q in questions:
            resp = client.post("/api/chat", json={"message": q})
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["success"] is True
            assert data["message"] == q

    def test_chat_error_response_format(self, client, chatbot):
        resp = client.post(
            "/api/chat",
            json={},
        )
        data = resp.get_json()
        assert "success" in data
        assert "error" in data
        assert data["success"] is False


# ---------------------------------------------------------------------------
# 4. Model-loaded state
# ---------------------------------------------------------------------------
class TestModelState:
    def test_model_loaded_in_health(self, client):
        resp = client.get("/health")
        data = resp.get_json()
        assert data["model_loaded"] is True

    def test_chat_works_after_health(self, client):
        client.get("/health")
        resp = client.post(
            "/api/chat",
            json={"message": "How do I reset my password?"},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True


# ---------------------------------------------------------------------------
# 5. Predict endpoint
# ---------------------------------------------------------------------------
class TestPredictEndpoint:
    def test_predict_requires_post(self, client):
        resp = client.get("/api/predict")
        assert resp.status_code == 405

    def test_predict_valid_request(self, client):
        resp = client.post(
            "/api/predict",
            json={"message": "How do I reset my password?"},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["intent"] == "password_reset"
        assert "confidence" in data
        assert "confidence_level" in data

    def test_predict_missing_message(self, client):
        resp = client.post("/api/predict", json={})
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# 6. Unsupported endpoint
# ---------------------------------------------------------------------------
class TestUnsupportedEndpoint:
    def test_unsupported_endpoint(self, client):
        resp = client.get("/does-not-exist")
        assert resp.status_code == 404

    def test_unsupported_method(self, client):
        resp = client.put("/api/chat")
        assert resp.status_code == 405
