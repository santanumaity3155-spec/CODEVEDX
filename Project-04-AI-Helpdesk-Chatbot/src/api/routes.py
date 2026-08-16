"""
Module 5: API routes for the AI Helpdesk Chatbot.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Any

from flask import Blueprint, jsonify, request, current_app

from chatbot import HelpdeskChatbot, get_response as module_get_response
from chatbot_config import ChatbotConfig
from config import APIConfig

_log = logging.getLogger("chatbot.api")

api_bp = Blueprint("api", __name__)

# Module-level chatbot instance (loaded once at startup).
_chatbot: HelpdeskChatbot | None = None


def get_chatbot() -> HelpdeskChatbot:
    """Return the shared chatbot instance (lazy-init)."""
    global _chatbot
    if _chatbot is None:
        config: ChatbotConfig = current_app.config["CHATBOT_CONFIG"]
        _chatbot = HelpdeskChatbot(config=config)
        _log.info(
            "Chatbot initialised with %d FAQ answers",
            len(_chatbot.faq.df),
        )
    return _chatbot


def _build_chat_response(chat_response, request_message: str) -> Dict[str, Any]:
    """Translate a Module 4 ChatResponse into the Module 5 API response schema."""
    return {
        "success": True,
        "message": request_message,
        "intent": chat_response.intent or "unknown",
        "confidence": chat_response.confidence,
        "response": chat_response.answer,
        "fallback": chat_response.is_fallback,
        "entities": chat_response.entities,
        "confidence_level": chat_response.confidence_level,
        "fallback_reason": chat_response.fallback_reason,
    }


def _build_error_response(message: str, status_code: int = 400) -> tuple[Dict[str, Any], int]:
    """Return a structured error payload."""
    return {
        "success": False,
        "error": message,
    }, status_code


@api_bp.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint.

    Returns the runtime status of the API and its dependent components.
    """
    start = time.perf_counter()
    bot = get_chatbot()
    latency = time.perf_counter() - start

    model_loaded = bot.classifier is not None
    faq_loaded = bot.faq is not None and len(bot.faq.df) > 0

    status = "healthy"
    if not model_loaded or not faq_loaded:
        status = "degraded"

    payload = {
        "status": status,
        "service": "AI Helpdesk Chatbot",
        "version": current_app.config["API_CONFIG"].app_version,
        "model_loaded": model_loaded,
        "faq_loaded": faq_loaded,
        "intents_supported": len(bot.classifier.classes) if model_loaded else 0,
        "latency_ms": round(latency * 1000, 2),
    }
    return jsonify(payload), 200


@api_bp.route("/chat", methods=["POST"])
def chat():
    """Primary chatbot endpoint.

    Request JSON:
        {"message": "<user question>"}

    Response JSON:
        {
            "success": true,
            "message": "<original message>",
            "intent": "<predicted intent>",
            "confidence": 0.70,
            "response": "<answer or fallback message>",
            "fallback": false,
            ...
        }
    """
    bot = get_chatbot()
    api_config: APIConfig = current_app.config["API_CONFIG"]

    # Validate content type
    if not request.is_json:
        return _build_error_response("Request must be application/json", 415)

    data = request.get_json(silent=True)
    if data is None:
        return _build_error_response("Invalid JSON body", 400)

    # Validate message field
    message = data.get("message")

    if message is None:
        return _build_error_response("Missing required field: message", 400)

    if not isinstance(message, str):
        message = str(message)

    stripped = message.strip()
    if not stripped:
        return _build_error_response("Message cannot be empty", 400)

    if len(stripped) > api_config.max_message_length:
        return _build_error_response(
            f"Message exceeds maximum length of {api_config.max_message_length} characters",
            413,
        )

    _log.info("API chat request received (len=%d)", len(message))

    try:
        start = time.perf_counter()
        chat_resp = bot.get_response(message)
        latency = time.perf_counter() - start

        payload = _build_chat_response(chat_resp, message)
        payload["latency_ms"] = round(latency * 1000, 2)

        status = 200
        if chat_resp.is_fallback:
            _log.warning(
                "Fallback intent=%s reason=%s conf=%.4f",
                chat_resp.intent,
                chat_resp.fallback_reason,
                chat_resp.confidence,
            )
        else:
            _log.info(
                "Predicted intent=%s confidence=%.4f",
                chat_resp.intent,
                chat_resp.confidence,
            )

        return jsonify(payload), status

    except Exception as exc:
        _log.error("Unhandled exception during chat processing: %s", exc, exc_info=True)
        return _build_error_response(
            "An internal error occurred. Please try again later.", 500
        )


@api_bp.route("/predict", methods=["POST"])
def predict():
    """Intent prediction endpoint.

    Returns only the intent and confidence without generating a full response.

    Request JSON:
        {"message": "<user question>"}

    Response JSON:
        {
            "success": true,
            "intent": "<predicted intent>",
            "confidence": 0.70,
            "confidence_level": "high"
        }
    """
    bot = get_chatbot()

    if not request.is_json:
        return _build_error_response("Request must be application/json", 415)

    data = request.get_json(silent=True)
    if data is None:
        return _build_error_response("Invalid JSON body", 400)

    message = data.get("message")
    if message is None:
        return _build_error_response("Missing required field: message", 400)

    if not isinstance(message, str):
        message = str(message)

    api_config: APIConfig = current_app.config["API_CONFIG"]
    if len(message) > api_config.max_message_length:
        return _build_error_response(
            f"Message exceeds maximum length of {api_config.max_message_length} characters",
            413,
        )

    try:
        prediction = bot.classify(message)
        return jsonify({
            "success": True,
            "intent": prediction.get("intent", "unknown"),
            "confidence": prediction.get("confidence", 0.0),
            "confidence_level": "high" if prediction.get("confidence", 0) >= 0.65 else (
                "medium" if prediction.get("confidence", 0) >= 0.52 else "low"
            ),
        }), 200

    except Exception as exc:
        _log.error("Prediction failed: %s", exc, exc_info=True)
        return _build_error_response(
            "An internal error occurred during prediction.", 500
        )
