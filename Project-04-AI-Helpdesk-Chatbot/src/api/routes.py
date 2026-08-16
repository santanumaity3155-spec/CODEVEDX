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
from admin import FAQManager, check_admin_auth, AdminAuthError, ValidationError, PersistenceError

_log = logging.getLogger("chatbot.api")

api_bp = Blueprint("api", __name__)
admin_bp = Blueprint("admin", __name__)

# Module-level chatbot instance (loaded once at startup).
_chatbot: HelpdeskChatbot | None = None
_faq_manager: FAQManager | None = None


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


def get_faq_manager() -> FAQManager:
    """Return the shared FAQ manager (lazy-init)."""
    global _faq_manager
    if _faq_manager is None:
        config: ChatbotConfig = current_app.config["CHATBOT_CONFIG"]
        chatbot = get_chatbot()

        def _refresh():
            chatbot.refresh_faq()

        _faq_manager = FAQManager(
            faq_csv=config.faq_nlp_ready_csv,
            config=config,
            refresh_callback=_refresh,
        )
    return _faq_manager


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


# ---------------------------------------------------------------------------
# Admin routes
# ---------------------------------------------------------------------------
def _admin_error(message: str, status_code: int = 400) -> tuple[Dict[str, Any], int]:
    """Return a structured admin error payload."""
    return {
        "success": False,
        "error": message,
    }, status_code


@admin_bp.route("/faqs", methods=["GET"])
def list_faqs():
    """List FAQ records with optional filtering.

    Query parameters:
        intent (optional): Filter by intent label.
        search (optional): Substring search in question text.
        limit (optional): Maximum number of records.
        offset (optional): Number of records to skip.

    Response:
        {
            "success": true,
            "count": 10,
            "faqs": [...]
        }
    """
    try:
        check_admin_auth(request)
    except AdminAuthError as exc:
        return _admin_error(str(exc), 403)

    manager = get_faq_manager()
    intent = request.args.get("intent")
    search = request.args.get("search")
    limit = request.args.get("limit", type=int)
    offset = request.args.get("offset", type=int) or 0

    try:
        faqs = manager.list_faqs(intent=intent, search=search, limit=limit, offset=offset)
        return jsonify({
            "success": True,
            "count": len(faqs),
            "faqs": faqs,
        }), 200
    except Exception as exc:
        _log.error("List FAQs failed: %s", exc, exc_info=True)
        return _admin_error("An internal error occurred.", 500)


@admin_bp.route("/faqs", methods=["POST"])
def create_faq():
    """Create a new FAQ record.

    Request JSON:
        {
            "question": "<question text>",
            "intent": "<intent label>",
            "answer": "<answer text>",
            "entity": "<optional entity>"
        }

    Response:
        {
            "success": true,
            "id": 295,
            "question": "...",
            "intent": "...",
            "answer": "...",
            "entity": "..."
        }
    """
    try:
        check_admin_auth(request)
    except AdminAuthError as exc:
        return _admin_error(str(exc), 403)

    if not request.is_json:
        return _admin_error("Request must be application/json", 415)

    data = request.get_json(silent=True)
    if data is None:
        return _admin_error("Invalid JSON body", 400)

    question = data.get("question")
    intent = data.get("intent")
    answer = data.get("answer")
    entity = data.get("entity")

    if question is None:
        return _admin_error("Missing required field: question", 400)
    if intent is None:
        return _admin_error("Missing required field: intent", 400)
    if answer is None:
        return _admin_error("Missing required field: answer", 400)

    manager = get_faq_manager()
    try:
        record = manager.create_faq(question, intent, answer, entity)
        return jsonify({"success": True, **record}), 201
    except ValidationError as exc:
        _log.warning("FAQ creation validation failed: %s", exc)
        return _admin_error(str(exc), 400)
    except PersistenceError as exc:
        _log.error("FAQ creation persistence failed: %s", exc)
        return _admin_error(str(exc), 500)


@admin_bp.route("/faqs/<int:faq_id>", methods=["PUT"])
def update_faq(faq_id: int):
    """Update an existing FAQ record.

    Request JSON (all fields optional; only provided fields are updated):
        {
            "question": "<new question>",
            "intent": "<new intent>",
            "answer": "<new answer>",
            "entity": "<new entity>"
        }

    Response:
        {
            "success": true,
            "id": 10,
            "question": "...",
            "intent": "...",
            "answer": "...",
            "entity": "..."
        }
    """
    try:
        check_admin_auth(request)
    except AdminAuthError as exc:
        return _admin_error(str(exc), 403)

    if not request.is_json:
        return _admin_error("Request must be application/json", 415)

    data = request.get_json(silent=True)
    if data is None:
        return _admin_error("Invalid JSON body", 400)

    # At least one field must be provided.
    if not any(k in data for k in ("question", "intent", "answer", "entity")):
        return _admin_error("No update fields provided.", 400)

    manager = get_faq_manager()
    try:
        record = manager.update_faq(
            faq_id=faq_id,
            question=data.get("question"),
            intent=data.get("intent"),
            answer=data.get("answer"),
            entity=data.get("entity"),
        )
        return jsonify({"success": True, **record}), 200
    except ValidationError as exc:
        _log.warning("FAQ update validation failed for id=%d: %s", faq_id, exc)
        if "No FAQ exists with ID" in str(exc):
            return _admin_error(str(exc), 404)
        return _admin_error(str(exc), 400)
    except PersistenceError as exc:
        _log.error("FAQ update persistence failed for id=%d: %s", faq_id, exc)
        return _admin_error(str(exc), 500)


@admin_bp.route("/faqs/<int:faq_id>", methods=["DELETE"])
def delete_faq(faq_id: int):
    """Delete an FAQ record by ID.

    Response:
        {
            "success": true,
            "message": "FAQ deleted."
        }
    """
    try:
        check_admin_auth(request)
    except AdminAuthError as exc:
        return _admin_error(str(exc), 403)

    manager = get_faq_manager()
    try:
        deleted = manager.delete_faq(faq_id)
        if not deleted:
            return _admin_error(f"No FAQ exists with ID {faq_id}.", 404)
        return jsonify({
            "success": True,
            "message": "FAQ deleted.",
        }), 200
    except PersistenceError as exc:
        _log.error("FAQ delete persistence failed for id=%d: %s", faq_id, exc)
        return _admin_error(str(exc), 500)


@admin_bp.route("/reload", methods=["POST"])
def reload_faq():
    """Reload the FAQ dataset from disk and refresh the chatbot.

    Response:
        {
            "success": true,
            "message": "FAQ data reloaded.",
            "faq_count": 294
        }
    """
    try:
        check_admin_auth(request)
    except AdminAuthError as exc:
        return _admin_error(str(exc), 403)

    try:
        manager = get_faq_manager()
        manager.reload()
        chatbot = get_chatbot()
        chatbot.refresh_faq()
        return jsonify({
            "success": True,
            "message": "FAQ data reloaded.",
            "faq_count": manager.count(),
        }), 200
    except Exception as exc:
        _log.error("FAQ reload failed: %s", exc, exc_info=True)
        return _admin_error("An internal error occurred during reload.", 500)

