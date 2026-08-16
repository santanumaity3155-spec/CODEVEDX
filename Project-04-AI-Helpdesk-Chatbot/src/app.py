"""
Module 5: Flask Application Factory
=====================================

Creates and configures the Flask application, registers the API blueprint,
and initialises the chatbot once at startup.

Run with:
    python src/app.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

from flask import Flask, jsonify
from flask_cors import CORS

from api.routes import api_bp
from chatbot_config import ChatbotConfig, get_logger
from config import APIConfig, get_api_config, setup_logging

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

_log = get_logger()


def create_app(config: APIConfig | None = None) -> Flask:
    """Application factory for the AI Helpdesk Chatbot API.

    Args:
        config: Optional APIConfig instance. When omitted, defaults are used.

    Returns:
        A fully configured Flask application instance.
    """
    cfg = config or get_api_config()

    # Configure logging early.
    setup_logging(cfg)
    _log.info("Starting %s v%s", cfg.app_name, cfg.app_version)

    app = Flask(__name__)
    app.config["API_CONFIG"] = cfg
    app.config["CHATBOT_CONFIG"] = cfg.chatbot
    app.config["JSON_SORT_KEYS"] = False

    # CORS
    CORS(app, resources={r"/api/*": {"origins": cfg.allowed_origins}})
    _log.info("CORS allowed origins: %s", cfg.allowed_origins)

    # Register blueprints.
    app.register_blueprint(api_bp, url_prefix=cfg.api_prefix)

    # Health-check alias at root (convenience).
    @app.route("/health", methods=["GET"])
    def root_health():
        from api.routes import health_check
        return health_check()

    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            "success": False,
            "error": "Endpoint not found",
            "message": "The requested resource was not found on this server.",
        }), 404

    @app.errorhandler(405)
    def method_not_allowed(error):
        return jsonify({
            "success": False,
            "error": "Method not allowed",
            "message": "The HTTP method is not allowed for this endpoint.",
        }), 405

    @app.errorhandler(500)
    def internal_error(error):
        _log.error("Unhandled server error: %s", error, exc_info=True)
        return jsonify({
            "success": False,
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again later.",
        }), 500

    _log.info("Flask application created successfully")
    return app


def main() -> int:
    """CLI entry point for running the development server."""
    cfg = get_api_config()

    # Pre-validate chatbot config so startup failures are loud.
    chatbot_problems = cfg.chatbot.validate()
    if chatbot_problems:
        _log.error("Chatbot configuration errors: %s", "; ".join(chatbot_problems))
        return 1

    app = create_app(cfg)

    _log.info(
        "Starting server on %s:%d (debug=%s)",
        cfg.host,
        cfg.port,
        cfg.debug,
    )
    app.run(host=cfg.host, port=cfg.port, debug=cfg.debug)
    return 0


if __name__ == "__main__":
    sys.exit(main())
