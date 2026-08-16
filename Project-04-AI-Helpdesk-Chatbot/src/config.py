"""
Module 5: Flask Application Configuration
==========================================

Extends the Module 4 ChatbotConfig with Flask-specific settings:
- host / port / debug
- CORS allowed origins
- logging configuration
- API metadata
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from chatbot_config import ChatbotConfig, get_logger


SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent


def _get_env_list(name: str, default: str) -> List[str]:
    """Parse a comma-separated environment variable into a list of strings."""
    raw = os.environ.get(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def _get_env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class APIConfig:
    """Flask-specific application settings.

    All values are derived from environment variables with safe defaults so
    the application works out of the box without a .env file.
    """

    # Application metadata
    app_name: str = "AI Helpdesk Chatbot API"
    app_version: str = "1.0.0"
    api_prefix: str = "/api"

    # Flask server settings
    host: str = os.environ.get("APP_HOST", "127.0.0.1")
    port: int = int(os.environ.get("APP_PORT", "5000"))
    debug: bool = _get_env_bool("APP_DEBUG", False)

    # CORS
    allowed_origins: List[str] = field(
        default_factory=lambda: _get_env_list(
            "ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
        )
    )

    # Logging
    log_level: str = os.environ.get("LOG_LEVEL", "INFO")
    log_format: str = (
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    log_date_format: str = "%Y-%m-%d %H:%M:%S"

    # Chatbot configuration (reuses Module 4 ChatbotConfig)
    chatbot: ChatbotConfig = field(default_factory=ChatbotConfig)

    # Validation
    max_message_length: int = 5000

    def __post_init__(self) -> None:
        # Ensure max_message_length respects chatbot config
        if self.max_message_length > self.chatbot.max_query_length:
            self.max_message_length = self.chatbot.max_query_length


def setup_logging(config: APIConfig) -> logging.Logger:
    """Configure application-wide logging.

    Returns the root 'chatbot' logger after attaching handlers.
    """
    root_logger = logging.getLogger("chatbot")
    root_logger.setLevel(getattr(logging, config.log_level.upper(), logging.INFO))

    if not root_logger.handlers:
        formatter = logging.Formatter(
            fmt=config.log_format,
            datefmt=config.log_date_format,
        )
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    return root_logger


def get_api_config() -> APIConfig:
    """Return the API configuration with default values."""
    return APIConfig()


def get_chatbot_config() -> ChatbotConfig:
    """Return the Module 4 chatbot configuration."""
    return ChatbotConfig()
