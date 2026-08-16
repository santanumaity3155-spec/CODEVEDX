"""
Test suite for Module 6: Admin FAQ Management.

Tests cover:
1. admin module imports
2. list FAQ
3. search FAQ
4. create FAQ
5. create validation
6. duplicate prevention
7. invalid intent rejection
8. missing question rejection
9. missing answer rejection
10. update FAQ
11. update validation
12. update nonexistent FAQ
13. delete FAQ
14. delete nonexistent FAQ
15. persistence
16. FAQ reload
17. chatbot sees updated FAQ
18. chatbot sees deleted FAQ correctly
19. error response format
20. logging behavior where practical
"""

from __future__ import annotations

import csv
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nlp_preprocessor import CANONICAL_INTENTS  # noqa: E402
from chatbot_config import ChatbotConfig  # noqa: E402
from admin import (  # noqa: E402
    FAQManager, ValidationError, PersistenceError, AdminAuthError,
    check_admin_auth,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
ORIGINAL_FAQ = PROJECT_ROOT / "data" / "processed" / "faq_nlp_ready.csv"


def _make_temp_csv(tmp_path: Path, filename: str = "faq_nlp_ready.csv") -> Path:
    """Copy the original FAQ CSV to a temp location and return the path."""
    dest = tmp_path / filename
    shutil.copy2(ORIGINAL_FAQ, dest)
    return dest


def _make_manager(tmp_path: Path):
    """Create a FAQManager backed by a temp copy of the FAQ dataset."""
    csv_path = _make_temp_csv(tmp_path)
    config = ChatbotConfig()
    # Override the FAQ path to point to the temp copy.
    config = ChatbotConfig(
        faq_nlp_ready_csv=csv_path,
        faq_dataset_csv=PROJECT_ROOT / "data" / "processed" / "faq_dataset.csv",
    )
    return FAQManager(faq_csv=csv_path, config=config)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a temporary directory for each test."""
    return tmp_path


@pytest.fixture
def manager(tmp_dir):
    """Provide a fresh FAQManager with a temp CSV copy."""
    return _make_manager(tmp_dir)


@pytest.fixture
def sample_faq_id(manager):
    """Create a sample FAQ and return its ID."""
    record = manager.create_faq(
        question="How do I fix my printer?",
        intent="technical_support",
        answer="Restart the printer and check the network connection.",
        entity="it_support",
    )
    return record["id"]


# ---------------------------------------------------------------------------
# 1. Admin module imports
# ---------------------------------------------------------------------------
class TestModuleImports:
    def test_faq_manager_imports(self):
        assert FAQManager is not None
        assert callable(FAQManager)

    def test_exception_classes_exist(self):
        assert issubclass(ValidationError, Exception)
        assert issubclass(PersistenceError, Exception)
        assert issubclass(AdminAuthError, Exception)

    def test_check_admin_auth_imports(self):
        assert callable(check_admin_auth)


# ---------------------------------------------------------------------------
# 2. List FAQ
# ---------------------------------------------------------------------------
class TestListFaq:
    def test_list_returns_records(self, manager):
        faqs = manager.list_faqs()
        assert len(faqs) >= 1
        for faq in faqs:
            assert "id" in faq
            assert "question" in faq
            assert "intent" in faq
            assert "answer" in faq

    def test_list_limit(self, manager):
        faqs = manager.list_faqs(limit=3)
        assert len(faqs) <= 3

    def test_list_offset(self, manager):
        all_faqs = manager.list_faqs()
        offset_faqs = manager.list_faqs(offset=1)
        assert len(offset_faqs) == len(all_faqs) - 1

    def test_list_filter_by_intent(self, manager):
        faqs = manager.list_faqs(intent="password_reset")
        for faq in faqs:
            assert faq["intent"] == "password_reset"

    def test_list_search(self, manager):
        faqs = manager.list_faqs(search="password")
        for faq in faqs:
            assert "password" in faq["question"].lower()

    def test_list_search_no_match(self, manager):
        faqs = manager.list_faqs(search="zzzznotfound")
        assert len(faqs) == 0


# ---------------------------------------------------------------------------
# 3. Search FAQ (covered in TestListFaq + additional edge cases)
# ---------------------------------------------------------------------------
class TestSearchFaq:
    def test_search_case_insensitive(self, manager):
        faqs = manager.list_faqs(search="Password")
        assert len(faqs) >= 1

    def test_search_partial_match(self, manager):
        faqs = manager.list_faqs(search="reset")
        assert len(faqs) >= 1


# ---------------------------------------------------------------------------
# 4. Create FAQ
# ---------------------------------------------------------------------------
class TestCreateFaq:
    def test_create_returns_record(self, manager):
        record = manager.create_faq(
            question="How do I connect to VPN?",
            intent="wifi_problems",
            answer="Use the company VPN client with your SSO credentials.",
            entity="wifi",
        )
        assert record["id"] >= 1
        assert record["question"] == "How do I connect to VPN?"
        assert record["intent"] == "wifi_problems"
        assert record["answer"] == "Use the company VPN client with your SSO credentials."
        assert record["entity"] == "wifi"

    def test_create_persists_to_csv(self, manager):
        record = manager.create_faq(
            question="How do I request a new keyboard?",
            intent="technical_support",
            answer="Submit a ticket in the IT portal under Hardware Request.",
        )
        # Reload and verify.
        manager.reload()
        fetched = manager.get_faq(record["id"])
        assert fetched is not None
        assert fetched["question"] == record["question"]

    def test_create_increments_id(self, manager):
        r1 = manager.create_faq(
            question="Q1", intent="help", answer="A1"
        )
        r2 = manager.create_faq(
            question="Q2", intent="help", answer="A2"
        )
        assert r2["id"] == r1["id"] + 1

    def test_create_increments_count(self, manager):
        initial = manager.count()
        manager.create_faq(
            question="New FAQ?", intent="help", answer="Help response."
        )
        assert manager.count() == initial + 1


# ---------------------------------------------------------------------------
# 5. Create validation
# ---------------------------------------------------------------------------
class TestCreateValidation:
    def test_create_missing_question(self, manager):
        with pytest.raises(ValidationError):
            manager.create_faq(
                question=None, intent="help", answer="Answer."
            )

    def test_create_empty_question(self, manager):
        with pytest.raises(ValidationError):
            manager.create_faq(
                question="   ", intent="help", answer="Answer."
            )

    def test_create_missing_intent(self, manager):
        with pytest.raises(ValidationError):
            manager.create_faq(
                question="Question?", intent=None, answer="Answer."
            )

    def test_create_empty_intent(self, manager):
        with pytest.raises(ValidationError):
            manager.create_faq(
                question="Question?", intent="  ", answer="Answer."
            )

    def test_create_missing_answer(self, manager):
        with pytest.raises(ValidationError):
            manager.create_faq(
                question="Question?", intent="help", answer=None
            )

    def test_create_empty_answer(self, manager):
        with pytest.raises(ValidationError):
            manager.create_faq(
                question="Question?", intent="help", answer="   "
            )

    def test_create_question_too_long(self, manager):
        with pytest.raises(ValidationError):
            manager.create_faq(
                question="x" * 1001,
                intent="help",
                answer="Answer.",
            )

    def test_create_answer_too_long(self, manager):
        with pytest.raises(ValidationError):
            manager.create_faq(
                question="Question?",
                intent="help",
                answer="x" * 5001,
            )


# ---------------------------------------------------------------------------
# 6. Duplicate prevention
# ---------------------------------------------------------------------------
class TestDuplicatePrevention:
    def test_duplicate_question_rejected(self, manager):
        manager.create_faq(
            question="Unique question text 12345",
            intent="help",
            answer="Answer.",
        )
        with pytest.raises(ValidationError, match="already exists"):
            manager.create_faq(
                question="Unique question text 12345",
                intent="help",
                answer="Different answer.",
            )

    def test_duplicate_case_insensitive(self, manager):
        manager.create_faq(
            question="My printer is broken",
            intent="technical_support",
            answer="Restart it.",
        )
        with pytest.raises(ValidationError):
            manager.create_faq(
                question="MY PRINTER IS BROKEN",
                intent="technical_support",
                answer="Restart it.",
            )

    def test_update_allows_same_question(self, manager, sample_faq_id):
        # Updating a record to keep its own question should be allowed.
        record = manager.get_faq(sample_faq_id)
        updated = manager.update_faq(
            faq_id=sample_faq_id,
            answer="New answer.",
        )
        assert updated["answer"] == "New answer."


# ---------------------------------------------------------------------------
# 7. Invalid intent rejection
# ---------------------------------------------------------------------------
class TestInvalidIntentRejection:
    def test_invalid_intent_rejected(self, manager):
        with pytest.raises(ValidationError, match="Invalid intent"):
            manager.create_faq(
                question="Question?",
                intent="not_a_real_intent",
                answer="Answer.",
            )

    def test_invalid_intent_update_rejected(self, manager, sample_faq_id):
        with pytest.raises(ValidationError, match="Invalid intent"):
            manager.update_faq(
                faq_id=sample_faq_id,
                intent="fake_intent_xyz",
            )

    def test_valid_intent_accepted(self, manager):
        for intent in list(CANONICAL_INTENTS)[:5]:
            record = manager.create_faq(
                question=f"Valid intent test {intent}",
                intent=intent,
                answer="Answer.",
            )
            assert record["intent"] == intent


# ---------------------------------------------------------------------------
# 8. Missing question rejection
# ---------------------------------------------------------------------------
class TestMissingQuestionRejection:
    def test_update_missing_question_ignored(self, manager, sample_faq_id):
        original = manager.get_faq(sample_faq_id)["question"]
        manager.update_faq(
            faq_id=sample_faq_id,
            question=None,
            answer="Updated answer.",
        )
        current = manager.get_faq(sample_faq_id)
        assert current["question"] == original


# ---------------------------------------------------------------------------
# 9. Missing answer rejection
# ---------------------------------------------------------------------------
class TestMissingAnswerRejection:
    def test_update_missing_answer_ignored(self, manager, sample_faq_id):
        original = manager.get_faq(sample_faq_id)["answer"]
        manager.update_faq(
            faq_id=sample_faq_id,
            answer=None,
            question="Updated question.",
        )
        current = manager.get_faq(sample_faq_id)
        assert current["answer"] == original


# ---------------------------------------------------------------------------
# 10. Update FAQ
# ---------------------------------------------------------------------------
class TestUpdateFaq:
    def test_update_question(self, manager, sample_faq_id):
        updated = manager.update_faq(
            faq_id=sample_faq_id,
            question="What is the new printer policy?",
        )
        assert updated["question"] == "What is the new printer policy?"

    def test_update_intent(self, manager, sample_faq_id):
        updated = manager.update_faq(
            faq_id=sample_faq_id,
            intent="help",
        )
        assert updated["intent"] == "help"

    def test_update_answer(self, manager, sample_faq_id):
        updated = manager.update_faq(
            faq_id=sample_faq_id,
            answer="This is the updated answer.",
        )
        assert updated["answer"] == "This is the updated answer."

    def test_update_entity(self, manager, sample_faq_id):
        updated = manager.update_faq(
            faq_id=sample_faq_id,
            entity="laptop",
        )
        assert updated["entity"] == "laptop"

    def test_update_multiple_fields(self, manager, sample_faq_id):
        updated = manager.update_faq(
            faq_id=sample_faq_id,
            question="Combined update question",
            intent="help",
            answer="Combined answer.",
            entity="help",
        )
        assert updated["question"] == "Combined update question"
        assert updated["intent"] == "help"
        assert updated["answer"] == "Combined answer."
        assert updated["entity"] == "help"


# ---------------------------------------------------------------------------
# 11. Update validation
# ---------------------------------------------------------------------------
class TestUpdateValidation:
    def test_update_all_fields_invalid(self, manager, sample_faq_id):
        with pytest.raises(ValidationError):
            manager.update_faq(
                faq_id=sample_faq_id,
                question="",
                intent="fake",
                answer="",
            )


# ---------------------------------------------------------------------------
# 12. Update nonexistent FAQ
# ---------------------------------------------------------------------------
class TestUpdateNonexistentFaq:
    def test_update_nonexistent_raises(self, manager):
        with pytest.raises(ValidationError, match="No FAQ exists with ID"):
            manager.update_faq(
                faq_id=999999,
                question="New question",
            )


# ---------------------------------------------------------------------------
# 13. Delete FAQ
# ---------------------------------------------------------------------------
class TestDeleteFaq:
    def test_delete_returns_true(self, manager, sample_faq_id):
        result = manager.delete_faq(sample_faq_id)
        assert result is True

    def test_delete_removes_record(self, manager, sample_faq_id):
        manager.delete_faq(sample_faq_id)
        fetched = manager.get_faq(sample_faq_id)
        assert fetched is None

    def test_delete_decrements_count(self, manager, sample_faq_id):
        count_before = manager.count()
        manager.delete_faq(sample_faq_id)
        assert manager.count() == count_before - 1

    def test_delete_persists(self, manager, sample_faq_id):
        manager.delete_faq(sample_faq_id)
        manager.reload()
        assert manager.get_faq(sample_faq_id) is None


# ---------------------------------------------------------------------------
# 14. Delete nonexistent FAQ
# ---------------------------------------------------------------------------
class TestDeleteNonexistentFaq:
    def test_delete_nonexistent_returns_false(self, manager):
        result = manager.delete_faq(999999)
        assert result is False


# ---------------------------------------------------------------------------
# 15. Persistence
# ---------------------------------------------------------------------------
class TestPersistence:
    def test_create_persists_to_csv(self, tmp_dir):
        manager = _make_manager(tmp_dir)
        record = manager.create_faq(
            question="Persistent FAQ?",
            intent="help",
            answer="Persistent answer.",
        )
        # Read the CSV directly.
        df = pd.read_csv(manager.faq_csv, encoding="utf-8")
        row = df[df["id"].astype(int) == record["id"]]
        assert len(row) == 1
        assert str(row.iloc[0]["question"]) == "Persistent FAQ?"

    def test_update_persists_to_csv(self, tmp_dir):
        manager = _make_manager(tmp_dir)
        record = manager.create_faq(
            question="Original question",
            intent="help",
            answer="Original answer.",
        )
        manager.update_faq(
            faq_id=record["id"],
            answer="Updated answer.",
        )
        df = pd.read_csv(manager.faq_csv, encoding="utf-8")
        row = df[df["id"].astype(int) == record["id"]]
        assert str(row.iloc[0]["answer"]) == "Updated answer."

    def test_delete_persists_to_csv(self, tmp_dir):
        manager = _make_manager(tmp_dir)
        record = manager.create_faq(
            question="To be deleted",
            intent="help",
            answer="Answer.",
        )
        manager.delete_faq(record["id"])
        df = pd.read_csv(manager.faq_csv, encoding="utf-8")
        assert len(df[df["id"].astype(int) == record["id"]]) == 0

    def test_safe_write_preserves_existing_records(self, tmp_dir):
        manager = _make_manager(tmp_dir)
        original_count = manager.count()
        record = manager.create_faq(
            question="Safe write test",
            intent="help",
            answer="Answer.",
        )
        manager.reload()
        assert manager.count() == original_count + 1
        assert manager.get_faq(record["id"]) is not None


# ---------------------------------------------------------------------------
# 16. FAQ reload
# ---------------------------------------------------------------------------
class TestFaqReload:
    def test_reload_updates_count(self, tmp_dir):
        manager = _make_manager(tmp_dir)
        initial = manager.count()
        manager.create_faq(
            question="Reload test",
            intent="help",
            answer="Answer.",
        )
        assert manager.count() == initial + 1
        # After reload, count stays the same because the temp CSV was mutated.
        manager.reload()
        assert manager.count() == initial + 1

    def test_reload_after_external_edit(self, tmp_dir):
        manager = _make_manager(tmp_dir)
        # Manually add a row to the CSV externally.
        df = pd.read_csv(manager.faq_csv, encoding="utf-8")
        new_row = {
            "id": int(df["id"].max()) + 1,
            "question": "Externally added",
            "clean_question": "externally added",
            "tokens": "externally added",
            "filtered_tokens": "externally added",
            "lemmatized_tokens": "externally added",
            "intent": "help",
            "answer": "Answer.",
            "entity": "help",
            "entities": "help",
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(manager.faq_csv, index=False, encoding="utf-8")
        manager.reload()
        assert manager.count() == len(pd.read_csv(manager.faq_csv, encoding="utf-8"))


# ---------------------------------------------------------------------------
# 17. Chatbot sees updated FAQ
# ---------------------------------------------------------------------------
class TestChatbotSeesUpdatedFaq:
    def test_chatbot_uses_updated_answer(self, tmp_dir):
        manager = _make_manager(tmp_dir)
        config = ChatbotConfig(faq_nlp_ready_csv=manager.faq_csv)
        from chatbot import HelpdeskChatbot

        bot = HelpdeskChatbot(config=config)

        # Create a new FAQ with a distinctive answer.
        record = manager.create_faq(
            question="How do I book a conference room?",
            intent="office_location",
            answer="Use the Outlook calendar to book conference rooms.",
            entity="location",
        )

        # Refresh chatbot.
        bot.refresh_faq()

        # Verify chatbot uses the new answer by checking the FAQ retriever directly.
        ans = bot.faq.get_answer("office_location", "How do I book a conference room?")
        assert ans is not None
        assert "Outlook" in ans


# ---------------------------------------------------------------------------
# 18. Chatbot sees deleted FAQ correctly
# ---------------------------------------------------------------------------
class TestChatbotSeesDeletedFaq:
    def test_chatbot_handles_deleted_faq(self, tmp_dir):
        manager = _make_manager(tmp_dir)
        config = ChatbotConfig(faq_nlp_ready_csv=manager.faq_csv)
        from chatbot import HelpdeskChatbot

        bot = HelpdeskChatbot(config=config)

        # Create and then delete an FAQ.
        record = manager.create_faq(
            question="Temporary FAQ for deletion test",
            intent="help",
            answer="Temporary answer.",
        )
        manager.delete_faq(record["id"])
        bot.refresh_faq()

        # Query should either fallback or return no_answer.
        resp = bot.get_response("Temporary FAQ for deletion test")
        assert resp.is_fallback or resp.fallback_reason == "no_answer"


# ---------------------------------------------------------------------------
# 19. Error response format
# ---------------------------------------------------------------------------
class TestErrorResponseFormat:
    def test_validation_error_format(self, manager):
        try:
            manager.create_faq(
                question="Q", intent="fake", answer="A"
            )
        except ValidationError as exc:
            msg = str(exc)
            assert "Invalid intent" in msg or "intent" in msg.lower()

    def test_auth_error_format(self):
        os.environ["ADMIN_API_KEY"] = "correct-key"
        try:
            with pytest.raises(AdminAuthError):
                check_admin_auth(type("Req", (), {
                    "headers": {"X-Admin-API-Key": "wrong"},
                })())
        finally:
            os.environ.pop("ADMIN_API_KEY", None)

    def test_no_auth_key_passes(self):
        # Should not raise when ADMIN_API_KEY is not set.
        os.environ.pop("ADMIN_API_KEY", None)
        try:
            check_admin_auth(type("Req", (), {"headers": {}})())
        except AdminAuthError:
            pytest.fail("check_admin_auth should pass when ADMIN_API_KEY is not set")


# ---------------------------------------------------------------------------
# 20. Logging behavior
# ---------------------------------------------------------------------------
class TestLoggingBehavior:
    def test_create_logs_info(self, manager, caplog):
        with caplog.at_level(logging.INFO, logger="chatbot"):
            manager.create_faq(
                question="Logging test FAQ",
                intent="help",
                answer="Logging answer.",
            )
        assert any("FAQ created" in rec.message for rec in caplog.records)

    def test_update_logs_info(self, manager, sample_faq_id, caplog):
        with caplog.at_level(logging.INFO, logger="chatbot"):
            manager.update_faq(
                faq_id=sample_faq_id,
                answer="Updated logging answer.",
            )
        assert any("FAQ updated" in rec.message for rec in caplog.records)

    def test_delete_logs_info(self, manager, sample_faq_id, caplog):
        with caplog.at_level(logging.INFO, logger="chatbot"):
            manager.delete_faq(sample_faq_id)
        assert any("FAQ deleted" in rec.message for rec in caplog.records)

    def test_validation_warning_logs(self, manager):
        import logging

        captured = []

        class ListHandler(logging.Handler):
            def emit(self, record):
                captured.append(record)

        handler = ListHandler()
        handler.setLevel(logging.WARNING)
        logger = logging.getLogger("chatbot")
        logger.addHandler(handler)
        try:
            try:
                manager.create_faq(
                    question="Q", intent="fake", answer="A"
                )
            except ValidationError:
                pass
            assert any(rec.levelno == logging.WARNING for rec in captured), (
                f"No WARNING logs captured: {[r.getMessage() for r in captured]}"
            )
        finally:
            logger.removeHandler(handler)


# ---------------------------------------------------------------------------
# Additional edge-case tests
# ---------------------------------------------------------------------------
class TestAdminEdgeCases:
    def test_get_faq_existing(self, manager):
        faqs = manager.list_faqs(limit=1)
        if not faqs:
            pytest.skip("No FAQs available")
        faq = manager.get_faq(faqs[0]["id"])
        assert faq is not None
        assert faq["id"] == faqs[0]["id"]

    def test_get_faq_missing(self, manager):
        assert manager.get_faq(999999) is None

    def test_count_initial(self, manager):
        assert manager.count() >= 1

    def test_supported_intents_nonempty(self, manager):
        intents = manager.supported_intents()
        assert len(intents) >= 1
        for intent in intents:
            assert intent in CANONICAL_INTENTS

    def test_create_with_empty_entity(self, manager):
        record = manager.create_faq(
            question="FAQ with no entity",
            intent="greetings",
            answer="Hello!",
            entity="",
        )
        assert record["entity"] == ""

    def test_entity_normalization(self, manager):
        record = manager.create_faq(
            question="Entity normalization test",
            intent="password_reset",
            answer="Answer.",
            entity="password_reset",
        )
        assert record["entity"] == "password"
