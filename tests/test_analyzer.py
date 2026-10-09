"""Tests for the analyzer pipeline (runs fully offline in mock LLM mode)."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from app.analyzer import analyze_document, answer_question, NOT_FOUND, get_llm
from app.models import AnalysisResult, Entity

SAMPLE_DOC = (
    "The University has announced that semester examinations will begin on "
    "October 15. Students must complete registration before October 5. "
    "Examination results will be declared within six weeks."
)


@pytest.fixture()
def local_only(monkeypatch):
    """Force the analyzer down the local-NLP path (no API key)."""
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    get_llm.cache_clear()
    yield
    get_llm.cache_clear()


def test_llm_disabled_without_key(local_only):
    assert get_llm().enabled is False


def test_analyze_document_returns_validated_result(local_only):
    result, stats = analyze_document(SAMPLE_DOC)
    assert isinstance(result, AnalysisResult)
    assert result.summary
    assert result.sentiment in {"positive", "negative", "neutral", "mixed"}
    assert stats["words"] > 10
    assert stats["engine"] == "local"


def test_local_summary_is_extracted_from_document(local_only):
    result, _ = analyze_document(SAMPLE_DOC)
    assert "mock" not in result.summary.lower()
    assert any(word in result.summary for word in ("University", "examinations", "registration"))


def test_analyze_empty_document_raises(local_only):
    with pytest.raises(ValueError):
        analyze_document("   \n  ")


def test_qa_uses_document_content(local_only):
    answer = answer_question(SAMPLE_DOC, "When do the examinations begin?")
    assert "October 15" in answer


def test_qa_returns_not_found_for_unrelated_question(local_only):
    answer = answer_question(SAMPLE_DOC, "Who is the president of France?")
    assert answer == NOT_FOUND


def test_qa_rejects_empty_question(local_only):
    with pytest.raises(ValueError):
        answer_question(SAMPLE_DOC, "   ")


def test_qa_requires_document(local_only):
    with pytest.raises(ValueError):
        answer_question("", "What is this about?")


def test_entity_model_validation():
    entity = Entity(text="October 15", type="DATE")
    assert entity.text == "October 15"
    with pytest.raises(Exception):
        Entity(text="x", type="D")  # type shorter than min_length


def test_not_found_message():
    assert NOT_FOUND == "The answer cannot be found in the provided document."
