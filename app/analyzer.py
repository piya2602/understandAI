"""Orchestrates preprocessing, prompting, LLM calls, and validation."""

from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Tuple

import yaml
from dotenv import load_dotenv
from pydantic import ValidationError

from .llm_client import LLMClient, LLMError
from .models import AnalysisResult, Entity, QAResult
from .nlp_processor import NlpProcessor

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPT_DIR = os.path.join(PROJECT_ROOT, "prompts")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "config.yaml")
NOT_FOUND = "The answer cannot be found in the provided document."

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@lru_cache(maxsize=1)
def get_nlp() -> NlpProcessor:
    return NlpProcessor(use_spacy=True)


@lru_cache(maxsize=1)
def get_llm() -> LLMClient:
    cfg = load_config()["llm"]
    api_key = os.environ.get("LLM_API_KEY")
    return LLMClient(
        base_url=cfg["base_url"],
        model=cfg["model"],
        api_key=api_key,
    )


def _load_prompt(name: str) -> str:
    with open(os.path.join(PROMPT_DIR, name), "r", encoding="utf-8") as fh:
        return fh.read()


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n\n[Document truncated due to length]"


def _local_analysis(nlp: NlpProcessor, text: str) -> AnalysisResult:
    """Full analysis using only local models — no network required."""
    summary = nlp.summarize(text, max_sentences=3)
    sentences = nlp.sentence_tokenize(text)
    return AnalysisResult(
        summary=summary or text[:300],
        sentiment=nlp.sentiment(text),
        keywords=nlp.keywords(text, top_n=8),
        entities=[Entity(**e) for e in nlp.entities(text)],
        important_information=[" ".join(s.split()) for s in sentences[:5]],
    )


def analyze_document(text: str) -> Tuple[AnalysisResult, dict]:
    """Analyze a document with the LLM when available, local NLP otherwise.

    Named entities always come from spaCy when the model is installed, because a
    supervised NER model types spans more consistently than the LLM.

    Returns (result, document statistics).
    Raises ValueError with a user-safe message on failure.
    """
    nlp = get_nlp()
    cfg = load_config()

    cleaned = nlp.clean_text(text)
    if not cleaned:
        raise ValueError("The provided document is empty.")

    stats = nlp.statistics(cleaned)
    stats["spacy_available"] = nlp.spacy_available
    local_entities = [Entity(**e) for e in nlp.entities(cleaned)]

    truncated = _truncate(cleaned, cfg["max_input_chars"])
    llm = get_llm()

    if not llm.enabled:
        stats["engine"] = "local"
        result = _local_analysis(nlp, truncated)
        if local_entities:
            result.entities = local_entities
        return result, stats

    template = _load_prompt("document_analysis.txt")
    prompt = template.replace("{document}", truncated)

    try:
        raw = llm.complete(prompt)
        result = AnalysisResult.model_validate(raw)
        stats["engine"] = "llm"
    except (LLMError, ValidationError):
        stats["engine"] = "local (LLM unavailable)"
        result = _local_analysis(nlp, truncated)

    if local_entities:
        result.entities = local_entities
    return result, stats


def answer_question(document: str, question: str) -> str:
    """Answer a question strictly from the document content."""
    nlp = get_nlp()

    question = (question or "").strip()
    if not question:
        raise ValueError("Please enter a question.")
    if not (document or "").strip():
        raise ValueError("Please analyze a document before asking questions.")

    cfg = load_config()
    doc = _truncate(nlp.clean_text(document), cfg["max_input_chars"])
    # Page markers are extraction artifacts, not document content.
    searchable = re.sub(r"\[Page \d+\]", " ", doc)

    llm = get_llm()
    if llm.enabled:
        template = _load_prompt("question_answering.txt")
        prompt = template.replace("{question}", question).replace("{document}", doc)
        try:
            raw = llm.complete(prompt, max_tokens=cfg["qa"]["max_tokens"])
            answer = QAResult.model_validate(raw).answer.strip()
            if len(answer) > 600:
                answer = answer[:600] + "…"
            return answer
        except (LLMError, ValidationError):
            pass  # fall through to local retrieval

    return nlp.retrieve_answer(question, searchable) or NOT_FOUND
