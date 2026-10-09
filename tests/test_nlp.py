"""Tests for the NLP preprocessing module."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.nlp_processor import NlpProcessor

SAMPLE = """The University has announced that semester examinations will begin
on October 15. Students must complete registration before October 5!!


Fee dues must be cleared early."""


def make_processor():
    return NlpProcessor(use_spacy=False)  # keep tests fast and dependency-free


def test_clean_text_collapses_whitespace():
    nlp = make_processor()
    cleaned = nlp.clean_text(SAMPLE)
    assert "\n\n\n" not in cleaned
    assert "\x0c" not in cleaned
    assert cleaned == cleaned.strip()


def test_sentence_tokenize_fallback():
    nlp = make_processor()
    sentences = nlp.sentence_tokenize(SAMPLE)
    assert len(sentences) == 3
    assert sentences[0].startswith("The University")


def test_keywords_drop_stopwords():
    nlp = make_processor()
    keywords = nlp.keywords(SAMPLE, top_n=10)
    assert "the" not in keywords
    assert "examinations" in " ".join(keywords) or "examination" in " ".join(keywords)


def test_statistics_counts():
    nlp = make_processor()
    stats = nlp.statistics(SAMPLE)
    assert stats["characters"] == len(SAMPLE)
    assert stats["sentences"] == 3
    assert stats["words"] > 20
    assert stats["avg_sentence_length"] > 0
