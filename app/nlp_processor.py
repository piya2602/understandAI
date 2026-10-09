"""Traditional NLP: preprocessing, spaCy NER, extractive summarization,
lexicon sentiment and TF-IDF answer retrieval. Runs fully offline."""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from typing import Dict, List, Optional

# Words too generic to be useful as keywords
_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any",
    "are", "as", "at", "be", "because", "been", "before", "being", "below", "between",
    "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each",
    "few", "for", "from", "further", "had", "has", "have", "having", "he", "her", "here",
    "hers", "him", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself",
    "just", "me", "more", "most", "my", "no", "nor", "not", "of", "off", "on", "once",
    "only", "or", "other", "our", "out", "over", "own", "same", "she", "should", "so",
    "some", "such", "than", "that", "the", "their", "theirs", "them", "then", "there",
    "these", "they", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom",
    "why", "will", "with", "would", "you", "your",
}

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


class NlpProcessor:
    """Text cleaning, segmentation, tokenization and statistics."""

    def __init__(self, use_spacy: bool = True):
        self._spacy = None
        self._vader = None
        if use_spacy:
            try:
                import spacy

                try:
                    self._spacy = spacy.load("en_core_web_sm")
                except OSError:
                    self._spacy = None  # model not downloaded; fall back
            except ImportError:
                self._spacy = None

    @property
    def spacy_available(self) -> bool:
        return self._spacy is not None

    # ---------- cleaning ----------

    def clean_text(self, text: str) -> str:
        """Normalize whitespace and strip junk characters."""
        text = text.replace("\x0c", " ")  # PDF form feed characters
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    # ---------- segmentation ----------

    def sentence_tokenize(self, text: str) -> List[str]:
        if self._spacy:
            return [s.text.strip() for s in self._spacy(text).sents if s.text.strip()]
        return [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", text)
            if s.strip()
        ]

    # ---------- named entities ----------

    def entities(self, text: str) -> List[Dict[str, str]]:
        """Named entities from the local spaCy model. Empty if the model is absent."""
        if not self._spacy:
            return []
        found: List[Dict[str, str]] = []
        seen = set()
        for ent in self._spacy(text).ents:
            label = " ".join(ent.text.split())
            key = (label.lower(), ent.label_)
            if not label or key in seen:
                continue
            seen.add(key)
            found.append({"text": label, "type": ent.label_})
        return found

    # ---------- tokens / keywords ----------

    def tokens(self, text: str) -> List[str]:
        return [t for t in re.findall(r"[A-Za-z][A-Za-z'-]+", text)]

    def keywords(self, text: str, top_n: int = 10) -> List[str]:
        toks = [
            t.lower().translate(_PUNCT_TABLE)
            for t in self.tokens(text)
        ]
        toks = [
            t for t in toks
            if t and t not in _STOPWORDS and len(t) > 2 and not t.isnumeric()
        ]
        return [w for w, _ in Counter(toks). most_common(top_n)]

    # ---------- stats ----------

    def statistics(self, text: str) -> dict:
        words = self.tokens(text)
        sentences = self.sentence_tokenize(text)
        return {
            "characters": len(text),
            "words": len(words),
            "sentences": len(sentences),
            "avg_sentence_length": round(len(words) / max(len(sentences), 1), 1),
        }

    # ---------- content words shared by the scoring helpers ----------

    def _content_words(self, text: str) -> List[str]:
        return [
            w for w in (t.lower() for t in self.tokens(text))
            if w not in _STOPWORDS and len(w) > 2
        ]

    # ---------- extractive summarization ----------

    def summarize(self, text: str, max_sentences: int = 3) -> str:
        """Pick the highest-scoring sentences by normalized term frequency.

        Extractive: it selects existing sentences rather than writing new text.
        """
        sentences = self.sentence_tokenize(text)
        if not sentences:
            return ""
        if len(sentences) <= max_sentences:
            return " ".join(" ".join(s.split()) for s in sentences)

        freq = Counter(self._content_words(text))
        if not freq:
            return " ".join(" ".join(s.split()) for s in sentences[:max_sentences])

        peak = max(freq.values())
        scored = []
        for index, sentence in enumerate(sentences):
            words = self._content_words(sentence)
            if not words:
                continue
            # Divide by sqrt(len) so long sentences don't win on length alone.
            score = sum(freq[w] / peak for w in words) / math.sqrt(len(words))
            scored.append((score, index, sentence))

        if not scored:
            return " ".join(" ".join(s.split()) for s in sentences[:max_sentences])

        scored.sort(key=lambda item: item[0], reverse=True)
        chosen = sorted(scored[:max_sentences], key=lambda item: item[1])
        return " ".join(" ".join(s.split()) for _, _, s in chosen)

    # ---------- lexicon sentiment ----------

    def sentiment(self, text: str) -> str:
        """Rule/lexicon sentiment via VADER. Returns neutral if unavailable."""
        if self._vader is None:
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

                self._vader = SentimentIntensityAnalyzer()
            except ImportError:
                return "neutral"

        scores = self._vader.polarity_scores(text)
        if scores["pos"] > 0.08 and scores["neg"] > 0.08:
            return "mixed"
        if scores["compound"] >= 0.05:
            return "positive"
        if scores["compound"] <= -0.05:
            return "negative"
        return "neutral"

    # ---------- TF-IDF answer retrieval ----------

    def retrieve_answer(self, question: str, text: str, min_score: float = 0.08) -> Optional[str]:
        """Return the sentence most similar to the question, or None if nothing matches."""
        sentences = self.sentence_tokenize(text)
        query = self._content_words(question)
        if not sentences or not query:
            return None

        docs = [self._content_words(s) for s in sentences]
        total = len(docs)
        doc_freq: Counter = Counter()
        for words in docs:
            doc_freq.update(set(words))

        def weights(words: List[str]) -> Dict[str, float]:
            counts = Counter(words)
            return {
                w: (counts[w] / len(words)) * math.log((total + 1) / (doc_freq.get(w, 0) + 1) + 1)
                for w in counts
            }

        q_vec = weights(query)
        q_norm = math.sqrt(sum(v * v for v in q_vec.values()))
        if not q_norm:
            return None

        best_sentence, best_score = None, 0.0
        for words, sentence in zip(docs, sentences):
            if not words:
                continue
            d_vec = weights(words)
            dot = sum(weight * d_vec.get(term, 0.0) for term, weight in q_vec.items())
            if not dot:
                continue
            d_norm = math.sqrt(sum(v * v for v in d_vec.values()))
            score = dot / (q_norm * d_norm) if d_norm else 0.0
            if score > best_score:
                best_score, best_sentence = score, sentence

        if best_sentence is None or best_score < min_score:
            return None
        return " ".join(best_sentence.split())
