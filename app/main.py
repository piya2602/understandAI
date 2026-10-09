"""NLP Insight — Intelligent Document Analyzer (Streamlit frontend).

Run with:  streamlit run app/main.py
"""

from __future__ import annotations

import os
import sys

import streamlit as st

# Make `app` package imports work when Streamlit runs this file directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.analyzer import analyze_document, answer_question  # noqa: E402
from app.pdf_utils import extract_pdf_text, is_pdf  # noqa: E402

APP_TITLE = "NLP Insight"
APP_SUBTITLE = "Intelligent Document Analyzer"

st.set_page_config(page_title=APP_TITLE, page_icon="🔍", layout="wide")


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "document" not in st.session_state:
    st.session_state.document = ""
if "result" not in st.session_state:
    st.session_state.result = None
if "stats" not in st.session_state:
    st.session_state.stats = None


# ---------------------------------------------------------------------------
# Sidebar — status + settings info
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Status")
    from app.analyzer import get_llm, get_nlp

    if get_llm().enabled:
        st.success("LLM API key loaded — hybrid mode (LLM + spaCy).")
    else:
        st.info(
            "**Local mode** — no `LLM_API_KEY` found. Summary, sentiment, "
            "entities and Q&A are produced entirely by local NLP models."
        )

    if not get_nlp().spacy_available:
        st.warning("spaCy model missing — run: python -m spacy download en_core_web_sm")

    st.divider()
    st.caption(
        "Pipeline:  \n"
        "Text / PDF  →  NLP preprocessing  →  Analysis  →  "
        "Validation  →  Dashboard"
    )


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
st.title(f"🔍 {APP_TITLE}")
st.caption(APP_SUBTITLE)

tab_paste, tab_upload = st.tabs(["📋 Paste Text", "📄 Upload File (.txt / .pdf)"])

with tab_paste:
    pasted = st.text_area(
        "Document text",
        value=st.session_state.document,
        height=220,
        placeholder="Paste your document here...",
        label_visibility="collapsed",
    )
    if st.button("Use pasted text", key="use_pasted"):
        st.session_state.document = pasted.strip()
        st.session_state.result = None
        st.session_state.stats = None
        if not st.session_state.document:
            st.error("Please provide a document before analyzing.")
        else:
            st.rerun()

with tab_upload:
    uploaded = st.file_uploader(
        "Upload a document",
        type=["txt", "pdf"],
        help="Text files (.txt) and PDF files (.pdf) are supported.",
    )
    if uploaded is not None:
        if is_pdf(uploaded):
            try:
                text = extract_pdf_text(uploaded)
            except ValueError as exc:
                st.error(str(exc))
                text = ""
        else:
            try:
                text = uploaded.read().decode("utf-8", errors="replace")
            except Exception:
                st.error("Could not read this file. Please upload a plain .txt file.")
                text = ""

        if text:
            st.success(f"Loaded **{uploaded.name}** — {len(text):,} characters.")
            with st.expander("Preview extracted text", expanded=False):
                st.text(text[:3000] + ("…" if len(text) > 3000 else ""))
            if st.button("Use this document", key="use_uploaded"):
                st.session_state.document = text
                st.session_state.result = None
                st.session_state.stats = None
                st.rerun()

doc = st.session_state.document
if doc:
    st.info(f"Current document: **{len(doc):,} characters**")
else:
    st.warning("No document loaded yet. Paste text or upload a .txt/.pdf file above.")


# ---------------------------------------------------------------------------
# Analyze
# ---------------------------------------------------------------------------
if st.button("🔎 Analyze Document", type="primary", disabled=not doc):
    with st.spinner("Analyzing document…"):
        try:
            result, stats = analyze_document(doc)
            st.session_state.result = result
            st.session_state.stats = stats
        except ValueError as exc:
            st.session_state.result = None
            st.session_state.stats = None
            st.error(str(exc))
        except Exception:
            st.error("Something went wrong while analyzing. Please try again.")


# ---------------------------------------------------------------------------
# Results dashboard
# ---------------------------------------------------------------------------
result = st.session_state.result
if result is not None:
    st.divider()
    st.subheader("📊 Analysis Results")

    # Summary + sentiment side by side
    col_sum, col_sent = st.columns([3, 1])
    with col_sum:
        st.markdown("**Summary**")
        st.info(result.summary)
    with col_sent:
        st.markdown("**Sentiment**")
        sentiment_icon = {
            "positive": "🟢 Positive",
            "negative": "🔴 Negative",
            "neutral": "⚪ Neutral",
            "mixed": "🟡 Mixed",
        }.get(result.sentiment, result.sentiment.title())
        st.metric("Overall", sentiment_icon.split(" ", 1)[1], sentiment_icon.split(" ", 1)[0])

    # Keywords
    st.markdown("**Keywords**")
    if result.keywords:
        st.markdown("  ".join(f"`{kw}`" for kw in result.keywords))
    else:
        st.caption("No keywords extracted.")

    # Important information
    st.markdown("**Important Information**")
    if result.important_information:
        for item in result.important_information:
            st.markdown(f"- {item}")
    else:
        st.caption("No important information extracted.")

    # Named entities
    st.markdown("**Named Entities**")
    if result.entities:
        st.table(
            {"Entity": [e.text for e in result.entities],
             "Type": [e.type for e in result.entities]}
        )
    else:
        st.caption("No entities found.")

    # Document statistics from the traditional NLP preprocessor
    if st.session_state.stats:
        with st.expander("📐 Document statistics (traditional NLP)"):
            s = st.session_state.stats
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Characters", f"{s['characters']:,}")
            c2.metric("Words", f"{s['words']:,}")
            c3.metric("Sentences", s["sentences"])
            c4.metric("Avg sentence length", s["avg_sentence_length"])

    # Raw JSON for the professor demo
    with st.expander("🧾 Raw validated JSON response"):
        st.json(result.model_dump())

    st.divider()

    # ------------------------------------------------------------------
    # Ask the document
    # ------------------------------------------------------------------
    st.subheader("💬 Ask the Document")
    question = st.text_input(
        "Question",
        placeholder="e.g. When do the examinations begin?",
        label_visibility="collapsed",
    )
    if st.button("Ask Question", type="secondary"):
        if not question.strip():
            st.error("Please enter a question.")
        else:
            with st.spinner("Searching the document…"):
                try:
                    answer = answer_question(doc, question)
                    st.markdown("**Answer**")
                    st.success(answer)
                except ValueError as exc:
                    st.error(str(exc))
                except Exception:
                    st.error("Something went wrong while answering. Please try again.")
