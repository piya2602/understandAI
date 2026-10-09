# NLP Insight — Intelligent Document Analyzer

A Streamlit app that analyzes documents (pasted text, `.txt`, or **`.pdf` upload**)
using a **hybrid NLP pipeline**: classical/statistical models run locally, and an
optional LLM adds abstractive summarization and generative question answering.

The app is fully functional **without an API key** — local models cover every
feature. A key only upgrades summarization, keyword phrasing and Q&A quality.

## Features

| Feature | Local (always available) | With `LLM_API_KEY` |
|---|---|---|
| Named entities | spaCy `en_core_web_sm` NER | spaCy (kept — more consistent typing) |
| Summary | Extractive sentence ranking | Abstractive |
| Sentiment | VADER lexicon + rules | LLM classification |
| Keywords | Frequency after stopword removal | Semantic key phrases |
| Question answering | TF-IDF sentence retrieval | Document-grounded generation |
| Statistics | Words, sentences, avg sentence length | — |

Other capabilities:

- 📋 Paste text, upload `.txt`, or upload `.pdf` (scanned PDFs are detected with a friendly message)
- 🧾 Raw validated JSON view (great for explaining the pipeline during a demo)
- ✅ Pydantic validation of every analysis result
- 🔁 Works with **any OpenAI-compatible API** (OpenRouter, OpenAI, Groq, Ollama, …)

## Pipeline

```
Text / PDF → NLP preprocessing → ┬→ local models ──────┐
                                 └→ prompt → LLM API ──┴→ validation → Streamlit UI
```

Named entity recognition always uses spaCy when the model is installed: a
supervised NER model assigns entity types more consistently than an LLM, which
tends to blur distinctions such as `TIME` vs `DATE`.

## Setup

```bash
# 1. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows (bash): source .venv/Scripts/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install the spaCy model (required for named entity recognition)
python -m spacy download en_core_web_sm

# 4. Optional: enable the LLM
cp .env.example .env             # then edit .env and add your API key
```

Without `LLM_API_KEY` the app runs in **local mode**: every feature still works,
powered by spaCy, VADER and TF-IDF retrieval.

## Run

```bash
streamlit run app/main.py
```

The sidebar shows which mode is active: **hybrid** (key found) or **local**.

## Configuration

Edit `config/config.yaml` to change the provider. The API key always comes from
the `LLM_API_KEY` environment variable — never put it in the YAML.

| Provider | base_url | model |
|---|---|---|
| OpenRouter | `https://openrouter.ai/api/v1` | `openai/gpt-4o-mini` |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` |
| Groq | `https://api.groq.com/openai/v1` | `llama-3.3-70b-versatile` |
| Ollama (local) | `http://localhost:11434/v1` | `llama3.1` (key can be any dummy text) |

Avoid router aliases such as `openrouter/auto`: they pick a different model on
every call, and reasoning models can spend the whole token budget on reasoning
and return empty content.

## Tests

```bash
pytest -v
```

All tests run offline against the local NLP path — no API key or network needed.

## Repository structure

```
UnderstandAI/
├── app/
│   ├── main.py          # Streamlit frontend
│   ├── analyzer.py      # hybrid pipeline orchestration
│   ├── llm_client.py    # OpenAI-compatible HTTP client
│   ├── nlp_processor.py # spaCy NER, summarization, sentiment, retrieval
│   ├── pdf_utils.py     # PDF text extraction
│   └── models.py        # pydantic response validation
├── prompts/             # prompt templates (outside source code)
├── config/config.yaml   # provider configuration
├── tests/               # pytest suites (run fully offline)
├── data/sample_documents/
├── .env.example
└── requirements.txt
```

## Error handling

Empty input, corrupted/password-protected PDFs, scanned PDFs, oversized
documents, missing API key, timeouts, rate limits, TLS interception on corporate
networks, and invalid LLM JSON are all handled with clear, user-safe messages —
no stack traces are ever shown. If the LLM fails for any reason, the app falls
back to local analysis instead of erroring.

## Note on corporate networks

If HTTPS requests fail with a certificate error, the `truststore` dependency
makes Python use the operating system certificate store, which resolves TLS
interception by corporate proxies.
