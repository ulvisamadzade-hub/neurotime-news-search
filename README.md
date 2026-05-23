# ORACULO — AI News Intelligence

An AI-powered news search assistant over ~20,000 scraped Azerbaijani news articles covering **May 10–15, 2026**. Users query in natural language via a web dashboard (ORACULO) or Telegram bot and receive semantically ranked results in seconds.

---

## What Was Built

| Component | Description |
|---|---|
| **ORACULO Web UI** | Dark-themed dashboard with article cards, circular relevance scores, and a live keywords sidebar |
| **Telegram Bot** | Natural-language interface with inline keyword buttons, concurrent-user support |
| **FastAPI Backend** | Serves the search API, keywords API, and static frontend |
| **Hybrid Search** | HyDE + BM25 + FAISS semantic search fused with Reciprocal Rank Fusion |
| **GPT Re-ranker** | GPT-4o-mini verifies each candidate is genuinely about the query |

---

## How Search Works (Full Pipeline)

```
User query
    │
    ▼
1. GPT-4o-mini  →  parse_query()
   Extracts: topic, date_from, date_to, sentiment
    │
    ▼
2. Date filter  →  O(n) scan of metadata
   Narrows candidate pool to articles in range
    │
    ▼
3. Bank entity filter  (if bank name detected)
   Hard-filters to articles mentioning the bank
    │
    ▼
4. HyDE  →  generate_hyde()
   GPT writes a hypothetical Azerbaijani news article on the topic.
   Embedding this is more semantically precise than embedding the raw query.
    │
    ├─── 5a. FAISS semantic search
    │         text-embedding-3-small → cosine similarity over filtered subset
    │
    └─── 5b. BM25 lexical search
              Azerbaijani-tokenised BM25Okapi over filtered subset
    │
    ▼
6. Reciprocal Rank Fusion (RRF)
   Merges semantic + lexical rankings without score normalisation
    │
    ▼
7. GPT Re-ranker  →  rerank()
   Sends top-20 candidates to GPT-4o-mini.
   GPT returns only indices that are DIRECTLY about the query.
   Also applies sentiment filtering (positive / negative).
    │
    ▼
8. Return top-k results
   Each result: title, source, url, published_at, snippet, relevance_score
```

### Why HyDE?

Raw short queries embed poorly. A hypothetical 3-4 sentence news article on the same topic produces a vector that sits much closer to real articles in embedding space. This significantly improves recall for named entities and domain-specific Azerbaijani terms.

### Why BM25 + RRF?

Semantic search misses exact keyword matches (e.g. specific person names, company codes). BM25 captures these. RRF merges both rankings without needing to normalise their scores — it only cares about rank positions.

### Why GPT Re-ranker?

FAISS + BM25 can surface articles that contain the right words but aren't really *about* the topic. The re-ranker acts as a strict relevance gate, discarding false positives before returning results to the user.

### Relevance Score

The displayed score is the **cosine similarity** between the HyDE embedding and the article embedding (normalised to [0, 1]). Typical strong matches score 0.65+; partial matches 0.45–0.65.

### Cost per Query

| Call | Model | Est. cost |
|---|---|---|
| parse_query | gpt-4o-mini | ~$0.00005 |
| generate_hyde | gpt-4o-mini | ~$0.00015 |
| embed | text-embedding-3-small | ~$0.00001 |
| rerank | gpt-4o-mini | ~$0.00020 |
| **Total** | | **~$0.00041 / query** |

---

## Architecture

```
┌─────────────────────────────────┐
│         ORACULO Web UI          │  browser → localhost:8000
│  (frontend/index.html)          │
└────────────────┬────────────────┘
                 │ HTTP
┌────────────────▼────────────────┐
│         FastAPI  (api.py)       │
│  POST /api/search               │
│  GET  /api/keywords             │
│  POST /api/keywords/search      │
└────────────────┬────────────────┘
                 │
┌────────────────▼────────────────┐
│      search.py  (pipeline)      │
│  parse_query → HyDE → FAISS     │
│  BM25 → RRF → rerank            │
└──────┬──────────────────────────┘
       │
  ┌────▼────┐  ┌──────────┐  ┌─────────┐
  │ FAISS   │  │ BM25     │  │ OpenAI  │
  │ index   │  │ index    │  │ API     │
  │.faiss   │  │ (memory) │  │         │
  └─────────┘  └──────────┘  └─────────┘

┌─────────────────────────────────┐
│      Telegram Bot (bot.py)      │
│  asyncio.to_thread → search()   │
│  watchdog auto-restart          │
└─────────────────────────────────┘
```

---

## Technologies

| Layer | Technology |
|---|---|
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector store | FAISS `IndexFlatIP` |
| Lexical search | BM25Okapi (`rank-bm25`) |
| Rank fusion | Reciprocal Rank Fusion (RRF) |
| Query parsing + HyDE + rerank | OpenAI `gpt-4o-mini` |
| Backend | Python 3.10+, FastAPI, Uvicorn |
| Telegram bot | `python-telegram-bot` v21 |
| Frontend | ORACULO — single-page HTML/CSS/JS dashboard |

---

## Setup

### 1. Prerequisites

- Python 3.10+
- OpenAI API key
- Telegram bot token (create via [@BotFather](https://t.me/botfather))

### 2. Install dependencies

```bash
cd neurotime
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and add your keys
```

`.env` must contain:
```
OPENAI_API_KEY=sk-proj-...
TELEGRAM_BOT_TOKEN=your_bot_token
```

### 4. Add the dataset

Place `news_data.xlsx` into the `data/` directory:
```
data/news_data.xlsx
```

### 5. Build the FAISS index

Run once — embeds all articles and saves the index locally.

```bash
python indexer.py
```

Creates `index.faiss` and `metadata.pkl` in the project root.

### 6. Run the API + web frontend

```bash
python api.py
```

Open [http://localhost:8000](http://localhost:8000) — ORACULO loads automatically.

### 7. Run the Telegram bot

In a separate terminal (or use the watchdog script):

```bash
# Direct
python bot.py

# With auto-restart watchdog
nohup bash -c 'while true; do python bot.py; sleep 3; done' > /tmp/bot.log 2>&1 &
```

---

## Usage

### ORACULO Web UI

- Type any natural-language query into the search bar
- Click example chips to run sample queries
- Results show title, source, date, category, snippet, and a circular AI relevance score
- Sidebar shows top keywords — toggle between **Global** (full dataset) and **Search** (current results)
- Clicking any keyword runs a new search for that term

### Telegram Bot

Send any message in natural language:
```
AccessBank haqqında xəbərlər
SOCAR may 12-də
may 13-15 arasında mənfi bank xəbərləri
```

Commands:
- `/keywords` — top keywords across full dataset
- `/help` — usage guide

After results, an inline button lets you see the most common keywords within those specific results.

---

## Known Limitations

- Dataset covers **May 10–15, 2026** only. Queries referencing other dates return no results.
- Keyword extraction is frequency-based with a stopword list — not a trained NER model, so some noise may appear.
- The FAISS index must be rebuilt if the dataset changes (`python indexer.py`).
- The Telegram bot uses long-polling; it requires a stable outbound internet connection.
- Each query makes ~4 OpenAI API calls sequentially (~3–6s total latency depending on OpenAI response times).
