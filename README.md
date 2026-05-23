# Neurotime AI News Search Assistant

An AI-powered news search assistant over ~20,000 scraped Azerbaijani news articles covering May 10–15, 2026. Users can query in natural language via a web interface or Telegram bot.

## What Was Built

- **Semantic search** over 20,915 articles using OpenAI `text-embedding-3-small` embeddings stored in a FAISS vector index
- **Date-aware query parsing** using GPT-4o-mini to extract topics, date ranges, and sentiment from natural-language queries
- **Keyword/entity extraction** across the full dataset or within search results
- **FastAPI backend** serving both the search API and the web frontend
- **Telegram bot** with natural-language interaction (no command-style interface)
- **Web frontend** — dark-themed single-page app with clickable keyword chips and relevance scores

## How Search Works

1. The user's query is sent to **GPT-4o-mini**, which extracts:
   - `topic` — the main subject to search for
   - `date_from` / `date_to` — parsed date range (if mentioned)
   - `sentiment` — positive/negative intent (if mentioned)
2. Articles are **filtered by date** first (O(n) scan of metadata)
3. The topic is **embedded** with `text-embedding-3-small` (1536-dim vector)
4. **Cosine similarity search** is performed over the filtered subset using FAISS `IndexFlatIP` (inner product on L2-normalized vectors)
5. Top-k results are returned with title, source, URL, date, snippet, and relevance score

### Relevance Score

The score is the **cosine similarity** between the query embedding and the article embedding (title + first 1500 chars of content), normalized to [0, 1]. A score of 1.0 means identical vectors; typical strong matches score 0.45–0.65 on this dataset.

### Cost Control

- Embeddings are computed **once** at index-build time (~$0.20 for 20,915 articles at $0.02/1M tokens)
- At query time, only **one embedding call** is made (for the user's topic), plus one GPT-4o-mini call for query parsing (~$0.0001 per query)
- The model never receives article content directly — only embeddings are used for retrieval

## Technologies

| Layer | Technology |
|---|---|
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector store | FAISS (`IndexFlatIP`) |
| Query parsing | OpenAI `gpt-4o-mini` |
| Backend | Python, FastAPI, Uvicorn |
| Telegram bot | `python-telegram-bot` v21 |
| Frontend | Vanilla HTML/CSS/JS |

## Setup

### 1. Prerequisites

- Python 3.10+
- An OpenAI API key
- A Telegram bot token (create via [@BotFather](https://t.me/botfather))

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

Place the provided `news_data.xlsx` file (from Neurotime) into the `data/` directory:
```
data/news_data.xlsx
```

### 5. Build the FAISS index

This step embeds all articles using OpenAI and saves the index locally. Run it once.

```bash
python indexer.py
```

This creates `index.faiss` and `metadata.pkl` in the project root.

### 6. Run the API + web frontend

```bash
python api.py
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

### 7. Run the Telegram bot (separate terminal)

```bash
python bot.py
```

## Usage

### Web interface

- Type any natural-language query into the search box
- Use example chips or the global keywords button
- Results show title, source domain, date, category, snippet, and relevance score
- Clicking a keyword chip runs a new search for that term

### Telegram bot

Send any message in natural language:
- `AccessBank haqqında xəbərlər`
- `SOCAR may 12-də`
- `may 13-15 arasında mənfi bank xəbərləri`
- `/keywords` — show top keywords across full dataset

After results, an inline button lets you see the most common keywords within those results.

## Known Limitations

- The dataset covers **May 10–15, 2026** only. Date queries referencing other periods return no results.
- Source field is derived from the URL domain (no explicit source column in the dataset).
- Keyword extraction is frequency-based with a stopword list; it does not use a trained NER model, so some noise may appear.
- The FAISS index must be rebuilt if the dataset changes (re-run `indexer.py`).
- The Telegram bot requires a public or polling-accessible network connection.
