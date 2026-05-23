import os
import json
import logging
import pickle
from datetime import datetime, date

import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "index.faiss")
META_PATH = os.path.join(BASE_DIR, "metadata.pkl")

_index = None
_metadata: list[dict] = []


def load_index():
    global _index, _metadata
    logger.info("Loading FAISS index from %s", INDEX_PATH)
    _index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        _metadata = pickle.load(f)
    logger.info("Index loaded: %d vectors, %d articles", _index.ntotal, len(_metadata))


def parse_query(query: str) -> dict:
    """Use GPT-4o-mini to extract structured parameters from a natural-language query."""
    system = (
        "You are a query parser for an Azerbaijani news search engine. "
        "The dataset covers May 10–15, 2026. "
        "When the user mentions dates, map them to this range. "
        "Extract search parameters and return ONLY valid JSON."
    )
    user = f"""Extract from this query:
- topic: main search topic as a short phrase (string)
- date_from: start date YYYY-MM-DD or null (dataset range: 2026-05-10 to 2026-05-15)
- date_to: end date YYYY-MM-DD or null (dataset range: 2026-05-10 to 2026-05-15)
- sentiment: "negative", "positive", or null
- category: relevant category hint or null (e.g. "İqtisadiyyat", "Siyasət", "İdman")

Query: "{query}"

Return only valid JSON with these 5 keys."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    try:
        parsed = json.loads(response.choices[0].message.content)
        logger.info("Query parsed — topic=%r date_from=%s date_to=%s sentiment=%s",
                    parsed.get("topic"), parsed.get("date_from"), parsed.get("date_to"), parsed.get("sentiment"))
        return parsed
    except Exception as e:
        logger.warning("Query parse failed (%s), falling back to raw query", e)
        return {"topic": query, "date_from": None, "date_to": None, "sentiment": None, "category": None}


def _embed(text: str) -> np.ndarray:
    response = client.embeddings.create(model="text-embedding-3-small", input=[text])
    vec = np.array([response.data[0].embedding], dtype=np.float32)
    faiss.normalize_L2(vec)
    return vec


def _to_date(val) -> date | None:
    if val is None:
        return None
    if isinstance(val, date):
        return val
    try:
        return datetime.strptime(str(val)[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def search(query: str, top_k: int = 10) -> dict:
    if _index is None:
        load_index()

    params = parse_query(query)
    topic = params.get("topic") or query
    date_from = _to_date(params.get("date_from"))
    date_to = _to_date(params.get("date_to"))
    cat_hint = (params.get("category") or "").strip().lower()

    # --- date + optional category filter ---
    filtered_indices = []
    for i, m in enumerate(_metadata):
        dt = m.get("created_at")
        if dt is not None:
            try:
                d = dt.date() if hasattr(dt, "date") else _to_date(dt)
            except Exception:
                d = None
        else:
            d = None

        if date_from and (d is None or d < date_from):
            continue
        if date_to and (d is None or d > date_to):
            continue

        if cat_hint:
            article_cat = str(m.get("category", "")).lower()
            if cat_hint not in article_cat:
                # soft skip — only skip if category is explicitly set and doesn't match
                pass  # don't hard-filter on category; let vector score decide

        filtered_indices.append(i)

    total_in_range = len(filtered_indices)

    logger.info("Search — query=%r filtered=%d/%d", query, len(filtered_indices), len(_metadata))

    if not filtered_indices:
        logger.info("No articles matched date filter")
        return {"results": [], "params": params, "total_in_range": 0}

    # --- embed query ---
    query_vec = _embed(topic)

    # --- vector search over filtered subset ---
    k = min(top_k, len(filtered_indices))

    if len(filtered_indices) < len(_metadata):
        # reconstruct vectors for the filtered subset
        sub_vecs = np.zeros((len(filtered_indices), _index.d), dtype=np.float32)
        for j, idx in enumerate(filtered_indices):
            _index.reconstruct(idx, sub_vecs[j])
        sub_index = faiss.IndexFlatIP(_index.d)
        sub_index.add(sub_vecs)
        scores, local_ids = sub_index.search(query_vec, k)
        result_indices = [filtered_indices[lid] for lid in local_ids[0] if lid >= 0]
        result_scores = [float(s) for s in scores[0] if s > -1]
    else:
        scores, ids = _index.search(query_vec, k)
        result_indices = [i for i in ids[0] if i >= 0]
        result_scores = [float(s) for s in scores[0]]

    results = []
    for idx, score in zip(result_indices, result_scores):
        m = _metadata[idx]
        dt = m.get("created_at")
        published = str(dt)[:19] if dt is not None else ""
        results.append(
            {
                "title": m.get("title", ""),
                "source": m.get("source", ""),
                "url": m.get("url", ""),
                "published_at": published,
                "snippet": m.get("snippet", ""),
                "category": m.get("category", ""),
                "relevance_score": round(score, 4),
            }
        )

    logger.info("Returning %d results (top score=%.4f)", len(results), results[0]["relevance_score"] if results else 0)
    return {"results": results, "params": params, "total_in_range": total_in_range}
