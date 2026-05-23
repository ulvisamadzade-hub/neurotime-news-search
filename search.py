import os
import json
import logging
import pickle
import re
from datetime import datetime, date

import numpy as np
import faiss
from openai import OpenAI
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
from banks import detect_bank

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=30.0)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "index.faiss")
META_PATH = os.path.join(BASE_DIR, "metadata.pkl")

ENTITY_FILTER_MIN = 5   # fall back to full set if entity filter yields fewer than this
MIN_SCORE = 0.38        # semantic score floor — results below this are dropped
SCORE_GAP = 0.08        # cut when consecutive semantic score drops by more than this
RRF_K = 60              # RRF constant (higher = less sensitive to top ranks)
CANDIDATE_MULTIPLIER = 5  # fetch top_k * this from each retriever before fusion
HYDE_MAX_TOKENS = 200   # max tokens for hypothetical document
RERANK_CANDIDATES = 20  # how many candidates to send to GPT re-ranker

_index = None
_metadata: list[dict] = []
_bm25: BM25Okapi | None = None


# ---------------------------------------------------------------------------
# Tokeniser (shared by BM25 index build and query-time scoring)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Lowercase + extract alphabetic tokens (handles Azerbaijani chars)."""
    return re.findall(r"[a-zəüöğşçıa-zа-яё]{2,}", text.lower())


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def load_index():
    global _index, _metadata, _bm25
    logger.info("Loading FAISS index from %s", INDEX_PATH)
    _index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        _metadata = pickle.load(f)
    logger.info("Index loaded: %d vectors, %d articles", _index.ntotal, len(_metadata))

    logger.info("Building BM25 index over %d articles…", len(_metadata))
    corpus = [
        _tokenize(f"{m.get('title', '')} {m.get('snippet', '')}")
        for m in _metadata
    ]
    _bm25 = BM25Okapi(corpus)
    logger.info("BM25 index ready")


# ---------------------------------------------------------------------------
# GPT helpers
# ---------------------------------------------------------------------------

def parse_query(query: str) -> dict:
    """Extract structured search params from a natural-language query."""
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
                    parsed.get("topic"), parsed.get("date_from"),
                    parsed.get("date_to"), parsed.get("sentiment"))
        return parsed
    except Exception as e:
        logger.warning("Query parse failed (%s), falling back to raw query", e)
        return {"topic": query, "date_from": None, "date_to": None,
                "sentiment": None, "category": None}


def generate_hyde(topic: str) -> str:
    """Generate a short hypothetical Azerbaijani news article for the topic.

    Embedding this paragraph instead of the raw query produces a vector that
    is semantically much closer to real articles on the same subject.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Sən Azərbaycan dilində xəbər məqaləsi yazan peşəkar jurnalistsən. "
                    "Verilən mövzu haqqında qısa, real bir xəbər mətni yaz.\n\n"
                    "Azərbaycanda əsas qurumlar:\n"
                    "- Neft: SOCAR (Azərbaycan Dövlət Neft Şirkəti)\n"
                    "- Dövlət bankı: ABB (Azərbaycan Beynəlxalq Bankı), Kapital Bank, Xalq Bank\n"
                    "- Hava yolu: AZAL (Azərbaycan Hava Yolları)\n"
                    "- Mərkəzi Bank: Azərbaycan Mərkəzi Bankı\n"
                    "- Şəhər: Bakı, Gəncə, Sumqayıt, Naxçıvan\n"
                    "Mövzu bu qurumlara aid olduqda onları mütləq xəbərdə işlət."
                ),
            },
            {
                "role": "user",
                "content": (
                    f'Mövzu: "{topic}"\n\n'
                    "3-4 cümlədən ibarət Azərbaycan xəbər məqaləsi yaz. "
                    "Konkret şirkət, qurum və şəxs adları işlət. "
                    "Rəsmi xəbər dili istifadə et."
                ),
            },
        ],
        max_tokens=HYDE_MAX_TOKENS,
        temperature=0.3,
    )
    doc = response.choices[0].message.content.strip()
    logger.info("HyDE document: %s…", doc[:120])
    return doc


# ---------------------------------------------------------------------------
# GPT Re-ranker
# ---------------------------------------------------------------------------

def rerank(query: str, candidates: list[dict], sentiment: str | None) -> list[dict]:
    """Use GPT-4o-mini to verify which candidates are genuinely about the query.

    Sends article titles + short snippets to GPT and asks it to return only
    the indices that are directly relevant. Also applies sentiment filtering
    when the user asked for positive/negative news.
    """
    if not candidates:
        return candidates

    articles_text = "\n\n".join(
        f"[{i}] Title: {a['title']}\nSnippet: {a['snippet'][:200]}"
        for i, a in enumerate(candidates)
    )

    sentiment_line = ""
    if sentiment == "negative":
        sentiment_line = "Only include articles with negative, risky, critical, or alarming content.\n"
    elif sentiment == "positive":
        sentiment_line = "Only include articles with positive, favorable, or optimistic content.\n"

    prompt = f"""You are a relevance judge for an Azerbaijani news search engine.

User query: "{query}"
{sentiment_line}
Below are {len(candidates)} candidate articles. Return ONLY the indices of articles that are DIRECTLY and GENUINELY about the query topic. Be strict — exclude articles that merely mention a related keyword in passing without the article being truly about the topic.

{articles_text}

Return a JSON object with a single key "relevant" containing a list of integer indices (0-based). Example: {{"relevant": [0, 3, 7]}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=200,
        )
        result = json.loads(response.choices[0].message.content)
        relevant_indices = [i for i in result.get("relevant", []) if isinstance(i, int) and i < len(candidates)]
        logger.info("Re-ranker: %d/%d candidates kept", len(relevant_indices), len(candidates))
        return [candidates[i] for i in relevant_indices]
    except Exception as e:
        logger.warning("Re-ranker failed (%s), returning all candidates", e)
        return candidates


# ---------------------------------------------------------------------------
# Embedding + date helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Core search
# ---------------------------------------------------------------------------

def search(query: str, top_k: int = 10) -> dict:
    if _index is None:
        load_index()

    # 1. Parse query
    params = parse_query(query)
    topic = params.get("topic") or query
    date_from = _to_date(params.get("date_from"))
    date_to = _to_date(params.get("date_to"))

    # 2. Date filter
    filtered_indices = []
    for i, m in enumerate(_metadata):
        dt = m.get("created_at")
        try:
            d = dt.date() if hasattr(dt, "date") else _to_date(dt)
        except Exception:
            d = None
        if date_from and (d is None or d < date_from):
            continue
        if date_to and (d is None or d > date_to):
            continue
        filtered_indices.append(i)

    total_in_range = len(filtered_indices)

    # 3. Bank entity filter
    bank = detect_bank(query)
    entity_applied = False
    if bank and filtered_indices:
        aliases_lower = [a.lower() for a in bank["aliases"]]
        entity_indices = [
            i for i in filtered_indices
            if any(
                alias in (
                    _metadata[i].get("title", "") + " " + _metadata[i].get("snippet", "")
                ).lower()
                for alias in aliases_lower
            )
        ]
        if len(entity_indices) == 0:
            logger.info("Bank entity filter: %r → 0 articles, returning empty", bank["canonical"])
            params["entity"] = bank["canonical"]
            return {"results": [], "params": params, "total_in_range": 0}
        else:
            logger.info("Bank entity filter: %r → %d articles (from %d)",
                        bank["canonical"], len(entity_indices), len(filtered_indices))
            filtered_indices = entity_indices
            entity_applied = True
            params["entity"] = bank["canonical"]

    logger.info("Search — query=%r candidates=%d entity_filter=%s",
                query, len(filtered_indices), entity_applied)

    if not filtered_indices:
        return {"results": [], "params": params, "total_in_range": 0}

    # 4. HyDE — generate hypothetical article, embed it
    hyde_doc = generate_hyde(topic)
    query_vec = _embed(hyde_doc)

    # 5. Semantic retrieval via FAISS over filtered subset
    n_candidates = min(top_k * CANDIDATE_MULTIPLIER, len(filtered_indices))

    if len(filtered_indices) < len(_metadata):
        sub_vecs = np.zeros((len(filtered_indices), _index.d), dtype=np.float32)
        for j, idx in enumerate(filtered_indices):
            _index.reconstruct(idx, sub_vecs[j])
        sub_index = faiss.IndexFlatIP(_index.d)
        sub_index.add(sub_vecs)
        scores, local_ids = sub_index.search(query_vec, n_candidates)
        sem_indices = [filtered_indices[lid] for lid in local_ids[0] if lid >= 0]
        sem_scores = {filtered_indices[lid]: float(s)
                      for lid, s in zip(local_ids[0], scores[0]) if lid >= 0}
    else:
        scores, ids = _index.search(query_vec, n_candidates)
        sem_indices = [i for i in ids[0] if i >= 0]
        sem_scores = {i: float(s) for i, s in zip(ids[0], scores[0]) if i >= 0}

    # 6. BM25 retrieval over filtered subset
    # For Azerbaijani topics (contain AZ-specific chars), use the raw topic tokens
    # so BM25 targets specific query terms. For English topics, use HyDE tokens
    # because the Azerbaijani hypothetical article bridges the language gap.
    _AZ_CHARS = set('əüöğşçı')
    query_tokens = _tokenize(topic)
    hyde_tokens = _tokenize(hyde_doc)
    topic_is_azerbaijani = any(c in _AZ_CHARS for t in query_tokens for c in t)
    bm25_tokens = query_tokens if (topic_is_azerbaijani and query_tokens) else hyde_tokens
    all_bm25 = _bm25.get_scores(bm25_tokens)
    bm25_filtered = sorted(
        [(idx, all_bm25[idx]) for idx in filtered_indices],
        key=lambda x: x[1], reverse=True
    )[:n_candidates]
    bm25_indices = [idx for idx, _ in bm25_filtered]
    bm25_scores = {idx: score for idx, score in bm25_filtered}

    # 7. Reciprocal Rank Fusion
    sem_rank = {idx: rank for rank, idx in enumerate(sem_indices)}
    bm25_rank = {idx: rank for rank, (idx, _) in enumerate(bm25_filtered)}

    all_candidates = set(sem_indices) | set(bm25_indices)
    rrf_scores = {
        idx: (1 / (RRF_K + sem_rank.get(idx, n_candidates)) +
              1 / (RRF_K + bm25_rank.get(idx, n_candidates)))
        for idx in all_candidates
    }

    # 8. Rank candidates
    if entity_applied:
        ranked = sorted(sem_scores.keys(), key=lambda x: sem_scores[x], reverse=True)
    else:
        ranked = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    # 9. Build candidate list for re-ranker (top RERANK_CANDIDATES)
    # Normalise BM25 scores so they're in a comparable range to semantic scores
    max_bm25 = max(bm25_scores.values()) if bm25_scores else 1.0
    candidates = []
    for idx in ranked[:RERANK_CANDIDATES]:
        m = _metadata[idx]
        dt = m.get("created_at")
        published = str(dt)[:19] if dt is not None else ""
        sem_s = sem_scores.get(idx, 0.0)
        bm25_s = bm25_scores.get(idx, 0.0)
        # Use semantic score when available; fall back to normalised BM25 for
        # lexical-only hits so they survive the post-rerank quality check.
        display_score = round(sem_s if sem_s > 0.0 else (bm25_s / max_bm25) * 0.3, 4)
        candidates.append({
            "title": m.get("title", ""),
            "source": m.get("source", ""),
            "url": m.get("url", ""),
            "published_at": published,
            "snippet": m.get("snippet", ""),
            "category": m.get("category", ""),
            "relevance_score": display_score,
            "_idx": idx,
        })

    # 10. GPT re-ranking — verify each candidate is genuinely relevant
    sentiment = params.get("sentiment")
    reranked = rerank(query, candidates, sentiment)

    # 11. Trust the re-ranker as the sole quality gate
    results = reranked

    # 12. Return top_k
    results = results[:top_k]
    for r in results:
        r.pop("_idx", None)

    logger.info("Returning %d results after re-ranking (top semantic=%.4f)",
                len(results), results[0]["relevance_score"] if results else 0)
    return {"results": results, "params": params, "total_in_range": total_in_range}
