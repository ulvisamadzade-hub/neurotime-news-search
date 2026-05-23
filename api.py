import os
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from search import load_index, search
from keywords import load_metadata, extract_keywords

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — loading index and metadata")
    load_index()
    load_metadata()
    logger.info("Startup complete")
    yield
    logger.info("Shutting down")


app = FastAPI(title="Neurotime News Search API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10


@app.post("/api/search")
async def search_news(req: SearchRequest, request: Request):
    client_ip = request.client.host if request.client else "unknown"
    logger.info("Search request from %s — query=%r top_k=%d", client_ip, req.query, req.top_k)
    try:
        result = await asyncio.to_thread(search, req.query, req.top_k)
        logger.info("Search complete — %d results returned", len(result.get("results", [])))
        return result
    except Exception as e:
        logger.error("Search error for query=%r: %s", req.query, e, exc_info=True)
        raise


@app.get("/api/keywords")
async def global_keywords(top_n: int = Query(default=20, ge=1, le=100)):
    logger.info("Global keywords request — top_n=%d", top_n)
    kw = await asyncio.to_thread(extract_keywords, top_n=top_n)
    return {"keywords": [{"term": k, "count": c} for k, c in kw]}


@app.post("/api/keywords/search")
async def keywords_for_search(req: SearchRequest):
    logger.info("Keywords-for-search request — query=%r", req.query)
    result = await asyncio.to_thread(search, req.query, 50)
    snippets = [f"{r['title']} {r['snippet']}" for r in result["results"]]
    kw = await asyncio.to_thread(extract_keywords, articles=snippets if snippets else None, top_n=20)
    return {"keywords": [{"term": k, "count": c} for k, c in kw]}


# Serve frontend as the last mount so API routes take priority
frontend_dir = os.path.join(BASE_DIR, "frontend")
if os.path.isdir(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
