import os
import pickle
import numpy as np
import faiss
import pandas as pd
from openai import OpenAI
from urllib.parse import urlparse
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "news_data.xlsx")
INDEX_PATH = os.path.join(BASE_DIR, "index.faiss")
META_PATH = os.path.join(BASE_DIR, "metadata.pkl")


def get_source(url):
    try:
        return urlparse(str(url)).netloc.replace("www.", "")
    except Exception:
        return "unknown"


def embed_batch(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [e.embedding for e in response.data]


def build_index():
    print("Loading dataset...")
    df = pd.read_excel(DATA_PATH, engine="openpyxl")
    df.columns = [c.strip().lower() for c in df.columns]

    # Normalise column names — the xlsx has: link, title, content, category, created_at
    df.rename(columns={"link": "url"}, inplace=True, errors="ignore")

    df["source"] = df["url"].apply(get_source)
    df["snippet"] = df["content"].fillna("").str[:300]
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df["title"] = df["title"].fillna("")
    df["content"] = df["content"].fillna("")
    df["category"] = df["category"].fillna("")

    # Build text for embedding: title + first 1500 chars of content
    texts = (df["title"] + " " + df["content"].str[:1500]).tolist()

    print(f"Embedding {len(texts)} articles with text-embedding-3-small...")
    all_embeddings = []
    batch_size = 100
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i : i + batch_size]
        batch = [t if t.strip() else " " for t in batch]
        embeddings = embed_batch(batch)
        all_embeddings.extend(embeddings)

    vectors = np.array(all_embeddings, dtype=np.float32)
    faiss.normalize_L2(vectors)

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    faiss.write_index(index, INDEX_PATH)
    print(f"FAISS index saved → {INDEX_PATH}  ({index.ntotal} vectors, dim={dim})")

    metadata = df[["title", "url", "source", "category", "created_at", "snippet"]].to_dict(
        orient="records"
    )
    with open(META_PATH, "wb") as f:
        pickle.dump(metadata, f)
    print(f"Metadata saved → {META_PATH}")


if __name__ == "__main__":
    build_index()
