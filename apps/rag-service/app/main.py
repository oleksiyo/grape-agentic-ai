import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


class RetrieveRequest(BaseModel):
    query: str
    disease_label: str | None = None
    top_k: int = 3


QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "grape_kb")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
KB_PATH = Path(os.getenv("KB_PATH", "/data/rag_kb"))
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "120"))

qdrant = QdrantClient(url=QDRANT_URL)


app = FastAPI(title="RAG Service", version="0.1.0")


INDEX_STATUS = "not_started"
INDEX_ERROR: str | None = None


def read_kb_documents() -> list[dict[str, str]]:
    docs: list[dict[str, str]] = []
    if not KB_PATH.exists():
        return docs

    for path in KB_PATH.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".txt", ".md"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            docs.append({"source": str(path.name), "text": text})
    return docs


def split_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def ollama_embedding(text: str) -> list[float]:
    with httpx.Client(timeout=60.0) as client:
        # Support both Ollama API variants used across versions.
        legacy_resp = client.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": OLLAMA_EMBED_MODEL, "prompt": text},
        )
        if legacy_resp.is_success:
            return legacy_resp.json()["embedding"]

        embed_resp = client.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": OLLAMA_EMBED_MODEL, "input": text},
        )
        if embed_resp.is_success:
            data = embed_resp.json()
            embeddings = data.get("embeddings") or []
            if embeddings and isinstance(embeddings[0], list):
                return embeddings[0]
            raise RuntimeError("ollama /api/embed response missing embeddings")

        legacy_body = legacy_resp.text.strip()[:300]
        embed_body = embed_resp.text.strip()[:300]
        raise RuntimeError(
            "ollama embedding failed; "
            f"/api/embeddings={legacy_resp.status_code} body={legacy_body}; "
            f"/api/embed={embed_resp.status_code} body={embed_body}"
        )


def ensure_collection(vector_size: int) -> None:
    collections = qdrant.get_collections().collections
    exists = any(col.name == QDRANT_COLLECTION for col in collections)
    if not exists:
        qdrant.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def get_points_count() -> int:
    try:
        return int(qdrant.count(collection_name=QDRANT_COLLECTION).count)
    except Exception:
        return 0


def index_kb_if_needed(force: bool = False) -> dict[str, int | bool]:
    docs = read_kb_documents()
    if not docs:
        return {"docs": 0, "points": 0, "skipped": True}

    # If collection already has points, skip re-indexing.
    if not force:
        try:
            count = qdrant.count(collection_name=QDRANT_COLLECTION).count
            if count > 0:
                return {"docs": len(docs), "points": int(count), "skipped": True}
        except Exception:
            pass

    first_embedding = ollama_embedding(docs[0]["text"][:CHUNK_SIZE])
    ensure_collection(vector_size=len(first_embedding))

    if force:
        try:
            qdrant.delete_collection(collection_name=QDRANT_COLLECTION)
        except Exception:
            pass
        ensure_collection(vector_size=len(first_embedding))

    points: list[PointStruct] = []
    for doc in docs:
        for chunk in split_chunks(doc["text"], CHUNK_SIZE, CHUNK_OVERLAP):
            emb = ollama_embedding(chunk)
            points.append(
                PointStruct(
                    id=str(uuid4()),
                    vector=emb,
                    payload={"source": doc["source"], "text": chunk},
                )
            )

    if points:
        qdrant.upsert(collection_name=QDRANT_COLLECTION, points=points)

    return {"docs": len(docs), "points": len(points), "skipped": False}


@app.on_event("startup")
def startup() -> None:
    global INDEX_STATUS, INDEX_ERROR

    try:
        result = index_kb_if_needed(force=False)
        INDEX_STATUS = "ready"
        INDEX_ERROR = None
        print(f"RAG startup indexing status: {result}")
    except Exception as exc:
        INDEX_STATUS = "failed"
        INDEX_ERROR = str(exc)
        print(f"RAG startup indexing failed: {exc}")
        # Service should stay alive even if index bootstrap fails.
        return


@app.get("/health")
def health() -> dict:
    try:
        qdrant.get_collections()
        qdrant_ok = True
    except Exception:
        qdrant_ok = False

    return {
        "status": "healthy" if (qdrant_ok and INDEX_STATUS != "failed") else "degraded",
        "service": "rag-service",
        "qdrant": qdrant_ok,
        "embed_model": OLLAMA_EMBED_MODEL,
        "qdrant_collection": QDRANT_COLLECTION,
        "kb_docs": len(read_kb_documents()),
        "indexed_points": get_points_count(),
        "index_status": INDEX_STATUS,
        "index_error": INDEX_ERROR,
    }


@app.post("/reindex")
def reindex() -> dict:
    global INDEX_STATUS, INDEX_ERROR

    try:
        result = index_kb_if_needed(force=True)
        INDEX_STATUS = "ready"
        INDEX_ERROR = None
        return {"success": True, "collection": QDRANT_COLLECTION, **result}
    except Exception as exc:
        INDEX_STATUS = "failed"
        INDEX_ERROR = str(exc)
        return {"success": False, "collection": QDRANT_COLLECTION, "error": str(exc)}


@app.post("/retrieve")
def retrieve(payload: RetrieveRequest) -> dict:
    enriched_query = payload.query
    if payload.disease_label:
        enriched_query = f"{payload.query}\nDisease label: {payload.disease_label}"

    try:
        query_vector = ollama_embedding(enriched_query)
        results = qdrant.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=query_vector,
            limit=max(1, payload.top_k),
        )
        items: list[dict[str, Any]] = []
        for hit in results:
            hit_payload = hit.payload or {}
            items.append(
                {
                    "source": hit_payload.get("source", "unknown"),
                    "score": float(hit.score),
                    "text": hit_payload.get("text", ""),
                }
            )
    except Exception as exc:
        return {
            "success": False,
            "query": payload.query,
            "disease_label": payload.disease_label,
            "items": [],
            "error": f"retrieval_failed: {exc}",
        }

    return {
        "success": True,
        "query": payload.query,
        "disease_label": payload.disease_label,
        "items": items,
    }
