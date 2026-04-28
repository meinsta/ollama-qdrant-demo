from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import requests
from qdrant_client import QdrantClient, models

DEFAULT_COLLECTION = "ollama_demo_docs"
DEFAULT_DATA_FILE = Path(__file__).with_name("sample_data.json")
DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DEFAULT_QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
DEFAULT_EMBED_MODEL = os.getenv("OLLAMA_MODEL", "nomic-embed-text")
DEFAULT_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2")


def load_documents(data_file: Path) -> List[Dict[str, Any]]:
    with data_file.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    if not isinstance(raw, list):
        raise ValueError(f"Expected a list of documents in {data_file}")

    required_fields = {"id", "title", "text", "category"}
    documents: List[Dict[str, Any]] = []
    for index, doc in enumerate(raw):
        if not isinstance(doc, dict):
            raise ValueError(f"Document at index {index} is not an object")

        missing = required_fields.difference(doc.keys())
        if missing:
            raise ValueError(
                f"Document at index {index} is missing required fields: {sorted(missing)}"
            )

        documents.append(
            {
                "id": int(doc["id"]),
                "title": str(doc["title"]),
                "text": str(doc["text"]),
                "category": str(doc["category"]),
            }
        )
    return documents


def _extract_embedding(data: Dict[str, Any]) -> Optional[List[float]]:
    direct = data.get("embedding")
    if isinstance(direct, list) and direct and isinstance(direct[0], (float, int)):
        return [float(value) for value in direct]

    nested = data.get("embeddings")
    if isinstance(nested, list) and nested:
        first = nested[0]
        if isinstance(first, list) and first and isinstance(first[0], (float, int)):
            return [float(value) for value in first]
        if isinstance(first, (float, int)):
            return [float(value) for value in nested]

    return None


def get_embedding(text: str, model: str, ollama_url: str) -> List[float]:
    base_url = ollama_url.rstrip("/")
    attempts = [
        (f"{base_url}/api/embed", {"model": model, "input": text}),
        (f"{base_url}/api/embeddings", {"model": model, "prompt": text}),
    ]

    errors: List[str] = []
    for url, payload in attempts:
        try:
            response = requests.post(url, json=payload, timeout=120)
        except requests.RequestException as exc:
            errors.append(f"{url}: {exc}")
            continue

        if response.status_code in (404, 405):
            errors.append(f"{url}: endpoint unavailable ({response.status_code})")
            continue

        try:
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            errors.append(f"{url}: {exc}")
            continue
        except ValueError as exc:
            errors.append(f"{url}: invalid JSON response ({exc})")
            continue

        embedding = _extract_embedding(data)
        if embedding:
            return embedding
        errors.append(f"{url}: response did not include an embedding vector")

    raise RuntimeError(
        "Unable to generate embeddings from Ollama. "
        "Ensure Ollama is running and the embedding model is available. "
        f"Details: {'; '.join(errors)}"
    )

def generate_text(prompt: str, model: str, ollama_url: str) -> str:
    url = f"{ollama_url.rstrip('/')}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}

    try:
        response = requests.post(url, json=payload, timeout=240)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            "Unable to generate a response from Ollama. "
            "Ensure Ollama is running and the generation model is available. "
            f"Details: {exc}"
        ) from exc

    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError(f"Ollama generation endpoint returned invalid JSON: {exc}") from exc

    text = data.get("response")
    if isinstance(text, str) and text.strip():
        return text.strip()

    raise RuntimeError("Ollama generation response did not include a non-empty 'response' field.")


def format_preview(text: str, max_length: int = 120) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_length:
        return cleaned
    return f"{cleaned[: max_length - 3]}..."


def search_documents(
    query: str,
    *,
    qdrant_url: str,
    ollama_url: str,
    collection: str,
    model: str,
    limit: int,
) -> List[Any]:
    client = QdrantClient(url=qdrant_url)
    query_vector = get_embedding(query, model, ollama_url)
    return client.search(
        collection_name=collection,
        query_vector=query_vector,
        limit=limit,
        with_payload=True,
    )


def build_rag_prompt(message: str, contexts: List[Dict[str, Any]]) -> str:
    if contexts:
        formatted_context = "\n\n".join(
            [
                (
                    f"[{index}] title={item['title']} category={item['category']} score={item['score']:.4f}\n"
                    f"{item['text']}"
                )
                for index, item in enumerate(contexts, start=1)
            ]
        )
    else:
        formatted_context = "No relevant context was retrieved."

    return (
        "You are a concise assistant. Answer the user using the provided context.\n"
        "If the context is insufficient, say so clearly.\n\n"
        f"Context:\n{formatted_context}\n\n"
        f"User question: {message}\n"
        "Answer:"
    )


def ingest_documents(args: argparse.Namespace) -> None:
    documents = load_documents(Path(args.data_file))
    if not documents:
        raise ValueError("No documents to ingest")

    client = QdrantClient(url=args.qdrant_url)
    probe_text = f"{documents[0]['title']}\n{documents[0]['text']}"
    probe_vector = get_embedding(probe_text, args.model, args.ollama_url)

    if client.collection_exists(args.collection):
        client.delete_collection(args.collection)

    client.create_collection(
        collection_name=args.collection,
        vectors_config=models.VectorParams(
            size=len(probe_vector), distance=models.Distance.COSINE
        ),
    )

    points: List[models.PointStruct] = []
    for doc in documents:
        text_for_embedding = f"{doc['title']}\n{doc['text']}"
        vector = get_embedding(text_for_embedding, args.model, args.ollama_url)
        payload = {
            "title": doc["title"],
            "text": doc["text"],
            "category": doc["category"],
        }
        points.append(
            models.PointStruct(
                id=doc["id"],
                vector=vector,
                payload=payload,
            )
        )

    client.upsert(collection_name=args.collection, points=points)
    print(
        f"Ingested {len(points)} documents into collection '{args.collection}' "
        f"using model '{args.model}'."
    )


def query_documents(args: argparse.Namespace) -> None:
    results = search_documents(
        args.query,
        qdrant_url=args.qdrant_url,
        ollama_url=args.ollama_url,
        collection=args.collection,
        model=args.model,
        limit=args.limit,
    )

    if not results:
        print("No results found.")
        return

    for index, result in enumerate(results, start=1):
        payload = result.payload or {}
        title = payload.get("title", "(untitled)")
        category = payload.get("category", "unknown")
        preview = format_preview(str(payload.get("text", "")))
        print(f"{index}. score={result.score:.4f} id={result.id}")
        print(f"   title: {title}")
        print(f"   category: {category}")
        print(f"   text: {preview}")


def traverse_documents(args: argparse.Namespace) -> None:
    client = QdrantClient(url=args.qdrant_url)
    offset = None
    shown = 0

    while True:
        points, next_offset = client.scroll(
            collection_name=args.collection,
            limit=args.batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        if not points:
            break

        for point in points:
            payload = point.payload or {}
            title = payload.get("title", "(untitled)")
            category = payload.get("category", "unknown")
            preview = format_preview(str(payload.get("text", "")))
            shown += 1
            print(f"{shown}. id={point.id} category={category} title={title}")
            print(f"   text: {preview}")

            if args.limit and shown >= args.limit:
                print(f"\nDisplayed {shown} points (limit reached).")
                return

        if next_offset is None:
            break
        offset = next_offset

    print(f"\nDisplayed {shown} points in total.")


def create_rag_app(
    *,
    qdrant_url: str,
    ollama_url: str,
    collection: str,
    embed_model: str,
    chat_model: str,
    default_limit: int,
) -> FastAPI:
    app = FastAPI(title="Tiny RAG Chat API", version="0.1.0")

    class ChatRequest(BaseModel):
        message: str = Field(min_length=1)
        limit: Optional[int] = Field(default=None, ge=1, le=10)

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/chat")
    def chat(request: ChatRequest) -> Dict[str, Any]:
        effective_limit = request.limit if request.limit is not None else default_limit
        try:
            results = search_documents(
                request.message,
                qdrant_url=qdrant_url,
                ollama_url=ollama_url,
                collection=collection,
                model=embed_model,
                limit=effective_limit,
            )
            contexts: List[Dict[str, Any]] = []
            for result in results:
                payload = result.payload or {}
                contexts.append(
                    {
                        "id": result.id,
                        "score": float(result.score),
                        "title": str(payload.get("title", "(untitled)")),
                        "category": str(payload.get("category", "unknown")),
                        "text": str(payload.get("text", "")),
                    }
                )

            prompt = build_rag_prompt(request.message, contexts)
            answer = generate_text(prompt, chat_model, ollama_url)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return {
            "answer": answer,
            "collection": collection,
            "chat_model": chat_model,
            "citations": [
                {
                    "id": item["id"],
                    "score": item["score"],
                    "title": item["title"],
                    "category": item["category"],
                    "text_preview": format_preview(item["text"], max_length=180),
                }
                for item in contexts
            ],
        }

    return app


def serve_chat_endpoint(args: argparse.Namespace) -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            "Uvicorn is required to run the API server. Install dependencies with "
            "'pip install -r requirements.txt'."
        ) from exc

    app = create_rag_app(
        qdrant_url=args.qdrant_url,
        ollama_url=args.ollama_url,
        collection=args.collection,
        embed_model=args.model,
        chat_model=args.chat_model,
        default_limit=args.retrieval_limit,
    )
    uvicorn.run(app, host=args.host, port=args.port)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simple Ollama + Qdrant demo with ingest/query/traverse/serve commands."
    )
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--model", default=DEFAULT_EMBED_MODEL)

    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser(
        "ingest", help="Create/recreate collection and ingest sample documents."
    )
    ingest_parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    ingest_parser.add_argument("--data-file", default=str(DEFAULT_DATA_FILE))
    ingest_parser.set_defaults(func=ingest_documents)

    query_parser = subparsers.add_parser(
        "query", help="Run semantic search against the collection."
    )
    query_parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    query_parser.add_argument("--query", required=True)
    query_parser.add_argument("--limit", type=int, default=3)
    query_parser.set_defaults(func=query_documents)

    traverse_parser = subparsers.add_parser(
        "traverse",
        help="Traverse points in the collection via Qdrant scroll pagination.",
    )
    traverse_parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    traverse_parser.add_argument("--batch-size", type=int, default=4)
    traverse_parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of points to print. 0 means no limit.",
    )
    traverse_parser.set_defaults(func=traverse_documents)

    serve_parser = subparsers.add_parser(
        "serve",
        help="Run a tiny RAG chat endpoint backed by Qdrant retrieval + Ollama generation.",
    )
    serve_parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    serve_parser.add_argument("--chat-model", default=DEFAULT_CHAT_MODEL)
    serve_parser.add_argument("--retrieval-limit", type=int, default=3)
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.set_defaults(func=serve_chat_endpoint)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except Exception as exc:  # pragma: no cover
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
