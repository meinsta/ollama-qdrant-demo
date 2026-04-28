# Ollama + Qdrant Tiny RAG Demo
A small, end-to-end retrieval-augmented generation (RAG) demo. Documents are embedded with [Ollama](https://ollama.com), stored in [Qdrant](https://qdrant.tech), and retrieved at query time to ground answers from a local chat model. Includes a CLI and a lightweight browser GUI with PDF/Markdown/text upload.
## Architecture
```
  user ──▶ GUI / curl ──▶ FastAPI (/chat, /ingest)
                          │
                          ├─ /ingest:
                          │    ├─ extract (pypdf for PDF, utf-8 for MD/TXT)
                          │    ├─ chunk   (word window with overlap, page-tracked)
                          │    ├─ embed   ──▶ Ollama (nomic-embed-text)
                          │    └─ upsert  ──▶ Qdrant (rich payload per chunk)
                          │
                          └─ /chat:
                               ├─ embed query    ──▶ Ollama
                               ├─ vector search ──▶ Qdrant (ollama_demo_docs)
                               └─ generate      ──▶ Ollama (llama3.2)
```
## Project layout
- `app.py` — CLI entrypoint with `ingest`, `ingest-file`, `query`, `traverse`, and `serve` subcommands
- `static/index.html` — single-file HTML/CSS/JS GUI served by `serve` (upload + chat)
- `sample_data.json` — 8 example documents used by `ingest`
- `requirements.txt` — Python dependencies (`qdrant-client`, `requests`, `fastapi`, `uvicorn`, `pydantic`, `pypdf`, `python-multipart`)
## Prerequisites
1. **Python 3.9+**
2. **Qdrant** running on `http://localhost:6333`
3. **Ollama** running on `http://localhost:11434`
4. An **embedding model** pulled in Ollama (default: `nomic-embed-text`, ~270 MB)
5. A **generation model** pulled in Ollama for `serve` (default: `llama3.2`, ~2 GB)
### Start Qdrant
```bash
docker run --rm -p 6333:6333 qdrant/qdrant
```
### Start Ollama and pull models
```bash
# install (macOS): brew install ollama
ollama serve &                # leave running in the background
ollama pull nomic-embed-text  # embeddings
ollama pull llama3.2          # chat model (only needed for `serve`)
```
## Install
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
> If you don't activate the venv, invoke its interpreter directly: `./.venv/bin/python app.py …`
## Usage
All commands accept the global flags `--qdrant-url`, `--ollama-url`, and `--model` (embedding model). Subcommands add their own.
### 1. Ingest sample data (clean-slate)
```bash
python app.py ingest
```
Drops + recreates the collection `ollama_demo_docs` with cosine distance and writes all 8 sample documents through the unified chunk pipeline. Each doc is chunked by words (default `--chunk-size 400`, `--chunk-overlap 50`) and stored with rich payload (see schema below). Override the source file with `--data-file path/to/file.json` (must be a JSON list of `{id, title, text, category}` objects).
### 2. Ingest your own documents (PDF / MD / TXT)
```bash
python app.py ingest-file path/to/whitepaper.pdf path/to/notes.md
```
Extracts text (per-page for PDFs via `pypdf`), chunks it, embeds each chunk with Ollama, and upserts the points into the collection. The collection is created lazily if it doesn't exist; existing chunks for the same source filename are replaced by default.
Useful flags:
- `--collection` (default: `ollama_demo_docs`)
- `--category foo` — payload label (default: `uploaded`)
- `--tags a,b,c` — comma-separated tag list stored in payload
- `--chunk-size N` (default: `400` words)
- `--chunk-overlap N` (default: `50` words)
- `--no-replace` — keep previously ingested chunks for the same source filename
Supported extensions: `.pdf`, `.md`, `.markdown`, `.txt`. Per-file size limit: 25 MB.
### 3. Semantic query
```bash
python app.py query --query "How do I run Qdrant locally?" --limit 3
```
Prints the top results with score, id, title, category, and a text preview.
### 4. Traverse stored points
```bash
python app.py traverse --batch-size 3
```
Iterates through every point using Qdrant scroll pagination (no similarity required). Use `--limit N` to cap the number of points printed.
### 5. Run the RAG chat endpoint + GUI
```bash
python app.py serve
```
Starts FastAPI/uvicorn on `http://127.0.0.1:8000` and exposes:
- `GET /` — browser GUI (upload card, question box, retrieval-limit input, answer + citations with source/page/chunk tags, live health pill)
- `GET /health` — liveness probe (`{"status":"ok"}`)
- `POST /chat` — JSON body `{"message": "…", "limit": 3}`; returns `{answer, collection, chat_model, citations:[…]}`
- `POST /ingest` — multipart upload (see below)
- `GET /docs` — auto-generated OpenAPI/Swagger UI
Open <http://127.0.0.1:8000> in a browser to use the GUI. Tip: ⌘/Ctrl+Enter sends from the textarea.
Example chat request:
```bash
curl -s http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message": "How do I run Qdrant locally?", "limit": 3}'
```
Example ingest request (multipart):
```bash
curl -s http://127.0.0.1:8000/ingest \
  -F 'files=@whitepaper.pdf' \
  -F 'files=@notes.md' \
  -F 'category=docs' \
  -F 'tags=pricing,2024' \
  -F 'replace=true' \
  -F 'chunk_size=400' \
  -F 'chunk_overlap=50'
```
Response shape:
```json
{
  "collection": "ollama_demo_docs",
  "embed_model": "nomic-embed-text",
  "total_chunks": 42,
  "results": [
    {"source": "whitepaper.pdf", "title": "whitepaper", "source_type": "pdf", "pages": 12, "chunks_ingested": 38},
    {"source": "notes.md", "title": "notes", "source_type": "markdown", "pages": null, "chunks_ingested": 4}
  ]
}
```
Useful `serve` flags:
- `--collection` (default: `ollama_demo_docs`)
- `--chat-model` (default: `llama3.2`)
- `--retrieval-limit` (default: `3`) — used when the request body omits `limit`
- `--host` (default: `127.0.0.1`), `--port` (default: `8000`)
## Chunk payload schema
Every chunk upserted into Qdrant carries the following payload (fields beyond the basics are what enable filtering, page-jump UI, and re-ingest by source):
- `title` — display title (filename stem for uploads, JSON `title` for sample docs)
- `text` — the chunk text
- `category` — user-provided label
- `source` — stable identifier (filename for uploads, `sample:<id>` for sample docs)
- `source_type` — `pdf` / `markdown` / `text` / `sample`
- `chunk_index`, `chunk_count` — position of this chunk within its source (0-indexed)
- `char_start`, `char_end` — character range within the joined source text
- `page_start`, `page_end` — 1-indexed page range (PDF only)
- `page_count` — total pages in the source PDF
- `tags` — list of strings
- `created_at` — ISO 8601 UTC timestamp of ingestion
Point IDs are deterministic UUIDv5 hashes of `(source, chunk_index)`, so re-ingesting the same source with `replace=true` overwrites prior chunks idempotently.
## Configuration via environment variables
- `QDRANT_URL` — default `http://localhost:6333`
- `OLLAMA_URL` — default `http://localhost:11434`
- `OLLAMA_MODEL` — embedding model, default `nomic-embed-text`
- `OLLAMA_CHAT_MODEL` — generation model used by `serve`, default `llama3.2`
CLI flags always override env vars.
## Troubleshooting
- **`zsh: command not found: python`** — use `python3` or the venv's interpreter (`./.venv/bin/python`).
- **`ModuleNotFoundError: No module named 'fastapi'` / `pypdf` / `multipart`** — the venv isn't active or deps aren't installed. Run `source .venv/bin/activate && pip install -r requirements.txt`, or call `./.venv/bin/python app.py …`.
- **`/ingest` returns `No extractable text found`** — the PDF is likely scanned images. Run it through OCR (e.g. `ocrmypdf input.pdf output.pdf`) and re-upload.
- **`/ingest` returns a vector-size error** — the collection was created with a different embedding model. Drop the collection (`python app.py ingest` recreates it) or switch back to the matching `--model`.
- **`/chat` returns `Unable to generate embeddings from Ollama`** — Ollama isn't running, or the embedding model isn't pulled. Check with `curl http://localhost:11434/api/tags` and `ollama list`.
- **`/chat` returns `Unable to generate a response from Ollama`** — the chat model (default `llama3.2`) isn't pulled. Run `ollama pull llama3.2`.
- **First `/chat` or `/ingest` request is slow** — Ollama needs to load models into memory on cold start; subsequent requests are much faster.
- **`urllib3 NotOpenSSLWarning` about LibreSSL** — cosmetic; safe to ignore on macOS system Python.
