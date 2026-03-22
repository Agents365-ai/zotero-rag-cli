# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`rak` — a local, CLI-first RAG search tool for Zotero libraries. Provides semantic search (ChromaDB + sentence-transformers), keyword search (SQLite FTS5) with Reciprocal Rank Fusion, PDF full-text extraction, collection/tag filtering, LLM Q&A via local models, and CSV/BibTeX export. Complements the `zot` CLI (zotero-cli-cc) which handles CRUD operations.

## Commands

```bash
# Install in development mode
uv pip install -e ".[dev]"

# Run all tests (excluding network-dependent)
pytest -m "not network"

# Run all tests including model downloads
pytest

# Run a single test file or test
pytest tests/test_searcher.py
pytest tests/test_bm25.py::test_search_ranking

# Run CLI
rak index                          # Incremental index (auto PDF extraction)
rak index --full                   # Full rebuild
rak status                         # Show index stats
rak clear --yes                    # Reset all indexes
rak config                         # Show settings
rak config llm_model mistral       # Set config value
rak search "query" --hybrid        # Hybrid search
rak search "q" --collection X --tag Y  # Filtered search
rak ask "question"                 # LLM Q&A (needs Ollama/LMStudio)
rak export "query" --format bibtex # Export as BibTeX
rak --json search "query"          # JSON output
rak completion zsh                 # Generate shell completions
```

## Architecture

All source is in `src/rak/`. The data flows through a pipeline:

```
zot CLI → indexer (fetch + parse + PDF extract) → embedder → vector store (ChromaDB) + BM25 (SQLite FTS5)
                                                                    ↓
                                      searcher (vector / hybrid with RRF fusion + collection/tag filters)
                                                                    ↓
                                                    formatter → CLI output / export / LLM Q&A
```

**Key modules and their roles:**

- **cli.py** — Click entry point (`rak`). Commands: `index`, `search`, `ask`, `chat`, `export`, `config`, `status`, `clear`, `completion`. Global flags: `--json`, `--model`.
- **config.py** — `RakConfig` dataclass. Paths via `platformdirs`, auto-detect Zotero storage, LLM settings. Persistent config via `config.json`.
- **embedder.py** — Wraps `SentenceTransformer`. `embed()` for single, `embed_batch()` for bulk (batch_size=32). Raises `ModelDownloadError` on failure.
- **store.py** — `VectorStore` wrapping ChromaDB persistent client. Collection `rak_papers`, cosine distance. Supports `where` filter for metadata queries.
- **bm25.py** — `BM25Index` using SQLite FTS5 virtual table for keyword search. Supports `delete()` for incremental updates.
- **indexer.py** — Orchestrates: `fetch_zot_items()` shells out to `zot --json --limit list`, `build_document_text()` concatenates title/authors/abstract/tags/pdf_text, `diff_items()` computes add/update/remove sets, `index_items()` supports both full and incremental modes via registry. Long documents are chunked into overlapping segments stored as separate vectors.
- **searcher.py** — `Searcher` with dependency-injected embedder/store/bm25. `build_where_filter()` builds ChromaDB metadata filters. `hybrid_search()` fuses vector + BM25 via `rrf_fuse(k=60)`. Chunk results are deduplicated to parent papers via `_deduplicate_chunks()`.
- **formatter.py** — Rich tables, JSON output, incremental stats, and ask result formatting.
- **pdf.py** — `extract_pdf_text()` via PyMuPDF, `find_pdf()` locates PDFs in Zotero storage. `chunk_text()` splits long text into overlapping word-level chunks (default 512 words, 64 overlap).
- **llm.py** — `LLMClient` wrapping OpenAI SDK for local LLM chat completions (Ollama/LMStudio).
- **export.py** — `to_csv()` and `to_bibtex()` formatters for search result export.
- **metadata.py** — `IndexMetadata` dataclass, `save_metadata()`/`load_metadata()` for tracking index state.
- **registry.py** — Content hash registry (`registry.json`) for incremental indexing. `compute_hash()`, `save_registry()`, `load_registry()`.
- **errors.py** — Custom exception hierarchy: `RakError` → `ZotNotFoundError`, `EmptyLibraryError`, `ModelDownloadError`.

**Design decisions:**
- All computation is local — no API keys needed for search. LLM Q&A uses local servers (Ollama/LMStudio).
- Data stored in platform-specific dir (`platformdirs.user_data_dir("rak")`): `chroma/` for vectors, `fts.sqlite` for keywords, `meta.json`, `registry.json`, `config.json`.
- `zot` CLI is a required external dependency for data ingestion.
- Indexing is incremental by default — content hashes detect new/changed/deleted items.
- PDF extraction auto-detects `~/Zotero/storage/` and is silently skipped if unavailable.
- Collections and tags stored as list metadata in ChromaDB for `$contains` filtering.

## Build System

Uses `hatchling`. Entry point: `rak = "rak.cli:main"`. Package located at `src/rak/` (src layout).

## Testing

98 tests. `@pytest.mark.network` marks tests requiring model downloads. CI runs `pytest -m "not network"`.
