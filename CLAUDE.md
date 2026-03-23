# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`rak` — a local, CLI-first RAG search tool for Zotero libraries. Provides semantic search (ChromaDB + sentence-transformers or API embeddings), keyword search (SQLite FTS5) with Reciprocal Rank Fusion, PDF/Markdown full-text extraction, collection/tag filtering, LLM Q&A via local or remote models, and CSV/BibTeX export. Complements the `zot` CLI (zotero-cli-cc) which handles CRUD operations.

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
rak index                          # Incremental index (auto PDF/MD extraction)
rak index --full                   # Full rebuild
rak status                         # Show index stats
rak clear --yes                    # Reset all indexes
rak config                         # Show settings
rak config llm_model mistral       # Set config value
rak search "query" --hybrid        # Hybrid search (vector + BM25)
rak search "query" --bm25          # Pure keyword search (no embedding model needed)
rak search "q" --collection X --tag Y  # Filtered search
rak ask "question"                 # LLM Q&A (needs Ollama/LMStudio/remote API)
rak export "query" --format bibtex # Export as BibTeX
rak --json search "query"          # JSON output
rak --verbose search "query"       # Debug logging
rak completion zsh                 # Generate shell completions
```

## Architecture

All source is in `src/rak/`. The data flows through a pipeline:

```
zot CLI → indexer (fetch + parse + PDF/MD extract) → embedder → vector store (ChromaDB) + BM25 (SQLite FTS5)
                                                                    ↓
                                      searcher (vector / hybrid with RRF fusion + collection/tag filters)
                                                                    ↓
                                                    formatter → CLI output / export / LLM Q&A
```

**Key modules and their roles:**

- **cli.py** — Click entry point (`rak`). Commands: `index`, `search`, `ask`, `chat`, `export`, `config`, `status`, `clear`, `completion`. Global flags: `--json`, `--model`, `--verbose`.
- **config.py** — `RakConfig` dataclass. Data stored in `~/Zotero/rak/`, auto-detect Zotero storage, LLM and embedding settings. Persistent config via `config.json`.
- **embedder.py** — Supports two providers: `local` (SentenceTransformer) and `api` (OpenAI-compatible `/v1/embeddings`). `embed()` for single, `embed_batch()` for bulk. Suppresses noisy model loading output.
- **store.py** — `VectorStore` wrapping ChromaDB persistent client. Collection `rak_papers`, cosine distance. `search()` clamps `n_results` to collection size. `get_ids_by_metadata()` returns IDs only.
- **bm25.py** — `BM25Index` using SQLite FTS5 virtual table. `add()` ensures uniqueness via delete-before-insert. `search_with_snippet()` returns FTS5 snippet highlights. Implements context manager for safe resource cleanup.
- **indexer.py** — Orchestrates: `fetch_zot_items()` shells out to `zot --json --limit list`, `build_document_text()` concatenates title/authors/abstract/tags/attachment_text, `diff_items()` computes add/update/remove sets with text cache to avoid redundant extraction, `index_items()` supports both full and incremental modes via registry. Long documents are chunked into overlapping segments stored as separate vectors.
- **searcher.py** — `Searcher` with dependency-injected embedder/store/bm25 (embedder and store are optional for BM25-only mode). `build_where_filter()` builds ChromaDB metadata filters. `bm25_search()` does pure keyword search without loading embeddings. `hybrid_search()` fuses vector + BM25 via `rrf_fuse(k=60)`. Chunk results are deduplicated to parent papers via `_deduplicate_chunks()`. Results include `snippet` from the best-matched chunk.
- **formatter.py** — Rich tables, JSON output (with optional snippets), incremental stats, and ask result formatting.
- **pdf.py** — `extract_pdf_text()` supports three providers: `pymupdf` (default, PyMuPDF), `mineru` (MinerU CLI), and `docling` (Docling CLI) for structured Markdown output with table/formula preservation. Falls back to PyMuPDF on failure. `extract_file_text()` handles PDF and Markdown, passing `provider` through. `find_attachments()` locates all PDF/MD files per Zotero item. `chunk_text()` splits text preferring paragraph/section boundaries (double newlines, markdown headings), merging small paragraphs up to `chunk_size` words and falling back to word-level overlap for oversized paragraphs. Validates `overlap < chunk_size`.
- **llm.py** — `LLMClient` wrapping OpenAI SDK for chat completions. Compatible with Ollama, LM Studio, OpenAI, DeepSeek, and any OpenAI-compatible endpoint.
- **export.py** — `to_csv()` and `to_bibtex()` formatters with BibTeX special character escaping and proper Zotero-to-BibTeX type mapping.
- **metadata.py** — `IndexMetadata` dataclass, `save_metadata()`/`load_metadata()` for tracking index state.
- **registry.py** — Content hash registry (`registry.json`) for incremental indexing. `compute_hash()`, `save_registry()`, `load_registry()`.
- **errors.py** — Custom exception hierarchy: `RakError` → `ZotNotFoundError`, `EmptyLibraryError`, `ModelDownloadError`.
- **mcp_server.py** — MCP server exposing `search_papers` and `index_status` tools for AI assistants (Cursor, LM Studio). Registers `atexit` handler for cleanup.

**Design decisions:**
- All computation is local by default — no API keys needed for search. Embedding and LLM can optionally use remote APIs.
- Data stored in `~/Zotero/rak/`: `chroma/` for vectors, `fts.sqlite` for keywords, `meta.json`, `registry.json`, `config.json`.
- `zot` CLI is a required external dependency for data ingestion.
- Indexing is incremental by default — content hashes detect new/changed/deleted items. PDF text is cached during indexing to avoid redundant extraction.
- All PDF and Markdown attachments per Zotero item are extracted and merged for indexing.
- BM25 `add()` uses delete-before-insert to prevent duplicate doc_ids.
- All CLI commands use `with BM25Index(...)` context manager to prevent SQLite connection leaks.
- `ask`/`chat` use search result snippets as LLM context instead of re-fetching documents.
- BibTeX export escapes special characters (`\ { } & % # _ ~ ^`).
- Config validates `chunk_overlap < chunk_size` at save time.
- MinerU and Docling integrations use their respective CLIs directly, keeping heavy dependencies out of `rak`'s install. Both fall back to PyMuPDF silently with a warning on failure.

**Configurable keys:**
`model_name`, `zot_command`, `llm_base_url`, `llm_model`, `llm_api_key`, `chunk_size`, `chunk_overlap`, `embedding_provider` (`local`/`api`), `embedding_base_url`, `embedding_api_key`, `pdf_provider` (`pymupdf`/`mineru`/`docling`).

## Build System

Uses `hatchling`. Entry point: `rak = "rak.cli:main"`, `rak-mcp = "rak.mcp_server:main"`. Package located at `src/rak/` (src layout).

## Testing

141 tests. `@pytest.mark.network` marks tests requiring model downloads. CI runs `pytest -m "not network"`.
