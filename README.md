# rak — RAG Knowledge Search for Zotero

Semantic and hybrid search over your Zotero library, powered by local embeddings. Ask questions with a local LLM.

## Install

```bash
uv tool install zotero-rag-cli
# or
pip install zotero-rag-cli
```

Requires `zot` ([zotero-cli-cc](https://github.com/Agents365-ai/zotero-cli-cc)) to be installed and working.

## Quick Start

```bash
# 1. Index your Zotero library (incremental by default, extracts PDF full text)
rak index

# 2. Semantic search
rak search "cell fate determination mechanisms"

# 3. Hybrid search (semantic + keyword BM25)
rak search "spatial transcriptomics" --hybrid

# 4. Ask a question (requires Ollama or LMStudio running locally)
rak ask "What are the main methods for single-cell clustering?"
```

## Commands

### Index

```bash
rak index                    # Incremental index (only new/changed items)
rak index --full             # Force full rebuild
rak index --limit 500        # Limit items fetched from zot
```

Automatically extracts PDF full text from `~/Zotero/storage/` if available.

### Search

```bash
rak search "single cell RNA sequencing methods"
rak search "CRISPR off-target effects" --hybrid
rak search "attention mechanism" --limit 5
rak search "RNA-seq" --collection "My Papers" --tag "methods"
rak --json search "spatial omics"
```

### Ask (LLM Q&A)

```bash
rak ask "What are the main findings about cell fate?"
rak ask "Compare CRISPR methods" --context 10 --hybrid
rak ask "Summarize spatial omics" --llm-model mistral --llm-url http://localhost:1234/v1
```

Requires a local OpenAI-compatible LLM server (Ollama, LMStudio, vLLM). Default: Ollama at `localhost:11434`.

### Export

```bash
rak export "single cell" --format csv                    # CSV to stdout
rak export "CRISPR" --format bibtex --output refs.bib    # BibTeX to file
rak export "RNA-seq" --hybrid --collection "Methods"     # With filters
```

### Config

```bash
rak config                           # Show all settings
rak config llm_model mistral         # Set LLM model persistently
rak config llm_base_url http://localhost:1234/v1
```

### Status & Clear

```bash
rak status                  # Show index stats (item count, model, last indexed)
rak clear                   # Delete all indexes (with confirmation)
rak clear --yes             # Skip confirmation
```

## How It Works

```
rak index                    rak search "query"          rak ask "question"
    │                            │                           │
    ▼                            ▼                           ▼
zot --json list              Embedder                    Searcher
    │                         │      │                       │
    ▼                    ┌────▼──┐   │                  Retrieved
Embedder + PDF           │Vector │   │                   Papers
    │                   │Search │   │                       │
    ▼                   └───┬───┘   │                       ▼
┌────────┐                  │       │                   Local LLM
│ChromaDB│ ◄────────────────┘       │                  (Ollama/etc)
└────────┘                          │                       │
┌────────┐              ┌───▼───┐   │                   Answer +
│FTS5 DB │ ◄────────────│ BM25  │◄──┘ (--hybrid)       Sources
└────────┘              └───┬───┘
                            │
                       RRF Fusion
                            │
                        Results
```

## Options

| Flag | Commands | Purpose |
|------|----------|---------|
| `--json` | Global | JSON output |
| `--model NAME` | Global | Embedding model (default: all-MiniLM-L6-v2) |
| `--hybrid` | search, ask, export | Enable hybrid search (vector + BM25) |
| `--limit N` | search, export | Number of results (default: 10) |
| `--collection NAME` | search, ask, export | Filter by Zotero collection |
| `--tag TAG` | search, ask, export | Filter by tag (repeatable, OR logic) |
| `--full` | index | Force full rebuild |
| `--context N` | ask | Number of context documents (default: 5) |
| `--llm-model NAME` | ask | Override LLM model |
| `--llm-url URL` | ask | Override LLM server URL |
| `--format csv\|bibtex` | export | Export format (default: csv) |
| `--output FILE` | export | Write to file instead of stdout |

## Embedding Models

| Model | Size | Best For |
|-------|------|----------|
| `all-MiniLM-L6-v2` (default) | ~80MB | English papers, fast |
| `nomic-ai/nomic-embed-text-v1.5` | ~270MB | Multilingual, more accurate |

Switch model:
```bash
rak --model nomic-ai/nomic-embed-text-v1.5 index --full
rak --model nomic-ai/nomic-embed-text-v1.5 search "query"
```

## Use with zot

`rak` is designed to work alongside [`zot`](https://github.com/Agents365-ai/zotero-cli-cc):

```bash
# zot: exact keyword search (fast, precise)
zot search "single cell"

# rak: semantic search (understands meaning)
rak search "methods for analyzing individual cell transcriptomes"

# rak hybrid: best of both
rak search "scRNA-seq clustering" --hybrid
```

## License

MIT
