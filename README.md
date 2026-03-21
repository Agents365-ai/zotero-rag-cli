# rak — RAG Knowledge Search for Zotero

Semantic and hybrid search over your Zotero library, powered by local embeddings.

## Install

```bash
uv tool install zotero-rag-cli
# or
pip install zotero-rag-cli
```

Requires `zot` ([zotero-cli-cc](https://github.com/Agents365-ai/zotero-cli-cc)) to be installed and working.

## Quick Start

```bash
# 1. Index your Zotero library
rak index

# 2. Semantic search
rak search "cell fate determination mechanisms"

# 3. Hybrid search (semantic + keyword BM25)
rak search "spatial transcriptomics" --hybrid
```

## How It Works

```
rak index                    rak search "query"
    │                            │
    ▼                            ▼
zot --json list              Embedder
    │                         │      │
    ▼                    ┌────▼──┐   │
Embedder                │Vector │   │
    │                   │Search │   │
    ▼                   └───┬───┘   │
┌────────┐                  │       │
│ChromaDB│ ◄────────────────┘       │
└────────┘                          │
┌────────┐              ┌───▼───┐   │
│FTS5 DB │ ◄────────────│ BM25  │◄──┘ (--hybrid)
└────────┘              └───┬───┘
                            │
                       RRF Fusion
                            │
                        Results
```

## Commands

### Index

```bash
# Index all items from zot (default: up to 5000)
rak index

# Limit items
rak index --limit 500
```

### Search

```bash
# Semantic search (vector similarity)
rak search "single cell RNA sequencing methods"

# Hybrid search (vector + BM25 keyword, fused with RRF)
rak search "CRISPR off-target effects" --hybrid

# Limit results
rak search "attention mechanism" --limit 5

# JSON output (for programmatic use)
rak --json search "spatial omics"
```

## Options

| Flag | Purpose |
|------|---------|
| `--json` | JSON output |
| `--model NAME` | Embedding model (default: all-MiniLM-L6-v2) |
| `--hybrid` | Enable hybrid search (vector + BM25) |
| `--limit N` | Number of results (default: 10) |

## Embedding Models

| Model | Size | Best For |
|-------|------|----------|
| `all-MiniLM-L6-v2` (default) | ~80MB | English papers, fast |
| `nomic-ai/nomic-embed-text-v1.5` | ~270MB | Multilingual, more accurate |

Switch model:
```bash
rak --model nomic-ai/nomic-embed-text-v1.5 index
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
