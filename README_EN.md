# rak — RAG Knowledge Search for Zotero

[中文](README.md)

Semantic and hybrid search over your Zotero library, powered by local embeddings. Ask questions with a local LLM. Everything runs locally — no API keys needed for search.

### Three Working Modes

| Mode | Command | For Whom | Use Case | LLM Required? |
|------|---------|----------|----------|:---:|
| **Search** | `rak search` / `rak export` | AI assistants / scripts | Retrieve paper data, `--json` output for programs | No |
| **Ask** | `rak ask` | Humans | Quick one-off question, get answer and go | Yes |
| **Chat** | `rak chat` | Humans | Deep multi-turn discussion over papers | Yes |

- **Search mode**: Pure local vector/keyword retrieval, fully offline, no API key needed. Claude Code uses `rak --json search` to get structured paper data
- **Ask mode**: Search + LLM answer in one shot. For quick questions in the terminal — use it like a search engine, get your answer and move on
- **Chat mode**: Search + LLM multi-turn conversation with context history. For sitting down and exploring papers in depth, with follow-up questions, `/search` to switch topics, `/tokens` to track usage

## Install

```bash
# Recommended
uv tool install zotero-rag-cli

# Or
pip install zotero-rag-cli
```

Requires `zot` ([zotero-cli-cc](https://github.com/Agents365-ai/zotero-cli-cc)) to be installed and working.

## Quick Start

```bash
# 1. Index your Zotero library (incremental, auto PDF extraction + chunking)
rak index

# 2. Semantic search
rak search "cell fate determination mechanisms"

# 3. Hybrid search (semantic + keyword BM25)
rak search "spatial transcriptomics" --hybrid

# 4. Ask a question (requires Ollama or LMStudio running locally)
rak ask "What are the main methods for single-cell clustering?"

# 5. Interactive multi-turn chat over your papers
rak chat
```

## Commands

### Index

```bash
rak index                    # Incremental index (only new/changed items)
rak index --full             # Force full rebuild
rak index --limit 500        # Limit items fetched from zot
```

Automatically extracts PDF full text from `~/Zotero/storage/` if available. Long documents are split into overlapping chunks (512 words, 64 overlap) for better Q&A accuracy.

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

Requires an LLM service. Supports local (Ollama, LMStudio, vLLM) or cloud (DeepSeek, OpenAI, any OpenAI-compatible API).

### Chat (Multi-turn Q&A)

```bash
rak chat                                # Start interactive session
rak chat --hybrid --context 10          # With hybrid search and more context
rak chat --collection "My Papers"       # Filtered to a collection
```

Interactive REPL for multi-turn conversations over your papers. Maintains conversation history across turns. Commands inside chat:

| Command | Purpose |
|---------|---------|
| `/search <query>` | Retrieve new papers and reset conversation |
| `/context` | Show current paper list |
| `/tokens` | Show estimated token usage and turn count |
| `/help` | Show available commands |
| `/quit` | Exit chat session |

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
rak config llm_api_key sk-xxx        # Set API key (for cloud LLMs)
```

#### LLM Configuration Examples

```bash
# DeepSeek (recommended cloud option)
rak config llm_base_url https://api.deepseek.com
rak config llm_model deepseek-chat
rak config llm_api_key sk-your-deepseek-key

# OpenAI
rak config llm_base_url https://api.openai.com/v1
rak config llm_model gpt-4o
rak config llm_api_key sk-your-openai-key

# Local Ollama (default, no API key needed)
rak config llm_base_url http://localhost:11434/v1
rak config llm_model llama3
rak config llm_api_key not-needed
```

### Status & Clear

```bash
rak status                  # Show index stats (item count, model, last indexed)
rak clear                   # Delete all indexes (with confirmation)
rak clear --yes             # Skip confirmation
```

### Shell Completions

```bash
rak completion bash          # Generate bash completions
rak completion zsh           # Generate zsh completions
rak completion fish          # Generate fish completions
rak completion               # Auto-detect shell

# Enable completions (add to your shell profile):
eval "$(rak completion)"
```

## Comparison with Similar Tools

| Feature | **rak** | [zotero-mcp](https://github.com/54yyyu/zotero-mcp) | [cookjohn/zotero-mcp](https://github.com/cookjohn/zotero-mcp) | [ZoteroBridge](https://github.com/Combjellyshen/ZoteroBridge) |
|---|:---:|:---:|:---:|:---:|
| **Semantic Search** | **✅** | ✅ | ✅ | ❌ |
| **Hybrid Search (Vector + BM25)** | **✅** | ❌ | ❌ | ❌ |
| **PDF Chunking** | **✅** | ❌ | ❌ | ❌ |
| **LLM Q&A** | **✅ Local** | Cloud API | Cloud API | Cloud API |
| **Multi-turn Chat** | **✅** | ❌ | ❌ | ❌ |
| **Streaming Responses** | **✅** | ❌ | ❌ | ❌ |
| **100% Local / No API Keys** | **✅** | ❌ | ❌ | ❌ |
| **CLI Terminal Use** | **✅** | ❌ | ❌ | ❌ |
| **MCP Protocol** | ❌ | ✅ | ✅ | ✅ |
| **Collection/Tag Filters** | **✅** | ✅ | ✅ | ✅ |
| **BibTeX/CSV Export** | **✅** | ❌ | ❌ | ❌ |
| **Incremental Indexing** | **✅** | N/A | N/A | N/A |
| **Shell Completions** | **✅** | ❌ | ❌ | ❌ |
| **JSON Output** | **✅** | N/A | N/A | N/A |
| **AI Coding Assistant** | **✅ Claude Code** | Claude/ChatGPT | Claude/Cursor | Claude/Cursor |
| **Language** | Python | Python | TypeScript | TypeScript |
| **Active** | ✅ 2026 | ✅ 2026 | ✅ 2026 | ✅ 2026 |

### Why rak?

> **The only CLI tool that provides local semantic search, hybrid retrieval, and LLM Q&A over your Zotero library — all without API keys or cloud services.**

- **Local**: All computation runs on your machine — embeddings, search, LLM
- **Fast**: Incremental indexing, millisecond vector search
- **Accurate**: PDF chunking + hybrid search (semantic + BM25 with RRF fusion)
- **Private**: Your papers never leave your machine
- **AI-Native**: Built for Claude Code, `--json` output for AI consumption
- **Terminal-Native**: Full CLI with shell completions, MCP tools can't run in terminal

## Architecture

```
┌─────────────────────────────────────┐
│          rak CLI (Click)            │
│ index│search│ask│chat│export│config │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│           Core Pipeline             │
│  Embedder (sentence-transformers)   │
│  + PDF Extractor + Text Chunker    │
└───────┬────────────────┬────────────┘
        │                │
   ┌────▼────┐    ┌──────▼──────┐
   │ChromaDB │    │ SQLite FTS5 │
   │(vectors)│    │  (keywords) │
   └────┬────┘    └──────┬──────┘
        │                │
   ┌────▼────────────────▼────┐
   │  Searcher (RRF Fusion)   │
   └────────────┬─────────────┘
                │
   ┌────────────▼─────────────┐
   │  Local LLM (Ollama/etc)  │
   │  ask / chat / stream     │
   └──────────────────────────┘
```

Data flows from `zot` CLI → indexer → embedder + PDF extractor → ChromaDB + BM25 → searcher → formatter / LLM.

## Options

| Flag | Commands | Purpose |
|------|----------|---------|
| `--json` | Global | JSON output |
| `--model NAME` | Global | Embedding model (default: all-MiniLM-L6-v2) |
| `--hybrid` | search, ask, chat, export | Enable hybrid search (vector + BM25) |
| `--limit N` | search, export | Number of results (default: 10) |
| `--collection NAME` | search, ask, chat, export | Filter by Zotero collection |
| `--tag TAG` | search, ask, chat, export | Filter by tag (repeatable, OR logic) |
| `--full` | index | Force full rebuild |
| `--context N` | ask, chat | Number of context documents (default: 5) |
| `--llm-model NAME` | ask, chat | Override LLM model |
| `--llm-url URL` | ask, chat | Override LLM server URL |
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
# zot: exact keyword search + CRUD (fast, precise)
zot search "single cell"
zot read ABC123
zot note ABC123 --add "Key finding: ..."

# rak: semantic search + LLM Q&A (understands meaning)
rak search "methods for analyzing individual cell transcriptomes"
rak ask "What are the main single-cell clustering approaches?"

# Best of both
rak search "scRNA-seq clustering" --hybrid
```

| Tool | Strength | Use When |
|------|----------|----------|
| `zot` | Exact match, CRUD, notes, tags | You know what you're looking for |
| `rak` | Semantic understanding, Q&A | You're exploring or asking questions |
| Both | Hybrid search | You want comprehensive results |

## Using with Claude Code

In any Claude Code session, use natural language:

```
Search my Zotero for papers about transformer attention
→ Claude runs: rak --json search "transformer attention"

What do my papers say about attention efficiency?
→ Claude runs: rak ask "What methods improve attention efficiency?"

Export papers about CRISPR as BibTeX
→ Claude runs: rak export "CRISPR" --format bibtex --output refs.bib
```

Add to `~/.claude/CLAUDE.md`:

```markdown
### Zotero RAG
- Use `rak` for semantic search and LLM Q&A over Zotero library
- Use `zot` for exact search, CRUD, notes, tags, exports
- Use `--json` flag when processing results programmatically
```

### Related Projects

- **[zotero-cli-cc](https://github.com/Agents365-ai/zotero-cli-cc)** — Zotero CLI for CRUD operations, required dependency for `rak`
- **[54yyyu/zotero-mcp](https://github.com/54yyyu/zotero-mcp)** — MCP-based Zotero integration with semantic search
- **[cookjohn/zotero-mcp](https://github.com/cookjohn/zotero-mcp)** — MCP Zotero integration for Claude/Cursor
- **[Combjellyshen/ZoteroBridge](https://github.com/Combjellyshen/ZoteroBridge)** — Zotero Bridge for AI coding assistants

---

## Support

<table>
  <tr>
    <td align="center">
      <img src="https://raw.githubusercontent.com/Agents365-ai/images_payment/main/qrcode/wechat-pay.png" width="180" alt="WeChat Pay">
      <br>
      <b>WeChat Pay</b>
    </td>
    <td align="center">
      <img src="https://raw.githubusercontent.com/Agents365-ai/images_payment/main/qrcode/alipay.png" width="180" alt="Alipay">
      <br>
      <b>Alipay</b>
    </td>
    <td align="center">
      <img src="https://raw.githubusercontent.com/Agents365-ai/images_payment/main/qrcode/buymeacoffee.png" width="180" alt="Buy Me a Coffee">
      <br>
      <b>Buy Me a Coffee</b>
    </td>
  </tr>
</table>

## License

MIT
