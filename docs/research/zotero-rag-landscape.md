# Zotero RAG 竞品调研

> 调研日期: 2026-03-22

## 竞品格局总览

### 第一梯队（1000+ stars）

| 项目 | Stars | 类型 | 语言 | 数据源 | 核心特点 |
|------|------:|------|------|--------|---------|
| [paper-qa](https://github.com/Future-House/paper-qa) | 8,294 | Python 库 | Python | Web API | ICLR 2025 论文，超人级科学 QA，`pip install paper-qa[zotero]` |
| [zotero-gpt](https://github.com/MuiseDestiny/zotero-gpt) | 6,994 | Zotero 插件 | TypeScript | 插件内 | 最流行 Zotero AI 插件，直接 LLM 对话，Markdown/LaTeX 渲染 |
| [papersgpt](https://github.com/papersgpt/papersgpt-for-zotero) | 2,189 | Zotero 插件 | JavaScript | 插件内 | 本地 RAG + 10+ LLM + MCP，批量分析 100+ 论文，C++ MCP 服务器 |
| [54yyyu/zotero-mcp](https://github.com/54yyyu/zotero-mcp) | 1,963 | MCP 服务器 | Python | 本地 API | 向量语义搜索，多 embedding 后端（本地/OpenAI/Gemini） |
| [Aria](https://github.com/lifan0127/ai-research-assistant) | 1,675 | Zotero 插件 | JavaScript | 插件内 | GPT-4 Vision 支持，笔记生成，聊天导出 |

### 第二梯队（100-999 stars）

| 项目 | Stars | 类型 | 语言 | 核心特点 |
|------|------:|------|------|---------|
| [cookjohn/zotero-mcp](https://github.com/cookjohn/zotero-mcp) | 510 | Zotero 插件+MCP | TypeScript | MCP 内嵌在 Zotero 插件中，语义搜索，全 CRUD |
| [The-Oracle-of-Zotero](https://github.com/Frost-group/The-Oracle-of-Zotero) | 440 | Python | Python | 先驱项目，LangChain + FAISS，已停更 |
| [llm-for-zotero](https://github.com/yilewang/llm-for-zotero) | 345 | Zotero 插件 | TypeScript | 极简设计，自带 LLM，PDF 阅读器侧边栏 |
| [zotero-chatgpt](https://github.com/kazgu/zotero-chatgpt) | 298 | Zotero 插件 | TypeScript | ChatGPT 集成 |
| [beaver-zotero](https://github.com/jlegewie/beaver-zotero) | 109 | Zotero 插件 | TypeScript | 免费语义搜索，PDF AI 标注，可分享聊天链接 |

### 第三梯队：值得关注的小项目

| 项目 | Stars | 核心亮点 |
|------|------:|---------|
| [zotero-paper-agent](https://github.com/windfollowingheart/zotero-paper-agent) | 97 | Zotero 插件，侧边栏聊天+文件上传 |
| [RAG-Assistant-for-Zotero](https://github.com/aahepburn/RAG-Assistant-for-Zotero) | 76 | **混合搜索（向量+BM25 RRF）**，ChromaDB，Ollama 全本地，Electron 桌面应用 |
| [ZotSeek](https://github.com/introfini/ZotSeek) | 64 | **100% 本地语义搜索**，内置 nomic-embed-text-v1.5，~70ms，section-aware |
| [kerim/zotero-code-execution](https://github.com/kerim/zotero-code-execution) | 43 | 多策略搜索，代码执行模式 |
| [seerai](https://github.com/dralkh/seerai) | 37 | Zotero 8 AI 插件，MCP + OCR + 表格提取 + Semantic Scholar |
| [ZoteroScholar](https://github.com/sanketsabharwal/ZoteroScholar) | 35 | Zotero + Ollama 全本地 Q&A |
| [Graph-RAG](https://github.com/zjkhurry/Graph-RAG) | 29 | **唯一 Graph RAG**，Neo4j 图数据库，引用网络可视化 |
| [zotero-rag/zotero-rag](https://github.com/zotero-rag/zotero-rag) | 12 | **唯一 Rust 实现**，Anthropic + Voyage AI，CLI 优先 |
| [deep-zotero](https://github.com/ccam80/deep-zotero) | 3 | MCP 13 工具，表格/图表提取（Claude Haiku Vision），引用图谱 |

### PyPI 生态

| 包名 | 说明 |
|------|------|
| `paper-qa` | 最成熟科学 RAG，`[zotero]` extra |
| `langchain-zotero-retriever` | 官方 LangChain Zotero Retriever |
| `zotero-mcp` | 54yyyu 的 MCP 服务器 |
| `pyzotero` | Zotero Web API Python 客户端（多数项目的基础依赖） |

---

## 技术方案对比

### Embedding 模型

| 项目 | 模型 | 本地/API | 大小 |
|------|------|---------|------|
| paper-qa | OpenAI | API | - |
| 54yyyu/zotero-mcp | all-MiniLM-L6-v2 (默认) | 本地 | ~80MB |
| ZotSeek | nomic-embed-text-v1.5 | 本地（内置） | ~270MB |
| RAG-Assistant | sentence-transformers | 本地 | ~80MB |
| zotero-rag (Rust) | Voyage AI | API | - |

### 向量存储

| 项目 | 向量库 |
|------|--------|
| paper-qa | 内存（可配置） |
| 54yyyu/zotero-mcp | ChromaDB |
| ZotSeek | SQLite (zotseek.sqlite) |
| RAG-Assistant | ChromaDB |
| Graph-RAG | Neo4j |
| deep-zotero | 内置 |

### 搜索策略

| 策略 | 项目 | 说明 |
|------|------|------|
| 纯向量搜索 | 54yyyu/zotero-mcp, paper-qa | embedding 相似度 |
| 混合搜索 (向量+BM25) | RAG-Assistant, ZotSeek | RRF 融合排序 |
| Graph RAG | zjkhurry/Graph-RAG | 引用关系图谱 + 向量 |
| Section-aware | ZotSeek, deep-zotero | 区分 Abstract/Methods/Results |

---

## 市场空白分析

### 1. CLI-first RAG（无人占据）
所有 RAG 项目都是 Zotero 插件或 Web App，没有一个是终端 CLI 工具。`rak` 可以填补这个空白，和 `zot` 形成互补。

### 2. 多数据源 RAG（无人占据）
没有项目把 Zotero + Obsidian + 本地 PDF 目录统一到一个 RAG pipeline。

### 3. 全本地 + 零配置 CLI（无人占据）
paper-qa 需要 OpenAI API，ZotSeek 是 Zotero 插件，没有一个像 `zot` 一样零配置的独立 RAG CLI。

### 4. Graph RAG（几乎空白）
只有一个已停更的项目用 Neo4j。论文引用关系图谱方向几乎无竞争。

---

## rak 的差异化定位

```
rak = CLI-first + 全本地 + 多数据源 + 零配置

竞品定位:
├── Zotero 插件 (zotero-gpt, papersgpt, ZotSeek) → 需要 Zotero 运行
├── MCP 服务器 (54yyyu, cookjohn) → 需要 AI 客户端
├── Python 库 (paper-qa) → 需要 API Key
└── rak → 终端 CLI，本地 embedding，zot 数据源，零配置
```

### 核心竞争力

1. **终端原生** — 唯一 CLI RAG 工具，和 `zot` 无缝配合
2. **全本地** — 内置 embedding 模型，无需 API Key，数据不出本机
3. **多数据源** — Zotero（通过 `zot --json`）+ Obsidian + 本地 PDF
4. **混合搜索** — 向量搜索 + BM25 + RRF 融合（参考 RAG-Assistant 和 ZotSeek）
5. **零配置** — `pip install zotero-rag-cli` 即可使用

### 技术选型建议

| 组件 | 推荐 | 理由 |
|------|------|------|
| Embedding | `all-MiniLM-L6-v2` | 80MB，效果好，社区标准 |
| 向量存储 | SQLite-vec 或 ChromaDB | SQLite-vec 零依赖，ChromaDB 更成熟 |
| 搜索策略 | 混合搜索 (向量+BM25+RRF) | ZotSeek 和 RAG-Assistant 验证了效果 |
| 数据源 | `zot --json` 管道 | 复用 zotero-cli-cc，无需重复实现 |
| LLM (可选) | Ollama / API | 纯搜索不需要 LLM，Q&A 模式可选 |
