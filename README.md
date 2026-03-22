# rak — Zotero 文献库 RAG 语义搜索

[English](README_EN.md)

## 简介

`rak` 是一个本地化的 Zotero RAG 搜索工具，专为 [Claude Code](https://claude.ai/code) 设计。

### 三种工作模式

| 模式 | 命令 | 给谁用 | 场景 | 需要 LLM？ |
|------|------|--------|------|:---:|
| **搜索模式** | `rak search` / `rak export` | AI 助手 / 脚本 | 检索论文信息，`--json` 输出供程序解析 | 不需要 |
| **问答模式** | `rak ask` | 人 | 快速问一个问题，拿到答案就走 | 需要 |
| **对话模式** | `rak chat` | 人 | 围绕论文深入讨论，多轮追问 | 需要 |

- **搜索模式**：纯本地向量/关键词检索，完全离线，无需 API Key。Claude Code 通过 `rak --json search` 获取结构化论文数据
- **问答模式**：搜索 + LLM 生成回答，一问一答。适合在终端快速提问，像搜索引擎一样用完即走
- **对话模式**：搜索 + LLM 多轮对话，维护上下文历史。适合坐下来深入探讨，支持追问、`/search` 切换话题、`/tokens` 查看用量

**核心特性：**
- **语义搜索**：基于 sentence-transformers 本地嵌入，理解查询意图
- **混合搜索**：向量搜索 + BM25 关键词搜索，RRF 融合排序
- **PDF 全文**：自动提取 PDF 全文并分块索引（512 词，64 词重叠）
- **多 LLM 支持**：本地（Ollama/LMStudio）或云端（DeepSeek/OpenAI/任何 OpenAI 兼容 API）
- **全程离线**：搜索模式无需 API Key，所有数据留在本地

## 安装

```bash
# 推荐
uv tool install zotero-rag-cli

# 或者
pip install zotero-rag-cli
```

需要先安装 `zot` ([zotero-cli-cc](https://github.com/Agents365-ai/zotero-cli-cc))。

## 快速开始

```bash
# 1. 索引 Zotero 文献库（增量索引，自动提取 PDF 全文 + 分块）
rak index

# 2. 语义搜索
rak search "细胞命运决定机制"

# 3. 混合搜索（语义 + 关键词 BM25）
rak search "spatial transcriptomics" --hybrid

# 4. 提问（需要本地运行 Ollama 或 LMStudio）
rak ask "单细胞聚类的主要方法有哪些？"

# 5. 交互式多轮对话
rak chat
```

## 命令一览

### 索引

```bash
rak index                    # 增量索引（仅处理新增/变更项目）
rak index --full             # 全量重建
rak index --limit 500        # 限制从 zot 获取的项目数
```

自动从 `~/Zotero/storage/` 提取 PDF 全文。长文档自动分割为重叠片段，提升问答准确率。

### 搜索

```bash
rak search "单细胞 RNA 测序方法"
rak search "CRISPR 脱靶效应" --hybrid
rak search "attention mechanism" --limit 5
rak search "RNA-seq" --collection "My Papers" --tag "methods"
rak --json search "spatial omics"
```

### 提问（LLM 问答）

```bash
rak ask "关于细胞命运的主要发现是什么？"
rak ask "比较 CRISPR 方法" --context 10 --hybrid
rak ask "总结空间组学" --llm-model mistral --llm-url http://localhost:1234/v1
```

需要 LLM 服务。支持本地（Ollama、LMStudio、vLLM）或云端（DeepSeek、OpenAI 等 OpenAI 兼容 API）。

### 多轮对话

```bash
rak chat                                # 启动交互会话
rak chat --hybrid --context 10          # 混合搜索 + 更多上下文
rak chat --collection "My Papers"       # 限定 collection
```

交互式 REPL，支持多轮对话和话题切换：

| 命令 | 功能 |
|------|------|
| `/search <查询>` | 检索新论文，重置对话 |
| `/context` | 显示当前论文列表 |
| `/tokens` | 显示估算 token 用量和对话轮次 |
| `/help` | 显示帮助信息 |
| `/quit` | 退出对话 |

### 导出

```bash
rak export "single cell" --format csv                    # CSV 输出
rak export "CRISPR" --format bibtex --output refs.bib    # BibTeX 导出到文件
rak export "RNA-seq" --hybrid --collection "Methods"     # 带过滤条件
```

### 配置

```bash
rak config                           # 显示所有设置
rak config llm_model mistral         # 持久化设置 LLM 模型
rak config llm_base_url http://localhost:1234/v1
rak config llm_api_key sk-xxx        # 设置 API Key（云端 LLM 需要）
```

#### LLM 配置示例

```bash
# DeepSeek（推荐云端方案）
rak config llm_base_url https://api.deepseek.com
rak config llm_model deepseek-chat
rak config llm_api_key sk-your-deepseek-key

# OpenAI
rak config llm_base_url https://api.openai.com/v1
rak config llm_model gpt-4o
rak config llm_api_key sk-your-openai-key

# 本地 Ollama（默认，无需 API Key）
rak config llm_base_url http://localhost:11434/v1
rak config llm_model llama3
rak config llm_api_key not-needed
```

### 状态与清除

```bash
rak status                  # 显示索引状态（项目数、模型、最后索引时间）
rak clear                   # 删除所有索引（需确认）
rak clear --yes             # 跳过确认
```

### Shell 补全

```bash
rak completion bash          # 生成 bash 补全脚本
rak completion zsh           # 生成 zsh 补全脚本
rak completion fish          # 生成 fish 补全脚本
rak completion               # 自动检测 shell

# 启用补全（添加到 shell 配置文件）：
eval "$(rak completion)"
```

## MCP Server（LM Studio / Cursor / Claude Desktop）

`rak` 内置 MCP Server，可在支持 MCP 协议的工具中直接调用搜索功能。

### 安装

```bash
pip install zotero-rag-cli[mcp]
```

### 提供的工具

| 工具 | 功能 |
|------|------|
| `search_papers` | 语义/混合搜索，返回论文标题、分数 |
| `index_status` | 查看索引状态 |

### 配置

在 LM Studio / Cursor / Claude Desktop 的 MCP 配置中添加：

```json
{
  "mcpServers": {
    "rak": {
      "command": "rak-mcp"
    }
  }
}
```

LM Studio 等工具的大模型拿到检索结果后自行判断和回答。

## 同类工具对比

| 特性 | **rak** | [zotero-mcp](https://github.com/54yyyu/zotero-mcp) | [cookjohn/zotero-mcp](https://github.com/cookjohn/zotero-mcp) | [ZoteroBridge](https://github.com/Combjellyshen/ZoteroBridge) |
|---|:---:|:---:|:---:|:---:|
| **语义搜索** | **✅** | ✅ | ✅ | ❌ |
| **混合搜索（向量 + BM25）** | **✅** | ❌ | ❌ | ❌ |
| **PDF 分块索引** | **✅** | ❌ | ❌ | ❌ |
| **LLM 问答** | **✅ 本地** | 云 API | 云 API | 云 API |
| **多轮对话** | **✅** | ❌ | ❌ | ❌ |
| **流式输出** | **✅** | ❌ | ❌ | ❌ |
| **100% 本地 / 无需 API Key** | **✅** | ❌ | ❌ | ❌ |
| **CLI 终端使用** | **✅** | ❌ | ❌ | ❌ |
| **MCP 协议** | **✅** | ✅ | ✅ | ✅ |
| **Collection/标签过滤** | **✅** | ✅ | ✅ | ✅ |
| **BibTeX/CSV 导出** | **✅** | ❌ | ❌ | ❌ |
| **增量索引** | **✅** | N/A | N/A | N/A |
| **Shell 补全** | **✅** | ❌ | ❌ | ❌ |
| **JSON 输出** | **✅** | N/A | N/A | N/A |
| **AI 编码助手集成** | **✅ Claude Code** | Claude/ChatGPT | Claude/Cursor | Claude/Cursor |
| **语言** | Python | Python | TypeScript | TypeScript |
| **活跃维护** | ✅ 2026 | ✅ 2026 | ✅ 2026 | ✅ 2026 |

### 为什么选择 rak？

> **唯一一个提供本地语义搜索、混合检索和 LLM 问答的 Zotero CLI 工具 — 全程无需 API Key 或云服务。**

- **本地化**：所有计算在本机运行 — 嵌入、搜索、LLM
- **高速**：增量索引，毫秒级向量搜索
- **精准**：PDF 分块 + 混合搜索（语义 + BM25 RRF 融合）
- **隐私**：论文数据永不离开本机
- **AI 原生**：专为 Claude Code 设计，`--json` 输出供 AI 解析
- **终端原生**：完整 CLI + Shell 补全，MCP 工具无法在终端中使用

## 架构

```
┌─────────────────────────────────────┐
│          rak CLI (Click)            │
│ index│search│ask│chat│export│config │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│           核心管线                   │
│  Embedder (sentence-transformers)   │
│  + PDF 提取器 + 文本分块器          │
└───────┬────────────────┬────────────┘
        │                │
   ┌────▼────┐    ┌──────▼──────┐
   │ChromaDB │    │ SQLite FTS5 │
   │(向量)    │    │ (关键词)     │
   └────┬────┘    └──────┬──────┘
        │                │
   ┌────▼────────────────▼────┐
   │  Searcher (RRF 融合排序)  │
   └────────────┬─────────────┘
                │
   ┌────────────▼─────────────┐
   │  本地 LLM (Ollama 等)     │
   │  ask / chat / stream     │
   └──────────────────────────┘
```

## 搭配 zot 使用

`rak` 专为搭配 [`zot`](https://github.com/Agents365-ai/zotero-cli-cc) 使用：

```bash
# zot：精确搜索 + CRUD（快速、精确）
zot search "single cell"
zot read ABC123
zot note ABC123 --add "关键发现：..."

# rak：语义搜索 + LLM 问答（理解语义）
rak search "分析单个细胞转录组的方法"
rak ask "主要的单细胞聚类方法有哪些？"

# 结合使用
rak search "scRNA-seq clustering" --hybrid
```

| 工具 | 优势 | 适用场景 |
|------|------|----------|
| `zot` | 精确匹配、CRUD、笔记、标签 | 明确知道要找什么 |
| `rak` | 语义理解、问答 | 探索性搜索或提问 |
| 两者结合 | 混合搜索 | 需要全面的检索结果 |

## 在 Claude Code 中使用

在任何 Claude Code 会话中，直接用自然语言请求：

```
帮我搜索 Zotero 中关于 transformer attention 的论文
→ Claude 自动运行: rak --json search "transformer attention"

我的论文中关于注意力效率有什么发现？
→ Claude 自动运行: rak ask "What methods improve attention efficiency?"

导出 CRISPR 相关论文为 BibTeX
→ Claude 自动运行: rak export "CRISPR" --format bibtex --output refs.bib
```

建议在 `~/.claude/CLAUDE.md` 中添加：

```markdown
### Zotero RAG
- 使用 `rak` 进行语义搜索和 LLM 问答
- 使用 `zot` 进行精确搜索、CRUD、笔记、标签、导出
- 处理结果时使用 `--json` 标志
```

### 相关项目

- **[zotero-cli-cc](https://github.com/Agents365-ai/zotero-cli-cc)** — Zotero CLI CRUD 工具，`rak` 的必要依赖
- **[54yyyu/zotero-mcp](https://github.com/54yyyu/zotero-mcp)** — 基于 MCP 协议的 Zotero 语义搜索集成
- **[cookjohn/zotero-mcp](https://github.com/cookjohn/zotero-mcp)** — MCP Zotero 集成，支持 Claude/Cursor
- **[Combjellyshen/ZoteroBridge](https://github.com/Combjellyshen/ZoteroBridge)** — Zotero Bridge AI 编码助手集成

---

## 支持作者

<table>
  <tr>
    <td align="center">
      <img src="https://raw.githubusercontent.com/Agents365-ai/images_payment/main/qrcode/wechat-pay.png" width="180" alt="微信支付">
      <br>
      <b>微信支付</b>
    </td>
    <td align="center">
      <img src="https://raw.githubusercontent.com/Agents365-ai/images_payment/main/qrcode/alipay.png" width="180" alt="支付宝">
      <br>
      <b>支付宝</b>
    </td>
    <td align="center">
      <img src="https://raw.githubusercontent.com/Agents365-ai/images_payment/main/qrcode/buymeacoffee.png" width="180" alt="Buy Me a Coffee">
      <br>
      <b>Buy Me a Coffee</b>
    </td>
  </tr>
</table>

## 许可证

MIT
