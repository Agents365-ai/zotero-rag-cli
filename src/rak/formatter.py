from __future__ import annotations

import json
from io import StringIO

from rich.console import Console
from rich.table import Table

from rak.searcher import SearchResult


def format_results(results: list[SearchResult], output_json: bool = False) -> str:
    if output_json:
        items = []
        for r in results:
            item = {"key": r.doc_id, "title": r.title, "score": round(r.score, 4), "source": r.source}
            if r.snippet:
                item["snippet"] = r.snippet
            items.append(item)
        return json.dumps(items, indent=2, ensure_ascii=False)
    buf = StringIO()
    console = Console(file=buf, force_terminal=False, width=120)
    table = Table(show_header=True, header_style="bold")
    table.add_column("Key", style="cyan", width=10)
    table.add_column("Title", width=60)
    table.add_column("Score", width=8, justify="right")
    table.add_column("Source", width=10)
    for r in results:
        table.add_row(r.doc_id, r.title, f"{r.score:.3f}", r.source)
    console.print(table)
    return buf.getvalue()


def format_index_stats(count: int, output_json: bool = False) -> str:
    if output_json:
        return json.dumps({"indexed": count})
    return f"Indexed {count} papers."


def format_incremental_stats(stats: dict, output_json: bool = False) -> str:
    if output_json:
        return json.dumps({k: v for k, v in stats.items() if k != "registry"})
    parts = []
    if stats["added"]:
        parts.append(f"{stats['added']} new")
    if stats["updated"]:
        parts.append(f"{stats['updated']} updated")
    if stats["removed"]:
        parts.append(f"{stats['removed']} removed")
    parts.append(f"{stats['unchanged']} unchanged")
    return ", ".join(parts) + "."


def format_ask_result(
    answer: str,
    sources: list[dict],
    output_json: bool = False,
) -> str:
    if output_json:
        return json.dumps({
            "answer": answer,
            "sources": [
                {"key": s["key"], "title": s["title"], "score": round(s["score"], 4)}
                for s in sources
            ],
        }, indent=2, ensure_ascii=False)
    lines = [answer, "", "Sources:"]
    for i, s in enumerate(sources, 1):
        lines.append(f"  {i}. {s['key']} - {s['title']} (score: {s['score']:.3f})")
    return "\n".join(lines)
