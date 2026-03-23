from __future__ import annotations

import click

from rak import __version__
from rak.config import RakConfig
from rak.errors import DimensionMismatchError, EmptyLibraryError, ModelDownloadError, ZotNotFoundError


@click.group()
@click.version_option(version=__version__, prog_name="rak")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
@click.option("--model", default=None, help="Embedding model name")
@click.option("--verbose", is_flag=True, help="Enable debug logging")
@click.pass_context
def main(ctx: click.Context, output_json: bool, model: str | None, verbose: bool) -> None:
    """rak — Semantic search over your Zotero library."""
    if verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s: %(message)s")
    ctx.ensure_object(dict)
    ctx.obj["json"] = output_json
    config = RakConfig()
    if model:
        config.model_name = model
    ctx.obj["config"] = config


def _create_embedder(config: RakConfig):
    """Create an Embedder from config."""
    from rak.embedder import Embedder
    return Embedder(config.model_name, provider=config.embedding_provider,
                    base_url=config.embedding_base_url, api_key=config.embedding_api_key)


def _run_search(config: RakConfig, query: str, limit: int,
                collection: str | None, tags: tuple[str, ...] | list[str],
                hybrid: bool, bm25_only: bool):
    """Execute a search and return (results, vector_store_or_none).

    Handles bm25-only vs vector/hybrid branching in one place.
    """
    from rak.bm25 import BM25Index
    from rak.searcher import Searcher

    tag_list = list(tags) if tags else None
    vector_store = None

    if bm25_only:
        with BM25Index(config.fts_db_path) as bm25:
            searcher = Searcher(None, None, bm25)
            results = searcher.bm25_search(query, limit=limit)
    else:
        from rak.store import VectorStore
        embedder = _create_embedder(config)
        vector_store = VectorStore(config.chroma_dir, embedder.dimension)
        with BM25Index(config.fts_db_path) as bm25:
            searcher = Searcher(embedder, vector_store, bm25)
            if hybrid:
                results = searcher.hybrid_search(query, limit=limit, collection=collection, tags=tag_list)
            else:
                results = searcher.vector_search(query, limit=limit, collection=collection, tags=tag_list)

    return results, vector_store


def _make_progress_bar():
    """Create a Rich progress bar with standard columns."""
    from rich.progress import Progress, BarColumn, TextColumn, MofNCompleteColumn, TimeRemainingColumn
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
    )


@main.command()
@click.option("--limit", default=5000, help="Max items to index from zot")
@click.option("--full", is_flag=True, help="Force full rebuild (ignore existing index)")
@click.pass_context
def index(ctx: click.Context, limit: int, full: bool) -> None:
    """Index Zotero library for semantic search."""
    from rak.bm25 import BM25Index
    from rak.formatter import format_incremental_stats, format_index_stats
    from rak.indexer import fetch_zot_items, index_items
    from rak.metadata import save_metadata
    from rak.registry import load_registry, save_registry
    from rak.store import VectorStore

    config: RakConfig = ctx.obj["config"]
    json_out = ctx.obj["json"]
    config.data_dir.mkdir(parents=True, exist_ok=True)

    try:
        click.echo("Loading embedding model...")
        embedder = _create_embedder(config)

        click.echo("Fetching items from zot...")
        items = fetch_zot_items(config.zot_command, limit=limit)
        click.echo(f"Found {len(items)} items.")

        vector_store = VectorStore(config.chroma_dir, embedder.dimension)

        storage_dir = config.zotero_storage_dir
        if storage_dir:
            click.echo(f"PDF extraction enabled: {storage_dir}")
            if config.pdf_provider != "pymupdf":
                click.echo(f"PDF provider: {config.pdf_provider} (slower but higher quality)")
            else:
                import shutil
                available = [p for p in ("mineru", "docling") if shutil.which(p)]
                if available:
                    click.echo(f"Tip: {', '.join(available)} detected. Use `rak config pdf_provider {available[0]}` for better PDF extraction.")
        else:
            click.echo("PDF extraction: Zotero storage not found, indexing metadata only.")

        registry = None if full else load_registry(config.data_dir)
        if registry is not None and not registry:
            registry = None

        with BM25Index(config.fts_db_path) as bm25, _make_progress_bar() as progress:
            task_desc = f"Indexing ({config.pdf_provider})" if config.pdf_provider != "pymupdf" else "Indexing"
            task = progress.add_task(task_desc, total=len(items))

            def on_progress(current: int, total: int) -> None:
                progress.update(task, completed=current)

            if registry is None:
                if full:
                    vector_store.clear()
                    bm25.clear()
                result = index_items(items, embedder, vector_store, bm25, on_progress, storage_dir=storage_dir, chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap, pdf_provider=config.pdf_provider)
                save_registry(config.data_dir, result["registry"])
                click.echo(format_index_stats(result["added"], output_json=json_out))
            else:
                result = index_items(items, embedder, vector_store, bm25, on_progress, registry=registry, storage_dir=storage_dir, chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap, pdf_provider=config.pdf_provider)
                save_registry(config.data_dir, result["registry"])
                click.echo(format_incremental_stats(result, output_json=json_out))

        save_metadata(config.data_dir, config.model_name, vector_store.count())
    except EmptyLibraryError as exc:
        click.echo(str(exc))
        ctx.exit(0)
    except (ZotNotFoundError, DimensionMismatchError) as exc:
        click.echo(f"Error: {exc}", err=True)
        ctx.exit(1)
    except ModelDownloadError as exc:
        click.echo(f"Error: {exc}", err=True)
        click.echo("Check your internet connection and try again.", err=True)
        ctx.exit(1)


@main.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show index status and metadata."""
    import json as json_mod
    from rak.metadata import load_metadata

    config: RakConfig = ctx.obj["config"]
    json_out = ctx.obj["json"]
    meta = load_metadata(config.data_dir)

    if meta is None:
        click.echo("No index found. Run 'rak index' first.")
        return

    if json_out:
        click.echo(json_mod.dumps({
            "item_count": meta.item_count,
            "model_name": meta.model_name,
            "last_indexed": meta.last_indexed,
            "data_directory": str(config.data_dir),
        }, indent=2))
    else:
        click.echo(f"Index: {meta.item_count} items")
        click.echo(f"Model: {meta.model_name}")
        click.echo(f"Last indexed: {meta.last_indexed}")
        click.echo(f"Data directory: {config.data_dir}")


@main.command()
@click.option("--yes", is_flag=True, help="Skip confirmation prompt")
@click.pass_context
def clear(ctx: click.Context, yes: bool) -> None:
    """Delete all indexes and reset."""
    import shutil
    from rak.metadata import META_FILENAME
    from rak.registry import REGISTRY_FILENAME

    config: RakConfig = ctx.obj["config"]
    targets = [config.chroma_dir, config.fts_db_path, config.data_dir / META_FILENAME, config.data_dir / REGISTRY_FILENAME]
    exists = [t for t in targets if t.exists()]

    if not exists:
        click.echo("Nothing to clear.")
        return

    if not yes:
        click.confirm("Delete all indexes?", abort=True)

    for target in exists:
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()

    click.echo("Cleared all indexes.")


@main.command("config")
@click.argument("key", required=False)
@click.argument("value", required=False)
@click.pass_context
def config_cmd(ctx: click.Context, key: str | None, value: str | None) -> None:
    """Show or set configuration values."""
    from rak.config import CONFIGURABLE_KEYS, load_config, save_config

    config: RakConfig = ctx.obj["config"]

    if key and value:
        if key not in CONFIGURABLE_KEYS:
            import difflib
            close = difflib.get_close_matches(key, CONFIGURABLE_KEYS, n=1, cutoff=0.5)
            hint = f" Did you mean '{close[0]}'?" if close else ""
            click.echo(f"Unknown config key: {key}.{hint}", err=True)
            click.echo(f"Valid keys: {', '.join(sorted(CONFIGURABLE_KEYS))}", err=True)
            ctx.exit(1)
            return
        try:
            save_config(config.data_dir, key, value)
        except ValueError as exc:
            click.echo(f"Error: {exc}", err=True)
            ctx.exit(1)
            return
        click.echo(f"{key} = {value}")
    elif key:
        if hasattr(config, key):
            click.echo(f"{key} = {getattr(config, key)}")
        else:
            click.echo(f"Unknown config key: {key}", err=True)
            ctx.exit(1)
    else:
        click.echo(f"model_name = {config.model_name}")
        click.echo(f"zot_command = {config.zot_command}")
        click.echo(f"llm_base_url = {config.llm_base_url}")
        click.echo(f"llm_model = {config.llm_model}")
        api_display = config.llm_api_key if config.llm_api_key == "not-needed" else config.llm_api_key[:8] + "..."
        click.echo(f"llm_api_key = {api_display}")
        click.echo(f"embedding_provider = {config.embedding_provider}")
        if config.embedding_provider == "api":
            click.echo(f"embedding_base_url = {config.embedding_base_url}")
            emb_key = config.embedding_api_key if config.embedding_api_key == "not-needed" else config.embedding_api_key[:8] + "..."
            click.echo(f"embedding_api_key = {emb_key}")
        click.echo(f"pdf_provider = {config.pdf_provider}")
        click.echo(f"data_dir = {config.data_dir}")
        click.echo(f"zotero_storage_dir = {config.zotero_storage_dir}")


def _resolve_key(key_or_title: str, vector_store, bm25) -> str | None:
    """Resolve a key or title query to a Zotero key.

    If the argument looks like a Zotero key (short alphanumeric, exists in index),
    return it directly. Otherwise treat it as a title search and let the user pick.
    """
    import re
    # Zotero keys are typically 8 alphanumeric chars (e.g., ABC12345)
    if re.match(r'^[A-Za-z0-9]{4,12}$', key_or_title):
        if vector_store.has(key_or_title) or vector_store.has(f"{key_or_title}_chunk_0"):
            return key_or_title

    # Treat as title search via BM25
    results = bm25.search(key_or_title, limit=10)
    if not results:
        click.echo(f"No papers found matching '{key_or_title}'.")
        return None

    # Get titles from vector store metadata
    ids = [r["id"] for r in results]
    # Filter to parent keys (strip chunk suffixes)
    parent_ids = list(dict.fromkeys(
        doc_id.split("_chunk_")[0] if "_chunk_" in doc_id else doc_id
        for doc_id in ids
    ))
    doc_data = vector_store.get(ids=parent_ids, include=["metadatas"])
    titles = {doc_id: meta.get("title", doc_id) for doc_id, meta in zip(doc_data["ids"], doc_data["metadatas"])}

    if len(titles) == 1:
        chosen = list(titles.keys())[0]
        click.echo(f"Matched: {titles[chosen]}")
        return chosen

    # Multiple matches — let user pick
    click.echo(f"Found {len(titles)} papers matching '{key_or_title}':")
    items = list(titles.items())
    for i, (doc_id, title) in enumerate(items, 1):
        click.echo(f"  {i}. [{doc_id}] {title}")

    try:
        choice = input("Select number (or Enter to cancel): ").strip()
    except (EOFError, KeyboardInterrupt):
        click.echo()
        return None
    if not choice:
        return None
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(items):
            return items[idx][0]
    except ValueError:
        pass
    click.echo("Invalid selection.")
    return None


@main.command()
@click.argument("key_or_title")
@click.option("--limit", default=10, help="Number of results")
@click.option("--collection", default=None, help="Filter by Zotero collection name")
@click.option("--tag", "tags", multiple=True, help="Filter by tag (repeatable, OR logic)")
@click.pass_context
def similar(ctx: click.Context, key_or_title: str, limit: int, collection: str | None, tags: tuple[str, ...]) -> None:
    """Find papers similar to a given one by its Zotero key or title.

    KEY_OR_TITLE can be a Zotero item key (e.g., ABC12345) or a title search
    query (e.g., "attention is all you need"). If multiple papers match,
    you will be prompted to select one.
    """
    from rak.bm25 import BM25Index
    from rak.formatter import format_results
    from rak.searcher import Searcher
    from rak.store import VectorStore

    config: RakConfig = ctx.obj["config"]
    json_out = ctx.obj["json"]

    try:
        embedder = _create_embedder(config)
        vector_store = VectorStore(config.chroma_dir, embedder.dimension)
        with BM25Index(config.fts_db_path) as bm25:
            key = _resolve_key(key_or_title, vector_store, bm25)
            if key is None:
                return

            searcher = Searcher(embedder, vector_store, bm25)
            results = searcher.similar_search(key, limit=limit, collection=collection, tags=list(tags) or None)

        if not results:
            click.echo(f"No similar papers found for '{key_or_title}'.")
            return

        output = format_results(results, output_json=json_out)
        if output.strip():
            click.echo(output)
    except (ModelDownloadError, DimensionMismatchError) as exc:
        click.echo(f"Error: {exc}", err=True)
        ctx.exit(1)


@main.command()
@click.option("--limit", default=5000, help="Max items to index from zot")
@click.pass_context
def reindex(ctx: click.Context, limit: int) -> None:
    """Clear indexes and rebuild from scratch. Useful after changing pdf_provider."""
    import shutil
    from rak.bm25 import BM25Index
    from rak.formatter import format_index_stats
    from rak.indexer import fetch_zot_items, index_items
    from rak.metadata import META_FILENAME, save_metadata
    from rak.registry import REGISTRY_FILENAME, save_registry
    from rak.store import VectorStore

    config: RakConfig = ctx.obj["config"]
    json_out = ctx.obj["json"]
    config.data_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Clear existing indexes
        for target in [config.chroma_dir, config.fts_db_path, config.data_dir / META_FILENAME, config.data_dir / REGISTRY_FILENAME]:
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
        click.echo("Cleared existing indexes.")

        click.echo("Loading embedding model...")
        embedder = _create_embedder(config)

        click.echo("Fetching items from zot...")
        items = fetch_zot_items(config.zot_command, limit=limit)
        click.echo(f"Found {len(items)} items.")

        vector_store = VectorStore(config.chroma_dir, embedder.dimension)

        storage_dir = config.zotero_storage_dir
        if storage_dir:
            click.echo(f"PDF extraction enabled: {storage_dir}")
            if config.pdf_provider != "pymupdf":
                click.echo(f"PDF provider: {config.pdf_provider} (slower but higher quality)")
        else:
            click.echo("PDF extraction: Zotero storage not found, indexing metadata only.")

        with BM25Index(config.fts_db_path) as bm25, _make_progress_bar() as progress:
            task_desc = f"Reindexing ({config.pdf_provider})" if config.pdf_provider != "pymupdf" else "Reindexing"
            task = progress.add_task(task_desc, total=len(items))

            def on_progress(current: int, total: int) -> None:
                progress.update(task, completed=current)

            result = index_items(items, embedder, vector_store, bm25, on_progress, storage_dir=storage_dir, chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap, pdf_provider=config.pdf_provider)
            save_registry(config.data_dir, result["registry"])
            click.echo(format_index_stats(result["added"], output_json=json_out))

        save_metadata(config.data_dir, config.model_name, vector_store.count())
    except EmptyLibraryError as exc:
        click.echo(str(exc))
        ctx.exit(0)
    except ZotNotFoundError as exc:
        click.echo(f"Error: {exc}", err=True)
        ctx.exit(1)
    except ModelDownloadError as exc:
        click.echo(f"Error: {exc}", err=True)
        click.echo("Check your internet connection and try again.", err=True)
        ctx.exit(1)


@main.command()
@click.argument("query")
@click.option("--hybrid", is_flag=True, help="Use hybrid search (vector + BM25)")
@click.option("--bm25", "bm25_only", is_flag=True, help="Pure keyword search (no embedding model needed)")
@click.option("--limit", default=10, help="Number of results")
@click.option("--collection", default=None, help="Filter by Zotero collection name")
@click.option("--tag", "tags", multiple=True, help="Filter by tag (repeatable, OR logic)")
@click.pass_context
def search(ctx: click.Context, query: str, hybrid: bool, bm25_only: bool, limit: int, collection: str | None, tags: tuple[str, ...]) -> None:
    """Semantic search over indexed papers."""
    from rak.formatter import format_results

    config: RakConfig = ctx.obj["config"]
    json_out = ctx.obj["json"]

    try:
        results, _ = _run_search(config, query, limit, collection, tags, hybrid, bm25_only)

        output = format_results(results, output_json=json_out)
        if output.strip():
            click.echo(output)
        else:
            click.echo("No results found.")
    except (ModelDownloadError, DimensionMismatchError) as exc:
        click.echo(f"Error: {exc}", err=True)
        ctx.exit(1)


@main.command()
@click.argument("question")
@click.option("--context", "context_n", default=5, help="Number of documents to retrieve")
@click.option("--hybrid", is_flag=True, help="Use hybrid search (vector + BM25)")
@click.option("--bm25", "bm25_only", is_flag=True, help="Pure keyword search (no embedding model needed)")
@click.option("--collection", default=None, help="Filter by Zotero collection name")
@click.option("--tag", "tags", multiple=True, help="Filter by tag (repeatable, OR logic)")
@click.option("--llm-model", default=None, help="Override LLM model name")
@click.option("--llm-url", default=None, help="Override LLM server URL")
@click.pass_context
def ask(
    ctx: click.Context,
    question: str,
    context_n: int,
    hybrid: bool,
    bm25_only: bool,
    collection: str | None,
    tags: tuple[str, ...],
    llm_model: str | None,
    llm_url: str | None,
) -> None:
    """Ask a question and get an answer based on your papers."""
    from rak.formatter import format_ask_result
    from rak.llm import LLMClient, LLMConnectionError, LLMServerError

    config: RakConfig = ctx.obj["config"]
    json_out = ctx.obj["json"]

    try:
        results, _ = _run_search(config, question, context_n, collection, tags, hybrid, bm25_only)

        if not results:
            click.echo("No relevant papers found for your question.")
            return

        context = []
        for r in results:
            context.append({
                "key": r.doc_id,
                "title": r.title,
                "text": r.snippet,
                "score": r.score,
            })

        base_url = llm_url or config.llm_base_url
        model = llm_model or config.llm_model
        api_key = config.llm_api_key
        llm = LLMClient(base_url=base_url, model=model, api_key=api_key)

        if json_out:
            answer = llm.ask(question, context)
            click.echo(format_ask_result(answer, context, output_json=True))
        else:
            import sys
            for token in llm.ask_stream(question, context):
                sys.stdout.write(token)
                sys.stdout.flush()
            sys.stdout.write("\n\n")
            click.echo("Sources:")
            for i, s in enumerate(context, 1):
                click.echo(f"  {i}. {s['key']} - {s['title']} (score: {s['score']:.3f})")
    except (ModelDownloadError, DimensionMismatchError) as exc:
        click.echo(f"Error: {exc}", err=True)
        ctx.exit(1)
    except (LLMConnectionError, LLMServerError) as exc:
        click.echo(f"Error: {exc}", err=True)
        ctx.exit(1)


@main.command()
@click.argument("query")
@click.option("--format", "fmt", type=click.Choice(["csv", "bibtex"]), default="csv", help="Export format")
@click.option("--hybrid", is_flag=True, help="Use hybrid search (vector + BM25)")
@click.option("--bm25", "bm25_only", is_flag=True, help="Pure keyword search (no embedding model needed)")
@click.option("--limit", default=10, help="Number of results")
@click.option("--collection", default=None, help="Filter by Zotero collection name")
@click.option("--tag", "tags", multiple=True, help="Filter by tag (repeatable, OR logic)")
@click.option("--output", "output_file", default=None, type=click.Path(), help="Write to file instead of stdout")
@click.pass_context
def export(
    ctx: click.Context,
    query: str,
    fmt: str,
    hybrid: bool,
    bm25_only: bool,
    limit: int,
    collection: str | None,
    tags: tuple[str, ...],
    output_file: str | None,
) -> None:
    """Export search results as CSV or BibTeX."""
    from rak.export import to_bibtex, to_csv

    config: RakConfig = ctx.obj["config"]

    try:
        results, vector_store = _run_search(config, query, limit, collection, tags, hybrid, bm25_only)

        if not results:
            click.echo("No results found.")
            return

        # Batch fetch metadata instead of N+1 individual calls
        meta_map: dict[str, dict] = {}
        if vector_store is not None:
            all_ids = [r.doc_id for r in results]
            doc_data = vector_store.get(ids=all_ids, include=["metadatas"])
            for doc_id, meta in zip(doc_data["ids"], doc_data["metadatas"]):
                meta_map[doc_id] = meta

        export_rows = []
        for r in results:
            meta = meta_map.get(r.doc_id, {})
            export_rows.append({
                "key": r.doc_id,
                "title": r.title,
                "score": r.score,
                "source": r.source,
                "date": meta.get("date", ""),
                "authors": meta.get("authors", ""),
                "item_type": meta.get("item_type", ""),
            })

        if fmt == "bibtex":
            output = to_bibtex(export_rows)
        else:
            output = to_csv(export_rows)

        if output_file:
            from pathlib import Path
            Path(output_file).write_text(output)
            click.echo(f"Exported {len(export_rows)} results to {output_file}")
        else:
            click.echo(output)
    except (ModelDownloadError, DimensionMismatchError) as exc:
        click.echo(f"Error: {exc}", err=True)
        ctx.exit(1)


@main.command()
@click.argument("shell", type=click.Choice(["bash", "zsh", "fish"]), required=False)
def completion(shell: str | None) -> None:
    """Generate shell completion script.

    Run: eval "$(rak completion bash)" or add to your shell profile.
    """
    import os
    import subprocess as _sp

    if shell is None:
        parent = os.environ.get("SHELL", "")
        if "zsh" in parent:
            shell = "zsh"
        elif "fish" in parent:
            shell = "fish"
        else:
            shell = "bash"

    env = {**os.environ, "_RAK_COMPLETE": f"{shell}_source"}
    result = _sp.run(["rak"], capture_output=True, text=True, env=env)
    click.echo(result.stdout)


@main.command()
@click.option("--context", "context_n", default=5, help="Number of documents to retrieve")
@click.option("--hybrid", is_flag=True, help="Use hybrid search (vector + BM25)")
@click.option("--bm25", "bm25_only", is_flag=True, help="Pure keyword search (no embedding model needed)")
@click.option("--collection", default=None, help="Filter by Zotero collection name")
@click.option("--tag", "tags", multiple=True, help="Filter by tag (repeatable, OR logic)")
@click.option("--llm-model", default=None, help="Override LLM model name")
@click.option("--llm-url", default=None, help="Override LLM server URL")
@click.pass_context
def chat(
    ctx: click.Context,
    context_n: int,
    hybrid: bool,
    bm25_only: bool,
    collection: str | None,
    tags: tuple[str, ...],
    llm_model: str | None,
    llm_url: str | None,
) -> None:
    """Interactive multi-turn Q&A over your papers."""
    import sys
    from contextlib import ExitStack
    from rak.bm25 import BM25Index
    from rak.chat import ChatSession
    from rak.llm import LLMClient, LLMConnectionError, LLMServerError
    from rak.searcher import Searcher

    config: RakConfig = ctx.obj["config"]

    try:
        stack = ExitStack()
        bm25 = stack.enter_context(BM25Index(config.fts_db_path))
        if bm25_only:
            searcher = Searcher(None, None, bm25)
        else:
            from rak.store import VectorStore
            embedder = _create_embedder(config)
            vector_store = VectorStore(config.chroma_dir, embedder.dimension)
            searcher = Searcher(embedder, vector_store, bm25)

        with stack:
            base_url = llm_url or config.llm_base_url
            model = llm_model or config.llm_model
            api_key = config.llm_api_key
            llm = LLMClient(base_url=base_url, model=model, api_key=api_key)

            tag_list = list(tags) if tags else None
            session = ChatSession(
                searcher=searcher, llm=llm, limit=context_n,
                collection=collection, tags=tag_list, hybrid=hybrid,
                bm25_only=bm25_only,
            )

            click.echo("Enter a search query to find papers (or /quit to exit):")
            try:
                query = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                click.echo()
                return
            if not query or query == "/quit":
                return

            session.search(query)
            if not session.context:
                click.echo("No papers found. Try a different query.")
                return

            click.echo(f"\nFound {len(session.context)} papers:")
            for i, doc in enumerate(session.context, 1):
                click.echo(f"  {i}. {doc['key']} - {doc['title']} (score: {doc['score']:.3f})")
            from rak.chat import HELP_TEXT
            click.echo(f"\nChat started. Type /help for commands.\n")

            while True:
                try:
                    user_input = input("> ").strip()
                except (EOFError, KeyboardInterrupt):
                    click.echo()
                    break

                if not user_input:
                    continue
                if user_input == "/quit":
                    break
                if user_input == "/help":
                    click.echo(HELP_TEXT)
                    continue
                if user_input == "/tokens":
                    click.echo(f"Estimated tokens: ~{session.token_count:,} | Turns: {session.turn_count}")
                    continue
                if user_input == "/context":
                    for i, doc in enumerate(session.context, 1):
                        click.echo(f"  {i}. {doc['key']} - {doc['title']} (score: {doc['score']:.3f})")
                    continue
                if user_input.startswith("/search "):
                    new_query = user_input[8:].strip()
                    if new_query:
                        session.search(new_query)
                        click.echo(f"\nFound {len(session.context)} papers:")
                        for i, doc in enumerate(session.context, 1):
                            click.echo(f"  {i}. {doc['key']} - {doc['title']} (score: {doc['score']:.3f})")
                        click.echo()
                    continue

                for token in session.ask(user_input):
                    sys.stdout.write(token)
                    sys.stdout.flush()
                sys.stdout.write("\n\n")

    except (ModelDownloadError, DimensionMismatchError) as exc:
        click.echo(f"Error: {exc}", err=True)
        ctx.exit(1)
    except (LLMConnectionError, LLMServerError) as exc:
        click.echo(f"Error: {exc}", err=True)
        ctx.exit(1)
