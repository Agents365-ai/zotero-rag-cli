from __future__ import annotations

import click

from rak import __version__
from rak.config import RakConfig
from rak.errors import EmptyLibraryError, ModelDownloadError, ZotNotFoundError


@click.group()
@click.version_option(version=__version__, prog_name="rak")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
@click.option("--model", default=None, help="Embedding model name")
@click.pass_context
def main(ctx: click.Context, output_json: bool, model: str | None) -> None:
    """rak — Semantic search over your Zotero library."""
    ctx.ensure_object(dict)
    ctx.obj["json"] = output_json
    config = RakConfig()
    if model:
        config.model_name = model
    ctx.obj["config"] = config


@main.command()
@click.option("--limit", default=5000, help="Max items to index from zot")
@click.option("--full", is_flag=True, help="Force full rebuild (ignore existing index)")
@click.pass_context
def index(ctx: click.Context, limit: int, full: bool) -> None:
    """Index Zotero library for semantic search."""
    from rak.bm25 import BM25Index
    from rak.embedder import Embedder
    from rak.formatter import format_incremental_stats, format_index_stats
    from rak.indexer import build_document_text, fetch_zot_items, index_items
    from rak.metadata import save_metadata
    from rak.registry import compute_hash, load_registry, save_registry
    from rak.store import VectorStore

    config: RakConfig = ctx.obj["config"]
    json_out = ctx.obj["json"]
    config.data_dir.mkdir(parents=True, exist_ok=True)

    try:
        click.echo("Loading embedding model...")
        embedder = Embedder(config.model_name)

        click.echo("Fetching items from zot...")
        items = fetch_zot_items(config.zot_command, limit=limit)
        click.echo(f"Found {len(items)} items.")

        vector_store = VectorStore(config.chroma_dir, embedder.dimension)
        bm25 = BM25Index(config.fts_db_path)

        storage_dir = config.zotero_storage_dir
        if storage_dir:
            click.echo(f"PDF extraction enabled: {storage_dir}")
        else:
            click.echo("PDF extraction: Zotero storage not found, indexing metadata only.")

        def on_progress(current: int, total: int) -> None:
            click.echo(f"  Indexed {current}/{total}...")

        registry = None if full else load_registry(config.data_dir)
        if registry is not None and not registry:
            registry = None

        if registry is None:
            if full:
                vector_store.clear()
                bm25.clear()
            count = index_items(items, embedder, vector_store, bm25, on_progress, storage_dir=storage_dir)
            new_registry = {}
            for item in items:
                key = item.get("key", "")
                if not key:
                    continue
                pdf_text = ""
                if storage_dir:
                    from rak.pdf import find_pdf, extract_pdf_text
                    pdf_path = find_pdf(storage_dir, key)
                    if pdf_path:
                        pdf_text = extract_pdf_text(pdf_path)
                text = build_document_text(item, pdf_text=pdf_text)
                if text.strip():
                    new_registry[key] = compute_hash(text)
            save_registry(config.data_dir, new_registry)
            bm25.close()
            click.echo(format_index_stats(count, output_json=json_out))
        else:
            result = index_items(items, embedder, vector_store, bm25, on_progress, registry=registry, storage_dir=storage_dir)
            save_registry(config.data_dir, result["registry"])
            bm25.close()
            click.echo(format_incremental_stats(result, output_json=json_out))

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
            click.echo(f"Unknown config key: {key}", err=True)
            click.echo(f"Valid keys: {', '.join(sorted(CONFIGURABLE_KEYS))}", err=True)
            ctx.exit(1)
            return
        save_config(config.data_dir, key, value)
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
        click.echo(f"data_dir = {config.data_dir}")
        click.echo(f"zotero_storage_dir = {config.zotero_storage_dir}")


@main.command()
@click.argument("query")
@click.option("--hybrid", is_flag=True, help="Use hybrid search (vector + BM25)")
@click.option("--limit", default=10, help="Number of results")
@click.option("--collection", default=None, help="Filter by Zotero collection name")
@click.option("--tag", "tags", multiple=True, help="Filter by tag (repeatable, OR logic)")
@click.pass_context
def search(ctx: click.Context, query: str, hybrid: bool, limit: int, collection: str | None, tags: tuple[str, ...]) -> None:
    """Semantic search over indexed papers."""
    from rak.bm25 import BM25Index
    from rak.embedder import Embedder
    from rak.formatter import format_results
    from rak.searcher import Searcher
    from rak.store import VectorStore

    config: RakConfig = ctx.obj["config"]
    json_out = ctx.obj["json"]

    try:
        embedder = Embedder(config.model_name)
        vector_store = VectorStore(config.chroma_dir, embedder.dimension)
        bm25 = BM25Index(config.fts_db_path)
        searcher = Searcher(embedder, vector_store, bm25)

        tag_list = list(tags) if tags else None
        if hybrid:
            results = searcher.hybrid_search(query, limit=limit, collection=collection, tags=tag_list)
        else:
            results = searcher.vector_search(query, limit=limit, collection=collection, tags=tag_list)

        bm25.close()
        output = format_results(results, output_json=json_out)
        if output.strip():
            click.echo(output)
        else:
            click.echo("No results found.")
    except ModelDownloadError as exc:
        click.echo(f"Error: {exc}", err=True)
        click.echo("Check your internet connection and try again.", err=True)
        ctx.exit(1)


@main.command()
@click.argument("question")
@click.option("--context", "context_n", default=5, help="Number of documents to retrieve")
@click.option("--hybrid", is_flag=True, help="Use hybrid search (vector + BM25)")
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
    collection: str | None,
    tags: tuple[str, ...],
    llm_model: str | None,
    llm_url: str | None,
) -> None:
    """Ask a question and get an answer based on your papers."""
    from rak.bm25 import BM25Index
    from rak.embedder import Embedder
    from rak.formatter import format_ask_result
    from rak.llm import LLMClient, LLMConnectionError
    from rak.searcher import Searcher
    from rak.store import VectorStore

    config: RakConfig = ctx.obj["config"]
    json_out = ctx.obj["json"]

    try:
        embedder = Embedder(config.model_name)
        vector_store = VectorStore(config.chroma_dir, embedder.dimension)
        bm25 = BM25Index(config.fts_db_path)
        searcher = Searcher(embedder, vector_store, bm25)

        tag_list = list(tags) if tags else None
        if hybrid:
            results = searcher.hybrid_search(question, limit=context_n, collection=collection, tags=tag_list)
        else:
            results = searcher.vector_search(question, limit=context_n, collection=collection, tags=tag_list)

        if not results:
            click.echo("No relevant papers found for your question.")
            return

        context = []
        for r in results:
            doc_data = vector_store._collection.get(ids=[r.doc_id], include=["documents"])
            doc_text = doc_data["documents"][0] if doc_data["documents"] else ""
            context.append({
                "key": r.doc_id,
                "title": r.title,
                "text": doc_text,
                "score": r.score,
            })

        bm25.close()

        base_url = llm_url or config.llm_base_url
        model = llm_model or config.llm_model
        llm = LLMClient(base_url=base_url, model=model)
        answer = llm.ask(question, context)

        click.echo(format_ask_result(answer, context, output_json=json_out))
    except ModelDownloadError as exc:
        click.echo(f"Error: {exc}", err=True)
        ctx.exit(1)
    except LLMConnectionError as exc:
        click.echo(f"Error: {exc}", err=True)
        ctx.exit(1)
