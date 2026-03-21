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
@click.pass_context
def index(ctx: click.Context, limit: int) -> None:
    """Index Zotero library for semantic search."""
    from rak.bm25 import BM25Index
    from rak.embedder import Embedder
    from rak.formatter import format_index_stats
    from rak.indexer import fetch_zot_items, index_items
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

        def on_progress(current: int, total: int) -> None:
            click.echo(f"  Indexed {current}/{total}...")

        count = index_items(items, embedder, vector_store, bm25, on_progress)
        bm25.close()
        click.echo(format_index_stats(count, output_json=json_out))
        from rak.metadata import save_metadata
        save_metadata(config.data_dir, config.model_name, count)
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
@click.option("--limit", default=10, help="Number of results")
@click.pass_context
def search(ctx: click.Context, query: str, hybrid: bool, limit: int) -> None:
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

        if hybrid:
            results = searcher.hybrid_search(query, limit=limit)
        else:
            results = searcher.vector_search(query, limit=limit)

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
