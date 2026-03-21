from __future__ import annotations

import click

from rak import __version__


@click.group()
@click.version_option(version=__version__, prog_name="rak")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
@click.pass_context
def main(ctx: click.Context, output_json: bool) -> None:
    """rak — Semantic search over your Zotero library."""
    ctx.ensure_object(dict)
    ctx.obj["json"] = output_json


@main.command()
def index() -> None:
    """Index Zotero library for semantic search."""
    click.echo("Indexing not yet implemented.")


@main.command()
@click.argument("query")
@click.option("--hybrid", is_flag=True, help="Use hybrid search (vector + BM25)")
@click.option("--limit", default=10, help="Number of results")
@click.pass_context
def search(ctx: click.Context, query: str, hybrid: bool, limit: int) -> None:
    """Semantic search over indexed papers."""
    click.echo("Search not yet implemented.")
