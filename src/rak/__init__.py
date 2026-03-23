from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("zotero-rag-cli")
except PackageNotFoundError:
    __version__ = "0.7.1"  # fallback for editable installs without metadata
