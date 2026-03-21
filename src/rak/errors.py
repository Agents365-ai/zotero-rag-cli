"""Custom exception hierarchy for rak."""
from __future__ import annotations


class RakError(Exception):
    """Base exception for rak."""


class ZotNotFoundError(RakError):
    """Raised when zot command is not found on PATH."""

    def __init__(self, command: str = "zot") -> None:
        super().__init__(f"'{command}' not found on PATH. Install: pip install zotero-cli-cc")


class EmptyLibraryError(RakError):
    """Raised when Zotero library contains no items."""

    def __init__(self) -> None:
        super().__init__("No items found in Zotero library.")


class ModelDownloadError(RakError):
    """Raised when model download or loading fails."""

    def __init__(self, model_name: str, reason: str = "") -> None:
        msg = f"Failed to load model '{model_name}'"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)
