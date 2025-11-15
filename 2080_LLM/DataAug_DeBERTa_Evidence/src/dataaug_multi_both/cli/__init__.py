# CLI package exposing the consolidated entry point for the app
from ..console import main  # re-export for poetry console script

__all__ = ["main"]
