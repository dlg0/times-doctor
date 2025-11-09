"""Safe I/O utilities for cross-platform file operations."""

from pathlib import Path
from typing import Union


def read_text(path: Union[str, Path], encoding: str = "utf-8", errors: str = "replace") -> str:
    """Safely read text file with proper encoding handling.

    Args:
        path: File path to read
        encoding: Text encoding (default: utf-8)
        errors: Error handling strategy (default: replace)

    Returns:
        File content as string
    """
    path = Path(path)
    return path.read_text(encoding=encoding, errors=errors)


def write_text(
    path: Union[str, Path], content: str, encoding: str = "utf-8", errors: str = "replace"
) -> None:
    """Safely write text file with proper encoding handling.

    Args:
        path: File path to write
        content: Text content to write
        encoding: Text encoding (default: utf-8)
        errors: Error handling strategy (default: replace)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding=encoding, errors=errors)


def normalize_path(path: Union[str, Path]) -> Path:
    """Normalize path for current OS.

    Args:
        path: Path to normalize

    Returns:
        Normalized Path object
    """
    return Path(path).resolve()


def quote_path(path: Union[str, Path]) -> str:
    """Quote a path for use in command line arguments.

    Args:
        path: Path to quote

    Returns:
        Quoted path string safe for subprocess
    """
    path_str = str(Path(path))

    if " " in path_str or "(" in path_str or ")" in path_str:
        return f'"{path_str}"'
    return path_str
