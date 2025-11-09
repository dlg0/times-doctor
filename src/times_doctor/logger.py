"""Centralized logging and console output for times-doctor."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from rich.console import Console
from rich.status import Status

_console: Console | None = None
_no_color: bool = False


def init_console(no_color: bool = False) -> None:
    """Initialize the global console with color settings."""
    global _console, _no_color
    _no_color = no_color
    _console = Console(force_terminal=not no_color, no_color=no_color, highlight=not no_color)


def get_console() -> Console:
    """Get the global console instance."""
    if _console is None:
        init_console()
    assert _console is not None  # noqa: S101
    return _console


def info(message: str, **kwargs: Any) -> None:
    """Print an info message."""
    get_console().print(message, **kwargs)


def success(message: str, **kwargs: Any) -> None:
    """Print a success message in green."""
    get_console().print(f"[green]{message}[/green]", **kwargs)


def warning(message: str, **kwargs: Any) -> None:
    """Print a warning message in yellow."""
    get_console().print(f"[yellow]{message}[/yellow]", **kwargs)


def error(message: str, **kwargs: Any) -> None:
    """Print an error message in red."""
    get_console().print(f"[red]{message}[/red]", **kwargs)


def dim(message: str, **kwargs: Any) -> None:
    """Print a dim/debug message."""
    get_console().print(f"[dim]{message}[/dim]", **kwargs)


@contextmanager
def spinner(message: str) -> Iterator[Status | None]:
    """Show a spinner while executing a block of code.

    Example:
        with spinner("Downloading files..."):
            download_files()
    """
    if _no_color:
        # Without color, just print the message
        info(message)
        yield None
    else:
        with get_console().status(message, spinner="dots") as status:
            yield status
