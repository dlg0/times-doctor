"""Centralized logging and console output for times-doctor."""
from rich.console import Console
from rich.spinner import Spinner
from rich.live import Live
from contextlib import contextmanager
from typing import Optional
import sys

_console: Optional[Console] = None
_no_color: bool = False


def init_console(no_color: bool = False):
    """Initialize the global console with color settings."""
    global _console, _no_color
    _no_color = no_color
    _console = Console(
        force_terminal=not no_color,
        no_color=no_color,
        highlight=not no_color
    )


def get_console() -> Console:
    """Get the global console instance."""
    if _console is None:
        init_console()
    return _console


def info(message: str, **kwargs):
    """Print an info message."""
    get_console().print(message, **kwargs)


def success(message: str, **kwargs):
    """Print a success message in green."""
    get_console().print(f"[green]{message}[/green]", **kwargs)


def warning(message: str, **kwargs):
    """Print a warning message in yellow."""
    get_console().print(f"[yellow]{message}[/yellow]", **kwargs)


def error(message: str, **kwargs):
    """Print an error message in red."""
    get_console().print(f"[red]{message}[/red]", **kwargs)


def dim(message: str, **kwargs):
    """Print a dim/debug message."""
    get_console().print(f"[dim]{message}[/dim]", **kwargs)


@contextmanager
def spinner(message: str):
    """Show a spinner while executing a block of code.
    
    Example:
        with spinner("Downloading files..."):
            download_files()
    """
    if _no_color:
        # Without color, just print the message
        info(message)
        yield
    else:
        with get_console().status(message, spinner="dots") as status:
            yield status
