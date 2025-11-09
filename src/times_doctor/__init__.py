try:
    from importlib.metadata import version

    __version__ = version("times-doctor")
except Exception:
    __version__ = "unknown"

__all__ = ["__version__"]
