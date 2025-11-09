"""LLM response caching to reduce API costs and improve performance."""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


def compute_cache_key(prompt: str, model: str, **kwargs: Any) -> str:
    """Compute stable cache key from prompt and parameters.

    Args:
        prompt: Full prompt text
        model: Model name (e.g., 'gpt-5-nano', 'claude-3-5-haiku-20241022')
        **kwargs: Additional parameters (temperature, reasoning_effort, etc.)

    Returns:
        SHA256 hash as hex string
    """
    # Sort kwargs for stable hashing
    params = {
        "model": model,
        "prompt": prompt,
        **{k: v for k, v in sorted(kwargs.items()) if v is not None},
    }

    # Create stable JSON representation
    stable_json = json.dumps(params, sort_keys=True, ensure_ascii=True)

    # Hash to create cache key
    return hashlib.sha256(stable_json.encode("utf-8")).hexdigest()


def get_cache_path(cache_key: str, cache_dir: Path) -> Path:
    """Get cache file path for a given cache key.

    Args:
        cache_key: Cache key hash
        cache_dir: Cache directory path

    Returns:
        Path to cache file
    """
    return cache_dir / f"cache_{cache_key}.json"


def read_cache(
    prompt: str, model: str, cache_dir: Path, **kwargs: Any
) -> Optional[tuple[str, dict[str, Any]]]:
    """Read cached LLM response if available.

    Args:
        prompt: Full prompt text
        model: Model name
        cache_dir: Cache directory path
        **kwargs: Additional parameters (temperature, reasoning_effort, etc.)

    Returns:
        Tuple of (response_text, metadata) if cached, None otherwise
    """
    cache_key = compute_cache_key(prompt, model, **kwargs)
    cache_file = get_cache_path(cache_key, cache_dir)

    if not cache_file.exists():
        return None

    try:
        with open(cache_file, encoding="utf-8") as f:
            cached_data = json.load(f)

        # Return cached response and metadata
        return cached_data.get("response", ""), cached_data.get("metadata", {})
    except Exception:
        # If cache read fails, treat as cache miss
        return None


def write_cache(
    prompt: str, model: str, response: str, metadata: dict[str, Any], cache_dir: Path, **kwargs: Any
) -> None:
    """Write LLM response to cache.

    Args:
        prompt: Full prompt text
        model: Model name
        response: LLM response text
        metadata: Response metadata (tokens, cost, etc.)
        cache_dir: Cache directory path
        **kwargs: Additional parameters (temperature, reasoning_effort, etc.)
    """
    cache_key = compute_cache_key(prompt, model, **kwargs)
    cache_file = get_cache_path(cache_key, cache_dir)

    # Ensure cache directory exists
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Prepare cache data
    cache_data = {
        "cache_key": cache_key,
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "params": {k: v for k, v in kwargs.items() if v is not None},
        "response": response,
        "metadata": metadata,
        "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16],
        "prompt_length": len(prompt),
    }

    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        # Don't fail the main operation if caching fails
        import logging

        logging.debug(f"Failed to write LLM cache: {e}")


def clear_cache(cache_dir: Path) -> int:
    """Clear all cached LLM responses.

    Args:
        cache_dir: Cache directory path

    Returns:
        Number of cache files deleted
    """
    if not cache_dir.exists():
        return 0

    count = 0
    cache_files = list(cache_dir.glob("cache_*.json"))
    errors = []
    for cache_file in cache_files:
        try:
            cache_file.unlink()
            count += 1
        except Exception as e:
            errors.append((cache_file, e))

    return count
