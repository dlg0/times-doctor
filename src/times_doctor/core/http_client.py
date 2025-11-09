"""HTTP client utilities with timeout and retry logic."""

import time
from typing import Any

from .exceptions import LlmError


def get_http_client():
    """Get configured httpx client with timeouts and retries.

    Returns:
        httpx.Client configured with appropriate timeouts
    """
    try:
        import httpx
    except ImportError:
        raise LlmError("httpx not installed")

    return httpx.Client(
        timeout=httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0),
        follow_redirects=True,
    )


def make_request_with_retry(
    method: str,
    url: str,
    headers: dict[str, str] | None = None,
    json: dict[str, Any] | None = None,
    max_retries: int = 3,
    backoff_factor: float = 2.0,
) -> Any:
    """Make HTTP request with exponential backoff retry.

    Args:
        method: HTTP method (GET, POST, etc.)
        url: Request URL
        headers: Optional request headers
        json: Optional JSON payload
        max_retries: Maximum number of retry attempts
        backoff_factor: Exponential backoff multiplier

    Returns:
        httpx.Response object

    Raises:
        LlmError: If request fails after all retries
    """
    try:
        import random

        import httpx
    except ImportError:
        raise LlmError("httpx not installed")

    last_exception: Exception | None = None

    for attempt in range(max_retries):
        try:
            client = get_http_client()
            response = client.request(method, url, headers=headers, json=json)

            # Handle rate limiting
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After", "60")
                try:
                    wait_time_int = int(retry_after)
                except ValueError:
                    wait_time_int = 60

                if attempt < max_retries - 1:
                    # Add jitter to avoid thundering herd
                    jitter = random.uniform(0, min(wait_time_int * 0.1, 5))
                    time.sleep(wait_time_int + jitter)
                    continue
                else:
                    raise LlmError(f"Rate limited after {max_retries} retries")

            # Handle server errors with retry
            if 500 <= response.status_code < 600:
                if attempt < max_retries - 1:
                    wait_time = float(backoff_factor**attempt)
                    # Add jitter (±25% of wait time)
                    jitter = random.uniform(-wait_time * 0.25, wait_time * 0.25)
                    time.sleep(max(0.1, wait_time + jitter))
                    continue
                else:
                    raise LlmError(
                        f"Server error {response.status_code} after {max_retries} retries"
                    )

            # Handle client errors (no retry)
            if 400 <= response.status_code < 500 and response.status_code != 429:
                raise LlmError(f"Client error {response.status_code}: {response.text[:200]}")

            # Success
            response.raise_for_status()
            return response

        except httpx.TimeoutException as e:
            last_exception = e
            if attempt < max_retries - 1:
                wait_time = float(backoff_factor**attempt)
                # Add jitter (±25% of wait time)
                jitter = random.uniform(-wait_time * 0.25, wait_time * 0.25)
                time.sleep(max(0.1, wait_time + jitter))
                continue
            else:
                raise LlmError(f"Request timeout after {max_retries} retries") from e

        except httpx.NetworkError as e:
            last_exception = e
            if attempt < max_retries - 1:
                wait_time = float(backoff_factor**attempt)
                # Add jitter (±25% of wait time)
                jitter = random.uniform(-wait_time * 0.25, wait_time * 0.25)
                time.sleep(max(0.1, wait_time + jitter))
                continue
            else:
                raise LlmError(f"Network error after {max_retries} retries") from e

    # Should not reach here, but just in case
    raise LlmError(f"Request failed after {max_retries} retries") from last_exception
