import json
import subprocess
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from . import redactor
from .config import get_config


def which(cmd: str) -> str | None:
    from shutil import which as _w

    return _w(cmd)


def log_llm_call(
    call_type: str, prompt: str, response: str, metadata: dict, log_dir: Path | None = None
) -> None:
    """Log LLM call details to _llm_calls folder for debugging.

    Args:
        call_type: Type of call (e.g., 'extraction_qa_check', 'review', 'diagnose')
        prompt: The prompt sent to the LLM
        response: The response from the LLM (or error message)
        metadata: Dict with model, provider, tokens, cost, error info, etc.
        log_dir: Directory to save logs (defaults to cwd/_llm_calls)
    """
    if log_dir is None:
        log_dir = Path.cwd() / "_llm_calls"

    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # milliseconds
    log_file = log_dir / f"{timestamp}_{call_type}.json"

    # Calculate context window stats
    model = metadata.get("model", "")
    input_tokens = metadata.get("input_tokens", 0)
    output_tokens = metadata.get("output_tokens", 0)
    cost = metadata.get("cost_usd", 0)

    # Model context window sizes
    context_windows = {
        "gpt-5-nano": 400000,
        "gpt-5-mini": 400000,
        "gpt-5": 400000,
        "gpt-5-pro": 400000,
        "claude-3-5-sonnet-20241022": 200000,
        "claude-3-5-haiku-20241022": 200000,
        "claude-3-opus-20240229": 200000,
        "claude-3-sonnet-20240229": 200000,
        "claude-3-haiku-20240307": 200000,
    }

    window_size = context_windows.get(model, 128000)  # default to 128k
    window_usage_pct = (input_tokens / window_size * 100) if window_size > 0 else 0

    log_data = {
        "timestamp": datetime.now().isoformat(),
        "call_type": call_type,
        "metadata": metadata,
        "prompt": redactor.redact_api_key(prompt),
        "response": redactor.redact_api_key(response),
        "prompt_length": len(prompt),
        "response_length": len(response),
        "tokens": {
            "input": input_tokens,
            "output": output_tokens,
            "total": input_tokens + output_tokens,
            "window_size": window_size,
            "window_usage_pct": round(window_usage_pct, 1),
        },
        "cost_usd": round(cost, 6),
    }

    log_data = redactor.redact_dict(log_data)

    try:
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        # Don't fail the main operation if logging fails
        print(f"[dim]Warning: Failed to log LLM call: {e}[/dim]")


def load_env():
    try:
        from dotenv import load_dotenv

        env_path = Path.cwd() / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            return True
    except ImportError:
        pass
    return False


def check_api_keys() -> dict:
    config = get_config()
    return {
        "openai": bool(config.openai_api_key),
        "anthropic": bool(config.anthropic_api_key),
        "amp": bool(config.amp_api_key or which(config.amp_cli)),
    }


def list_openai_models() -> list[str]:
    """List available OpenAI models for chat completions."""
    config = get_config()
    if not config.openai_api_key:
        return []

    # GPT-5 reasoning models with effort levels
    reasoning_models = ["gpt-5", "gpt-5-pro", "gpt-5-mini", "gpt-5-nano"]
    effort_levels = ["minimal", "low", "medium", "high"]

    models = []
    # Add reasoning models with effort combinations
    for base in reasoning_models:
        for effort in effort_levels:
            models.append(f"{base} (reasoning/{effort})")

    # Add non-reasoning chat variants
    models.extend(
        [
            "gpt-5-chat (non-reasoning)",
            "gpt-5-mini-chat (non-reasoning)",
            "gpt-5-nano-chat (non-reasoning)",
        ]
    )

    return models


def list_anthropic_models() -> list[str]:
    """List available Anthropic models."""
    return [
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
    ]


class LLMResult:
    def __init__(
        self,
        text: str,
        provider: str,
        used: bool,
        model: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
    ):
        self.text = text
        self.provider = provider
        self.used = used
        self.model = model
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cost_usd = cost_usd


def _call_cli(cli: str, prompt: str) -> str:
    """Call external CLI tool with prompt input.

    Args:
        cli: Path to CLI executable
        prompt: Text prompt to send via stdin

    Returns:
        CLI output text, or empty string on error
    """
    try:
        p = subprocess.run(
            [cli],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,  # Don't raise on non-zero exit, just return empty
        )
        if p.returncode != 0:
            # Log stderr if available for debugging
            if p.stderr:
                import logging

                logging.debug(f"CLI {cli} failed with code {p.returncode}: {p.stderr}")
            return ""
        return p.stdout.strip()
    except subprocess.TimeoutExpired:
        import logging

        logging.warning(f"CLI {cli} timed out after 120s")
        return ""
    except FileNotFoundError:
        import logging

        logging.warning(f"CLI executable not found: {cli}")
        return ""
    except Exception as e:
        import logging

        logging.debug(f"Unexpected error calling CLI {cli}: {e}")
        return ""


def _call_openai_responses_api(
    prompt: str,
    model: str = "gpt-5-nano",
    reasoning_effort: str = "medium",
    stream_callback: Callable[[str], None] | None = None,
    log_dir: Path | None = None,
    use_cache: bool = True,
    instructions: str | None = None,
    text_format: type[Any] | None = None,
) -> tuple[str, dict[str, Any]] | tuple[Any, dict[str, Any]]:
    """Call OpenAI GPT-5 Responses API with optional streaming support.

    Args:
        prompt: The user input/data to send to the API (goes to 'input' field)
        model: Model name to use
        reasoning_effort: Reasoning effort level
        stream_callback: Optional callback for streaming responses
        log_dir: Directory for logging
        use_cache: Whether to use cached responses (default: True)
        instructions: System-level instructions (goes to 'instructions' field)
        text_format: Optional Pydantic model for structured output

    Returns:
        Tuple of (response_text, metadata) or (parsed_model, metadata) if text_format provided
    """
    start_time = time.time()
    config = get_config()
    key = config.openai_api_key
    if not key:
        return "", {}

    # Check cache first (skip if streaming or cache disabled)
    if use_cache and not stream_callback:
        from .llm_cache import read_cache

        cache_dir = (log_dir or Path.cwd() / "_llm_calls") / "cache"
        cached = read_cache(
            prompt=prompt,
            model=model,
            cache_dir=cache_dir,
            reasoning_effort=reasoning_effort,
        )
        if cached:
            text, meta = cached
            # Add cache hit indicator to metadata
            meta["cached"] = True
            meta["cache_hit"] = True
            print(
                f"[dim]LLM: {model} (cached) | {meta.get('input_tokens', 0):,}→{meta.get('output_tokens', 0):,} tok | $0.0000 (cache hit)[/dim]"
            )
            return text, meta

    # Use OpenAI SDK for structured output with Pydantic models
    # Only supports gpt-5 models
    if text_format:
        try:
            from openai import OpenAI as OpenAIClient
        except ImportError as e:
            raise RuntimeError(f"OpenAI SDK not available for structured output: {e}")

        # Enforce gpt-5 only for structured output
        if not model.startswith("gpt-5"):
            raise ValueError(
                f"Structured output (text_format) only supports gpt-5 models, got: {model}"
            )

        client = OpenAIClient(api_key=key)

        # Build input as message array (official SDK format)
        input_messages = []
        if instructions:
            input_messages.append({"role": "system", "content": instructions})
        input_messages.append({"role": "user", "content": prompt})

        # Build kwargs for SDK call
        kwargs = {
            "model": model,
            "input": input_messages,
            "text_format": text_format,
        }

        # Add reasoning parameter (gpt-5 only, no "summary" key)
        if reasoning_effort:
            kwargs["reasoning"] = {"effort": reasoning_effort}

        # Call SDK and handle errors strictly (no fallback)
        try:
            if stream_callback:
                # Streaming with structured output using SDK event API
                with client.responses.stream(**kwargs) as stream:
                    # Use SDK's event handler for text deltas
                    for event in stream:
                        if hasattr(event, "type") and event.type == "response.output_text.delta":
                            if hasattr(event, "delta"):
                                stream_callback(event.delta)

                    # Get final response
                    response = stream.get_final_response()
            else:
                # Non-streaming
                response = client.responses.parse(**kwargs)

            # Extract parsed output
            parsed_output = response.output_parsed

            # Validate parse succeeded
            if not isinstance(parsed_output, text_format):
                raise ValueError(
                    f"Structured parse failed: expected {text_format.__name__}, "
                    f"got {type(parsed_output).__name__}"
                )

        except Exception as e:
            # Raise exception instead of falling back
            raise RuntimeError(
                f"OpenAI structured output parse failed for model {model}: {e}"
            ) from e

        # Extract usage and cost info
        usage = response.usage
        input_tokens = usage.input_tokens if usage else 0
        output_tokens = usage.output_tokens if usage else 0

        # Calculate cost
        cost_per_1k_input = {
            "gpt-5-nano": 0.0001,
            "gpt-5-mini": 0.0005,
            "gpt-5-pro": 0.01,
            "gpt-5": 0.005,
        }
        cost_per_1k_output = {
            "gpt-5-nano": 0.0004,
            "gpt-5-mini": 0.0015,
            "gpt-5-pro": 0.03,
            "gpt-5": 0.015,
        }

        model_key = model
        if model_key not in cost_per_1k_input:
            for key_prefix in cost_per_1k_input:
                if model.startswith(key_prefix):
                    model_key = key_prefix
                    break

        input_cost = input_tokens / 1000 * cost_per_1k_input.get(model_key, 0.0001)
        output_cost = output_tokens / 1000 * cost_per_1k_output.get(model_key, 0.0004)

        metadata = {
            "model": model,
            "provider": "openai",
            "reasoning_effort": reasoning_effort,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cost_usd": input_cost + output_cost,
            "duration_seconds": round(time.time() - start_time, 2),
        }

        # Print concise log
        window_pct = input_tokens / 400000 * 100
        effort_str = f" ({reasoning_effort})" if reasoning_effort else ""
        print(
            f"[dim]LLM: {model}{effort_str} | {input_tokens:,}→{output_tokens:,} tok ({window_pct:.1f}% window) | {metadata['duration_seconds']:.1f}s | ${metadata['cost_usd']:.4f}[/dim]"
        )

        return parsed_output, metadata

    # Fall through to httpx-based implementation for non-structured or streaming
    try:
        import json as json_module

        import httpx
    except Exception:
        return "", {}

    url = "https://api.openai.com/v1/responses"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    # Build payload with proper field usage
    payload: dict[str, Any] = {
        "model": model,
        "input": prompt,  # User input as string (not message array!)
        "reasoning": {"effort": reasoning_effort, "summary": "auto"},
        "store": True,
        "stream": bool(stream_callback),
    }

    # Add instructions if provided (system-level guidance)
    if instructions:
        payload["instructions"] = instructions

    # Default text output (structured output handled by SDK above)
    payload["text"] = {"format": {"type": "text"}, "verbosity": "medium"}

    try:
        # Longer timeout for reasoning models
        timeout_seconds = 300 if "gpt-5" in model or "pro" in model else 120

        # Log raw request
        if log_dir is None:
            log_dir = Path.cwd() / "_llm_calls"
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        request_log = log_dir / f"{timestamp}_request.json"
        with open(request_log, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "url": url,
                    "headers": {k: v for k, v in headers.items() if k != "Authorization"},
                    "payload": payload,
                },
                f,
                indent=2,
            )
        print(f"  → Logged request to {request_log}")

        if stream_callback:
            # Streaming mode
            full_text = ""
            input_tokens = 0
            output_tokens = 0

            with httpx.stream(
                "POST", url, headers=headers, json=payload, timeout=timeout_seconds
            ) as r:
                if r.status_code != 200:
                    # Fall back to non-streaming
                    payload["stream"] = False
                    return _call_openai_responses_api(
                        prompt, model=model, reasoning_effort=reasoning_effort, stream_callback=None
                    )

                for line in r.iter_lines():
                    if not line or line == "data: [DONE]":
                        continue
                    if line.startswith("data: "):
                        try:
                            event = json_module.loads(line[6:])

                            # Handle response.output_text.delta events
                            if event.get("type") == "response.output_text.delta":
                                delta = event.get("delta", "")
                                if delta:
                                    full_text += delta
                                    stream_callback(delta)

                            # Extract usage from response.done event
                            elif event.get("type") == "response.done":
                                usage = event.get("usage", {})
                                input_tokens = usage.get("input_tokens", 0)
                                output_tokens = usage.get("output_tokens", 0)

                        except Exception:
                            continue

            # Log streaming response
            response_log = log_dir / f"{timestamp}_response.json"
            with open(response_log, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "streaming": True,
                        "full_text": full_text,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                    },
                    f,
                    indent=2,
                )
            print(f"  ← Logged response to {response_log}")

            # Estimate tokens if not provided
            if input_tokens == 0:
                input_tokens = len(prompt) // 4
            if output_tokens == 0:
                output_tokens = len(full_text) // 4

            # Calculate cost
            cost_per_1k_input = {
                "gpt-5-nano": 0.0001,
                "gpt-5-mini": 0.0005,
                "gpt-5-pro": 0.01,
                "gpt-5": 0.005,
            }
            cost_per_1k_output = {
                "gpt-5-nano": 0.0004,
                "gpt-5-mini": 0.0015,
                "gpt-5-pro": 0.03,
                "gpt-5": 0.015,
            }

            model_key = model
            if model_key not in cost_per_1k_input:
                for key_prefix in cost_per_1k_input:
                    if model.startswith(key_prefix):
                        model_key = key_prefix
                        break

            input_cost = input_tokens / 1000 * cost_per_1k_input.get(model_key, 0.0001)
            output_cost = output_tokens / 1000 * cost_per_1k_output.get(model_key, 0.0004)

            metadata = {
                "model": model,
                "provider": "openai",
                "reasoning_effort": reasoning_effort,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "cost_usd": input_cost + output_cost,
                "duration_seconds": round(time.time() - start_time, 2),
            }

            # Print concise log
            in_tok_stream = cast(int, metadata["input_tokens"])
            out_tok_stream = cast(int, metadata["output_tokens"])
            dur_stream = cast(float, metadata["duration_seconds"])
            cost_stream = cast(float, metadata["cost_usd"])
            window_pct_stream = (
                (in_tok_stream / 400000 * 100)
                if model.startswith("gpt-5")
                else (in_tok_stream / 200000 * 100)
            )
            effort_str = f" ({reasoning_effort})" if reasoning_effort else ""
            print(
                f"[dim]LLM: {model}{effort_str} | {in_tok_stream:,}→{out_tok_stream:,} tok ({window_pct_stream:.1f}% window) | {dur_stream:.1f}s | ${cost_stream:.4f}[/dim]"
            )

            # Write to cache (streaming mode)
            if use_cache:
                from .llm_cache import write_cache

                cache_dir = (log_dir or Path.cwd() / "_llm_calls") / "cache"
                write_cache(
                    prompt=prompt,
                    model=model,
                    response=full_text.strip(),
                    metadata=metadata,
                    cache_dir=cache_dir,
                    reasoning_effort=reasoning_effort,
                )

            return full_text.strip(), metadata

        # Non-streaming mode (original behavior)
        r = httpx.post(url, headers=headers, json=payload, timeout=timeout_seconds)

        # Log raw response
        response_log = log_dir / f"{timestamp}_response.json"
        with open(response_log, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "status_code": r.status_code,
                    "headers": dict(r.headers),
                    "body": r.json() if r.status_code == 200 else r.text,
                },
                f,
                indent=2,
            )
        print(f"[dim]  ← Logged response to {response_log}[/dim]")

        if r.status_code == 200:
            data = r.json()
            usage = data.get("usage", {})

            # GPT-5 responses API pricing (check most specific models first)
            cost_per_1k_input = {
                "gpt-5-nano": 0.0001,
                "gpt-5-mini": 0.0005,
                "gpt-5-pro": 0.01,
                "gpt-5": 0.005,
            }
            cost_per_1k_output = {
                "gpt-5-nano": 0.0004,
                "gpt-5-mini": 0.0015,
                "gpt-5-pro": 0.03,
                "gpt-5": 0.015,
            }

            # Find matching price by exact key match first, then prefix
            model_key = model
            if model_key not in cost_per_1k_input:
                # Try prefix matching (most specific first)
                for key_prefix in cost_per_1k_input:
                    if model.startswith(key_prefix):
                        model_key = key_prefix
                        break

            input_cost = (
                usage.get("input_tokens", 0) / 1000 * cost_per_1k_input.get(model_key, 0.0001)
            )
            output_cost = (
                usage.get("output_tokens", 0) / 1000 * cost_per_1k_output.get(model_key, 0.0004)
            )

            metadata = {
                "model": model,
                "provider": "openai",
                "reasoning_effort": reasoning_effort,
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "cost_usd": input_cost + output_cost,
                "duration_seconds": round(time.time() - start_time, 2),
            }

            # Print concise log
            in_tok_nonstream = cast(int, metadata["input_tokens"])
            out_tok_nonstream = cast(int, metadata["output_tokens"])
            dur_nonstream = cast(float, metadata["duration_seconds"])
            cost_nonstream = cast(float, metadata["cost_usd"])
            window_pct_nonstream = (
                (in_tok_nonstream / 400000 * 100)
                if model.startswith("gpt-5")
                else (in_tok_nonstream / 200000 * 100)
            )
            effort_str = f" ({reasoning_effort})" if reasoning_effort else ""
            print(
                f"[dim]LLM: {model}{effort_str} | {in_tok_nonstream:,}→{out_tok_nonstream:,} tok ({window_pct_nonstream:.1f}% window) | {dur_nonstream:.1f}s | ${cost_nonstream:.4f}[/dim]"
            )

            # Write to cache (non-streaming mode)
            if use_cache:
                from .llm_cache import write_cache

                cache_dir = (log_dir or Path.cwd() / "_llm_calls") / "cache"

            # Extract text from response - GPT-5 Responses API uses 'output' field
            output = data.get("output", [])
            print(
                f"[dim]DEBUG: output type={type(output)}, len={len(output) if isinstance(output, list) else 'N/A'}[/dim]"
            )

            # Try to find the message with actual content (skip reasoning messages)
            text_content = ""
            if output and isinstance(output, list):
                for i, item in enumerate(output):
                    if not isinstance(item, dict):
                        continue

                    # Skip reasoning-only messages
                    item_type = item.get("type", "")
                    if item_type == "reasoning":
                        continue

                    if i == 0:
                        print(f"[dim]DEBUG: output[{i}] keys={list(item.keys())}[/dim]")

                    content = item.get("content", [])
                    if i == 0:
                        print(
                            f"[dim]DEBUG: content type={type(content)}, len={len(content) if isinstance(content, list) else 'N/A'}[/dim]"
                        )

                    if content and isinstance(content, list) and len(content) > 0:
                        if i == 0:
                            print(f"[dim]DEBUG: content[0]={content[0]}[/dim]")
                        text_content = content[0].get("text", "")
                        if text_content:
                            break

                    # Fall back to summary field when content is empty
                    summary = item.get("summary", "")
                    if summary:
                        text_content = summary
                        break

            print(f"[dim]DEBUG: text_content length={len(text_content)}[/dim]")

            # Write to cache before returning
            if use_cache:
                write_cache(
                    prompt=prompt,
                    model=model,
                    response=text_content,
                    metadata=metadata,
                    cache_dir=cache_dir,
                    reasoning_effort=reasoning_effort,
                )

            return text_content, metadata
        else:
            error_msg = f"OpenAI Responses API error {r.status_code}"
            try:
                error_data = r.json()
                if "error" in error_data:
                    error_msg = (
                        f"{error_msg}: {error_data['error'].get('message', 'Unknown error')}"
                    )
            except:
                pass
            print(f"[dim]{error_msg}[/dim]")
            return "", {}
    except Exception as e:
        print(f"[dim]OpenAI Responses API exception: {str(e)}[/dim]")
        return "", {}


def _call_openai_api(
    prompt: str,
    model: str = "",
    stream_callback: Callable[[str], None] | None = None,
    log_dir: Path | None = None,
) -> tuple[str, dict[str, Any]]:
    config = get_config()
    key = config.openai_api_key
    if not key:
        return "", {}
    try:
        import json as json_module

        import httpx
    except Exception:
        if which("openai"):
            return _call_cli("openai", prompt), {"model": "openai-cli", "provider": "openai-cli"}
        return "", {}

    if not model:
        model = config.openai_model

    # Parse model string to extract base model and reasoning effort
    base_model = model
    reasoning_effort = None

    if " (reasoning/" in model:
        # Extract: "gpt-5-mini (reasoning/medium)" -> base="gpt-5-mini", effort="medium"
        base_model = model.split(" (reasoning/")[0]
        reasoning_effort = model.split(" (reasoning/")[1].rstrip(")")
    elif " (non-reasoning)" in model:
        # Extract: "gpt-5-chat (non-reasoning)" -> base="gpt-5-chat"
        base_model = model.split(" (non-reasoning)")[0]

    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {key}"}
    payload = {
        "model": base_model,
        "messages": [
            {"role": "system", "content": "You are a concise LP solver expert."},
            {"role": "user", "content": prompt},
        ],
        "stream": bool(stream_callback),
    }

    # Add reasoning parameter for reasoning models
    if reasoning_effort:
        payload["reasoning"] = {"effort": reasoning_effort}

    # GPT-5 models only support temperature=1 (default), don't include it
    # Older models support configurable temperature
    if not base_model.startswith("gpt-5"):
        payload["temperature"] = config.openai_temperature

    try:
        # Longer timeout for reasoning models (GPT-5, etc.)
        timeout_seconds = 300 if base_model.startswith("gpt-5") or "pro" in base_model else 120

        if stream_callback:
            # Try streaming mode first
            try:
                full_text = ""
                input_tokens = 0
                output_tokens = 0

                with httpx.stream(
                    "POST", url, headers=headers, json=payload, timeout=timeout_seconds
                ) as r:
                    if r.status_code != 200:
                        # Fall back to non-streaming on error
                        from rich import print as rprint

                        rprint(
                            "\n[yellow]⚠ Streaming not available (requires org verification), falling back to non-streaming mode...[/yellow]"
                        )
                        payload["stream"] = False
                        return _call_openai_api(prompt, model=model, stream_callback=None)

                    for line in r.iter_lines():
                        if not line or line == "data: [DONE]":
                            continue
                        if line.startswith("data: "):
                            try:
                                chunk = json_module.loads(line[6:])
                                delta = chunk.get("choices", [{}])[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    full_text += content
                                    stream_callback(content)

                                # Extract usage from the last chunk if available
                                if "usage" in chunk:
                                    usage = chunk["usage"]
                                    input_tokens = usage.get("prompt_tokens", 0)
                                    output_tokens = usage.get("completion_tokens", 0)
                            except Exception:
                                continue
            except Exception:
                # Fall back to non-streaming on any error
                from rich import print as rprint

                rprint(
                    "\n[yellow]⚠ Streaming error, falling back to non-streaming mode...[/yellow]"
                )
                payload["stream"] = False
                return _call_openai_api(prompt, model=model, stream_callback=None)

            # If we didn't get usage data from stream, estimate tokens
            if input_tokens == 0:
                # Rough estimate: ~4 chars per token
                input_tokens = len(prompt) // 4
            if output_tokens == 0:
                output_tokens = len(full_text) // 4

            # Calculate cost
            cost_per_1k_input = {
                "gpt-5": 0.005,
                "gpt-5-pro": 0.01,
                "gpt-5-mini": 0.0005,
                "gpt-5-nano": 0.0001,
                "gpt-4o": 0.0025,
                "gpt-4o-mini": 0.00015,
                "gpt-4-turbo": 0.01,
                "gpt-4": 0.03,
                "gpt-3.5-turbo": 0.0005,
            }
            cost_per_1k_output = {
                "gpt-5": 0.015,
                "gpt-5-pro": 0.03,
                "gpt-5-mini": 0.0015,
                "gpt-5-nano": 0.0004,
                "gpt-4o": 0.01,
                "gpt-4o-mini": 0.0006,
                "gpt-4-turbo": 0.03,
                "gpt-4": 0.06,
                "gpt-3.5-turbo": 0.0015,
            }

            model_key = model
            for key_prefix in cost_per_1k_input:
                if model.startswith(key_prefix):
                    model_key = key_prefix
                    break

            input_cost = input_tokens / 1000 * cost_per_1k_input.get(model_key, 0.0005)
            output_cost = output_tokens / 1000 * cost_per_1k_output.get(model_key, 0.0015)

            metadata = {
                "model": model,
                "temperature": 1.0 if base_model.startswith("gpt-5") else config.openai_temperature,
                "provider": "openai",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "cost_usd": input_cost + output_cost,
            }
            if reasoning_effort:
                metadata["reasoning_effort"] = reasoning_effort

            return full_text.strip(), metadata
        else:
            # Non-streaming mode (original behavior)

            # Log raw request
            if log_dir is None:
                log_dir = Path.cwd() / "_llm_calls"
            log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            request_log = log_dir / f"{timestamp}_request.json"
            with open(request_log, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "url": url,
                        "headers": {k: v for k, v in headers.items() if k != "Authorization"},
                        "payload": payload,
                    },
                    f,
                    indent=2,
                )
            print(f"  → Logged request to {request_log}")

            r = httpx.post(url, headers=headers, json=payload, timeout=timeout_seconds)

            # Log raw response
            response_log = log_dir / f"{timestamp}_response.json"
            with open(response_log, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "status_code": r.status_code,
                        "headers": dict(r.headers),
                        "body": r.json() if r.status_code == 200 else r.text,
                    },
                    f,
                    indent=2,
                )
            print(f"[dim]  ← Logged response to {response_log}[/dim]")

            if r.status_code == 200:
                data = r.json()
                usage = data.get("usage", {})

                # Calculate cost (pricing as of 2025)
                cost_per_1k_input = {
                    "gpt-5": 0.005,
                    "gpt-5-pro": 0.01,
                    "gpt-5-mini": 0.0005,
                    "gpt-5-nano": 0.0001,
                    "gpt-4o": 0.0025,
                    "gpt-4o-mini": 0.00015,
                    "gpt-4-turbo": 0.01,
                    "gpt-4": 0.03,
                    "gpt-3.5-turbo": 0.0005,
                }
                cost_per_1k_output = {
                    "gpt-5": 0.015,
                    "gpt-5-pro": 0.03,
                    "gpt-5-mini": 0.0015,
                    "gpt-5-nano": 0.0004,
                    "gpt-4o": 0.01,
                    "gpt-4o-mini": 0.0006,
                    "gpt-4-turbo": 0.03,
                    "gpt-4": 0.06,
                    "gpt-3.5-turbo": 0.0015,
                }

                model_base = model.split("-")[0:2]
                model_key = "-".join(model_base) if len(model_base) >= 2 else model
                for key_prefix in cost_per_1k_input:
                    if model.startswith(key_prefix):
                        model_key = key_prefix
                        break

                input_cost = (
                    usage.get("prompt_tokens", 0) / 1000 * cost_per_1k_input.get(model_key, 0.0005)
                )
                output_cost = (
                    usage.get("completion_tokens", 0)
                    / 1000
                    * cost_per_1k_output.get(model_key, 0.0015)
                )

                metadata = {
                    "model": model,
                    "temperature": 1.0
                    if base_model.startswith("gpt-5")
                    else config.openai_temperature,
                    "provider": "openai",
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                    "cost_usd": input_cost + output_cost,
                }
                if reasoning_effort:
                    metadata["reasoning_effort"] = reasoning_effort

                return data["choices"][0]["message"]["content"].strip(), metadata
            else:
                # Return error info for better debugging
                error_msg = f"OpenAI API error {r.status_code}"
                try:
                    error_data = r.json()
                    if "error" in error_data:
                        error_msg = (
                            f"{error_msg}: {error_data['error'].get('message', 'Unknown error')}"
                        )
                except:
                    pass
                print(f"[dim]{error_msg}[/dim]")
                return "", {}
    except Exception as e:
        print(f"[dim]OpenAI API exception: {str(e)}[/dim]")
        return "", {}


def _call_anthropic_api(
    prompt: str,
    model: str = "",
    stream_callback: Callable[[str], None] | None = None,
    log_dir: Path | None = None,
    use_cache: bool = True,
) -> tuple[str, dict[str, Any]]:
    """Call Anthropic API directly with optional streaming support.

    Args:
        prompt: The prompt to send to the API
        model: Model name to use
        stream_callback: Optional callback for streaming responses
        log_dir: Directory for logging
        use_cache: Whether to use cached responses (default: True)

    Returns:
        Tuple of (response_text, metadata)
    """
    start_time = time.time()
    config = get_config()
    key = config.anthropic_api_key
    if not key:
        return "", {}

    # Check cache first (skip if streaming or cache disabled)
    if use_cache and not stream_callback:
        from .llm_cache import read_cache

        cache_dir = (log_dir or Path.cwd() / "_llm_calls") / "cache"
        model_to_use = model or "claude-3-5-sonnet-20241022"
        cached = read_cache(
            prompt=prompt,
            model=model_to_use,
            cache_dir=cache_dir,
            temperature=config.anthropic_temperature,
        )
        if cached:
            text, meta = cached
            # Add cache hit indicator to metadata
            meta["cached"] = True
            meta["cache_hit"] = True
            print(
                f"[dim]LLM: {model_to_use} (cached) | {meta.get('input_tokens', 0):,}→{meta.get('output_tokens', 0):,} tok | $0.0000 (cache hit)[/dim]"
            )
            return text, meta

    try:
        import json as json_module

        import httpx
    except Exception:
        return "", {}

    if not model:
        model = "claude-3-5-sonnet-20241022"
    temperature = config.anthropic_temperature

    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": 4096,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
        "stream": bool(stream_callback),
    }

    try:
        # Longer timeout for reasoning models
        timeout_seconds = 300 if "opus" in model or "sonnet" in model else 120

        if stream_callback:
            # Streaming mode
            full_text = ""
            input_tokens = 0
            output_tokens = 0

            with httpx.stream(
                "POST", url, headers=headers, json=payload, timeout=timeout_seconds
            ) as r:
                if r.status_code != 200:
                    # Fall back to non-streaming
                    payload["stream"] = False
                    return _call_anthropic_api(prompt, model=model, stream_callback=None)

                for line in r.iter_lines():
                    if not line:
                        continue

                    # Anthropic streaming format: "event: ...\ndata: {...}"
                    if line.startswith("data: "):
                        try:
                            data = json_module.loads(line[6:])

                            # Handle different event types
                            if data.get("type") == "content_block_delta":
                                delta = data.get("delta", {})
                                if delta.get("type") == "text_delta":
                                    text = delta.get("text", "")
                                    if text:
                                        full_text += text
                                        stream_callback(text)

                            # Extract usage from message_delta event
                            elif data.get("type") == "message_delta":
                                usage = data.get("usage", {})
                                output_tokens = usage.get("output_tokens", output_tokens)

                            # Extract input tokens from message_start event
                            elif data.get("type") == "message_start":
                                message = data.get("message", {})
                                usage = message.get("usage", {})
                                input_tokens = usage.get("input_tokens", 0)

                        except Exception:
                            continue

            # Estimate tokens if not provided
            if input_tokens == 0:
                input_tokens = len(prompt) // 4
            if output_tokens == 0:
                output_tokens = len(full_text) // 4

            # Calculate cost
            pricing = {
                "claude-3-5-sonnet-20241022": (0.003, 0.015),
                "claude-3-5-haiku-20241022": (0.0008, 0.004),
                "claude-3-opus-20240229": (0.015, 0.075),
                "claude-3-sonnet-20240229": (0.003, 0.015),
                "claude-3-haiku-20240307": (0.00025, 0.00125),
            }

            cost_per_1k_input, cost_per_1k_output = pricing.get(model, (0.003, 0.015))
            input_cost = input_tokens / 1000 * cost_per_1k_input
            output_cost = output_tokens / 1000 * cost_per_1k_output

            metadata = {
                "model": model,
                "temperature": temperature,
                "provider": "anthropic",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "cost_usd": input_cost + output_cost,
                "duration_seconds": round(time.time() - start_time, 2),
            }

            # Write to cache (streaming mode)
            if use_cache:
                from .llm_cache import write_cache

                cache_dir = (log_dir or Path.cwd() / "_llm_calls") / "cache"
                write_cache(
                    prompt=prompt,
                    model=model,
                    response=full_text.strip(),
                    metadata=metadata,
                    cache_dir=cache_dir,
                    temperature=temperature,
                )

            return full_text.strip(), metadata

        else:
            # Non-streaming mode (original behavior)
            r = httpx.post(url, headers=headers, json=payload, timeout=timeout_seconds)
            if r.status_code == 200:
                data = r.json()
                usage = data.get("usage", {})

                # Anthropic pricing (as of 2024)
                pricing = {
                    "claude-3-5-sonnet-20241022": (0.003, 0.015),
                    "claude-3-5-haiku-20241022": (0.0008, 0.004),
                    "claude-3-opus-20240229": (0.015, 0.075),
                    "claude-3-sonnet-20240229": (0.003, 0.015),
                    "claude-3-haiku-20240307": (0.00025, 0.00125),
                }

                cost_per_1k_input, cost_per_1k_output = pricing.get(model, (0.003, 0.015))
                input_cost = usage.get("input_tokens", 0) / 1000 * cost_per_1k_input
                output_cost = usage.get("output_tokens", 0) / 1000 * cost_per_1k_output

                metadata = {
                    "model": model,
                    "temperature": temperature,
                    "provider": "anthropic",
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
                    "cost_usd": input_cost + output_cost,
                    "duration_seconds": round(time.time() - start_time, 2),
                }

                # Print concise log
                in_tok_anthropic = cast(int, metadata["input_tokens"])
                out_tok_anthropic = cast(int, metadata["output_tokens"])
                dur_anthropic = cast(float, metadata["duration_seconds"])
                cost_anthropic = cast(float, metadata["cost_usd"])
                window_pct_anthropic = in_tok_anthropic / 200000 * 100
                print(
                    f"[dim]LLM: {model} | {in_tok_anthropic:,}→{out_tok_anthropic:,} tok ({window_pct_anthropic:.1f}% window) | {dur_anthropic:.1f}s | ${cost_anthropic:.4f}[/dim]"
                )

                content = data.get("content", [])
                if content and len(content) > 0:
                    response_text = content[0].get("text", "")

                    # Write to cache (non-streaming mode)
                    if use_cache:
                        from .llm_cache import write_cache

                        cache_dir = (log_dir or Path.cwd() / "_llm_calls") / "cache"
                        write_cache(
                            prompt=prompt,
                            model=model,
                            response=response_text,
                            metadata=metadata,
                            cache_dir=cache_dir,
                            temperature=temperature,
                        )

                    return response_text, metadata
            return "", {}
    except Exception as e:
        print(f"[dim]Anthropic API exception: {str(e)}[/dim]")
        return "", {}


def _call_anthropic_cli(prompt: str) -> str:
    cli = "claude"
    if which(cli):
        return _call_cli(cli, prompt)
    return ""


def _call_amp_cli(prompt: str) -> str:
    config = get_config()
    cli = config.amp_cli
    if which(cli):
        return _call_cli(cli, prompt)
    return ""


def summarize(diagnostics: dict[str, Any], provider: str = "auto") -> LLMResult:
    from .prompts import build_llm_prompt

    prompt = build_llm_prompt(diagnostics)

    def done(name: str, text: str, meta: dict[str, Any] | None = None) -> LLMResult:
        if meta:
            return LLMResult(
                text=text,
                provider=name,
                used=bool(text),
                model=meta.get("model", ""),
                input_tokens=meta.get("input_tokens", 0),
                output_tokens=meta.get("output_tokens", 0),
                cost_usd=meta.get("cost_usd", 0.0),
            )
        return LLMResult(text=text, provider=name, used=bool(text))

    prov = (provider or "auto").lower()
    if prov == "none":
        return done("none", "")

    if prov in ("auto", "openai"):
        t, meta = _call_openai_api(prompt)
        if t:
            return done("openai", t, meta)

    if prov in ("auto", "anthropic"):
        t = _call_anthropic_cli(prompt)
        if t:
            return done("anthropic_cli", t, {"model": "claude-cli", "provider": "anthropic"})

    if prov in ("auto", "amp"):
        t = _call_amp_cli(prompt)
        if t:
            return done("amp_cli", t, {"model": "amp-cli", "provider": "amp"})

    return done("none", "")


def _filter_run_log(
    file_content: str, progress_callback: Callable[[int, int, str], None] | None = None
) -> dict[str, Any]:
    """Filter run_log.txt to keep only condensed diagnostic content.

    Filtering rules:
    - Skip everything before "starting execution" or "Restarting execution"
    - Drop lines containing "DMoves" or "PMoves"
    - Drop lines starting with "Iteration:"
    - Drop lines starting with "Elapsed time ="
    - Deduplicate exact duplicates
    - Condense repetitive lines that differ only in numbers

    Returns dict with "filtered_content" key containing the filtered text.
    """

    lines = file_content.split("\n")

    # Find the start line
    start_idx = 0
    for i, line in enumerate(lines):
        if "starting execution" in line.lower() or "Restarting execution" in line:
            start_idx = i
            break

    # Step 1: Collect errors/warnings from beginning (before execution)
    pre_execution_important = []
    for i in range(0, start_idx):
        line = lines[i]
        # Keep errors, warnings, and their context (but be specific to avoid noise)
        if (
            "*** Error" in line
            or "*** Warning" in line
            or "Domain violation" in line
            or "*** " in line
        ):
            pre_execution_important.append(line)

    # Step 2: Basic filtering from execution start
    filtered_lines = []
    for i in range(start_idx, len(lines)):
        line = lines[i]

        # Skip lines matching basic filter rules
        if (
            "DMoves" in line
            or "PMoves" in line
            or line.startswith("Iteration:")
            or line.startswith("Elapsed time =")
        ):
            continue

        filtered_lines.append(line)

    # Combine pre-execution important items with filtered execution log
    if pre_execution_important:
        all_lines = (
            pre_execution_important
            + ["", "--- [Pre-execution errors above] ---", ""]
            + filtered_lines
        )
    else:
        all_lines = filtered_lines

    # Step 2: Condense repetitive patterns
    condensed_lines = _condense_repetitive_lines(all_lines)

    if progress_callback:
        orig_lines = len(lines)
        after_basic = len(filtered_lines)
        final_count = len(condensed_lines)
        progress_callback(
            1, 1, f"Filtered run_log: {orig_lines} → {after_basic} → {final_count} lines"
        )

    # Return filtered content directly
    return {
        "filtered_content": "\n".join(condensed_lines),
        "sections": [],  # Empty sections since we're returning filtered content
    }


def _condense_repetitive_lines(lines: list) -> list:
    """Condense lines that differ only in numeric values.

    Groups consecutive similar lines and shows:
    - One example of the line
    - Count and sample values if repeated

    Also handles multi-line patterns (e.g., error messages spanning 2+ lines)
    """
    import re

    if not lines:
        return lines

    # First pass: handle multi-line error patterns (e.g., "*** Error" followed by indented description)
    lines = _condense_multiline_errors(lines)

    # Create pattern by replacing numbers with placeholder
    def make_pattern(line: str) -> str:
        # Replace integers with {N}
        pattern = re.sub(r"\b\d+\b", "{N}", line)
        # Replace floats with {F}
        pattern = re.sub(r"\b\d+\.\d+\b", "{F}", pattern)
        return pattern

    # Extract all numbers from a line
    def extract_numbers(line: str) -> list:
        return re.findall(r"\b\d+(?:\.\d+)?\b", line)

    result: list[str] = []
    current_group: list[str] = []
    current_pattern: str | None = None

    for line in lines:
        # Skip empty lines
        if not line.strip():
            if current_group:
                result.extend(_format_group(current_group, current_pattern))
                current_group = []
                current_pattern = None
            result.append(line)
            continue

        pattern = make_pattern(line)

        # If pattern matches current group, add to group
        if pattern == current_pattern:
            current_group.append(line)
        else:
            # Flush previous group
            if current_group:
                result.extend(_format_group(current_group, current_pattern))

            # Start new group
            current_group = [line]
            current_pattern = pattern

    # Flush final group
    if current_group:
        result.extend(_format_group(current_group, current_pattern))

    return result


def _condense_multiline_errors(lines: list) -> list:
    """Condense multi-line error patterns like:
    *** Error 170 in file.dd
        Domain violation for element
    (repeated many times)

    Returns condensed version with one example + count.
    """
    import re

    result = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Check if this is an error/warning line
        if line.startswith("*** Error") or line.startswith("*** Warning"):
            # Look ahead to see if next line is indented (continuation)
            if i + 1 < len(lines) and lines[i + 1].startswith("    "):
                # Extract the error pattern (remove numbers)
                error_pattern = re.sub(r"\b\d+\b", "{N}", line)
                desc_line = lines[i + 1]

                # Count consecutive occurrences of this error+description pair
                count = 0
                examples: list[str] = []
                j = i

                while j + 1 < len(lines):
                    if (
                        re.sub(r"\b\d+\b", "{N}", lines[j]) == error_pattern
                        and j + 1 < len(lines)
                        and lines[j + 1] == desc_line
                    ):
                        count += 1
                        # Collect first few examples with varying numbers
                        if len(examples) < 5:
                            nums = re.findall(r"\b\d+\b", lines[j])
                            examples.extend(nums)
                        j += 2  # Skip error + description
                    else:
                        break

                if count >= 3:
                    # Show condensed version
                    result.append(lines[i])
                    result.append(lines[i + 1])

                    # Get unique numbers from examples
                    unique_nums = []
                    seen = set()
                    for num in examples:
                        if num not in seen:
                            seen.add(num)
                            unique_nums.append(num)

                    if unique_nums:
                        if len(unique_nums) <= 6:
                            nums_str = ", ".join(unique_nums)
                        else:
                            nums_str = (
                                ", ".join(unique_nums[:3])
                                + f", ... ({len(unique_nums)} unique values)"
                            )
                        result.append(
                            f"  [↑ repeated {count} times with error numbers: {nums_str}]"
                        )
                    else:
                        result.append(f"  [↑ repeated {count} times]")

                    i = j  # Skip past all the duplicates
                    continue

        # Not a condensable error, keep as-is
        result.append(line)
        i += 1

    return result


def _format_group(group: list[str], pattern: str | None) -> list[str]:
    """Format a group of similar lines.

    If group has 1-2 items: return as-is
    If group has 3+ items: return one example + summary
    """
    if len(group) <= 2:
        return group

    # Show first occurrence
    first_line = group[0]

    # Extract varying numbers from all lines in group
    import re

    all_numbers = []
    for line in group:
        nums = re.findall(r"\b\d+(?:\.\d+)?\b", line)
        all_numbers.extend(nums)

    # Get unique numbers (preserve order)
    seen = set()
    unique_nums = []
    for num in all_numbers:
        if num not in seen:
            seen.add(num)
            unique_nums.append(num)

    # Build summary
    if len(unique_nums) <= 6:
        nums_str = ", ".join(unique_nums)
    else:
        nums_str = ", ".join(unique_nums[:3]) + f", ... ({len(unique_nums)} values)"

    summary = f"  [↑ repeated {len(group)} times with values: {nums_str}]"

    return [first_line, summary]


def _extract_lst_pages(
    file_content: str, progress_callback: Callable[[int, int, str], None] | None = None
) -> dict[str, Any]:
    """Extract and condense sections from GAMS .lst file using LST parser.

    Uses the LST parser to extract semantic sections and aggregate repetitive content.
    """
    import tempfile
    from pathlib import Path

    from .lst_parser import process_lst_file

    # Count original lines
    original_line_count = len(file_content.split("\n"))

    # Write content to a temporary file (LST parser expects a file path)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".lst", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(file_content)
        tmp_path = Path(tmp.name)

    try:
        if progress_callback:
            progress_callback(1, 3, "Parsing LST file sections...")

        # Parse the LST file
        result = process_lst_file(tmp_path)

        if progress_callback:
            progress_callback(2, 3, f"Extracted {len(result['sections'])} sections")

        # Build markdown output
        output_parts = []

        # Add metadata
        output_parts.append("# LST File Analysis\n")
        output_parts.append("## Metadata\n")
        for key, value in result["metadata"].items():
            output_parts.append(f"- **{key}**: {value}\n")
        output_parts.append("\n")

        # Add each section
        for section_name, section_data in result["sections"].items():
            output_parts.append(f"## {section_name}\n\n")

            # If there's a text summary, use it
            if "text_summary" in section_data:
                output_parts.append(section_data["text_summary"])
                output_parts.append("\n\n")

            # For compilation sections with errors
            if "errors" in section_data:
                output_parts.append("### Error Summary\n\n")
                for error_code, error_info in section_data["errors"].items():
                    output_parts.append(
                        f"**Error {error_code}**: {error_info['count']:,} occurrences\n\n"
                    )

                    # Show top patterns
                    if error_info["elements"]:
                        output_parts.append("Top error patterns:\n")
                        top_patterns = sorted(
                            error_info["elements"].items(), key=lambda x: x[1], reverse=True
                        )[:10]

                        for pattern, count in top_patterns:
                            output_parts.append(f"- `{pattern}`: {count:,} occurrences\n")
                        output_parts.append("\n")

                    # Show sample errors
                    if error_info["samples"] and len(error_info["samples"]) > 0:
                        output_parts.append("<details>\n")
                        output_parts.append(
                            "<summary>Sample errors (click to expand)</summary>\n\n"
                        )
                        for i, sample in enumerate(error_info["samples"][:5], 1):
                            output_parts.append(f"**Sample {i}:**\n")
                            output_parts.append(
                                f"```\n{sample.get('context', 'No context')}\n```\n\n"
                            )
                        output_parts.append("</details>\n\n")

            # For sections with summary dict
            elif "summary" in section_data and isinstance(section_data["summary"], dict):
                output_parts.append("### Summary\n\n")
                for key, value in section_data["summary"].items():
                    output_parts.append(f"- **{key}**: {value}\n")
                output_parts.append("\n")

            # For other sections with content (keep short excerpts)
            elif "content" in section_data:
                content = section_data["content"]
                if len(content) > 2000:
                    output_parts.append(f"```\n{content[:2000]}\n... (truncated)\n```\n\n")
                else:
                    output_parts.append(f"```\n{content}\n```\n\n")

        extracted_text = "".join(output_parts)
        condensed_line_count = len(extracted_text.split("\n"))

        if progress_callback:
            progress_callback(
                3, 3, f"Condensed LST: {original_line_count} → {condensed_line_count} lines"
            )

        # Build sections list for backward compatibility
        sections_list = []
        for section_name in result["sections"]:
            sections_list.append(
                {
                    "name": section_name,
                    "start_line": 1,  # Not meaningful anymore with semantic sections
                    "end_line": 1,  # Not meaningful anymore with semantic sections
                }
            )

        return {"sections": sections_list, "extracted_text": extracted_text}

    finally:
        # Clean up temp file
        tmp_path.unlink(missing_ok=True)


def chunk_text_by_lines(
    text: str, max_chars: int = 100000, overlap_lines: int = 50
) -> list[tuple[str, int, int]]:
    """Split text into overlapping chunks.

    Args:
        text: Full text content
        max_chars: Maximum characters per chunk
        overlap_lines: Number of lines to overlap between chunks

    Returns:
        List of (chunk_text, start_line, end_line) tuples
    """
    lines = text.split("\n")
    chunks = []

    start_idx = 0
    while start_idx < len(lines):
        # Calculate how many lines fit in this chunk
        chunk_lines: list[str] = []
        char_count = 0
        end_idx = start_idx

        for i in range(start_idx, len(lines)):
            line = lines[i]
            if char_count + len(line) + 1 > max_chars and chunk_lines:
                break
            chunk_lines.append(line)
            char_count += len(line) + 1  # +1 for newline
            end_idx = i

        chunk_text = "\n".join(chunk_lines)
        chunks.append((chunk_text, start_idx + 1, end_idx + 1))  # 1-indexed line numbers

        # Move start to next chunk with overlap
        if end_idx >= len(lines) - 1:
            break
        start_idx = max(start_idx + 1, end_idx - overlap_lines)

    return chunks


def condense_qa_check(
    file_content: str, progress_callback: Callable[[int, int, str], None] | None = None
) -> str:
    """Condense QA_CHECK.LOG using rule-based parsing (no LLM required).

    Uses structured parsing to extract and deduplicate events by severity,
    message, and index sets. Similar to how LST and run_log files are processed.

    Args:
        file_content: Full QA_CHECK.LOG content
        progress_callback: Optional callback for progress updates

    Returns:
        Formatted condensed text with grouped warnings/errors
    """
    from .qa_check_parser import condense_events, format_condensed_output, iter_events

    if progress_callback:
        progress_callback(0, 2, "Parsing QA_CHECK.LOG events")

    # Parse events from the file content (as lines)
    lines = file_content.split("\n")
    original_line_count = len(lines)
    events = iter_events(lines, index_allow=None, min_severity="INFO")

    if progress_callback:
        progress_callback(1, 2, "Condensing events")

    # Condense events
    summary_rows, message_counts, all_index_keys = condense_events(events)

    # Format output
    condensed_output = format_condensed_output(summary_rows, message_counts, all_index_keys)
    condensed_line_count = len(condensed_output.split("\n"))

    if progress_callback:
        progress_callback(
            2, 2, f"Condensed QA_CHECK: {original_line_count} → {condensed_line_count} lines"
        )

    return condensed_output


def extract_condensed_sections(
    file_content: str,
    file_type: str,
    log_dir: Path | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Extract condensed diagnostic sections from a log file using fast LLM.

    Args:
        file_content: Full file content
        file_type: One of 'qa_check', 'run_log', 'lst'
        log_dir: Directory to save LLM call logs
        progress_callback: Optional callback(current, total, message) for progress updates

    Returns:
        dict with 'sections' list of {name, start_line, end_line}

    Raises:
        RuntimeError: If extraction fails
    """
    import re

    from .prompts import build_extraction_prompt

    # Special handling for .lst files - they're already paginated
    if file_type == "lst":
        return _extract_lst_pages(file_content, progress_callback)

    # Special handling for run_log - simple filtering rules
    if file_type == "run_log":
        return _filter_run_log(file_content, progress_callback)

    # Check if chunking is needed (300k chars ~= 75k tokens, staying well under 400k token limit)
    needs_chunking = len(file_content) > 300000

    if needs_chunking:
        chunks = chunk_text_by_lines(file_content, max_chars=300000, overlap_lines=50)
        if progress_callback:
            progress_callback(0, len(chunks), f"Processing {len(chunks)} chunks")

        all_sections = []

        for i, (chunk_text, start_line, end_line) in enumerate(chunks, 1):
            if progress_callback:
                progress_callback(
                    i, len(chunks), f"Chunk {i}/{len(chunks)} (lines {start_line}-{end_line})"
                )

            # Add line numbers to chunk
            chunk_lines = chunk_text.split("\n")
            numbered_content = "\n".join(
                f"{start_line + j}: {line}" for j, line in enumerate(chunk_lines)
            )

            prompt = build_extraction_prompt(numbered_content, file_type)

            # Call LLM for this chunk
            api_keys = check_api_keys()
            text = ""
            meta: dict[str, Any] = {}

            if api_keys["openai"]:
                text, meta = _call_openai_responses_api(
                    prompt,
                    model="gpt-5-nano",
                    reasoning_effort="minimal",
                    log_dir=log_dir,
                    use_cache=True,
                )
            elif api_keys["anthropic"]:
                text, meta = _call_anthropic_api(
                    prompt, model="claude-3-5-haiku-20241022", log_dir=log_dir, use_cache=True
                )
            else:
                error_msg = "No OpenAI or Anthropic API key found. Extraction requires OPENAI_API_KEY or ANTHROPIC_API_KEY in .env file"
                log_llm_call(
                    f"extraction_{file_type}_chunk{i}", prompt, "", {"error": error_msg}, log_dir
                )
                raise RuntimeError(error_msg)

            # Log the call
            log_llm_call(f"extraction_{file_type}_chunk{i}", prompt, text, meta, log_dir)

            if not text:
                continue  # Skip empty responses

            # Parse JSON response
            json_match = re.search(r'\{[\s\S]*"sections"[\s\S]*\}', text)
            json_str = json_match.group(0) if json_match else text

            try:
                result = json.loads(json_str)
                if "sections" in result and isinstance(result["sections"], list):
                    all_sections.extend(result["sections"])
            except Exception:
                continue  # Skip failed parses, don't fail whole extraction

        # Merge and deduplicate sections
        merged_sections = _merge_sections(all_sections)
        return {"sections": merged_sections}

    else:
        # Original single-call logic for small files
        lines = file_content.split("\n")
        numbered_content = "\n".join(f"{i + 1}: {line}" for i, line in enumerate(lines))

        prompt = build_extraction_prompt(numbered_content, file_type)

        # Determine which fast model to use
        api_keys = check_api_keys()

        text = ""
        meta = {}
        error_msg = ""

        if api_keys["openai"]:
            text, meta = _call_openai_responses_api(
                prompt,
                model="gpt-5-nano",
                reasoning_effort="minimal",
                log_dir=log_dir,
                use_cache=True,
            )
        elif api_keys["anthropic"]:
            text, meta = _call_anthropic_api(
                prompt, model="claude-3-5-haiku-20241022", log_dir=log_dir, use_cache=True
            )
        else:
            error_msg = "No OpenAI or Anthropic API key found. Extraction requires OPENAI_API_KEY or ANTHROPIC_API_KEY in .env file"
            log_llm_call(f"extraction_{file_type}", prompt, "", {"error": error_msg}, log_dir)
            raise RuntimeError(error_msg)

        # Log the call
        log_llm_call(f"extraction_{file_type}", prompt, text, meta, log_dir)

        if not text:
            error_msg = f"Failed to extract condensed sections from {file_type}. LLM returned empty response."
            raise RuntimeError(error_msg)

        # Parse JSON response
        json_match = re.search(r'\{[\s\S]*"sections"[\s\S]*\}', text)
        json_str = json_match.group(0) if json_match else text

        try:
            parsed_result: dict[str, Any] = json.loads(json_str)
            if "sections" not in parsed_result or not isinstance(parsed_result["sections"], list):
                raise ValueError("Invalid JSON structure")
            return parsed_result
        except Exception as e:
            error_msg = (
                f"Failed to parse extraction JSON from {file_type}: {e}\nLLM response: {text[:500]}"
            )
            raise RuntimeError(error_msg)


def _merge_sections(sections: list[dict]) -> list[dict]:
    """Merge and deduplicate extracted sections.

    Removes overlapping sections and sorts by start_line.
    """
    if not sections:
        return []

    # Sort by start_line
    sorted_sections = sorted(sections, key=lambda s: s.get("start_line", 0))

    # Merge overlapping or adjacent sections with same name
    merged: list[dict[str, Any]] = []
    for section in sorted_sections:
        if not merged:
            merged.append(section)
            continue

        last = merged[-1]
        # If sections overlap or are adjacent and have similar names, merge them
        if section.get("start_line", 0) <= last.get("end_line", 0) + 10 and _similar_names(
            last.get("name", ""), section.get("name", "")
        ):
            # Extend the last section
            last["end_line"] = max(last.get("end_line", 0), section.get("end_line", 0))
            # Combine names if different
            if last.get("name") != section.get("name"):
                last["name"] = f"{last.get('name')} / {section.get('name')}"
        else:
            merged.append(section)

    return merged


def _similar_names(name1: str, name2: str) -> bool:
    """Check if two section names are similar enough to merge."""
    if name1 == name2:
        return True
    # Check if one is substring of other
    return bool(name1 in name2 or name2 in name1)


def create_condensed_markdown(file_lines: list[str], sections: list[dict], file_type: str) -> str:
    """Create markdown file with extracted condensed sections.

    Args:
        file_lines: Original file content as list of lines
        sections: List of {name, start_line, end_line} dicts
        file_type: One of 'qa_check', 'run_log', 'lst'

    Returns:
        Markdown formatted string
    """
    md_lines = []

    # Title
    file_name_map = {"qa_check": "QA_CHECK.LOG", "run_log": "Run Log", "lst": "LST File"}
    md_lines.append(f"# {file_name_map.get(file_type, file_type)} - Condensed")
    md_lines.append("")

    for section in sections:
        name = section.get("name", "Unknown Section")
        start = section.get("start_line", 1)
        end = section.get("end_line", 1)

        md_lines.append(f"## {name} (Lines {start}-{end})")
        md_lines.append("")
        md_lines.append("```")

        # Extract lines (convert to 0-indexed)
        for i in range(start - 1, min(end, len(file_lines))):
            if i >= 0 and i < len(file_lines):
                md_lines.append(file_lines[i])

        md_lines.append("```")
        md_lines.append("")

    return "\n".join(md_lines)


def review_files(
    qa_check: str,
    run_log: str,
    lst_content: str,
    provider: str = "auto",
    model: str = "",
    stream_callback: Callable[[str], None] | None = None,
    log_dir: Path | None = None,
    use_cache: bool = True,
) -> LLMResult:
    from .prompts import build_review_prompt

    prompt = build_review_prompt(qa_check, run_log, lst_content)

    def done(name: str, text: str, meta: dict[str, Any] | None = None) -> LLMResult:
        # Log the call
        log_meta: dict[str, Any] = meta if meta else {"provider": name}
        log_llm_call("review", prompt, text, log_meta, log_dir)

        if meta:
            return LLMResult(
                text=text,
                provider=name,
                used=bool(text),
                model=meta.get("model", ""),
                input_tokens=meta.get("input_tokens", 0),
                output_tokens=meta.get("output_tokens", 0),
                cost_usd=meta.get("cost_usd", 0.0),
            )
        return LLMResult(text=text, provider=name, used=bool(text))

    prov = (provider or "auto").lower()
    if prov == "none":
        return done("none", "")

    if prov in ("auto", "openai"):
        # Use GPT-5 with high reasoning effort for review via Responses API
        # Responses API supports streaming with the stream parameter
        t, meta = _call_openai_responses_api(
            prompt,
            model="gpt-5",
            reasoning_effort="high",
            stream_callback=stream_callback,
            log_dir=log_dir,
            use_cache=use_cache,
        )
        if t:
            return done("openai", t, meta)

    if prov in ("auto", "anthropic"):
        # Use Sonnet for review (best balance of quality/cost)
        if not model:
            model = "claude-3-5-sonnet-20241022"
        t, meta = _call_anthropic_api(
            prompt,
            model=model,
            stream_callback=stream_callback,
            log_dir=log_dir,
            use_cache=use_cache,
        )
        if t:
            return done("anthropic", t, meta)

        t = _call_anthropic_cli(prompt)
        if t:
            return done("anthropic_cli", t, {"model": "claude-cli", "provider": "anthropic"})

    if prov in ("auto", "amp"):
        t = _call_amp_cli(prompt)
        if t:
            return done("amp_cli", t, {"model": "amp-cli", "provider": "amp"})

    return done("none", "")


def review_solver_options(
    qa_check: str,
    run_log: str,
    lst_content: str,
    opt_content: str,
    solver: str = "cplex",
    provider: str = "auto",
    model: str = "",
    stream_callback: Callable[[str], None] | None = None,
    log_dir: Path | None = None,
    use_cache: bool = True,
) -> LLMResult:
    from .prompts import build_solver_options_review_prompt
    from .solver_models import SolverDiagnosis

    instructions, input_data = build_solver_options_review_prompt(
        qa_check, run_log, lst_content, opt_content, solver
    )

    def done(
        name: str, text: str, meta: dict[str, Any] | None = None, structured: Any = None
    ) -> LLMResult:
        # Log the call (use structured output as JSON if available)
        log_meta: dict[str, Any] = meta if meta else {"provider": name}
        log_text = text if not structured else structured.model_dump_json(indent=2)
        log_llm_call("solver_options_review", input_data, log_text, log_meta, log_dir)

        if meta:
            return LLMResult(
                text=text,
                provider=name,
                used=bool(text),
                model=meta.get("model", ""),
                input_tokens=meta.get("input_tokens", 0),
                output_tokens=meta.get("output_tokens", 0),
                cost_usd=meta.get("cost_usd", 0.0),
            )
        return LLMResult(text=text, provider=name, used=bool(text))

    # Only OpenAI supported for structured output
    prov = (provider or "auto").lower()

    if prov not in ("auto", "openai"):
        raise ValueError(
            "review-solver-options only supports OpenAI provider with structured output. "
            "Please set OPENAI_API_KEY and use --llm openai or auto (default)"
        )

    # Use GPT-5 with high reasoning effort for solver option review with structured output
    result, meta = _call_openai_responses_api(
        input_data,
        model="gpt-5",
        reasoning_effort="high",
        stream_callback=stream_callback,
        log_dir=log_dir,
        use_cache=use_cache,
        instructions=instructions,
        text_format=SolverDiagnosis,
    )

    if not result:
        return done("none", "")

    # Handle both structured and text responses
    if isinstance(result, SolverDiagnosis):
        # Convert structured output to text for backward compatibility
        text = result.model_dump_json(indent=2)
        return done("openai", text, meta, structured=result)
    else:
        return done("openai", result, meta)


def review_qa_check_fixes(
    prompt: str,
    provider: str = "auto",
    model: str = "",
    stream_callback: Callable[[str], None] | None = None,
    log_dir: Path | None = None,
    use_cache: bool = True,
) -> LLMResult:
    """Review QA_CHECK.LOG and provide actionable fix recommendations using oracle.

    Args:
        prompt: Complete prompt with QA issues, run_log, and lst excerpts
        provider: LLM provider (only 'openai' or 'auto' supported for reasoning)
        model: Specific model override (default: gpt-5 with medium reasoning)
        stream_callback: Optional callback for streaming output
        log_dir: Directory for LLM call logs
        use_cache: Whether to use LLM response cache

    Returns:
        LLMResult with fix recommendations text and metadata
    """

    def done(name: str, text: str, meta: dict[str, Any] | None = None) -> LLMResult:
        # Log the call
        log_meta: dict[str, Any] = meta if meta else {"provider": name}
        log_llm_call("review_qa_check_fixes", prompt, text, log_meta, log_dir)

        if meta:
            return LLMResult(
                text=text,
                provider=name,
                used=bool(text),
                model=meta.get("model", ""),
                input_tokens=meta.get("input_tokens", 0),
                output_tokens=meta.get("output_tokens", 0),
                cost_usd=meta.get("cost_usd", 0.0),
            )
        return LLMResult(text=text, provider=name, used=bool(text))

    # Only OpenAI supported for reasoning
    prov = (provider or "auto").lower()

    if prov not in ("auto", "openai"):
        raise ValueError(
            "review-qa-check only supports OpenAI provider with reasoning. "
            "Please set OPENAI_API_KEY and use --llm openai or auto (default)"
        )

    # Use GPT-5 with medium reasoning effort for QA fix recommendations
    result, meta = _call_openai_responses_api(
        prompt,
        model=model or "gpt-5",
        reasoning_effort="medium",
        stream_callback=stream_callback,
        log_dir=log_dir,
        use_cache=use_cache,
        instructions="You are the Oracle, a TIMES/Veda QA expert. Return only actionable remediation steps per QA issue with file references and fix instructions.",
    )

    if not result:
        return done("none", "")

    return done("openai", result, meta)
