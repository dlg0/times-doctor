import os, subprocess, json
from pathlib import Path
from datetime import datetime

def which(cmd: str):
    from shutil import which as _w
    return _w(cmd)

def log_llm_call(call_type: str, prompt: str, response: str, metadata: dict, log_dir: Path = None):
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
    
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "call_type": call_type,
        "metadata": metadata,
        "prompt": prompt,
        "response": response,
        "prompt_length": len(prompt),
        "response_length": len(response)
    }
    
    try:
        with open(log_file, 'w', encoding='utf-8') as f:
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
    load_env()
    return {
        "openai": bool(os.environ.get("OPENAI_API_KEY")),
        "anthropic": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "amp": bool(os.environ.get("AMP_API_KEY") or which(os.environ.get("AMP_CLI", "amp")))
    }

def list_openai_models() -> list[str]:
    """List available OpenAI models for chat completions."""
    key = os.environ.get("OPENAI_API_KEY","")
    if not key:
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
    models.extend([
        "gpt-5-chat (non-reasoning)",
        "gpt-5-mini-chat (non-reasoning)",
        "gpt-5-nano-chat (non-reasoning)"
    ])
    
    return models

def list_anthropic_models() -> list[str]:
    """List available Anthropic models."""
    return [
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022", 
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307"
    ]

class LLMResult:
    def __init__(self, text: str, provider: str, used: bool, model: str = "", input_tokens: int = 0, output_tokens: int = 0, cost_usd: float = 0.0):
        self.text = text
        self.provider = provider
        self.used = used
        self.model = model
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cost_usd = cost_usd

def _call_cli(cli: str, prompt: str) -> str:
    try:
        p = subprocess.run([cli], input=prompt.encode('utf-8'), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
        return p.stdout.decode('utf-8', errors='ignore').strip()
    except Exception:
        return ""

def _call_openai_api(prompt: str, model: str = "", stream_callback=None) -> tuple[str, dict]:
    key = os.environ.get("OPENAI_API_KEY","")
    if not key:
        return "", {}
    try:
        import httpx
    except Exception:
        if which("openai"):
            return _call_cli("openai", prompt), {"model": "openai-cli", "provider": "openai-cli"}
        return "", {}
    
    if not model:
        model = os.environ.get("OPENAI_MODEL","gpt-5-mini")
    
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
            {"role": "user", "content": prompt}
        ],
        "stream": True if stream_callback else False
    }
    
    # Add reasoning parameter for reasoning models
    if reasoning_effort:
        payload["reasoning"] = {"effort": reasoning_effort}
    
    # GPT-5 models only support temperature=1 (default), don't include it
    # Older models support configurable temperature
    if not base_model.startswith("gpt-5"):
        temperature = float(os.environ.get("OPENAI_TEMPERATURE", "0.2"))
        payload["temperature"] = temperature
    
    try:
        # Longer timeout for reasoning models (GPT-5, etc.)
        timeout_seconds = 300 if base_model.startswith("gpt-5") or "pro" in base_model else 120
        
        if stream_callback:
            # Try streaming mode first
            try:
                full_text = ""
                input_tokens = 0
                output_tokens = 0
                
                with httpx.stream("POST", url, headers=headers, json=payload, timeout=timeout_seconds) as r:
                    if r.status_code != 200:
                        # Fall back to non-streaming on error
                        from rich import print as rprint
                        rprint("\n[yellow]⚠ Streaming not available (requires org verification), falling back to non-streaming mode...[/yellow]")
                        payload["stream"] = False
                        return _call_openai_api(prompt, model=model, stream_callback=None)
                
                    for line in r.iter_lines():
                        if not line or line == "data: [DONE]":
                            continue
                        if line.startswith("data: "):
                            try:
                                import json
                                chunk = json.loads(line[6:])
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
            except Exception as e:
                # Fall back to non-streaming on any error
                from rich import print as rprint
                rprint(f"\n[yellow]⚠ Streaming error, falling back to non-streaming mode...[/yellow]")
                payload["stream"] = False
                return _call_openai_api(prompt, model=model, stream_callback=None)
            
            # Calculate cost
            cost_per_1k_input = {
                "gpt-5": 0.005, "gpt-5-pro": 0.01, "gpt-5-mini": 0.0005, "gpt-5-nano": 0.0001,
                "gpt-4o": 0.0025, "gpt-4o-mini": 0.00015, "gpt-4-turbo": 0.01, "gpt-4": 0.03, "gpt-3.5-turbo": 0.0005
            }
            cost_per_1k_output = {
                "gpt-5": 0.015, "gpt-5-pro": 0.03, "gpt-5-mini": 0.0015, "gpt-5-nano": 0.0004,
                "gpt-4o": 0.01, "gpt-4o-mini": 0.0006, "gpt-4-turbo": 0.03, "gpt-4": 0.06, "gpt-3.5-turbo": 0.0015
            }
            
            model_key = model
            for key_prefix in cost_per_1k_input.keys():
                if model.startswith(key_prefix):
                    model_key = key_prefix
                    break
            
            input_cost = input_tokens / 1000 * cost_per_1k_input.get(model_key, 0.0005)
            output_cost = output_tokens / 1000 * cost_per_1k_output.get(model_key, 0.0015)
            
            metadata = {
                "model": model,
                "temperature": 1.0 if base_model.startswith("gpt-5") else float(os.environ.get("OPENAI_TEMPERATURE", "0.2")),
                "provider": "openai",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "cost_usd": input_cost + output_cost
            }
            if reasoning_effort:
                metadata["reasoning_effort"] = reasoning_effort
            
            return full_text.strip(), metadata
        else:
            # Non-streaming mode (original behavior)
            r = httpx.post(url, headers=headers, json=payload, timeout=timeout_seconds)
            if r.status_code == 200:
                data = r.json()
                usage = data.get("usage", {})
                
                # Calculate cost (pricing as of 2025)
                cost_per_1k_input = {
                    "gpt-5": 0.005, "gpt-5-pro": 0.01, "gpt-5-mini": 0.0005, "gpt-5-nano": 0.0001,
                    "gpt-4o": 0.0025, "gpt-4o-mini": 0.00015, "gpt-4-turbo": 0.01, "gpt-4": 0.03, "gpt-3.5-turbo": 0.0005
                }
                cost_per_1k_output = {
                    "gpt-5": 0.015, "gpt-5-pro": 0.03, "gpt-5-mini": 0.0015, "gpt-5-nano": 0.0004,
                    "gpt-4o": 0.01, "gpt-4o-mini": 0.0006, "gpt-4-turbo": 0.03, "gpt-4": 0.06, "gpt-3.5-turbo": 0.0015
                }
                
                model_base = model.split("-")[0:2]
                model_key = "-".join(model_base) if len(model_base) >= 2 else model
                for key_prefix in cost_per_1k_input.keys():
                    if model.startswith(key_prefix):
                        model_key = key_prefix
                        break
                
                input_cost = usage.get("prompt_tokens", 0) / 1000 * cost_per_1k_input.get(model_key, 0.0005)
                output_cost = usage.get("completion_tokens", 0) / 1000 * cost_per_1k_output.get(model_key, 0.0015)
                
                metadata = {
                    "model": model,
                    "temperature": 1.0 if base_model.startswith("gpt-5") else float(os.environ.get("OPENAI_TEMPERATURE", "0.2")),
                    "provider": "openai",
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                    "cost_usd": input_cost + output_cost
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
                        error_msg = f"{error_msg}: {error_data['error'].get('message', 'Unknown error')}"
                except:
                    pass
                print(f"[dim]{error_msg}[/dim]")
                return "", {}
    except Exception as e:
        print(f"[dim]OpenAI API exception: {str(e)}[/dim]")
        return "", {}

def _call_anthropic_api(prompt: str, model: str = "") -> tuple[str, dict]:
    """Call Anthropic API directly."""
    key = os.environ.get("ANTHROPIC_API_KEY","")
    if not key:
        return "", {}
    
    try:
        import httpx
    except Exception:
        return "", {}
    
    if not model:
        model = "claude-3-5-sonnet-20241022"
    temperature = float(os.environ.get("ANTHROPIC_TEMPERATURE", "0.2"))
    
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    payload = {
        "model": model,
        "max_tokens": 4096,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}]
    }
    
    try:
        # Longer timeout for reasoning models
        timeout_seconds = 300 if "opus" in model or "sonnet" in model else 120
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
                "claude-3-haiku-20240307": (0.00025, 0.00125)
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
                "cost_usd": input_cost + output_cost
            }
            
            content = data.get("content", [])
            if content and len(content) > 0:
                return content[0].get("text", ""), metadata
        return "", {}
    except Exception:
        return "", {}

def _call_anthropic_cli(prompt: str) -> str:
    cli = os.environ.get("ANTHROPIC_CLI") or "claude"
    if which(cli):
        return _call_cli(cli, prompt)
    return ""

def _call_amp_cli(prompt: str) -> str:
    cli = os.environ.get("AMP_CLI") or "amp"
    if which(cli):
        return _call_cli(cli, prompt)
    return ""

def summarize(diagnostics: dict, provider: str = "auto") -> LLMResult:
    from .prompts import build_llm_prompt
    prompt = build_llm_prompt(diagnostics)

    def done(name, text, meta=None):
        if meta:
            return LLMResult(
                text=text, 
                provider=name, 
                used=bool(text),
                model=meta.get("model", ""),
                input_tokens=meta.get("input_tokens", 0),
                output_tokens=meta.get("output_tokens", 0),
                cost_usd=meta.get("cost_usd", 0.0)
            )
        return LLMResult(text=text, provider=name, used=bool(text))

    prov = (provider or "auto").lower()
    if prov == "none":
        return done("none", "")

    if prov in ("auto","openai"):
        t, meta = _call_openai_api(prompt)
        if t: return done("openai", t, meta)

    if prov in ("auto","anthropic"):
        t = _call_anthropic_cli(prompt)
        if t: return done("anthropic_cli", t, {"model": "claude-cli", "provider": "anthropic"})

    if prov in ("auto","amp"):
        t = _call_amp_cli(prompt)
        if t: return done("amp_cli", t, {"model": "amp-cli", "provider": "amp"})

    return done("none", "")

def extract_useful_sections(file_content: str, file_type: str, log_dir: Path = None) -> dict:
    """Extract useful diagnostic sections from a log file using fast LLM.
    
    Args:
        file_content: Full file content
        file_type: One of 'qa_check', 'run_log', 'lst'
        log_dir: Directory to save LLM call logs
    
    Returns:
        dict with 'sections' list of {name, start_line, end_line}
    
    Raises:
        RuntimeError: If extraction fails
    """
    from .prompts import build_extraction_prompt
    import re
    
    # Add line numbers to content
    lines = file_content.split('\n')
    numbered_content = '\n'.join(f"{i+1}: {line}" for i, line in enumerate(lines))
    
    prompt = build_extraction_prompt(numbered_content, file_type)
    
    # Determine which fast model to use
    api_keys = check_api_keys()
    
    text = ""
    meta = {}
    error_msg = ""
    
    if api_keys["openai"]:
        # Use fast GPT-5 chat model (non-reasoning)
        text, meta = _call_openai_api(prompt, model="gpt-5-mini-chat")
    elif api_keys["anthropic"]:
        # Use Haiku (fastest Anthropic model)
        text, meta = _call_anthropic_api(prompt, model="claude-3-5-haiku-20241022")
    else:
        error_msg = "No OpenAI or Anthropic API key found. Extraction requires OPENAI_API_KEY or ANTHROPIC_API_KEY in .env file"
        log_llm_call(f"extraction_{file_type}", prompt, "", {"error": error_msg}, log_dir)
        raise RuntimeError(error_msg)
    
    # Log the call
    log_llm_call(f"extraction_{file_type}", prompt, text, meta, log_dir)
    
    if not text:
        error_msg = f"Failed to extract useful sections from {file_type}. LLM returned empty response."
        raise RuntimeError(error_msg)
    
    # Parse JSON response
    # Try to extract JSON from response (in case LLM wrapped it in markdown)
    json_match = re.search(r'\{[\s\S]*"sections"[\s\S]*\}', text)
    if json_match:
        json_str = json_match.group(0)
    else:
        json_str = text
    
    try:
        result = json.loads(json_str)
        if "sections" not in result or not isinstance(result["sections"], list):
            raise ValueError("Invalid JSON structure")
        return result
    except Exception as e:
        error_msg = f"Failed to parse extraction JSON from {file_type}: {e}\nLLM response: {text[:500]}"
        raise RuntimeError(error_msg)

def create_useful_markdown(file_lines: list[str], sections: list[dict], file_type: str) -> str:
    """Create markdown file with extracted useful sections.
    
    Args:
        file_lines: Original file content as list of lines
        sections: List of {name, start_line, end_line} dicts
        file_type: One of 'qa_check', 'run_log', 'lst'
    
    Returns:
        Markdown formatted string
    """
    md_lines = []
    
    # Title
    file_name_map = {
        'qa_check': 'QA_CHECK.LOG',
        'run_log': 'Run Log',
        'lst': 'LST File'
    }
    md_lines.append(f"# {file_name_map.get(file_type, file_type)} - Useful Sections")
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

def review_files(qa_check: str, run_log: str, lst_content: str, provider: str = "auto", model: str = "", stream_callback=None, log_dir: Path = None) -> LLMResult:
    from .prompts import build_review_prompt
    prompt = build_review_prompt(qa_check, run_log, lst_content)

    def done(name, text, meta=None):
        # Log the call
        log_meta = meta if meta else {"provider": name}
        log_llm_call("review", prompt, text, log_meta, log_dir)
        
        if meta:
            return LLMResult(
                text=text, 
                provider=name, 
                used=bool(text),
                model=meta.get("model", ""),
                input_tokens=meta.get("input_tokens", 0),
                output_tokens=meta.get("output_tokens", 0),
                cost_usd=meta.get("cost_usd", 0.0)
            )
        return LLMResult(text=text, provider=name, used=bool(text))

    prov = (provider or "auto").lower()
    if prov == "none":
        return done("none", "")

    if prov in ("auto","openai"):
        t, meta = _call_openai_api(prompt, model=model, stream_callback=stream_callback)
        if t: return done("openai", t, meta)

    if prov in ("auto","anthropic"):
        # Try API first, fall back to CLI
        t, meta = _call_anthropic_api(prompt, model=model)
        if t: return done("anthropic", t, meta)
        
        t = _call_anthropic_cli(prompt)
        if t: return done("anthropic_cli", t, {"model": "claude-cli", "provider": "anthropic"})

    if prov in ("auto","amp"):
        t = _call_amp_cli(prompt)
        if t: return done("amp_cli", t, {"model": "amp-cli", "provider": "amp"})

    return done("none", "")
