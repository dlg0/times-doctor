import os, subprocess
from pathlib import Path

def which(cmd: str):
    from shutil import which as _w
    return _w(cmd)

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

def _call_openai_api(prompt: str) -> tuple[str, dict]:
    key = os.environ.get("OPENAI_API_KEY","")
    if not key:
        return "", {}
    try:
        import httpx
    except Exception:
        if which("openai"):
            return _call_cli("openai", prompt), {"model": "openai-cli", "provider": "openai-cli"}
        return "", {}
    
    model = os.environ.get("OPENAI_MODEL","gpt-4o-mini")
    temperature = float(os.environ.get("OPENAI_TEMPERATURE", "0.2"))
    
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {key}"}
    payload = {
        "model": model, 
        "messages": [
            {"role": "system", "content": "You are a concise LP solver expert."},
            {"role": "user", "content": prompt}
        ], 
        "temperature": temperature
    }
    
    try:
        r = httpx.post(url, headers=headers, json=payload, timeout=60)
        if r.status_code == 200:
            data = r.json()
            usage = data.get("usage", {})
            
            # Calculate cost (pricing as of 2024)
            cost_per_1k_input = {"gpt-4o": 0.0025, "gpt-4o-mini": 0.00015, "gpt-4-turbo": 0.01, "gpt-3.5-turbo": 0.0005}
            cost_per_1k_output = {"gpt-4o": 0.01, "gpt-4o-mini": 0.0006, "gpt-4-turbo": 0.03, "gpt-3.5-turbo": 0.0015}
            
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
                "temperature": temperature,
                "provider": "openai",
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "cost_usd": input_cost + output_cost
            }
            
            return data["choices"][0]["message"]["content"].strip(), metadata
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

def review_files(qa_check: str, run_log: str, lst_content: str, provider: str = "auto") -> LLMResult:
    from .prompts import build_review_prompt
    prompt = build_review_prompt(qa_check, run_log, lst_content)

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
