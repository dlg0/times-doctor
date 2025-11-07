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
    def __init__(self, text: str, provider: str, used: bool):
        self.text = text
        self.provider = provider
        self.used = used

def _call_cli(cli: str, prompt: str) -> str:
    try:
        p = subprocess.run([cli], input=prompt.encode('utf-8'), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
        return p.stdout.decode('utf-8', errors='ignore').strip()
    except Exception:
        return ""

def _call_openai_api(prompt: str) -> str:
    key = os.environ.get("OPENAI_API_KEY","")
    if not key:
        return ""
    try:
        import httpx
    except Exception:
        if which("openai"):
            return _call_cli("openai", prompt)
        return ""
    model = os.environ.get("OPENAI_MODEL","gpt-4o-mini")
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {key}"}
    payload = {"model": model, "messages":[{"role":"system","content":"You are a concise LP solver expert."},
                                           {"role":"user","content": prompt}], "temperature": 0.2}
    try:
        r = httpx.post(url, headers=headers, json=payload, timeout=60)
        if r.status_code == 200:
            data = r.json()
            return data["choices"][0]["message"]["content"].strip()
        return ""
    except Exception:
        return ""

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

    def done(name, text):
        return LLMResult(text=text, provider=name, used=bool(text))

    prov = (provider or "auto").lower()
    if prov == "none":
        return done("none", "")

    if prov in ("auto","openai"):
        t = _call_openai_api(prompt)
        if t: return done("openai", t)

    if prov in ("auto","anthropic"):
        t = _call_anthropic_cli(prompt)
        if t: return done("anthropic_cli", t)

    if prov in ("auto","amp"):
        t = _call_amp_cli(prompt)
        if t: return done("amp_cli", t)

    return done("none", "")

def review_files(qa_check: str, run_log: str, lst_content: str, provider: str = "auto") -> LLMResult:
    from .prompts import build_review_prompt
    prompt = build_review_prompt(qa_check, run_log, lst_content)

    def done(name, text):
        return LLMResult(text=text, provider=name, used=bool(text))

    prov = (provider or "auto").lower()
    if prov == "none":
        return done("none", "")

    if prov in ("auto","openai"):
        t = _call_openai_api(prompt)
        if t: return done("openai", t)

    if prov in ("auto","anthropic"):
        t = _call_anthropic_cli(prompt)
        if t: return done("anthropic_cli", t)

    if prov in ("auto","amp"):
        t = _call_amp_cli(prompt)
        if t: return done("amp_cli", t)

    return done("none", "")
