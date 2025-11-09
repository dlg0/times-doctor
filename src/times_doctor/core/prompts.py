import hashlib
import json
import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)


def _find_prompts_dir() -> Path:
    """Find the prompts/ directory relative to this module."""
    base = Path(__file__).resolve()
    for parent in [base.parent.parent.parent, *base.parents]:
        prompts_dir = parent / "prompts"
        if prompts_dir.exists() and prompts_dir.is_dir():
            return prompts_dir
    raise FileNotFoundError("Could not locate prompts/ directory")


def _validate_placeholders(content: str, required: set[str]) -> tuple[set[str], set[str]]:
    """Validate that template contains exactly the required placeholders.
    
    Returns:
        (missing, extra) sets of placeholder names
    """
    found = set(re.findall(r'\{(\w+)\}', content))
    missing = required - found
    extra = found - required
    return missing, extra


def load_prompt_template(
    name: str,
    version: str | None = None,
    required_placeholders: set[str] | None = None,
    strict_hash: bool | None = None
) -> str | None:
    """Load a prompt template with versioning and validation.
    
    Args:
        name: Prompt template name (e.g., 'qa_check_compress')
        version: Specific version to load (e.g., 'v1'). If None, uses manifest current.
        required_placeholders: Set of placeholder names that must be in template (e.g., {'file_type'})
        strict_hash: Enforce SHA256 checksum validation. None = use env var PROMPT_STRICT_HASH (default True in CI)
    
    Returns:
        Template content or None if not found (caller should use fallback)
    
    Raises:
        ValueError: If placeholders don't match required set or checksum fails in strict mode
    """
    if strict_hash is None:
        strict_hash = os.getenv("PROMPT_STRICT_HASH", "1") == "1"
    
    # Check for environment override (e.g., PROMPT_VERSION_QA_CHECK_COMPRESS=v2)
    env_key = f"PROMPT_VERSION_{name.upper()}".replace("-", "_")
    version_override = os.getenv(env_key)
    if version_override:
        version = version_override
        logger.info(f"Prompt version override via {env_key}: {version}")
    
    try:
        prompts_dir = _find_prompts_dir()
    except FileNotFoundError:
        logger.warning(f"Prompts directory not found for {name}")
        return None
    
    manifest_path = prompts_dir / "manifest.json"
    expected_hash = None
    
    # Load manifest to get current version and hash
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            prompt_config = manifest.get(name, {})
            
            if version is None:
                version = prompt_config.get("current")
            
            if version:
                version_info = prompt_config.get("versions", {}).get(version, {})
                expected_hash = version_info.get("sha256")
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to read manifest for {name}: {e}")
    
    # If still no version, try to find latest vN
    if not version:
        prompt_dir = prompts_dir / name
        if prompt_dir.exists():
            versions = sorted(prompt_dir.glob("v*.txt"), reverse=True)
            if versions:
                version = versions[0].stem
                logger.info(f"Using fallback version {version} for {name}")
    
    if not version:
        logger.warning(f"No version found for prompt {name}")
        return None
    
    # Load template file
    template_path = prompts_dir / name / f"{version}.txt"
    if not template_path.exists():
        logger.warning(f"Template file not found: {template_path}")
        return None
    
    content = template_path.read_text(encoding="utf-8")
    
    # Validate checksum
    if expected_hash:
        actual_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if actual_hash != expected_hash:
            msg = f"Checksum mismatch for {name}/{version}: expected {expected_hash[:16]}..., got {actual_hash[:16]}..."
            if strict_hash:
                raise ValueError(msg)
            else:
                logger.warning(msg)
    
    # Validate placeholders
    if required_placeholders is not None:
        missing, extra = _validate_placeholders(content, required_placeholders)
        if missing or extra:
            details = []
            if missing:
                details.append(f"missing: {{{', '.join(sorted(missing))}}}")
            if extra:
                details.append(f"unexpected: {{{', '.join(sorted(extra))}}}")
            raise ValueError(f"Placeholder validation failed for {name}/{version}: {'; '.join(details)}")
    
    # Log usage for observability
    logger.debug(f"Loaded prompt: {name}/{version} (hash: {expected_hash[:8] if expected_hash else 'unknown'}...)")
    
    return content


def build_llm_prompt(diagnostics: dict) -> str:
    """Build diagnostic prompt for LP solver issues."""
    status = diagnostics.get("status", {})
    ranges = diagnostics.get("ranges", {})
    mixed = diagnostics.get("mixed_currency_files", [])
    used_barrier = diagnostics.get("used_barrier_noXO", False)

    # Build context section
    context_lines = []
    for k, v in status.items():
        context_lines.append(f"- {k}: {v}")
    
    if ranges:
        m = ranges.get("matrix")
        b = ranges.get("bound")
        r = ranges.get("rhs")
        if r:
            context_lines.append(f"- Range RHS min={r[0]:.3e}, max={r[1]:.3e}")
        if b:
            context_lines.append(f"- Range Bound min={b[0]:.3e}, max={b[1]:.3e}")
        if m:
            context_lines.append(f"- Range Matrix min={m[0]:.3e}, max={m[1]:.3e}")
    
    context_lines.append(f"- Used barrier without crossover: {used_barrier}")
    
    if mixed:
        context_lines.append("- Mixed currencies seen in: " + ", ".join(mixed))
    
    # Try to load template, fall back to inline
    template = load_prompt_template("llm_prompt", required_placeholders={"context"})
    if template:
        return template.replace("{context}", "\n".join(context_lines))
    
    # Inline fallback
    lines = []
    lines.append("You are an LP solver expert helping diagnose a TIMES/Veda run.")
    lines.append("Return a short, numbered action plan (<=12 items). Keep it concrete and tool-ready.")
    lines.append("Context:")
    lines.extend(context_lines)
    lines.append("Constraints: Focus on practical steps (e.g., switch to dual simplex, unify currency to AUD25, fix tiny coefficients).")
    return "\n".join(lines)


def build_qa_check_compress_prompt(file_content: str) -> str:
    """Build prompt for compressing QA_CHECK.LOG into concise grouped warnings/errors."""
    template = load_prompt_template("qa_check_compress")
    if not template:
        # Fallback inline version
        template = """You are analyzing a TIMES/Veda QA_CHECK.LOG file that contains warnings and errors.

Your task: Reduce this to a concise warning/error list where instead of a row for every region/tech combo for the same warning/error, just give a single row for that warning/error but indicate the set of indices it applies to.

Format each warning/error like this:

WARNING: Delayed Process but PASTInvestment.
    Regions: ADE, CAN, CQ, CVIC, LV, MEL
    Processes: EE_NatGas*, EE_RePro*, EE_Solar*

SEVERE ERROR: Defective sum of FX and UP FLO_SHAREs in Group
    Regions: NSW
    Processes: IT_Gas_Exp, IT_Pet_Ref
    Vintage: 2015
    Commodity Group: IT_Met_Pro_NRGI

Instructions:
- Group identical warnings/errors together
- List all regions, processes, commodities, vintages etc. that the warning applies to
- Use wildcards (*) where appropriate to show patterns
- Keep severity level (WARNING, ERROR, SEVERE ERROR, etc.)
- Keep the warning/error message verbatim
- Sort by severity (SEVERE ERROR first, then ERROR, then WARNING)

End your output with:

---
See QA_CHECK.LOG for full detail
"""
    
    return f"{template}\n\nFile content:\n```\n{file_content}\n```"


def build_extraction_prompt(file_content: str, file_type: str) -> str:
    """Build prompt for fast LLM to extract condensed line ranges from log files.
    
    Args:
        file_content: The full file content with line numbers
        file_type: One of 'run_log', 'lst'
    """
    template = load_prompt_template("extraction_sections", required_placeholders={"file_type"})
    if not template:
        # Fallback inline version
        template = """You are analyzing a TIMES/Veda {file_type} file to identify ONLY the useful diagnostic sections.

Your task:
1. Identify line ranges containing useful diagnostic information
2. Return a JSON object mapping section names to line ranges

INCLUDE line ranges for:
- Error messages and warnings
- Solver status and termination information
- Range statistics (min/max values, matrix statistics)
- Infeasibility or optimality information
- Unusual solver output or failures
- Summary sections at start or end

EXCLUDE line ranges for:
- Routine progress messages
- Repetitive iteration logs
- Verbose table dumps
- License/copyright boilerplate
- Duplicate information

Return ONLY valid JSON in this exact format:
{
  "sections": [
    {"name": "Error Messages", "start_line": 45, "end_line": 52},
    {"name": "Range Statistics", "start_line": 234, "end_line": 256}
  ]
}
"""
    
    prompt = template.replace("{file_type}", file_type.upper())
    return f"{prompt}\n\nFile content:\n```\n{file_content}\n```"


def build_review_prompt(qa_check: str, run_log: str, lst_content: str) -> str:
    """Build review prompt for TIMES run diagnostics."""
    template = load_prompt_template("review")
    if not template:
        # Fallback inline version
        lines = []
        lines.append("You are an expert TIMES/Veda energy model diagnostician. Review the following files from a TIMES run and provide:")
        lines.append("1. A concise human-readable summary of what is happening in this run")
        lines.append("2. Any errors, warnings, or issues detected")
        lines.append("3. A ranked list of recommended actions the user should take")
        lines.append("")
        lines.append("Focus on practical, actionable advice. Be specific about what files, settings, or data need attention.")
        template = "\n".join(lines)
    
    # Append file content sections
    sections = []
    sections.append("")
    sections.append("=== QA_CHECK.LOG (CONDENSED) ===")
    sections.append(qa_check if qa_check else "(file not found)")
    sections.append("")
    sections.append("=== RUN LOG (CONDENSED) ===")
    sections.append(run_log if run_log else "(file not found)")
    sections.append("")
    sections.append("=== LST FILE (CONDENSED EXCERPTS) ===")
    sections.append(lst_content if lst_content else "(file not found)")
    sections.append("")
    sections.append("Provide your analysis in markdown format with clear sections.")
    
    return template + "\n".join(sections)
