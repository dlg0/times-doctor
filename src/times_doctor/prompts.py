from pathlib import Path

def _load_prompt_template(name: str) -> str:
    """Load a prompt template from the prompts/ directory."""
    # Try relative to this module first
    prompt_file = Path(__file__).parent.parent.parent / "prompts" / f"{name}.txt"
    if prompt_file.exists():
        return prompt_file.read_text(encoding="utf-8")
    
    # Fallback to inline version if file not found
    return None

def build_llm_prompt(diagnostics: dict) -> str:
    status = diagnostics.get("status", {})
    ranges = diagnostics.get("ranges", {})
    mixed = diagnostics.get("mixed_currency_files", [])
    used_barrier = diagnostics.get("used_barrier_noXO", False)

    lines = []
    lines.append("You are an LP solver expert helping diagnose a TIMES/Veda run.")
    lines.append("Return a short, numbered action plan (<=12 items). Keep it concrete and tool-ready.")
    lines.append("Context:")
    for k,v in status.items():
        lines.append(f"- {k}: {v}")
    if ranges:
        m = ranges.get("matrix")
        b = ranges.get("bound")
        r = ranges.get("rhs")
        if r: lines.append(f"- Range RHS min={r[0]:.3e}, max={r[1]:.3e}")
        if b: lines.append(f"- Range Bound min={b[0]:.3e}, max={b[1]:.3e}")
        if m: lines.append(f"- Range Matrix min={m[0]:.3e}, max={m[1]:.3e}")
    lines.append(f"- Used barrier without crossover: {used_barrier}")
    if mixed:
        lines.append("- Mixed currencies seen in: " + ", ".join(mixed))
    lines.append("Constraints: Focus on practical steps (e.g., switch to dual simplex, unify currency to AUD25, fix tiny coefficients).")
    return "\n".join(lines)

def build_qa_check_compress_prompt(file_content: str) -> str:
    """Build prompt for compressing QA_CHECK.LOG into concise grouped warnings/errors."""
    template = _load_prompt_template("qa_check_compress")
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
    """Build prompt for fast LLM to extract useful line ranges from log files.
    
    Args:
        file_content: The full file content with line numbers
        file_type: One of 'run_log', 'lst'
    """
    template = _load_prompt_template("extraction_sections")
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
    lines = []
    lines.append("You are an expert TIMES/Veda energy model diagnostician. Review the following files from a TIMES run and provide:")
    lines.append("1. A concise human-readable summary of what is happening in this run")
    lines.append("2. Any errors, warnings, or issues detected")
    lines.append("3. A ranked list of recommended actions the user should take")
    lines.append("")
    lines.append("Focus on practical, actionable advice. Be specific about what files, settings, or data need attention.")
    lines.append("")
    lines.append("=== QA_CHECK.LOG ===")
    lines.append(qa_check[:8000] if qa_check else "(file not found)")
    lines.append("")
    lines.append("=== RUN LOG ===")
    lines.append(run_log[:8000] if run_log else "(file not found)")
    lines.append("")
    lines.append("=== LST FILE (excerpts) ===")
    lines.append(lst_content[:8000] if lst_content else "(file not found)")
    lines.append("")
    lines.append("Provide your analysis in markdown format with clear sections.")
    return "\n".join(lines)
