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

def build_extraction_prompt(file_content: str, file_type: str) -> str:
    """Build prompt for fast LLM to extract useful line ranges from log files.
    
    Args:
        file_content: The full file content with line numbers
        file_type: One of 'qa_check', 'run_log', 'lst'
    """
    lines = []
    lines.append(f"You are analyzing a TIMES/Veda {file_type.upper()} file to identify ONLY the useful diagnostic sections.")
    lines.append("")
    lines.append("Your task:")
    lines.append("1. Identify line ranges containing useful diagnostic information")
    lines.append("2. Return a JSON object mapping section names to line ranges")
    lines.append("")
    lines.append("INCLUDE line ranges for:")
    lines.append("- Error messages and warnings")
    lines.append("- Solver status and termination information")
    lines.append("- Range statistics (min/max values, matrix statistics)")
    lines.append("- Infeasibility or optimality information")
    lines.append("- Unusual solver output or failures")
    lines.append("- Summary sections at start or end")
    lines.append("")
    lines.append("EXCLUDE line ranges for:")
    lines.append("- Routine progress messages")
    lines.append("- Repetitive iteration logs")
    lines.append("- Verbose table dumps")
    lines.append("- License/copyright boilerplate")
    lines.append("- Duplicate information")
    lines.append("")
    lines.append("Return ONLY valid JSON in this exact format:")
    lines.append("{")
    lines.append('  "sections": [')
    lines.append('    {"name": "Error Messages", "start_line": 45, "end_line": 52},')
    lines.append('    {"name": "Range Statistics", "start_line": 234, "end_line": 256}')
    lines.append("  ]")
    lines.append("}")
    lines.append("")
    lines.append("File content:")
    lines.append("```")
    lines.append(file_content)
    lines.append("```")
    return "\n".join(lines)

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
