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
