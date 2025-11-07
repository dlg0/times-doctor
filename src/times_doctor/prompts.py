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
