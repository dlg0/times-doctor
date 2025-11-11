"""Utilities for rendering .opt files with enforced algorithm settings."""

import logging

from times_doctor.core.solver_models import OptFileConfig, OptParameter

log = logging.getLogger(__name__)


def extract_solver_algorithm(cplex_opt_content: str) -> dict[str, str]:
    """Extract lpmethod and solutiontype from a cplex.opt file.

    Args:
        cplex_opt_content: Raw content of cplex.opt file

    Returns:
        Dict with 'lpmethod' and 'solutiontype' keys (empty string if not found)
    """
    result = {"lpmethod": "", "solutiontype": ""}

    if not cplex_opt_content:
        return result

    # Parse line by line, ignoring comments
    for line in cplex_opt_content.splitlines():
        # Strip inline comments (anything after $)
        if "$" in line:
            line = line.split("$")[0]

        line = line.strip()
        if not line or line.startswith("*"):
            continue

        # Split on whitespace to get parameter name and value
        parts = line.split()
        if len(parts) >= 2:
            param_name = parts[0].lower()
            param_value = parts[1]

            if param_name == "lpmethod":
                result["lpmethod"] = param_value
            elif param_name == "solutiontype":
                result["solutiontype"] = param_value

    return result


def render_opt_lines(
    config: OptFileConfig, base_algorithm: dict[str, str], warn_on_override: bool = True
) -> list[str]:
    """Render .opt file lines with enforced algorithm settings.

    This function ensures that every generated .opt file includes the base
    algorithm settings (lpmethod and solutiontype) from the original run,
    regardless of whether the LLM included them.

    Args:
        config: OptFileConfig from LLM with parameters to set
        base_algorithm: Dict with 'lpmethod' and 'solutiontype' from original cplex.opt
        warn_on_override: If True, log warning when LLM provided conflicting values

    Returns:
        List of formatted .opt file lines ready to write
    """
    lines = []

    # Add description as comment header
    if config.description:
        lines.append(f"* {config.description}")
        lines.append("*")

    # Build dict of LLM-provided parameters (case-insensitive)
    llm_params: dict[str, OptParameter] = {}
    for param in config.parameters:
        llm_params[param.name.strip().lower()] = param

    # Define required algorithm parameters with reasons
    required_params = []

    if base_algorithm.get("lpmethod"):
        lpmethod_reason = (
            "Use barrier algorithm"
            if base_algorithm["lpmethod"] == "4"
            else "Solver algorithm (from original run)"
        )
        required_params.append(
            OptParameter(name="lpmethod", value=base_algorithm["lpmethod"], reason=lpmethod_reason)
        )

    if base_algorithm.get("solutiontype"):
        solutiontype_reason = (
            "Barrier without crossover"
            if base_algorithm["solutiontype"] == "2"
            else "Solution type (from original run)"
        )
        required_params.append(
            OptParameter(
                name="solutiontype",
                value=base_algorithm["solutiontype"],
                reason=solutiontype_reason,
            )
        )

    # Inject required parameters first, checking for conflicts
    for req_param in required_params:
        param_key = req_param.name.lower()

        if param_key in llm_params:
            llm_value = llm_params[param_key].value.strip()
            req_value = req_param.value.strip()

            if llm_value != req_value and warn_on_override:
                log.warning(
                    f"{config.filename}: LLM suggested {req_param.name}={llm_value}, "
                    f"overriding to {req_value} (from original run)"
                )

        # Add parameter comment on its own line, then the parameter value
        lines.append(f"* {req_param.reason}")
        lines.append(f"{req_param.name} {req_param.value}")

    # Add remaining LLM parameters (excluding the ones we already handled)
    required_keys = {p.name.lower() for p in required_params}
    for param in config.parameters:
        key = param.name.strip().lower()
        if key not in required_keys:
            # Add parameter comment on its own line, then the parameter value
            lines.append(f"* {param.reason}")
            lines.append(f"{param.name} {param.value}")

    return lines
