"""Solver option validation and correction for LLM-generated configurations."""

from .cplex_validator import CplexOptionsValidator
from .solver_models import OptFileConfig, SolverDiagnosis


def validate_solver_diagnosis(
    diagnosis: SolverDiagnosis, solver: str = "cplex"
) -> tuple[bool, list[str]]:
    """Validate all opt configurations in a SolverDiagnosis.

    Args:
        diagnosis: SolverDiagnosis from LLM with opt configurations
        solver: Solver type ('cplex' or 'gurobi')

    Returns:
        (is_valid, error_messages) where error_messages lists all validation issues
    """
    if solver != "cplex":
        return True, []

    validator = CplexOptionsValidator()
    all_errors = []

    for config in diagnosis.opt_configurations:
        options = {param.name: param.value for param in config.parameters}
        result = validator.validate(options)

        if not result.is_valid:
            all_errors.append(f"\n**{config.filename}**:")
            for error in result.errors:
                msg = f"  - {error.key}: {error.message}"
                if error.suggested_fix:
                    msg += f" (try: {error.suggested_fix})"
                all_errors.append(msg)

    return len(all_errors) == 0, all_errors


def normalize_opt_config(config: OptFileConfig, solver: str = "cplex") -> OptFileConfig:
    """Normalize option names and values in an OptFileConfig.

    Args:
        config: OptFileConfig with potentially non-canonical names/values
        solver: Solver type ('cplex' or 'gurobi')

    Returns:
        New OptFileConfig with normalized options (only valid ones)
    """
    if solver != "cplex":
        return config

    validator = CplexOptionsValidator()
    options = {param.name: param.value for param in config.parameters}
    result = validator.validate(options)

    # Build new parameters list with only valid, normalized options
    from .solver_models import OptParameter

    normalized_params = []
    for param in config.parameters:
        # Find the canonical name
        canonical, _ = validator._resolve_name(param.name)
        if canonical and canonical in result.normalized_options:
            normalized_params.append(
                OptParameter(
                    name=canonical,
                    value=str(result.normalized_options[canonical]),
                    reason=param.reason,
                )
            )

    return OptFileConfig(
        solver=config.solver,
        filename=config.filename,
        description=config.description,
        parameters=normalized_params,
    )


def build_validation_feedback(error_messages: list[str]) -> str:
    """Build feedback message for LLM when validation fails.

    Args:
        error_messages: List of validation error messages

    Returns:
        Formatted feedback string to send to LLM
    """
    feedback = [
        "VALIDATION ERRORS - The following CPLEX options are invalid:\n",
        *error_messages,
        "\nPlease fix these errors and regenerate the configurations with valid CPLEX options only.",
        "Refer to the CPLEX options metadata for valid option names, types, and value ranges.",
    ]
    return "\n".join(feedback)
