"""Pydantic models for structured solver diagnosis outputs from OpenAI Responses API."""

from pydantic import BaseModel, Field


class OptParameter(BaseModel):
    """A single parameter in a CPLEX .opt file."""

    name: str = Field(description="Parameter name (e.g., 'epopt', 'eprhs')")
    value: str = Field(description="Parameter value (e.g., '1e-7', '4')")
    reason: str = Field(description="Why this parameter is set to this value")


class OptFileConfig(BaseModel):
    """A complete CPLEX .opt configuration file."""

    filename: str = Field(
        pattern=r"^[a-z][a-z0-9_]*\.opt$",
        description="Filename like 'tight_tolerances.opt' (lowercase, underscores, no number prefix)",
    )
    description: str = Field(description="Brief description of what this configuration tests")
    parameters: list[OptParameter] = Field(
        min_length=1, description="List of CPLEX parameters to set"
    )


class SolverDiagnosis(BaseModel):
    """Structured diagnosis of why solver stopped at feasible (not proven optimal)."""

    summary: str = Field(
        description="Concise explanation of why solver stopped at feasible rather than proven optimal"
    )
    opt_configurations: list[OptFileConfig] = Field(
        min_length=10,
        max_length=15,
        description="Different cplex.opt configurations to test (10-15 variants)",
    )
    action_plan: list[str] = Field(
        min_length=3,
        max_length=12,
        description="Ranked action items for user to follow",
    )
