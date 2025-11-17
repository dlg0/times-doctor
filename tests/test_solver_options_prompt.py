"""Test solver options prompt and output validation."""

import re

import pytest

from times_doctor.core.solver_models import OptFileConfig, SolverDiagnosis


def extract_opt_files_from_text(text: str) -> list[dict]:
    """Extract opt file configurations from LLM output text.

    This mimics the extraction logic in the CLI.
    """
    opt_files = []
    pattern = re.compile(
        r"===OPT_FILE:\s*(\S+)\s*\n(.*?)===END_OPT_FILE",
        re.DOTALL | re.IGNORECASE,
    )

    for match in pattern.finditer(text):
        filename = match.group(1).strip()
        content = match.group(2).strip()

        # Parse parameters
        parameters = []
        for line in content.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Parse "param value $ comment" format
            parts = line.split("$", 1)
            param_line = parts[0].strip()
            comment = parts[1].strip() if len(parts) > 1 else ""

            if param_line:
                param_parts = param_line.split(None, 1)
                if len(param_parts) == 2:
                    parameters.append(
                        {"name": param_parts[0], "value": param_parts[1], "reason": comment}
                    )

        opt_files.append({"filename": filename, "parameters": parameters})

    return opt_files


class TestSolverOptionsPrompt:
    """Tests for solver options review prompt and output validation."""

    def test_prompt_includes_lpmethod_requirement(self):
        """Verify the prompt explicitly requires lpmethod in every opt file."""
        from times_doctor.core.prompts import load_prompt_template

        template = load_prompt_template("solver_options_review")
        assert template is not None, "Solver options review prompt not found"

        # Check for explicit lpmethod requirement
        assert "lpmethod 4" in template, "Prompt should explicitly require lpmethod 4"
        assert "solutiontype 2" in template, "Prompt should explicitly require solutiontype 2"
        assert (
            "EVERY .opt file MUST include" in template or "must include" in template.lower()
        ), "Prompt should strongly emphasize lpmethod requirement"

    def test_prompt_examples_include_lpmethod(self):
        """Verify the prompt examples show lpmethod in opt files."""
        from times_doctor.core.prompts import load_prompt_template

        template = load_prompt_template("solver_options_review")
        assert template is not None

        # Extract example opt files from prompt
        opt_examples = extract_opt_files_from_text(template)

        assert len(opt_examples) > 0, "Prompt should include example opt files"

        for example in opt_examples:
            param_names = [p["name"] for p in example["parameters"]]
            assert "lpmethod" in param_names, f"Example {example['filename']} missing lpmethod"
            assert (
                "solutiontype" in param_names
            ), f"Example {example['filename']} missing solutiontype"


class TestSolverDiagnosisValidation:
    """Tests for validating LLM-generated SolverDiagnosis output."""

    def validate_opt_file_config(self, config: OptFileConfig) -> list[str]:
        """Validate a single opt file configuration.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Check filename format
        if not re.match(r"^[a-z][a-z0-9_]*\.opt$", config.filename):
            errors.append(
                f"{config.filename}: Invalid filename format (should be lowercase with underscores)"
            )

        # Check for required parameters
        param_names = [p.name.lower() for p in config.parameters]

        if "lpmethod" not in param_names:
            errors.append(f"{config.filename}: Missing required parameter 'lpmethod'")

        if "solutiontype" not in param_names:
            errors.append(f"{config.filename}: Missing required parameter 'solutiontype'")

        # Check lpmethod value (should be 4 for barrier)
        for param in config.parameters:
            if param.name.lower() == "lpmethod" and param.value != "4":
                errors.append(
                    f"{config.filename}: lpmethod should be 4 (barrier), got {param.value}"
                )

            if param.name.lower() == "solutiontype" and param.value != "2":
                errors.append(
                    f"{config.filename}: solutiontype should be 2 (no crossover), got {param.value}"
                )

        # Check that parameters have reasons
        errors.extend(
            f"{config.filename}: Parameter {param.name} missing meaningful reason"
            for param in config.parameters
            if not param.reason or len(param.reason) < 5
        )

        return errors

    def test_structured_output_schema(self):
        """Test that SolverDiagnosis schema is correctly defined."""
        # Create a minimal valid instance
        diagnosis = SolverDiagnosis(
            summary="Test summary",
            opt_configurations=[
                OptFileConfig(
                    filename="test.opt",
                    description="Test config",
                    parameters=[
                        {"name": "lpmethod", "value": "4", "reason": "Use barrier"},
                        {"name": "solutiontype", "value": "2", "reason": "No crossover"},
                    ],
                )
                for _ in range(10)  # min_length=10
            ],
            action_plan=["Step 1", "Step 2", "Step 3"],
        )

        # Should not raise
        assert diagnosis.summary == "Test summary"
        assert len(diagnosis.opt_configurations) == 10
        assert len(diagnosis.action_plan) == 3

    def test_validate_solver_diagnosis(self):
        """Test validation of a complete SolverDiagnosis object."""
        diagnosis = SolverDiagnosis(
            summary="Solver stopped at feasible due to loose tolerances",
            opt_configurations=[
                OptFileConfig(
                    filename=f"config_{i}.opt",
                    description=f"Test configuration {i}",
                    parameters=[
                        {"name": "lpmethod", "value": "4", "reason": "Use barrier algorithm"},
                        {
                            "name": "solutiontype",
                            "value": "2",
                            "reason": "Barrier without crossover",
                        },
                        {
                            "name": "epopt",
                            "value": "1e-7",
                            "reason": "Tighter optimality tolerance",
                        },
                    ],
                )
                for i in range(10)
            ],
            action_plan=["Test tolerances", "Review results", "Iterate"],
        )

        # Validate each opt file
        all_errors = []
        for config in diagnosis.opt_configurations:
            errors = self.validate_opt_file_config(config)
            all_errors.extend(errors)

        assert len(all_errors) == 0, f"Validation errors: {all_errors}"

    def test_detect_missing_lpmethod(self):
        """Test that validator catches missing lpmethod."""
        config = OptFileConfig(
            filename="missing_lpmethod.opt",
            description="Test",
            parameters=[{"name": "epopt", "value": "1e-7", "reason": "Tighter tolerance"}],
        )

        errors = self.validate_opt_file_config(config)
        assert len(errors) > 0
        assert any("lpmethod" in e.lower() for e in errors)

    def test_detect_wrong_lpmethod_value(self):
        """Test that validator catches wrong lpmethod value."""
        config = OptFileConfig(
            filename="wrong_lpmethod.opt",
            description="Test",
            parameters=[
                {"name": "lpmethod", "value": "2", "reason": "Wrong - should be 4"},
                {"name": "solutiontype", "value": "2", "reason": "No crossover"},
            ],
        )

        errors = self.validate_opt_file_config(config)
        assert len(errors) > 0
        assert any("lpmethod should be 4" in e for e in errors)

    def test_detect_missing_solutiontype(self):
        """Test that validator catches missing solutiontype."""
        config = OptFileConfig(
            filename="missing_solutiontype.opt",
            description="Test",
            parameters=[{"name": "lpmethod", "value": "4", "reason": "Use barrier"}],
        )

        errors = self.validate_opt_file_config(config)
        assert len(errors) > 0
        assert any("solutiontype" in e.lower() for e in errors)


@pytest.mark.llm
class TestSolverOptionsLLMIntegration:
    """Integration tests that actually call the LLM (expensive, run manually).

    To run these tests:
        export OPENAI_API_KEY=sk-...
        pytest -m llm -xvs --no-cov

    These tests use gpt-5 with low reasoning for balance (~30s, ~$0.05 per test).
    """

    def test_llm_generates_valid_opt_files(self):
        """Test that LLM actually generates valid opt files with lpmethod.

        NOTE: This test uses GPT-5 with low reasoning for balance of speed/cost/quality.
        - gpt-5-nano was too weak to follow structured output requirements reliably
        - gpt-5 low: ~30s, ~$0.05 per test
        - Production uses gpt-5 high: ~5min, ~$0.50 per run
        """
        import os

        from times_doctor.core.llm import _call_openai_responses_api
        from times_doctor.core.prompts import build_solver_options_review_prompt
        from times_doctor.core.solver_models import SolverDiagnosis

        # Skip if no API key
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        # Build the prompt
        qa_check = "No major issues found."
        run_log = "Solver stopped at FEASIBLE."
        lst_content = "Matrix range: 1e-6 to 1e6"
        cplex_opt = "* Default CPLEX options"

        instructions, input_data = build_solver_options_review_prompt(
            qa_check, run_log, lst_content, cplex_opt
        )

        # Call LLM with moderate settings for testing
        # Using gpt-5 with low reasoning (faster/cheaper than high, but reliable)
        result, meta = _call_openai_responses_api(
            input_data,
            model="gpt-5",
            reasoning_effort="low",
            instructions=instructions,
            text_format=SolverDiagnosis,
        )

        assert result, "LLM should return result"
        assert isinstance(result, SolverDiagnosis), "Should return structured SolverDiagnosis"

        # Validate all opt files
        validator = TestSolverDiagnosisValidation()
        all_errors = []
        for config in result.opt_configurations:
            errors = validator.validate_opt_file_config(config)
            all_errors.extend(errors)

        assert len(all_errors) == 0, "LLM generated invalid opt files:\n" + "\n".join(all_errors)
