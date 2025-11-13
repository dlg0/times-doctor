"""Tests for solver validation integration."""

from times_doctor.core.solver_models import OptFileConfig, SolverDiagnosis
from times_doctor.core.solver_validation import (
    build_validation_feedback,
    normalize_opt_config,
    validate_solver_diagnosis,
)


class TestSolverValidation:
    """Test suite for solver validation integration."""

    def test_validate_valid_diagnosis(self):
        """Test validation of a valid SolverDiagnosis."""
        diagnosis = SolverDiagnosis(
            summary="Test diagnosis",
            opt_configurations=[
                OptFileConfig(
                    filename="test.opt",
                    description="Test config",
                    parameters=[
                        {"name": "lpmethod", "value": "4", "reason": "Barrier"},
                        {"name": "epopt", "value": "1e-7", "reason": "Tight tolerance"},
                    ],
                )
                for _ in range(10)
            ],
            action_plan=["Step 1", "Step 2", "Step 3"],
        )

        is_valid, errors = validate_solver_diagnosis(diagnosis, solver="cplex")
        assert is_valid
        assert len(errors) == 0

    def test_validate_invalid_option_names(self):
        """Test validation catches invalid option names."""
        diagnosis = SolverDiagnosis(
            summary="Test diagnosis",
            opt_configurations=[
                OptFileConfig(
                    filename="test.opt",
                    description="Test config",
                    parameters=[
                        {"name": "lpmethod", "value": "4", "reason": "Barrier"},
                        {"name": "invalidoption", "value": "1", "reason": "Bad option"},
                    ],
                )
                for _ in range(10)
            ],
            action_plan=["Step 1", "Step 2", "Step 3"],
        )

        is_valid, errors = validate_solver_diagnosis(diagnosis, solver="cplex")
        assert not is_valid
        assert len(errors) > 0
        assert any("invalidoption" in str(e).lower() for e in errors)

    def test_validate_invalid_option_values(self):
        """Test validation catches invalid option values."""
        diagnosis = SolverDiagnosis(
            summary="Test diagnosis",
            opt_configurations=[
                OptFileConfig(
                    filename="test.opt",
                    description="Test config",
                    parameters=[
                        {"name": "lpmethod", "value": "999", "reason": "Invalid value"},
                        {"name": "epopt", "value": "not_a_number", "reason": "Bad type"},
                    ],
                )
                for _ in range(10)
            ],
            action_plan=["Step 1", "Step 2", "Step 3"],
        )

        is_valid, errors = validate_solver_diagnosis(diagnosis, solver="cplex")
        assert not is_valid
        assert len(errors) > 0

    def test_validate_skips_non_cplex(self):
        """Test validation skips non-CPLEX solvers."""
        diagnosis = SolverDiagnosis(
            summary="Test diagnosis",
            opt_configurations=[
                OptFileConfig(
                    solver="gurobi",
                    filename="test.opt",
                    description="Test config",
                    parameters=[{"name": "anythinghere", "value": "123", "reason": "Whatever"}],
                )
                for _ in range(10)
            ],
            action_plan=["Step 1", "Step 2", "Step 3"],
        )

        is_valid, errors = validate_solver_diagnosis(diagnosis, solver="gurobi")
        assert is_valid
        assert len(errors) == 0

    def test_normalize_valid_config(self):
        """Test normalizing a config with valid options."""
        config = OptFileConfig(
            filename="test.opt",
            description="Test",
            parameters=[
                {"name": "lpmethod", "value": "4", "reason": "Barrier"},
                {"name": "EPOPT", "value": "1e-7", "reason": "Tight"},  # Mixed case
            ],
        )

        normalized = normalize_opt_config(config, solver="cplex")
        assert len(normalized.parameters) == 2
        param_names = [p.name for p in normalized.parameters]
        assert "lpmethod" in param_names
        assert "epopt" in param_names
        assert "EPOPT" not in param_names  # Should be normalized to lowercase

    def test_normalize_removes_invalid_options(self):
        """Test normalization removes invalid options."""
        config = OptFileConfig(
            filename="test.opt",
            description="Test",
            parameters=[
                {"name": "lpmethod", "value": "4", "reason": "Valid"},
                {"name": "invalidoption", "value": "1", "reason": "Invalid"},
                {"name": "epopt", "value": "1e-7", "reason": "Valid"},
            ],
        )

        normalized = normalize_opt_config(config, solver="cplex")
        param_names = [p.name for p in normalized.parameters]
        assert "lpmethod" in param_names
        assert "epopt" in param_names
        assert "invalidoption" not in param_names

    def test_normalize_resolves_synonyms(self):
        """Test normalization resolves synonyms to canonical names."""
        # Find a synonym in the metadata
        from times_doctor.core.cplex_validator import CplexOptionsValidator

        validator = CplexOptionsValidator()
        synonym_name = None
        canonical_name = None

        for canonical, meta in validator.metadata.items():
            if "synonyms" in meta and meta["synonyms"]:
                synonym_name = meta["synonyms"][0]
                canonical_name = canonical
                break

        if synonym_name:
            config = OptFileConfig(
                filename="test.opt",
                description="Test",
                parameters=[
                    {"name": synonym_name, "value": "1", "reason": "Using synonym"},
                ],
            )

            normalized = normalize_opt_config(config, solver="cplex")
            if normalized.parameters:  # May be empty if value is invalid
                param_names = [p.name for p in normalized.parameters]
                assert canonical_name in param_names or len(param_names) == 0

    def test_normalize_fixes_boolean_values(self):
        """Test normalization converts boolean values to 0/1."""
        # Find a boolean option
        from times_doctor.core.cplex_validator import CplexOptionsValidator

        validator = CplexOptionsValidator()
        bool_option = None

        for name, meta in validator.metadata.items():
            if meta.get("type") == "boolean":
                bool_option = name
                break

        if bool_option:
            config = OptFileConfig(
                filename="test.opt",
                description="Test",
                parameters=[
                    {"name": bool_option, "value": "yes", "reason": "Enable feature"},
                ],
            )

            normalized = normalize_opt_config(config, solver="cplex")
            assert len(normalized.parameters) == 1
            assert normalized.parameters[0].value in ("0", "1")

    def test_normalize_skips_non_cplex(self):
        """Test normalization skips non-CPLEX configs."""
        config = OptFileConfig(
            solver="gurobi",
            filename="test.opt",
            description="Test",
            parameters=[{"name": "anythinghere", "value": "123", "reason": "Whatever"}],
        )

        normalized = normalize_opt_config(config, solver="gurobi")
        assert normalized == config  # Should be unchanged

    def test_build_validation_feedback(self):
        """Test building validation feedback message."""
        errors = [
            "\n**test.opt**:",
            "  - invalidopt: Unknown option 'invalidopt' (try: epopt)",
            "  - lpmethod: Value 999 not allowed",
        ]

        feedback = build_validation_feedback(errors)
        assert "VALIDATION ERRORS" in feedback
        assert "invalidopt" in feedback
        assert "lpmethod" in feedback
        assert "fix these errors" in feedback.lower()

    def test_normalize_preserves_metadata(self):
        """Test that normalization preserves filename and description."""
        config = OptFileConfig(
            filename="my_config.opt",
            description="Custom description",
            parameters=[{"name": "lpmethod", "value": "4", "reason": "Barrier"}],
        )

        normalized = normalize_opt_config(config, solver="cplex")
        assert normalized.filename == "my_config.opt"
        assert normalized.description == "Custom description"

    def test_normalize_preserves_reasons(self):
        """Test that normalization preserves parameter reasons."""
        config = OptFileConfig(
            filename="test.opt",
            description="Test",
            parameters=[
                {"name": "lpmethod", "value": "4", "reason": "Use barrier for large LPs"},
                {"name": "epopt", "value": "1e-7", "reason": "Tight optimality tolerance"},
            ],
        )

        normalized = normalize_opt_config(config, solver="cplex")
        reasons = [p.reason for p in normalized.parameters]
        assert "Use barrier for large LPs" in reasons
        assert "Tight optimality tolerance" in reasons

    def test_validation_with_multiple_configs(self):
        """Test validation handles multiple configurations."""
        diagnosis = SolverDiagnosis(
            summary="Test",
            opt_configurations=[
                OptFileConfig(
                    filename=f"config_{i}.opt",
                    description=f"Config {i}",
                    parameters=[
                        {"name": "lpmethod", "value": "4", "reason": "Valid"},
                        {"name": f"invalid_{i}", "value": "1", "reason": "Invalid"},
                    ],
                )
                for i in range(10)
            ],
            action_plan=["Step 1", "Step 2", "Step 3"],
        )

        is_valid, errors = validate_solver_diagnosis(diagnosis, solver="cplex")
        assert not is_valid
        # Should have errors for all 10 configs
        assert len([e for e in errors if "config_" in str(e)]) >= 10
