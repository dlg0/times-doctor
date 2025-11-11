"""Tests for .opt file rendering with enforced algorithm settings."""

from times_doctor.core.opt_renderer import extract_solver_algorithm, render_opt_lines
from times_doctor.core.solver_models import OptFileConfig, OptParameter


class TestExtractSolverAlgorithm:
    """Tests for extract_solver_algorithm."""

    def test_extract_both_parameters(self):
        """Extract lpmethod and solutiontype from valid cplex.opt."""
        cplex_opt = """
* CPLEX solver configuration
lpmethod 4       $ Use barrier algorithm
solutiontype 2   $ Barrier without crossover
epopt 1e-6
"""
        result = extract_solver_algorithm(cplex_opt)
        assert result["lpmethod"] == "4"
        assert result["solutiontype"] == "2"

    def test_extract_with_inline_comments(self):
        """Extract parameters that have inline $ comments."""
        cplex_opt = "lpmethod 4 $ barrier\nsolutiontype 2 $ no crossover"
        result = extract_solver_algorithm(cplex_opt)
        assert result["lpmethod"] == "4"
        assert result["solutiontype"] == "2"

    def test_extract_case_insensitive(self):
        """Extract parameters regardless of case."""
        cplex_opt = "LPMethod 4\nSOLUTIONTYPE 2"
        result = extract_solver_algorithm(cplex_opt)
        assert result["lpmethod"] == "4"
        assert result["solutiontype"] == "2"

    def test_extract_only_lpmethod(self):
        """Extract when only lpmethod is present."""
        cplex_opt = "lpmethod 2\nepopt 1e-8"
        result = extract_solver_algorithm(cplex_opt)
        assert result["lpmethod"] == "2"
        assert result["solutiontype"] == ""

    def test_extract_empty_content(self):
        """Handle empty cplex.opt content."""
        result = extract_solver_algorithm("")
        assert result["lpmethod"] == ""
        assert result["solutiontype"] == ""

    def test_extract_ignores_star_comments(self):
        """Ignore lines starting with *."""
        cplex_opt = """
* lpmethod 99 $ this is a comment
lpmethod 4
* solutiontype 99
solutiontype 2
"""
        result = extract_solver_algorithm(cplex_opt)
        assert result["lpmethod"] == "4"
        assert result["solutiontype"] == "2"


class TestRenderOptLines:
    """Tests for render_opt_lines."""

    def test_inject_base_algorithm_when_llm_omits(self):
        """Inject lpmethod and solutiontype when LLM doesn't provide them."""
        config = OptFileConfig(
            filename="tight_tolerances.opt",
            description="Tighter tolerances",
            parameters=[
                OptParameter(name="epopt", value="1e-8", reason="Tight optimality"),
                OptParameter(name="eprhs", value="1e-8", reason="Tight feasibility"),
            ],
        )
        base_algorithm = {"lpmethod": "4", "solutiontype": "2"}

        lines = render_opt_lines(config, base_algorithm, warn_on_override=False)

        # Check header
        assert "* Tighter tolerances" in lines
        assert "*" in lines

        # Check algorithm parameters (comment + value on separate lines)
        assert "* Use barrier algorithm" in lines
        assert "lpmethod 4" in lines
        assert "* Barrier without crossover" in lines
        assert "solutiontype 2" in lines

        # Check other parameters
        assert "* Tight optimality" in lines
        assert "epopt 1e-8" in lines
        assert "* Tight feasibility" in lines
        assert "eprhs 1e-8" in lines

    def test_preserve_order_algorithm_first(self):
        """Ensure algorithm parameters come before other parameters."""
        config = OptFileConfig(
            filename="test.opt",
            description="Test config",
            parameters=[
                OptParameter(name="epopt", value="1e-7", reason="First param"),
                OptParameter(name="eprhs", value="1e-7", reason="Second param"),
            ],
        )
        base_algorithm = {"lpmethod": "4", "solutiontype": "2"}

        lines = render_opt_lines(config, base_algorithm)

        # Find indices (only count actual parameter lines, not comments)
        lpmethod_idx = next(i for i, line in enumerate(lines) if line.startswith("lpmethod"))
        solutiontype_idx = next(
            i for i, line in enumerate(lines) if line.startswith("solutiontype")
        )
        epopt_idx = next(i for i, line in enumerate(lines) if line.startswith("epopt"))

        assert lpmethod_idx < epopt_idx
        assert solutiontype_idx < epopt_idx

    def test_override_conflicting_llm_values(self):
        """Override when LLM provides different algorithm values."""
        config = OptFileConfig(
            filename="test.opt",
            description="Test",
            parameters=[
                OptParameter(name="lpmethod", value="2", reason="LLM suggestion"),
                OptParameter(name="solutiontype", value="1", reason="LLM suggestion"),
                OptParameter(name="epopt", value="1e-7", reason="Tolerance"),
            ],
        )
        base_algorithm = {"lpmethod": "4", "solutiontype": "2"}

        lines = render_opt_lines(config, base_algorithm, warn_on_override=False)

        # Should use base algorithm values, not LLM values
        assert "lpmethod 4" in lines
        assert "solutiontype 2" in lines

        # Should not duplicate lpmethod/solutiontype (only one non-comment line each)
        assert sum(1 for line in lines if line.startswith("lpmethod")) == 1
        assert sum(1 for line in lines if line.startswith("solutiontype")) == 1

        # Other params should still be there
        assert "epopt 1e-7" in lines

    def test_case_insensitive_deduplication(self):
        """Deduplicate parameters case-insensitively."""
        config = OptFileConfig(
            filename="test.opt",
            description="Test",
            parameters=[
                OptParameter(name="LPMethod", value="2", reason="LLM suggestion"),
                OptParameter(name="epopt", value="1e-7", reason="Tolerance"),
            ],
        )
        base_algorithm = {"lpmethod": "4", "solutiontype": "2"}

        lines = render_opt_lines(config, base_algorithm)

        # Should only have one lpmethod parameter line (not case-sensitive duplicate)
        lpmethod_lines = [line for line in lines if line.lower().startswith("lpmethod")]
        assert len(lpmethod_lines) == 1

    def test_preserve_llm_matching_values(self):
        """Don't warn when LLM provides matching values."""
        config = OptFileConfig(
            filename="test.opt",
            description="Test",
            parameters=[
                OptParameter(name="lpmethod", value="4", reason="Barrier"),
                OptParameter(name="solutiontype", value="2", reason="No crossover"),
                OptParameter(name="epopt", value="1e-7", reason="Tolerance"),
            ],
        )
        base_algorithm = {"lpmethod": "4", "solutiontype": "2"}

        lines = render_opt_lines(config, base_algorithm)

        # Should still inject at top, no duplicates
        assert sum(1 for line in lines if line.lower().startswith("lpmethod")) == 1
        assert sum(1 for line in lines if line.lower().startswith("solutiontype")) == 1

    def test_handle_empty_base_algorithm(self):
        """Handle case where base algorithm extraction failed."""
        config = OptFileConfig(
            filename="test.opt",
            description="Test",
            parameters=[
                OptParameter(name="epopt", value="1e-7", reason="Tolerance"),
            ],
        )
        base_algorithm = {"lpmethod": "", "solutiontype": ""}

        lines = render_opt_lines(config, base_algorithm)

        # Should only have description and the parameter from LLM
        assert "* Test" in lines
        assert "* Tolerance" in lines
        assert "epopt 1e-7" in lines

        # Should not have lpmethod/solutiontype parameter lines
        assert not any(line.startswith("lpmethod") for line in lines)
        assert not any(line.startswith("solutiontype") for line in lines)

    def test_preserve_non_barrier_algorithm(self):
        """Preserve non-barrier algorithms from original run."""
        config = OptFileConfig(
            filename="test.opt",
            description="Test",
            parameters=[
                OptParameter(name="epopt", value="1e-7", reason="Tolerance"),
            ],
        )
        # Original run used dual simplex
        base_algorithm = {"lpmethod": "2", "solutiontype": "1"}

        lines = render_opt_lines(config, base_algorithm)

        # Should preserve original algorithm (not force barrier)
        assert "* Solver algorithm (from original run)" in lines
        assert "lpmethod 2" in lines
        assert "* Solution type (from original run)" in lines
        assert "solutiontype 1" in lines
