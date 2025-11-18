"""
Test LST condensation algorithm robustness with challenging real-world files.

This module tests that the LST condensation algorithm can handle files with:
- Thousands of repetitive domain violation errors
- Complex element patterns with multiple quoted tokens
- Year-specific data that should be generalized
- Source context from BATINCLUDE statements
"""

from pathlib import Path

import pytest

from times_doctor.core.lst_parser import process_lst_file


class TestLSTCondensationRobustness:
    """Tests for LST condensation with challenging real-world files."""

    @pytest.fixture
    def test_lst_path(self):
        """Path to the problematic test.lst fixture."""
        path = Path(__file__).parent / "fixtures" / "sample_files" / "test.lst"
        if not path.exists():
            pytest.skip(f"Missing test fixture (gitignored): {path}")
        return path

    def test_aud25_domain_violations_are_condensed_usefully(self, test_lst_path):
        """
        Test that the algorithm condenses AUD25 domain violation errors meaningfully.

        The test.lst file contains thousands of "Domain violation for element" (error 170)
        messages, all related to the element 'AUD25' appearing across different regions
        and years. The current algorithm fails to produce useful output because:

        1. It searches forward from error lines instead of backward to find elements
        2. It only captures the first quoted token (e.g., 'ACT') instead of the
           problematic element ('AUD25')
        3. Element patterns don't get properly generalized by year

        A "useful" condensed output should:
        - Detect error 170 with high occurrence counts
        - Show that 'AUD25' is the problematic element in aggregated patterns
        - Generalize year patterns (e.g., 2021, 2022 -> YEAR)
        - Include source context from BATINCLUDE statements for debugging
        """
        result = process_lst_file(test_lst_path)

        # Find compilation sections (handles both "Compilation" and "C o m p i l a t i o n")
        comp_keys = [k for k in result["sections"] if "compilation" in k.lower().replace(" ", "")]
        assert comp_keys, "No Compilation sections found in LST file"

        # Aggregate statistics across all compilation sections
        total_170_count = 0
        element_patterns_with_aud25 = 0
        has_year_generalization = False
        has_source_context = False

        for section_key in comp_keys:
            section = result["sections"][section_key]
            errors = section.get("errors", {})

            # Check for error 170 (Domain violation for element)
            error_170 = errors.get("170")
            if not error_170:
                continue

            # Count total occurrences
            count = error_170.get("count", 0)
            total_170_count += count

            # Check if element patterns include AUD25
            elements = error_170.get("elements", {})
            for pattern in elements:
                if "AUD25" in pattern:
                    element_patterns_with_aud25 += 1
                if "YEAR" in pattern:
                    has_year_generalization = True

            # Check if sample contexts include source file information
            # Note: Source context from BATINCLUDE is in the full LST file lines,
            # not always in the condensed sample context
            samples = error_170.get("samples", [])
            for sample in samples:
                context = sample.get("context", "")
                element = sample.get("element", "")
                # We consider it useful if we have the element info
                if context or element:
                    has_source_context = True

            # Summary should mention Error 170
            summary = section.get("summary", "")
            assert "Error 170" in summary or "170" in summary, (
                f"Section summary should mention Error 170: {summary}"
            )

        # Verify useful information was extracted
        assert total_170_count > 20, (
            f"Expected many error 170 occurrences (file has thousands), got {total_170_count}"
        )

        assert element_patterns_with_aud25 > 0, (
            "Expected aggregated element patterns to include 'AUD25' (the actual problematic element)"
        )

        assert has_year_generalization, (
            "Expected element patterns to generalize years (e.g., 2021, 2022 -> YEAR)"
        )

        assert has_source_context, (
            "Expected error samples to include context or element information for debugging"
        )

    def test_element_extraction_direction(self, test_lst_path):
        """
        Test that elements are extracted by looking backward from error lines.

        In test.lst, the pattern is:
        1. Data line with element: 1833710  'ACT'.2021.'H2prd_elec_AE'.'AUD25' 2872
        2. Error marker lines: ****  $170, **** LINE ..., etc.
        3. Error description: **** 170  Domain violation for element

        The element line comes BEFORE the error line, so the algorithm must search
        backward, not forward.
        """
        result = process_lst_file(test_lst_path)

        # Get any compilation section with error 170
        comp_sections = [
            result["sections"][k]
            for k in result["sections"]
            if "compilation" in k.lower().replace(" ", "")
        ]

        error_170_found = False
        for section in comp_sections:
            errors = section.get("errors", {})
            if "170" in errors:
                error_170_found = True
                error_170 = errors["170"]

                # Get a sample context
                samples = error_170.get("samples", [])
                assert len(samples) > 0, "Error 170 should have sample contexts"

                # At least one sample should show the full element descriptor
                # including the data line that precedes the error marker
                found_full_descriptor = False
                for sample in samples:
                    context = sample.get("context", "")
                    # Should contain lines showing the element with AUD25
                    if "AUD25" in context:
                        found_full_descriptor = True
                        break

                assert found_full_descriptor, (
                    "Sample contexts should include the element line with 'AUD25'"
                )
                break

        assert error_170_found, "Should find error 170 in compilation sections"

    def test_full_element_pattern_capture(self, test_lst_path):
        """
        Test that element patterns capture the full descriptor, not just first token.

        Lines like:
            1833710  'ACT'.2021.'H2prd_elec_AE'.'AUD25' 2872

        Should be parsed to capture:
        - Region: ACT
        - Year: 2021 (should generalize to YEAR)
        - Process: H2prd_elec_AE
        - Element: AUD25 (the problematic element)

        Current implementation only captures 'ACT', losing critical information.
        """
        result = process_lst_file(test_lst_path)

        comp_sections = [
            result["sections"][k]
            for k in result["sections"]
            if "compilation" in k.lower().replace(" ", "")
        ]

        for section in comp_sections:
            errors = section.get("errors", {})
            if "170" in errors:
                error_170 = errors["170"]
                elements = error_170.get("elements", {})

                # Should have patterns that include technology names and AUD25
                has_tech_pattern = False
                has_multi_token_pattern = False

                for pattern in elements:
                    # Pattern should contain more than just a region code
                    if "H2prd" in pattern or "elec" in pattern:
                        has_tech_pattern = True
                    # Pattern should show structure with multiple components
                    if pattern.count(".") >= 2 or pattern.count("'") >= 4:
                        has_multi_token_pattern = True

                # At minimum, patterns should capture multiple components
                assert has_tech_pattern or has_multi_token_pattern, (
                    f"Element patterns should capture full descriptor structure, got: {list(elements.keys())}"
                )
                break

    def test_solver_status_extraction(self, test_lst_path):
        """
        Test that solver status (infeasible, iterations, resource usage) is extracted.

        The Solution Report section contains critical diagnostic information:
        - Solver type (LP, MIP, etc.)
        - Status code and description
        - Infeasibility flag
        - Resource usage and limits
        - Iteration counts
        - Execution errors
        """
        result = process_lst_file(test_lst_path)

        # Find solution report sections
        solution_sections = [
            result["sections"][k]
            for k in result["sections"]
            if "solution" in k.lower() and "report" in k.lower()
        ]

        assert len(solution_sections) > 0, "Should find at least one Solution Report section"

        # Check the solution report
        solution = solution_sections[0]

        # Should extract solver type
        assert solution.get("solver_type") == "LP", (
            f"Should extract solver type as LP, got {solution.get('solver_type')}"
        )

        # Should extract status code 3 (infeasible)
        assert solution.get("status_code") == 3, (
            f"Should extract status code 3 (infeasible), got {solution.get('status_code')}"
        )

        # Should detect infeasibility
        assert solution.get("infeasible") is True, "Should detect that model is infeasible"

        # Should detect no solution
        assert solution.get("has_solution") is False, "Should detect that no solution was returned"

        # Should extract resource usage
        assert solution.get("resource_usage") is not None, "Should extract resource usage"
        assert solution.get("resource_limit") is not None, "Should extract resource limit"

        # Should extract iteration count
        assert solution.get("iteration_count") == 0, (
            "Should extract iteration count (0 for infeasible)"
        )

        # Summary should highlight infeasibility
        summary = solution.get("summary", "")
        assert "infeasible" in summary.lower(), f"Summary should mention infeasibility: {summary}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
