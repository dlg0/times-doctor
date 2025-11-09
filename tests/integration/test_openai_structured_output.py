"""Test OpenAI Responses API structured output capabilities.

Starting from the exact example in OpenAI docs, then incrementally
bridging to our actual use case (SolverDiagnosis).
"""

import os

import pytest
from pydantic import BaseModel

from times_doctor.core.solver_models import SolverDiagnosis


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Requires OPENAI_API_KEY")
class TestOpenAIStructuredOutput:
    """Incremental tests from docs example to our use case."""

    def test_01_simple_structured_output_gpt5(self):
        """Test 1: Simple structured output with gpt-5."""
        from openai import OpenAI

        class CalendarEvent(BaseModel):
            name: str
            date: str
            participants: list[str]

        client = OpenAI()

        response = client.responses.parse(
            model="gpt-5",
            input=[
                {"role": "system", "content": "Extract the event information."},
                {
                    "role": "user",
                    "content": "Alice and Bob are going to a science fair on Friday.",
                },
            ],
            text_format=CalendarEvent,
        )

        event = response.output_parsed

        # Verify we got a CalendarEvent object
        assert isinstance(event, CalendarEvent)
        assert event.name
        assert event.date
        assert len(event.participants) > 0
        print(f"✓ Test 1 passed: {event}")

    def test_02_gpt5_with_reasoning_effort(self):
        """Test 2: Simple structured output with gpt-5 and reasoning effort."""
        from openai import OpenAI

        class CalendarEvent(BaseModel):
            name: str
            date: str
            participants: list[str]

        client = OpenAI()

        response = client.responses.parse(
            model="gpt-5",
            input=[
                {"role": "system", "content": "Extract the event information."},
                {
                    "role": "user",
                    "content": "Alice and Bob are going to a science fair on Friday.",
                },
            ],
            text_format=CalendarEvent,
            reasoning={"effort": "medium"},
        )

        event = response.output_parsed

        # Verify we got a CalendarEvent object
        assert isinstance(event, CalendarEvent)
        assert event.name
        assert event.date
        assert len(event.participants) > 0
        print(f"✓ Test 2 passed with gpt-5 + reasoning: {event}")

    def test_03_more_complex_nested_schema(self):
        """Test 3: More complex nested schema similar to our OptFile structure."""
        from openai import OpenAI

        class Parameter(BaseModel):
            name: str
            value: str
            reason: str

        class Configuration(BaseModel):
            filename: str
            description: str
            parameters: list[Parameter]

        class Analysis(BaseModel):
            summary: str
            configurations: list[Configuration]

        client = OpenAI()

        response = client.responses.parse(
            model="gpt-5",
            input=[
                {
                    "role": "system",
                    "content": "Generate 2 CPLEX optimizer configurations with parameters.",
                },
                {
                    "role": "user",
                    "content": "Create two test configs: one with tight tolerances, one with loose.",
                },
            ],
            text_format=Analysis,
        )

        analysis = response.output_parsed

        # Verify structure
        assert isinstance(analysis, Analysis)
        assert analysis.summary
        assert len(analysis.configurations) >= 2
        assert all(isinstance(c, Configuration) for c in analysis.configurations)
        assert all(len(c.parameters) > 0 for c in analysis.configurations)
        print(f"✓ Test 3 passed: {len(analysis.configurations)} configs generated")

    @pytest.mark.slow
    def test_04_our_solver_diagnosis_model_gpt5_brief(self):
        """Test 4: Our actual SolverDiagnosis model with gpt-5 (brief prompt)."""
        from openai import OpenAI

        client = OpenAI()

        response = client.responses.parse(
            model="gpt-5",
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are a CPLEX solver expert. Generate a brief diagnosis "
                        "with 10-12 solver configurations to test. Be concise."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "A TIMES model returned FEASIBLE but NOT PROVEN OPTIMAL. "
                        "The barrier method stopped early. Generate 10-12 test configurations."
                    ),
                },
            ],
            text_format=SolverDiagnosis,
        )

        diagnosis = response.output_parsed

        # Verify structure
        assert isinstance(diagnosis, SolverDiagnosis)
        assert diagnosis.summary
        assert 10 <= len(diagnosis.opt_configurations) <= 15
        assert len(diagnosis.action_plan) >= 3
        print(
            f"✓ Test 4 passed: {len(diagnosis.opt_configurations)} configs, {len(diagnosis.action_plan)} actions"
        )

    @pytest.mark.slow
    def test_05_our_solver_diagnosis_model_gpt5_with_reasoning(self):
        """Test 5: SolverDiagnosis with gpt-5 and reasoning effort (KEY TEST)."""
        from openai import OpenAI

        client = OpenAI()

        response = client.responses.parse(
            model="gpt-5",
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are a CPLEX solver expert. Generate a diagnosis "
                        "with 10-12 solver configurations to test. Be concise."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "A TIMES model returned FEASIBLE but NOT PROVEN OPTIMAL. "
                        "The barrier method stopped early. Generate 10-12 test configurations."
                    ),
                },
            ],
            text_format=SolverDiagnosis,
            reasoning={"effort": "high"},
        )

        diagnosis = response.output_parsed

        # Verify structure
        assert isinstance(diagnosis, SolverDiagnosis)
        assert diagnosis.summary
        assert 10 <= len(diagnosis.opt_configurations) <= 15
        assert len(diagnosis.action_plan) >= 3
        print(
            f"✓ Test 5 passed with gpt-5 + reasoning: {len(diagnosis.opt_configurations)} configs, {len(diagnosis.action_plan)} actions"
        )

    def test_06_large_input_with_diagnostics(self):
        """Test 6: Large diagnostic input similar to our actual use case."""
        from openai import OpenAI

        client = OpenAI()

        # Simulate condensed diagnostic data
        large_input = """
=== CURRENT cplex.opt CONFIGURATION ===
scaind 1
lpmethod 4
baralg 1
epopt 1e-3

=== QA_CHECK.LOG (CONDENSED) ===
SEVERE: Defective sum of FX and UP FLO_SHAREs (1338 occurrences)
WARNING: NCAP_AF FX + LO/UP at same TS-level (38000 occurrences)

=== RUN LOG (CONDENSED) ===
CPLEX status: 6 (non-optimal)
Barrier iterations: 450
Solution available but not proven optimal

=== LST FILE (CONDENSED) ===
Matrix coefficient range: 7.1e-13 to 1.7e+05
Objective value: 1.234e+10
""".strip()

        response = client.responses.parse(
            model="gpt-5",
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are a CPLEX solver expert. Analyze the diagnostic data "
                        "and generate 10-12 solver configurations to test."
                    ),
                },
                {"role": "user", "content": large_input},
            ],
            text_format=SolverDiagnosis,
        )

        diagnosis = response.output_parsed

        # Verify we can handle large inputs
        assert isinstance(diagnosis, SolverDiagnosis)
        assert len(diagnosis.opt_configurations) >= 10
        assert all(len(c.parameters) > 0 for c in diagnosis.opt_configurations)
        print(f"✓ Test 6 passed with large input: {len(diagnosis.opt_configurations)} configs")
