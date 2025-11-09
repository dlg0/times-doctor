"""Fake LLM provider for testing without API calls."""


class FakeLlmProvider:
    """Mock LLM provider that returns canned responses."""

    def __init__(self, response_template=None):
        self.calls = []
        self.response_template = response_template or "# Analysis\n\nMock LLM response."

    def condense(self, files: dict[str, str]) -> str:
        """Mock condense operation."""
        self.calls.append(("condense", files))
        return "\n\n".join(f"## {name}\n{content[:200]}" for name, content in files.items())

    def analyze(self, condensed: str, model: str = "gpt-5") -> str:
        """Mock analyze operation."""
        self.calls.append(("analyze", condensed, model))
        return self.response_template

    def reset(self):
        """Reset call history."""
        self.calls = []
