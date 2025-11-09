"""Type definitions and protocols for times-doctor."""

from pathlib import Path
from typing import Any, Protocol


class LlmProvider(Protocol):
    """Protocol for LLM providers (OpenAI, Anthropic, etc.)."""

    def condense(self, files: dict[str, str], model: str) -> str:
        """Condense multiple files into a summary.

        Args:
            files: Dict mapping filenames to their content
            model: Model identifier to use for condensing

        Returns:
            Condensed text summary
        """
        ...

    def analyze(self, condensed: str, model: str, prompt: str) -> str:
        """Analyze condensed content with LLM.

        Args:
            condensed: Condensed text to analyze
            model: Model identifier to use for analysis
            prompt: Analysis prompt/instructions

        Returns:
            Analysis result as text
        """
        ...


class GamsRunner(Protocol):
    """Protocol for GAMS execution."""

    def run_gams(
        self,
        driver_file: Path,
        run_dir: Path,
        times_source: Path,
        threads: int,
        options: dict[str, Any],
    ) -> int:
        """Execute GAMS with given parameters.

        Args:
            driver_file: Path to .run or .gms file
            run_dir: Working directory for GAMS run
            times_source: Path to TIMES source code
            threads: Number of threads to use
            options: Additional GAMS options

        Returns:
            GAMS exit code
        """
        ...
