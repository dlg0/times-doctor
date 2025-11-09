"""CPLEX progress monitoring for barrier and simplex solves."""

import math
import re
from typing import Any


class BarrierProgressTracker:
    """Track barrier solve progress based on complementarity (mu) reduction."""

    def __init__(self, mu_target: float = 1e-8):
        """
        Initialize tracker.

        Args:
            mu_target: Target complementarity for convergence (matches typical barrier tolerance)
        """
        self.mu_target = mu_target
        self.mu0: float | None = None  # First observed mu
        self.in_crossover = False

    def update_mu(self, mu: float) -> float | None:
        """
        Update with new mu value and return progress percentage.

        Args:
            mu: Current complementarity value

        Returns:
            Progress as fraction 0.0-1.0, or None if not calculable
        """
        if self.mu0 is None:
            self.mu0 = mu

        if mu <= 0 or self.mu0 <= 0 or self.mu_target <= 0:
            return None

        numerator = math.log10(self.mu0) - math.log10(mu)
        denominator = math.log10(self.mu0) - math.log10(self.mu_target)

        if denominator <= 0:
            return None

        pct = numerator / denominator
        return max(0.0, min(1.0, pct))


# Regex patterns for CPLEX output
RE_MU = re.compile(r"(?:mu|complementarity)\s*=?\s*([0-9.eE+\-]+)", re.I)
RE_ITER = re.compile(r"\b(?:barrier|iter(?:ation)?)\b.*?(\d+)", re.I)
RE_PRIMAL_INFEAS = re.compile(r"primal\s+infeas(?:ibility)?\s*=?\s*([0-9.eE+\-]+)", re.I)
RE_DUAL_INFEAS = re.compile(r"dual\s+infeas(?:ibility)?\s*=?\s*([0-9.eE+\-]+)", re.I)
RE_CROSSOVER = re.compile(r"crossover", re.I)
RE_SIMPLEX = re.compile(r"\b(?:simplex|primal|dual)\b", re.I)


def parse_cplex_line(line: str) -> dict[str, Any] | None:
    """
    Parse a CPLEX iteration log line.

    Args:
        line: Single line from CPLEX output

    Returns:
        Dict with parsed values, or None if not an iteration line
    """
    result: dict[str, Any] = {}

    # Check for crossover
    if RE_CROSSOVER.search(line):
        result["phase"] = "crossover"

    # Extract mu (complementarity)
    mu_match = RE_MU.search(line)
    if mu_match:
        result["mu"] = float(mu_match.group(1))
        if "phase" not in result:
            result["phase"] = "barrier"

    # Extract iteration number
    iter_match = RE_ITER.search(line)
    if iter_match:
        result["iteration"] = iter_match.group(1)

    # Extract primal infeasibility
    pinf_match = RE_PRIMAL_INFEAS.search(line)
    if pinf_match:
        result["primal_infeas"] = float(pinf_match.group(1))

    # Extract dual infeasibility
    dinf_match = RE_DUAL_INFEAS.search(line)
    if dinf_match:
        result["dual_infeas"] = float(dinf_match.group(1))

    # Check for simplex
    if not result.get("phase") and RE_SIMPLEX.search(line):
        result["phase"] = "simplex"

    return result if result else None


def format_progress_line(
    parsed: dict[str, Any],
    progress_pct: float | None = None,
    tracker: BarrierProgressTracker | None = None,
) -> str:
    """
    Format a progress line for display.

    Args:
        parsed: Parsed CPLEX line from parse_cplex_line()
        progress_pct: Optional pre-calculated progress percentage (0.0-1.0)
        tracker: Optional tracker to calculate progress from mu

    Returns:
        Formatted string for display
    """
    phase = parsed.get("phase", "solving")
    iteration = parsed.get("iteration", "?")

    # Calculate progress if we have mu and a tracker
    if progress_pct is None and tracker and "mu" in parsed:
        if parsed.get("phase") == "crossover":
            tracker.in_crossover = True
        progress_pct = tracker.update_mu(parsed["mu"])

    # Format progress indicator
    in_crossover = bool(tracker and tracker.in_crossover)
    if progress_pct is not None and not in_crossover and phase == "barrier":
        pct_str = f"{int(progress_pct * 100)}%"
    else:
        pct_str = "–"

    # Build the line
    parts = [f"[{phase} {pct_str}] it={iteration}"]

    if "mu" in parsed:
        parts.append(f"mu={parsed['mu']:.2e}")

    if "primal_infeas" in parsed and "dual_infeas" in parsed:
        parts.append(f"Pinf={parsed['primal_infeas']:.2e} Dinf={parsed['dual_infeas']:.2e}")

    return " ".join(parts)


def scan_log_for_progress(
    log_lines: list[str], tracker: BarrierProgressTracker | None = None
) -> list[str]:
    """
    Scan log lines and return formatted progress lines.

    Args:
        log_lines: List of log lines to scan
        tracker: Optional tracker (created if not provided)

    Returns:
        List of formatted progress strings
    """
    if tracker is None:
        tracker = BarrierProgressTracker()

    progress_lines: list[str] = []

    for line in log_lines:
        parsed = parse_cplex_line(line)
        if parsed:
            formatted = format_progress_line(parsed, tracker=tracker)
            progress_lines.append(formatted)

    return progress_lines
