"""Interactive job cancellation controller for times-doctor scan command."""

from __future__ import annotations

import contextlib
import signal
import threading
from types import FrameType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rich.console import Console

    from times_doctor.job_control import JobRegistry
    from times_doctor.multi_run_progress import MultiRunProgressMonitor


class InteractiveCancelController:
    """Controller for interactive job cancellation via Ctrl-C.

    Handles SIGINT signals safely by setting a flag that the main loop checks,
    rather than doing I/O directly in the signal handler. This ensures responsive
    and reliable cancellation across platforms.
    """

    def __init__(
        self, job_registry: JobRegistry, monitor: MultiRunProgressMonitor, console: Console
    ) -> None:
        self.job_registry = job_registry
        self.monitor = monitor
        self.console = console
        self.sigint_event = threading.Event()
        self._old_handler: Any = None
        self._lock = threading.Lock()  # avoid concurrent menus

    def _sigint_handler(self, signum: int, frame: FrameType | None) -> None:  # noqa: ARG002
        """Minimal signal handler that only sets an event.

        Do not do I/O or blocking operations in signal handlers - just set a flag
        and let the main loop handle it safely.
        """
        self.sigint_event.set()

    def install(self) -> InteractiveCancelController:
        """Install the SIGINT handler on the main thread."""
        if threading.current_thread() is not threading.main_thread():
            return self
        self._old_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._sigint_handler)
        return self

    def uninstall(self) -> None:
        """Restore the previous SIGINT handler."""
        if self._old_handler is not None:
            with contextlib.suppress(Exception):
                signal.signal(signal.SIGINT, self._old_handler)
            self._old_handler = None

    def should_open_menu(self) -> bool:
        """Check if Ctrl-C was pressed and menu should be shown."""
        return self.sigint_event.is_set()

    def clear_request(self) -> None:
        """Clear the Ctrl-C request flag."""
        self.sigint_event.clear()

    def _format_job_list(self, jobs: list[str]) -> str:
        """Format running jobs as a numbered list."""
        if not jobs:
            return "  (no running jobs)"
        width = len(str(len(jobs)))
        lines = [f"  {i:>{width}}) {name}" for i, name in enumerate(jobs, start=1)]
        return "\n".join(lines)

    def _parse_selection(self, user_input: str, jobs: list[str]) -> set[str]:
        """Parse user selection into set of job names.

        Accepts:
        - Empty string: no selection
        - 'a', 'all', '*': all jobs
        - '1,3,5': comma-separated numbers
        - '2-4': ranges
        - '1,3-5,7': combinations
        """
        s = user_input.strip().lower()
        if not s:
            return set()
        if s in {"a", "all", "*"}:
            return set(jobs)

        idxs: set[int] = set()
        for tok in s.split(","):
            tok = tok.strip()
            if not tok:
                continue
            if "-" in tok:
                with contextlib.suppress(Exception):
                    lo, hi = tok.split("-", 1)
                    lo_i, hi_i = int(lo), int(hi)
                    for k in range(min(lo_i, hi_i), max(lo_i, hi_i) + 1):
                        idxs.add(k)
            else:
                with contextlib.suppress(Exception):
                    idxs.add(int(tok))

        selected = {jobs[i - 1] for i in idxs if 1 <= i <= len(jobs)}
        return selected

    def show_menu(self) -> None:
        """Show interactive menu to select jobs to cancel.

        Called from the main loop (not from signal handler) to ensure safe I/O.
        """
        with self._lock:
            self.clear_request()

            # Pause live display for clean input UX
            with self.monitor.pause_display():
                running = self.job_registry.get_running_jobs()
                self.console.print(
                    "\n[bold magenta]Cancel jobs[/bold magenta] (Ctrl-C again to reopen):"
                )
                self.console.print("Select which running jobs to cancel:")
                self.console.print(self._format_job_list(running))
                self.console.print(
                    "\nEnter: numbers (1,3,5), ranges (2-4), 'a' for all, or just Enter to resume"
                )

                try:
                    # Input runs on main thread now; safe to block briefly
                    user_input = input("> ").strip()
                except KeyboardInterrupt:
                    # If user hits Ctrl-C while typing, just return to display
                    self.console.print("[dim]Cancelled menu; resuming[/dim]\n")
                    return

                chosen = self._parse_selection(user_input, running)
                if not chosen:
                    self.console.print("[dim]No selection; resuming[/dim]\n")
                    return

                # Cancel selected jobs
                for name in sorted(chosen):
                    # Avoid crashes; continue cancelling others
                    with contextlib.suppress(Exception):
                        self.job_registry.cancel(name, grace_seconds=10.0)

                self.console.print(
                    f"[yellow]Requested cancellation of {len(chosen)} job(s).[/yellow]\n"
                )
