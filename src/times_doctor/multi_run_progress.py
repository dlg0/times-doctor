"""Multi-run progress monitoring for parallel GAMS execution."""

import threading
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
from rich.table import Table
from rich.console import Console
from . import logger as log
from rich.live import Live
from . import cplex_progress


class RunStatus(Enum):
    """Status of a single run."""
    WAITING = "waiting"
    STARTING = "starting"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class RunProgress:
    """Progress state for a single run."""
    name: str
    status: RunStatus = RunStatus.WAITING
    phase: str = "–"
    progress_pct: Optional[float] = None
    iteration: str = "–"
    mu: Optional[float] = None
    primal_infeas: Optional[float] = None
    dual_infeas: Optional[float] = None
    error_msg: Optional[str] = None
    tracker: cplex_progress.BarrierProgressTracker = field(default_factory=cplex_progress.BarrierProgressTracker)
    
    def format_progress(self) -> str:
        """Format progress percentage for display."""
        if self.progress_pct is not None and self.phase == "barrier":
            return f"{int(self.progress_pct * 100)}%"
        return "–"
    
    def format_iteration(self) -> str:
        """Format iteration info for display."""
        if self.iteration != "–":
            return f"it={self.iteration}"
        return "–"
    
    def format_details(self) -> str:
        """Format detailed solver info."""
        parts = []
        if self.mu is not None:
            parts.append(f"μ={self.mu:.2e}")
        if self.primal_infeas is not None and self.dual_infeas is not None:
            parts.append(f"P={self.primal_infeas:.2e}")
            parts.append(f"D={self.dual_infeas:.2e}")
        return " ".join(parts) if parts else "–"


class MultiRunProgressMonitor:
    """
    Thread-safe monitor for tracking progress of multiple GAMS runs.
    
    Can be used for:
    - Single run (datacheck)
    - Sequential runs (scan without --parallel)
    - Parallel runs (scan with --parallel)
    """
    
    def __init__(self, run_names: list[str], title: str = "Progress"):
        """
        Initialize monitor.
        
        Args:
            run_names: List of run identifiers (e.g., ["dual", "sift", "bar_nox"])
            title: Display title for the progress table
        """
        self.title = title
        self.runs: dict[str, RunProgress] = {
            name: RunProgress(name=name) for name in run_names
        }
        self.lock = threading.Lock()
        self.console = log.get_console()
        self.live: Optional[Live] = None
        self._should_stop = False
    
    def update_status(self, run_name: str, status: RunStatus, error_msg: Optional[str] = None):
        """Update the status of a run."""
        with self.lock:
            if run_name in self.runs:
                self.runs[run_name].status = status
                if error_msg:
                    self.runs[run_name].error_msg = error_msg
    
    def update_cplex_progress(self, run_name: str, parsed: dict):
        """
        Update CPLEX progress for a run.
        
        Args:
            run_name: Run identifier
            parsed: Parsed CPLEX line from cplex_progress.parse_cplex_line()
        """
        with self.lock:
            if run_name not in self.runs:
                return
            
            run = self.runs[run_name]
            run.status = RunStatus.RUNNING
            
            # Update phase
            if 'phase' in parsed:
                run.phase = parsed['phase']
            
            # Update iteration
            if 'iteration' in parsed:
                run.iteration = parsed['iteration']
            
            # Update mu and calculate progress
            if 'mu' in parsed:
                run.mu = parsed['mu']
                if parsed.get('phase') == 'crossover':
                    run.tracker.in_crossover = True
                run.progress_pct = run.tracker.update_mu(parsed['mu'])
            
            # Update infeasibilities
            if 'primal_infeas' in parsed:
                run.primal_infeas = parsed['primal_infeas']
            if 'dual_infeas' in parsed:
                run.dual_infeas = parsed['dual_infeas']
    
    def get_table(self) -> Table:
        """Generate Rich table for display."""
        table = Table(title=self.title, show_header=True, header_style="bold cyan")
        
        table.add_column("Run", style="cyan", no_wrap=True)
        table.add_column("Status", style="yellow")
        table.add_column("Phase", style="magenta")
        table.add_column("Progress", justify="right", style="green")
        table.add_column("Iteration", justify="right")
        table.add_column("Details", style="dim")
        
        with self.lock:
            for run_name, run in self.runs.items():
                # Color-code status
                status_style = {
                    RunStatus.WAITING: "dim",
                    RunStatus.STARTING: "yellow",
                    RunStatus.RUNNING: "green",
                    RunStatus.COMPLETED: "bold green",
                    RunStatus.FAILED: "bold red",
                }[run.status]
                
                status_text = f"[{status_style}]{run.status.value}[/{status_style}]"
                
                # Show error message if failed
                details = run.error_msg if run.error_msg else run.format_details()
                
                table.add_row(
                    run_name,
                    status_text,
                    run.phase,
                    run.format_progress(),
                    run.format_iteration(),
                    details
                )
        
        return table
    
    def start_live_display(self):
        """Start the live display."""
        if self.live is None:
            self.live = Live(self.get_table(), console=self.console, refresh_per_second=2)
            self.live.start()
    
    def stop_live_display(self):
        """Stop the live display."""
        if self.live:
            self.live.stop()
            self.live = None
    
    def update_display(self):
        """Update the live display (call periodically)."""
        if self.live:
            self.live.update(self.get_table())
    
    def __enter__(self):
        """Context manager entry."""
        self.start_live_display()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_live_display()
        return False
    
    def all_completed(self) -> bool:
        """Check if all runs are completed or failed."""
        with self.lock:
            return all(
                run.status in (RunStatus.COMPLETED, RunStatus.FAILED)
                for run in self.runs.values()
            )
    
    def get_summary(self) -> dict[str, int]:
        """Get summary of run statuses."""
        with self.lock:
            summary = {
                "total": len(self.runs),
                "waiting": 0,
                "starting": 0,
                "running": 0,
                "completed": 0,
                "failed": 0,
            }
            for run in self.runs.values():
                summary[run.status.value] += 1
            return summary
