import contextlib
import csv
import gc
import os
import re
import shutil
import signal
import stat
import subprocess
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer
from rich import print
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text

from . import __version__, cplex_progress, gurobi_progress
from . import logger as log
from .core import llm as llm_mod
from .core.cost_estimator import estimate_cost, estimate_tokens
from .job_control import JobRegistry, terminate_process_tree
from .multi_run_progress import MultiRunProgressMonitor, RunStatus

if TYPE_CHECKING:
    from .core.solver_models import SolverDiagnosis

app = typer.Typer(add_completion=False)
console = Console()


def version_callback(value: bool) -> None:
    if value:
        print(f"times-doctor version {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version",
        "-v",
        help="Show version and exit",
        callback=version_callback,
        is_eager=True,
    ),
    no_color: bool = typer.Option(
        False, "--no-color", help="Disable colored output", envvar="NO_COLOR"
    ),
) -> None:
    log.init_console(no_color=no_color)


def _make_writable_and_retry(func, path, exc_info):
    """Helper for shutil.rmtree to handle read-only files on Windows."""
    with contextlib.suppress(Exception):
        os.chmod(path, stat.S_IWRITE)
    with contextlib.suppress(Exception):
        func(path)


def remove_tree_robust(
    path: Path, retries: int = 15, base_sleep: float = 0.1, max_sleep: float = 2.0
) -> bool:
    """
    Remove directory tree with Windows-safe retry and fallback strategies.

    On Windows, files can be locked by:
    - GAMS processes that haven't fully released handles
    - Windows file system delays
    - Antivirus software scanning files

    This function:
    1. Tries to delete with chmod on read-only files (up to ~23s with backoff)
    2. If still locked, renames to .stale.<timestamp> quarantine
    3. Returns False only if both delete and rename fail

    Args:
        path: Directory to remove
        retries: Number of retry attempts (default: 15)
        base_sleep: Initial sleep duration in seconds (default: 0.1)
        max_sleep: Maximum sleep duration in seconds (default: 2.0)

    Returns:
        True if deleted or quarantined, False if completely failed
    """
    if not path.exists():
        return True

    for i in range(retries):
        try:
            shutil.rmtree(path, onerror=_make_writable_and_retry)
            return True
        except FileNotFoundError:
            return True
        except (PermissionError, OSError):
            gc.collect()  # Help release Python-side handles
            sleep_time = min(max_sleep, base_sleep * (2**i))
            time.sleep(sleep_time)

    # Quarantine rename as fallback
    try:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        quarantine = path.with_name(f"{path.name}.stale.{ts}")
        path.rename(quarantine)
        console.print(
            f"[yellow]Could not delete {path.name} (locked by another process); moved to {quarantine.name}[/yellow]"
        )
        return True
    except Exception as e:
        console.print(f"[red]Failed to delete or rename {path.name}: {e}[/red]")
        return False


def write_default_opt(solver: str, path: Path, threads: int = 7) -> None:
    """
    Write default solver .opt file with conservative settings.

    Args:
        solver: Solver type ('cplex' or 'gurobi')
        path: Path to write .opt file
        threads: Number of threads for solver to use
    """
    lines = []
    if solver == "cplex":
        lines = [
            "names yes",
            f"threads {threads}",
            "lpmethod 2",
            "solutiontype 1",
            "eprhs 1.0e-06",
            "epopt 1.0e-06",
            "scaind -1",
            "aggind 1",
            "simdisplay 2",
        ]
    elif solver == "gurobi":
        lines = [
            f"Threads {threads}",
            "Method 1",
            "Crossover 1",
            "FeasibilityTol 1e-06",
            "OptimalityTol 1e-06",
            "NumericFocus 1",
        ]

    path.write_text("\n".join(lines) + "\n")


def ensure_display_options(opt_path: Path) -> None:
    """
    Ensure CPLEX .opt file has simdisplay and bardisplay for progress tracking.

    Args:
        opt_path: Path to CPLEX .opt file
    """
    try:
        txt = opt_path.read_text(encoding="utf-8", errors="ignore")
        lower = txt.lower()
        lines = []
        if "simdisplay" not in lower:
            lines.append("simdisplay 2")
        if "bardisplay" not in lower:
            lines.append("bardisplay 2")
        if lines:
            suffix = "\n" if not txt.endswith("\n") else ""
            opt_path.write_text(txt + suffix + "\n".join(lines) + "\n", encoding="utf-8")
    except Exception:
        pass  # Silently fail if we can't update the file


def ensure_threads_option(opt_path: Path, solver: str, threads: int) -> None:
    """
    Ensure solver .opt file sets Threads/threads to the requested value.

    Updates all existing occurrences (case-insensitive) or appends if missing.
    Preserves all other lines, comments, and formatting.

    Args:
        opt_path: Path to solver .opt file
        solver: Solver type ('cplex' or 'gurobi')
        threads: Number of threads to set
    """
    import re
    import time

    def try_write_opt_file(path: Path, content: str) -> bool:
        """Try to write to opt file, return True if successful."""
        try:
            path.write_text(content, encoding="utf-8")
            return True
        except (PermissionError, OSError):
            return False

    # Read file content
    try:
        txt = opt_path.read_text(encoding="utf-8", errors="ignore")
    except FileNotFoundError:
        # If no file exists, create a minimal one
        line = f"threads {threads}\n" if solver.lower() == "cplex" else f"Threads {threads}\n"
        if not try_write_opt_file(opt_path, line):
            console.print(
                f"[yellow]Warning: Cannot write to {opt_path.name} (file may be open in another program)[/yellow]"
            )
            if typer.confirm("Close the file and press Enter to retry", default=True):
                for _attempt in range(3):
                    time.sleep(0.5)
                    if try_write_opt_file(opt_path, line):
                        console.print(f"[green]✓ Successfully updated {opt_path.name}[/green]")
                        return
                console.print(
                    f"[red]Error: Still cannot write to {opt_path.name}. Please close it and restart the scan.[/red]"
                )
                raise typer.Exit(1)
        return
    except (PermissionError, OSError):
        console.print(
            f"[yellow]Warning: Cannot read {opt_path.name} (file may be open in another program)[/yellow]"
        )
        if typer.confirm("Close the file and press Enter to retry", default=True):
            for attempt in range(3):
                time.sleep(0.5)
                try:
                    txt = opt_path.read_text(encoding="utf-8", errors="ignore")
                    break
                except (PermissionError, OSError):
                    if attempt == 2:
                        console.print(
                            f"[red]Error: Still cannot read {opt_path.name}. Please close it and restart the scan.[/red]"
                        )
                        raise typer.Exit(1)
            else:
                raise typer.Exit(1)
        else:
            raise typer.Exit(1)

    target = "threads"  # Match case-insensitively
    lines_out = []
    updated_any = False

    for orig in txt.splitlines():
        s = orig.lstrip()
        # Keep full-line comments and blank lines as-is
        if not s or s[0] in ("*", "#", "$"):
            lines_out.append(orig)
            continue

        # Split off inline comments for safe rewrite
        # Gurobi uses '#', CPLEX often uses '$'; support both
        idx_hash = orig.find("#")
        idx_dlr = orig.find("$")
        split_idxs = [i for i in (idx_hash, idx_dlr) if i != -1]
        cpos = min(split_idxs) if split_idxs else -1
        code = orig if cpos == -1 else orig[:cpos]
        inline_comment = "" if cpos == -1 else orig[cpos:]

        # Match name [=|space] value; preserve original name and separator
        m = re.match(r"^(\s*)([A-Za-z][\w\-]*)(\s*(?:=|\s)\s*)(.*\S)?\s*$", code)
        if m and m.group(2).lower() == target:
            prefix, name, sep, _ = m.groups()
            new_code = f"{prefix}{name}{sep}{threads}"
            # Keep single space before inline comment if any existed
            new_line = new_code + ("" if not inline_comment else (" " + inline_comment.lstrip()))
            lines_out.append(new_line)
            updated_any = True
        else:
            lines_out.append(orig)

    if not updated_any:
        # Ensure file ends with newline, then append threads line
        if lines_out and lines_out[-1] != "":
            lines_out.append("")
        appended = f"threads {threads}" if solver.lower() == "cplex" else f"Threads {threads}"
        lines_out.append(appended)

    # Try to write the updated content
    new_content = "\n".join(lines_out) + "\n"
    if not try_write_opt_file(opt_path, new_content):
        console.print(
            f"[yellow]Warning: Cannot write to {opt_path.name} (file may be open in another program)[/yellow]"
        )
        if typer.confirm("Close the file and press Enter to retry", default=True):
            for _attempt in range(3):
                time.sleep(0.5)
                if try_write_opt_file(opt_path, new_content):
                    console.print(f"[green]✓ Successfully updated {opt_path.name}[/green]")
                    return
            console.print(
                f"[red]Error: Still cannot write to {opt_path.name}. Please close it and restart the scan.[/red]"
            )
            raise typer.Exit(1)
        else:
            raise typer.Exit(1)


def resolve_opt_file(run_dir: Path, solver: str, opt_file: str | None) -> Path | None:
    """
    Resolve .opt file from various sources.

    Args:
        run_dir: Run directory to search in
        solver: Solver type ('cplex' or 'gurobi')
        opt_file: Optional path or name to resolve

    Returns:
        Path to .opt file, or None if not found
    """
    # If explicit path provided
    if opt_file:
        opt_path = Path(opt_file)
        if opt_path.exists():
            return opt_path

        # Try as a name in _td_opt_files/<solver>/
        opt_files_dir = run_dir / "_td_opt_files" / solver
        if opt_files_dir.exists():
            candidate = (
                opt_files_dir / f"{opt_file}.opt"
                if not opt_file.endswith(".opt")
                else opt_files_dir / opt_file
            )
            if candidate.exists():
                return candidate

    # Look for existing solver.opt in run_dir
    existing = run_dir / f"{solver}.opt"
    if existing.exists():
        return existing

    return None


def ensure_solver_option(opt_path: Path, name: str, value: str) -> None:
    """
    Ensure solver .opt file has 'name value' (case-insensitive name).
    Updates existing entry or appends if missing. Preserves comments/format.

    Args:
        opt_path: Path to solver .opt file
        name: Option name to set
        value: Option value to set
    """
    import re

    def try_write(p: Path, content: str) -> bool:
        try:
            p.write_text(content, encoding="utf-8")
            return True
        except (PermissionError, OSError):
            return False

    try:
        txt = opt_path.read_text(encoding="utf-8", errors="ignore")
    except FileNotFoundError:
        line = f"{name} {value}\n"
        if not try_write(opt_path, line):
            console.print(f"[yellow]Warning: Cannot write to {opt_path.name}[/yellow]")
        return

    lines_out = []
    updated = False
    target = name.lower()

    for orig in txt.splitlines():
        s = orig.lstrip()
        if not s or s[0] in ("*", "#", "$"):
            lines_out.append(orig)
            continue
        idx_hash = orig.find("#")
        idx_dlr = orig.find("$")
        split_idxs = [i for i in (idx_hash, idx_dlr) if i != -1]
        cpos = min(split_idxs) if split_idxs else -1
        code = orig if cpos == -1 else orig[:cpos]
        inline_comment = "" if cpos == -1 else orig[cpos:]

        m = re.match(r"^(\s*)([A-Za-z][\w\-]*)(\s*(?:=|\s)\s*)(.*\S)?\s*$", code)
        if m and m.group(2).lower() == target:
            prefix, name_tok, sep, _ = m.groups()
            new_code = f"{prefix}{name_tok}{sep}{value}"
            new_line = new_code + ("" if not inline_comment else (" " + inline_comment.lstrip()))
            lines_out.append(new_line)
            updated = True
        else:
            lines_out.append(orig)

    if not updated:
        if lines_out and lines_out[-1] != "":
            lines_out.append("")
        lines_out.append(f"{name} {value}")

    new_content = "\n".join(lines_out) + "\n"
    if not try_write(opt_path, new_content):
        console.print(f"[yellow]Warning: Cannot write to {opt_path.name}[/yellow]")


def detect_solver_from_files(run_log: str, lst_text: str) -> str | None:
    """
    Detect which solver was used from log files.

    Args:
        run_log: Run log content
        lst_text: LST file content

    Returns:
        'cplex', 'gurobi', or None if not detected
    """
    s = (run_log or "") + "\n" + (lst_text or "")
    if re.search(r"(?i)GAMS/\s*GUROBI|\bGurobi\b|LP\s*=\s*GUROBI", s):
        return "gurobi"
    if re.search(r"(?i)GAMS/\s*CPLEX|\bCPLEX\b|LP\s*=\s*CPLEX", s):
        return "cplex"
    return None


def detect_iis_present(solver: str | None, run_log: str, lst_text: str) -> tuple[bool, str | None]:
    """
    Check if IIS/Conflict Refiner output is present in log files.

    Args:
        solver: Solver type ('cplex', 'gurobi', or None to check both)
        run_log: Run log content
        lst_text: LST file content

    Returns:
        Tuple of (iis_found, reason_string)
    """
    t = (run_log or "") + "\n" + (lst_text or "")
    if (solver == "gurobi" or solver is None) and re.search(
        r"(?i)Computing Irreducible Inconsistent Subsystem|IIS computed|Non-minimal IIS computed",
        t,
    ):
        return True, "Gurobi IIS markers found"
    if (solver == "cplex" or solver is None) and re.search(
        r"(?i)IIS found|Irreducible Inconsistent Subsystem|Conflict Refiner", t
    ):
        return True, "CPLEX IIS/conflict markers found"
    return False, None


def detect_times_version(lst_path: Path | None) -> str | None:
    """Detect TIMES version from existing .lst file."""
    if not lst_path or not lst_path.exists():
        return None
    try:
        text = read_text(lst_path)
        m = re.search(r"TIMES\s+--\s+VERSION\s+([\d\.]+)", text)
        return m.group(1) if m else None
    except Exception:
        return None


def get_times_source(version: str | None = None) -> Path:
    """Get TIMES source code, downloading specific version if necessary."""
    cache_dir = Path.home() / ".cache" / "times-doctor"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Determine directory name based on version
    times_src = cache_dir / f"TIMES_model-{version}" if version else cache_dir / "TIMES_model"

    # Check if already downloaded
    if times_src.exists() and (times_src / "initsys.mod").exists():
        return times_src

    # Download with version tag if specified
    download_msg = f"Downloading TIMES source code{f' v{version}' if version else ''}..."

    try:
        with log.spinner(download_msg):
            cmd = ["git", "clone", "--depth=1"]
            if version:
                cmd.extend(["--branch", f"v{version}"])
            cmd.extend(["https://github.com/etsap-TIMES/TIMES_model.git", str(times_src)])

            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
        log.success(f"Downloaded TIMES source to {times_src}")
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            "Git clone timed out after 300s. Check network connection or try manual download from https://github.com/etsap-TIMES/TIMES_model"
        )
    except subprocess.CalledProcessError as e:
        err_msg = e.stderr if e.stderr else ""
        if version and "not found" in err_msg.lower():
            log.warning(f"Version v{version} not found, trying latest 4.x...")
            # Fallback to latest if specific version not found
            return get_times_source(version=None)
        raise RuntimeError(
            f"Failed to download TIMES source. Install git or download manually from https://github.com/etsap-TIMES/TIMES_model\nError: {err_msg}"
        )

    return times_src


def pick_driver_gms(run_dir: Path) -> Path:
    """Find the main .run or .gms driver file."""
    run_files = sorted(run_dir.glob("*.run"), key=lambda p: p.stat().st_size, reverse=True)
    if run_files:
        return run_files[0]

    gms_files = sorted(run_dir.glob("*.gms"), key=lambda p: p.stat().st_size, reverse=True)
    if gms_files:
        return gms_files[0]

    raise FileNotFoundError("No .run or .gms driver found in run dir")


def latest_lst(run_dir: Path) -> Path | None:
    # Find .lst files but exclude .condensed.lst files
    lsts = [p for p in run_dir.glob("*.lst") if ".condensed" not in p.name]
    lsts = sorted(lsts, key=lambda p: p.stat().st_mtime, reverse=True)
    return lsts[0] if lsts else None


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def write_opt(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def ensure_out(run_dir: Path) -> Path:
    out = run_dir / "times_doctor_out"
    out.mkdir(exist_ok=True)
    return out


def _render_solver_diagnosis(
    diagnosis: "SolverDiagnosis",
    created_files: list[Path],
    md_path: Path,
) -> None:
    """Render a clean summary of solver diagnosis to terminal."""
    from textwrap import shorten

    from rich.box import SIMPLE_HEAD
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    console = Console()

    # 1) Diagnosis summary panel
    console.print(
        Panel(
            Text(diagnosis.summary.strip()),
            title="[bold]Diagnosis[/bold]",
            border_style="cyan",
        )
    )

    # 2) Action plan panel
    steps = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(diagnosis.action_plan))
    console.print(
        Panel(
            Text(steps),
            title="[bold]Action Plan[/bold]",
            border_style="green",
        )
    )

    # 3) Generated .opt files table
    table = Table(
        "File", "Params", "Description", box=SIMPLE_HEAD, title="Generated Configurations"
    )
    max_desc = max(30, min(80, console.width - 35))
    for cfg in diagnosis.opt_configurations:
        table.add_row(
            f"[cyan]{cfg.filename}[/cyan]",
            str(len(cfg.parameters)),
            shorten(cfg.description, width=max_desc, placeholder="…"),
        )
    console.print(table)

    # 4) Footer info
    print(f"\n[green]✓ Detailed report:[/green] {md_path}")
    if created_files:
        out_dir = created_files[0].parent
        print(f"[green]✓ Created {len(created_files)} .opt files:[/green] {out_dir}/")


def run_gams_with_progress(
    cmd: list[str],
    cwd: str,
    max_lines: int = 30,
    monitor: MultiRunProgressMonitor | None = None,
    run_name: str | None = None,
    timeout_seconds: int | None = None,
    cancel_event: threading.Event | None = None,
    job_registry: "JobRegistry | None" = None,
    parse_line_func: Any | None = None,
    progress_tracker: Any | None = None,
) -> int:
    """
    Run GAMS subprocess with robust error handling and live output monitoring.

    Subprocess Execution Strategy:
    - Uses subprocess.Popen for streaming output (required for live display)
    - Arguments passed as list (proper quoting, no shell injection risk)
    - Explicit timeout handling with graceful termination
    - Cross-platform path handling via pathlib.Path
    - Comprehensive error handling for FileNotFoundError and other exceptions

    GAMS Path Resolution:
    - First: Use explicit --gams-path if provided by user
    - Then: Check GAMS_PATH environment variable
    - Finally: Search system PATH for 'gams' executable
    - On failure: Raise FileNotFoundError with actionable guidance

    Args:
        cmd: GAMS command and arguments as list (e.g., ['gams', 'model.gms'])
        cwd: Working directory (normalized to Path internally)
        max_lines: Max lines to show in standalone display
        monitor: Optional MultiRunProgressMonitor for coordinated display
        run_name: Name of this run (required if monitor is provided)
        timeout_seconds: Optional timeout in seconds (prevents hangs)
        parse_line_func: Optional solver-specific line parser (defaults to CPLEX)
        progress_tracker: Optional solver-specific progress tracker (defaults to CPLEX)

    Returns:
        GAMS exit code (0 = success)

    Raises:
        FileNotFoundError: GAMS executable not found
        TimeoutError: Process exceeded timeout_seconds
        KeyboardInterrupt: User pressed CTRL-C
    """
    import threading
    from pathlib import Path

    # Validate parameters
    if monitor and not run_name:
        raise ValueError("run_name is required when monitor is provided")

    # Use standalone display or report to monitor
    use_monitor = monitor is not None

    if not use_monitor:
        # Log the command being run (only in standalone mode)
        console.print(f"[dim]Working directory: {cwd}[/dim]")
        console.print(f"[dim]Command: {' '.join(cmd)}[/dim]")
    else:
        # Update monitor status
        if monitor and run_name:
            monitor.update_status(run_name, RunStatus.STARTING)

    # Find the _run_log.txt file that will be created
    cwd_path = Path(cwd)
    log_files_before = set(cwd_path.glob("*_run_log.txt"))

    try:
        from .job_control import create_process_group_kwargs

        # Set status to "starting" before spawn so cancel menu can see it
        if job_registry and run_name:
            job_registry.set_status(run_name, "starting")

        # Create process in its own group for clean cancellation
        pg_kwargs = create_process_group_kwargs()

        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
            bufsize=0,
            **pg_kwargs,
        )

        # Register process with job registry if provided
        if job_registry and run_name:
            job_registry.attach_popen(run_name, proc)
            job_registry.set_status(run_name, "running")

        # Update monitor with parent PID
        if use_monitor and monitor and run_name:
            monitor.update_pids(run_name, proc.pid, None)

        # Start child process tracker thread (if psutil available)
        def _child_tracker():
            try:
                import psutil

                _has_psutil = True
            except Exception:
                _has_psutil = False

            if not _has_psutil:
                return

            try:
                parent = psutil.Process(proc.pid)
            except Exception:
                return

            # Track while parent alive
            while proc.poll() is None:
                try:
                    children = parent.children(recursive=True)
                    # Filter to known GAMS worker names
                    worker_names = {"gamscmex.exe", "gmsgennx.exe"}
                    filtered = []
                    for c in children:
                        try:
                            name = c.name().lower()
                            if name in worker_names:
                                filtered.append(c.pid)
                        except Exception:
                            pass

                    if job_registry and run_name:
                        job_registry.update_child_pids(run_name, filtered)
                    if use_monitor and monitor and run_name:
                        monitor.update_pids(run_name, None, filtered)
                except Exception:
                    pass
                time.sleep(0.5)

        threading.Thread(target=_child_tracker, daemon=True).start()

        # Start exit watcher thread to ensure final status update
        def _exit_watcher():
            rc = proc.wait()
            # If cancellation already marked, don't override
            if cancel_event and cancel_event.is_set():
                return
            if use_monitor and monitor and run_name:
                status = RunStatus.COMPLETED if rc == 0 else RunStatus.FAILED
                error_msg = None if rc == 0 else f"Exit code: {rc}"
                monitor.update_status(run_name, status, error_msg)
            if job_registry and run_name:
                job_registry.set_status(run_name, "completed" if rc == 0 else "failed")

        threading.Thread(target=_exit_watcher, daemon=True).start()

        if not use_monitor:
            console.print(f"[dim]GAMS process started (PID: {proc.pid})[/dim]")

        # Check if cancelled immediately after spawn (before entering main loop)
        if cancel_event and cancel_event.is_set():
            if use_monitor and monitor and run_name:
                monitor.update_status(run_name, RunStatus.CANCELLED)
            if job_registry and run_name:
                job_registry.cancel(run_name, grace_seconds=5)
            else:
                terminate_process_tree(proc, grace_seconds=5)
            raise KeyboardInterrupt("Job cancelled before start")

    except FileNotFoundError:
        error_msg = f"GAMS executable not found: {cmd[0]}"
        if use_monitor and monitor and run_name:
            monitor.update_status(run_name, RunStatus.FAILED, error_msg)
        else:
            console.print(f"[red]Error: {error_msg}[/red]")
            console.print(
                "[yellow]Make sure GAMS is installed and in PATH, or use --gams-path[/yellow]"
            )
        raise
    except Exception as e:
        error_msg = f"Error starting GAMS: {e}"
        if use_monitor and monitor and run_name:
            monitor.update_status(run_name, RunStatus.FAILED, error_msg)
        else:
            console.print(f"[red]{error_msg}[/red]")
        raise

    lines: list[str] = []
    log_file_lines: list[str] = []
    current_log_file = None
    display_text = Text("Starting GAMS...", style="dim")

    # Solver progress tracking (defaults to CPLEX if not specified)
    if parse_line_func is None:
        parse_line_func = cplex_progress.parse_cplex_line
    if progress_tracker is None:
        progress_tracker = cplex_progress.BarrierProgressTracker()

    def read_output():
        for line in iter(proc.stdout.readline, b""):
            try:
                line_str = line.decode("utf-8", errors="ignore").rstrip()
                if line_str:
                    lines.append(line_str)
            except Exception as e:
                console.print(f"[red]Error reading GAMS output: {e}[/red]")
                pass

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()

    def handle_interrupt(signum, frame):
        console.print("\n[yellow]Received CTRL-C, terminating GAMS process...[/yellow]")
        terminate_process_tree(proc, grace_seconds=3)
        raise KeyboardInterrupt()

    # Only register signal handler from main thread when not orchestrated by scan (monitor=None)
    # When scan orchestrates runs, it handles SIGINT at a higher level
    old_handler = None
    signal_handler_installed = False
    use_monitor = monitor is not None
    if (not use_monitor) and threading.current_thread() is threading.main_thread():
        try:
            old_handler = signal.signal(signal.SIGINT, handle_interrupt)
            signal_handler_installed = True
        except (ValueError, RuntimeError):
            # Signal handling not available in this context
            pass

    start_time = time.monotonic()
    timed_out = False

    try:
        try:
            import psutil

            has_psutil = True
        except ImportError:
            has_psutil = False

        iterations = 0
        last_log_check = 0

        # Standalone mode variables
        if not use_monitor:
            nonlocal_progress_line: list[str | None] = [None]
            live_display = Live(display_text, console=console, refresh_per_second=2)
            live_display.start()

        try:
            while proc.poll() is None:
                # Check for cancellation
                if cancel_event and cancel_event.is_set():
                    if use_monitor and monitor and run_name:
                        monitor.update_status(run_name, RunStatus.CANCELLED)
                    else:
                        console.print("[yellow]Run cancelled. Terminating...[/yellow]")
                    if job_registry and run_name:
                        job_registry.set_status(run_name, "cancelled")
                        job_registry.cancel(run_name, grace_seconds=5)
                    else:
                        terminate_process_tree(proc, grace_seconds=5)
                    raise KeyboardInterrupt("Job cancelled")

                # Check timeout
                if timeout_seconds and (time.monotonic() - start_time) > timeout_seconds:
                    timed_out = True
                    timeout_msg = f"Timed out after {timeout_seconds}s"
                    if use_monitor and monitor and run_name:
                        monitor.update_status(run_name, RunStatus.FAILED, timeout_msg)
                    else:
                        console.print(f"[yellow]Run {timeout_msg}. Terminating...[/yellow]")
                    if job_registry and run_name:
                        job_registry.set_status(run_name, "failed")
                        job_registry.cancel(run_name, grace_seconds=5)
                    else:
                        terminate_process_tree(proc, grace_seconds=5)
                    break

                iterations += 1

                # Try to find and tail log files if no stdout
                if iterations - last_log_check > 4:  # Check every 2 seconds
                    last_log_check = iterations
                    if not current_log_file:
                        # Try _run_log.txt first, then .lst files as fallback
                        log_files_after = set(cwd_path.glob("*_run_log.txt"))
                        new_files = log_files_after - log_files_before
                        if not new_files:
                            # Fallback to .lst files
                            log_files_after = set(cwd_path.glob("*.lst"))
                            new_files = log_files_after - set()  # All .lst files

                        if new_files:
                            current_log_file = max(new_files, key=lambda p: p.stat().st_mtime)
                            if not use_monitor:
                                live_display.stop()
                                console.print(f"[dim]Monitoring log file: {current_log_file}[/dim]")
                                live_display.start()

                    # Read from log file if available
                    if current_log_file and current_log_file.exists():
                        try:
                            with open(current_log_file, encoding="utf-8", errors="ignore") as f:
                                new_content = f.readlines()
                                if len(new_content) > len(log_file_lines):
                                    # Get new lines before updating log_file_lines
                                    old_len = len(log_file_lines)
                                    log_file_lines = new_content
                                    # Show last N non-empty lines
                                    lines = [l.rstrip() for l in new_content if l.strip()]

                                    # Check for solver progress in new lines
                                    for new_line in new_content[old_len:]:
                                        parsed = parse_line_func(new_line)
                                        if parsed:
                                            if use_monitor and monitor and run_name:
                                                # Report to monitor
                                                monitor.update_cplex_progress(run_name, parsed)
                                            else:
                                                # Update standalone display
                                                formatted = cplex_progress.format_progress_line(
                                                    parsed, tracker=progress_tracker
                                                )
                                                nonlocal_progress_line[0] = formatted
                        except Exception:
                            pass

                # Update standalone display
                if not use_monitor:
                    if lines:
                        display_lines = lines[-max_lines:]

                        # Add solver progress line at the top if we have one
                        if nonlocal_progress_line[0]:
                            display_lines = [
                                f"[bold cyan]{nonlocal_progress_line[0]}[/bold cyan]",
                                "",
                            ] + display_lines

                        display_text = Text("\n".join(display_lines))
                        live_display.update(display_text)
                    else:
                        # Show periodic heartbeat if no output
                        if iterations % 10 == 0:
                            elapsed = iterations * 0.5
                            # Check if process is actually doing work (including child processes)
                            if has_psutil:
                                try:
                                    p = psutil.Process(proc.pid)
                                    p.cpu_percent(interval=0.1)
                                    p.memory_info().rss / 1024 / 1024

                                    # Check for child processes (GAMS spawns gmsgennx.exe on Windows)
                                    children = p.children(recursive=True)
                                    child_info = ""
                                    if children:
                                        child_cpu = sum(
                                            c.cpu_percent(interval=0.1) for c in children
                                        )
                                        child_mem = (
                                            sum(c.memory_info().rss for c in children) / 1024 / 1024
                                        )
                                        # Calculate approximate core usage (CPU% / 100)
                                        cores_used = child_cpu / 100
                                        child_info = f" + {len(children)} worker(s) using ~{cores_used:.1f} cores, {child_mem:.0f}MB"

                                    display_text = Text(
                                        f"GAMS running... ({elapsed:.0f}s{child_info})",
                                        style="dim yellow",
                                    )
                                except:
                                    display_text = Text(
                                        f"Waiting for GAMS output... ({elapsed:.0f}s, {len(lines)} lines)",
                                        style="dim yellow",
                                    )
                            else:
                                display_text = Text(
                                    f"Waiting for GAMS output... ({elapsed:.0f}s, {len(lines)} lines)",
                                    style="dim yellow",
                                )
                            live_display.update(display_text)

                # Update monitor display if in monitor mode
                if use_monitor and monitor:
                    monitor.update_display()

                time.sleep(0.1)

            # One final update after process ends (standalone mode only)
            if not use_monitor and lines:
                display_lines = lines[-max_lines:]
                display_text = Text("\n".join(display_lines))
                live_display.update(display_text)

        finally:
            # Clean up standalone display
            if not use_monitor:
                live_display.stop()
    finally:
        # Restore signal handler only if we installed one
        if signal_handler_installed and old_handler is not None:
            with contextlib.suppress(ValueError, RuntimeError):
                signal.signal(signal.SIGINT, old_handler)

        # Update final status in monitor mode
        if use_monitor and not timed_out and monitor and run_name:
            # Check if already marked as cancelled
            if cancel_event and cancel_event.is_set():
                pass  # Already set to CANCELLED above
            elif proc.returncode == 0:
                monitor.update_status(run_name, RunStatus.COMPLETED)
                if job_registry and run_name:
                    job_registry.set_status(run_name, "completed")
            else:
                monitor.update_status(run_name, RunStatus.FAILED, f"Exit code: {proc.returncode}")
                if job_registry and run_name:
                    job_registry.set_status(run_name, "failed")

    reader.join(timeout=1)

    # Raise TimeoutError if timed out
    if timed_out:
        raise TimeoutError(f"GAMS run timed out after {timeout_seconds}s")

    # Only print diagnostics in standalone mode
    if not use_monitor:
        console.print(f"[dim]GAMS process exited with code: {proc.returncode}[/dim]")
        console.print(f"[dim]Total output lines captured: {len(lines)}[/dim]")

    return proc.returncode


def parse_range_stats(text: str) -> dict:
    sec = re.search(r"RANGE STATISTICS.*?(RHS.*?)(?:\n\n|\r\n\r\n)", text, flags=re.S | re.I)
    if not sec:
        return {}
    block = sec.group(0)

    def grab(name):
        m = re.search(rf"{name}.*?\[\s*([\-\d\.Ee\+]+)\s*,\s*([\-\d\.Ee\+]+)\s*\]", block)
        return (float(m.group(1)), float(m.group(2))) if m else None

    return {"rhs": grab("RHS"), "bound": grab("Bound"), "matrix": grab("Matrix")}


def parse_statuses(text: str) -> dict:
    m = re.search(r"MODEL STATUS\s*:\s*(.+)", text)
    s = re.search(r"SOLVER STATUS\s*:\s*(.+)", text)
    c = re.search(r"LP status\s*\((\d+)\)\s*:\s*([^\n\r]+)", text, flags=re.I)
    obj = re.search(r"Objective\s*=\s*([-+0-9Ee\.]+)", text)
    return {
        "model_status": (m.group(1).strip() if m else ""),
        "solver_status": (s.group(1).strip() if s else ""),
        "lp_status_code": (c.group(1) if c else ""),
        "lp_status_text": (c.group(2).strip() if c else ""),
        "objective": (obj.group(1) if obj else ""),
    }


def grep_mixed_currencies(run_dir: Path) -> list[str]:
    hits = []
    for pat in ["*parscen*.csv", "*parscen*.txt", "*SysSettings*.csv", "*SysSettings*.txt"]:
        for f in run_dir.glob(pat):
            try:
                txt = read_text(f)
            except Exception:
                continue
            if "AUD14" in txt and "AUD25" in txt:
                hits.append(str(f))
    return hits


def suggest_fixes(
    status: dict, ranges: dict, mixed_cur_files: list[str], used_barrier_noXO: bool
) -> list[str]:
    tips = []
    if (
        status.get("lp_status_text", "").lower().startswith("non-optimal")
        or status.get("lp_status_code") == "6"
    ):
        if used_barrier_noXO:
            tips.append(
                "Barrier without crossover returned status 6. Re-run with dual simplex (lpmethod 2, solutiontype 1) to certify OPTIMAL."
            )
        tips.append(
            "Run a short diagnostic with 'datacheck 2' to print range statistics and identify tiny coefficients/ill-conditioning."
        )
    matrix = ranges.get("matrix")
    if matrix:
        mn, mx = matrix
        if mn != 0.0 and abs(mn) < 1e-12:
            tips.append(
                f"Matrix min coefficient is {mn:.2e}. Rescale inputs: unify currencies, fix unit conversions, or drop near-zero coefficients."
            )
        if mx != 0.0 and abs(mx) > 1e8:
            tips.append(
                f"Matrix max coefficient is {mx:.2e}. Large range may cause numerical issues. Normalize units if possible."
            )
    if mixed_cur_files:
        tips.append(
            "Mixed currencies detected (e.g., AUD14 and AUD25). Standardise to AUD25 and ensure G_CUREX/syssettings include it; remove the older currency rows."
        )
    if not tips:
        tips.append(
            "No obvious red flags found. If still unstable, try 'lpmethod 5' (sifting) or tighten tolerances to 1e-7 and re-test."
        )
    return tips


@app.command()
def datacheck(
    run_dir: str,
    gams_path: str | None = typer.Option(
        None, "--gams-path", help="Path to gams.exe (defaults to 'gams' in PATH)"
    ),
    threads: int = typer.Option(7, help="Number of threads for CPLEX to use during datacheck"),
) -> None:
    """
    Rerun model with CPLEX datacheck mode for detailed diagnostics.

    Creates '_td_datacheck' directory, copies your run, and runs GAMS
    with CPLEX 'datacheck 2' enabled. This generates range statistics
    and identifies numerical issues WITHOUT solving the full model.

    Much faster than full solve. Identifies:
      - Matrix coefficient ranges (min/max)
      - Bound ranges
      - RHS ranges
      - Numerical conditioning issues

    After datacheck completes, run 'review' again to analyze results.

    \b
    Example:
      times-doctor datacheck data/065Nov25-annualupto2045/parscen

    \b
    Created directories:
      <run_dir>/_td_datacheck/       - Datacheck run copy
      <run_dir>/_td_datacheck/*.lst  - Listing with range stats
    """
    rd = Path(run_dir).resolve()

    gams_cmd = gams_path if gams_path else "gams"

    # Detect TIMES version
    lst = latest_lst(rd)
    times_version = detect_times_version(lst)
    times_src = get_times_source(version=times_version)

    pick_driver_gms(rd)
    tmp = rd / "_td_datacheck"

    if tmp.exists():
        print(f"[yellow]Removing existing datacheck directory: {tmp}[/yellow]")
        shutil.rmtree(tmp)

    print(f"[yellow]Creating datacheck run directory: {tmp}[/yellow]")
    shutil.copytree(rd, tmp)

    # Write CPLEX options file with datacheck enabled
    write_opt(
        tmp / "cplex.opt",
        [
            "datacheck 2",
            f"threads {threads}",
            "names yes",
            "lpmethod 2",
            "solutiontype 1",
            "scaind -1",
            "aggind 1",
            "eprhs 1.0e-06",
            "epopt 1.0e-06",
            "numericalemphasis 1",
            "simdisplay 2",
            "bardisplay 2",
        ],
    )

    tmp_driver = pick_driver_gms(tmp)
    dd_dir = rd.parent
    restart_file = times_src / "_times.g00"
    gdx_dir = tmp / "GAMSSAVE"
    gdx_file = gdx_dir / f"{tmp.name}.gdx"

    print(
        f"[yellow]Running GAMS datacheck with {threads} threads (this may take several minutes)...[/yellow]"
    )
    run_gams_with_progress(
        [
            gams_cmd,
            tmp_driver.name,
            f"r={restart_file}",
            f"idir1={tmp}",
            f"idir2={times_src}",
            f"idir3={dd_dir}",
            f"gdx={gdx_file}",
            "gdxcompress=1",
            "LP=CPLEX",
            "OPTFILE=1",
            "LOGOPTION=2",
            f"logfile={tmp.name}_run_log.txt",
            f"--GDXPATH={gdx_dir}/",
            "--ERR_ABORT=NO",
        ],
        cwd=str(tmp),
    )

    print("\n[green]✓ GAMS datacheck complete[/green]")
    print(f"[green]✓ Datacheck output saved to: {tmp}[/green]")

    # Parse and display basic status
    lst = latest_lst(tmp)
    if lst:
        lst_text = read_text(lst)
        ranges = parse_range_stats(lst_text)
        status = parse_statuses(lst_text)

        if ranges:
            print("\n[bold cyan]Range Statistics:[/bold cyan]")
            for k, (mn, mx) in ranges.items():
                print(f"  {k:8s}: min {mn:.3e}, max {mx:.3e}")

        if status:
            print("\n[bold cyan]Status:[/bold cyan]")
            print(f"  Model Status:  {status.get('model_status', 'N/A')}")
            print(f"  Solver Status: {status.get('solver_status', 'N/A')}")
            print(f"  LP Status:     {status.get('lp_status_text', 'N/A')}")

    print("\n[bold yellow]Next step:[/bold yellow]")
    print(
        f"  Run 'times-doctor review {run_dir}' to analyze the datacheck results with LLM assistance."
    )
    print(
        "  The review command will prompt you to select between the original run and this datacheck run."
    )


@app.command()
def rerun(
    run_dir: str,
    solver: str = typer.Option("cplex", "--solver", help="Solver to use: cplex|gurobi"),
    opt_file: str | None = typer.Option(
        None, "--opt-file", help="Path or name of .opt file to use"
    ),
    threads: int = typer.Option(
        7, help="Number of threads for solver to use (used in default .opt)"
    ),
    gams_path: str | None = typer.Option(
        None, "--gams-path", help="Path to gams.exe (defaults to 'gams' in PATH)"
    ),
    label: str | None = typer.Option(None, "--label", help="Optional label for rerun directory"),
    with_iis: bool = typer.Option(
        False,
        "--with-iis",
        help="Enable IIS/Conflict Refiner after solve (adds 'iis 1' to solver .opt)",
    ),
) -> None:
    """
    Rerun model with chosen solver and options.

    Creates '_td_rerun/<solver>/<label-or-timestamp>' directory, copies your run,
    and runs GAMS with the specified solver.

    If --opt-file is provided, uses that configuration.
    Otherwise writes default solver options.

    After rerun completes, you can run 'review-solver-options' to generate
    alternative configurations, then use 'scan' to test them all.

    \b
    Example:
      times-doctor rerun data/065Nov25-annualupto2045/parscen --solver gurobi
      times-doctor rerun data/065Nov25-annualupto2045/parscen --solver cplex --opt-file tight_tolerances
      times-doctor rerun data/065Nov25-annualupto2045/parscen --solver cplex --with-iis

    \b
    Created directories:
      <run_dir>/_td_rerun/<solver>/<label>/       - Rerun directory
      <run_dir>/_td_rerun/<solver>/<label>/*.lst  - Listing with solver output
    """
    rd = Path(run_dir).resolve()
    solver = solver.lower()

    if solver not in ("cplex", "gurobi"):
        console.print(f"[red]Error: solver must be 'cplex' or 'gurobi', got '{solver}'[/red]")
        raise typer.Exit(1)

    gams_cmd = gams_path if gams_path else "gams"

    # Detect TIMES version
    lst = latest_lst(rd)
    times_version = detect_times_version(lst)
    times_src = get_times_source(version=times_version)

    pick_driver_gms(rd)

    # Create rerun directory
    rerun_base = rd / "_td_rerun" / solver
    rerun_base.mkdir(parents=True, exist_ok=True)

    if label:
        tmp = rerun_base / label
    else:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        tmp = rerun_base / ts

    if tmp.exists():
        console.print(f"[yellow]Removing existing rerun directory: {tmp}[/yellow]")
        remove_tree_robust(tmp)

    console.print(f"[yellow]Creating rerun directory: {tmp}[/yellow]")
    shutil.copytree(rd, tmp)

    # Resolve and prepare .opt file
    opt_dst = tmp / f"{solver}.opt"
    resolved_opt = resolve_opt_file(rd, solver, opt_file)

    if resolved_opt:
        console.print(f"[yellow]Using {solver}.opt from: {resolved_opt}[/yellow]")
        shutil.copy(resolved_opt, opt_dst)
    else:
        console.print(f"[yellow]Writing default {solver}.opt with {threads} threads[/yellow]")
        write_default_opt(solver, opt_dst, threads)

    # Enable IIS if requested
    if with_iis:
        ensure_solver_option(opt_dst, "iis", "1")
        if solver == "cplex":
            ensure_solver_option(opt_dst, "names", "yes")
        console.print("[green]✓ Enabled IIS/Conflict Refiner (iis 1)[/green]")

    tmp_driver = pick_driver_gms(tmp)
    dd_dir = rd.parent
    restart_file = times_src / "_times.g00"
    gdx_dir = tmp / "GAMSSAVE"
    gdx_file = gdx_dir / f"{tmp.name}.gdx"

    console.print(
        f"[yellow]Running GAMS with {solver.upper()} solver (this may take several minutes)...[/yellow]"
    )

    # Choose parser and tracker based on solver
    if solver == "gurobi":
        parse_func = gurobi_progress.parse_gurobi_line
        tracker: Any = gurobi_progress.BarrierProgressTracker()
    else:
        parse_func = cplex_progress.parse_cplex_line
        tracker = cplex_progress.BarrierProgressTracker()

    run_gams_with_progress(
        [
            gams_cmd,
            tmp_driver.name,
            f"r={restart_file}",
            f"idir1={tmp}",
            f"idir2={times_src}",
            f"idir3={dd_dir}",
            f"gdx={gdx_file}",
            "gdxcompress=1",
            f"LP={solver.upper()}",
            "OPTFILE=1",
            "LOGOPTION=2",
            f"logfile={tmp.name}_run_log.txt",
            f"--GDXPATH={gdx_dir}/",
            "--ERR_ABORT=NO",
        ],
        cwd=str(tmp),
        parse_line_func=parse_func,
        progress_tracker=tracker,
    )

    console.print(f"\n[green]✓ GAMS {solver.upper()} run complete[/green]")
    console.print(f"[green]✓ Output saved to: {tmp}[/green]")

    # Parse and display basic status
    lst = latest_lst(tmp)
    if lst:
        lst_text = read_text(lst)
        status = parse_statuses(lst_text)

        if status:
            console.print("\n[bold cyan]Status:[/bold cyan]")
            console.print(f"  Model Status:  {status.get('model_status', 'N/A')}")
            console.print(f"  Solver Status: {status.get('solver_status', 'N/A')}")
            console.print(f"  LP Status:     {status.get('lp_status_text', 'N/A')}")

    console.print("\n[bold yellow]Next step:[/bold yellow]")
    console.print(
        f"  Run 'times-doctor review-solver-options {run_dir} --solver {solver}' to generate alternative configurations."
    )
    console.print(
        f"  Then run 'times-doctor scan {run_dir} --solver {solver}' to test all configurations."
    )


@app.command()
def scan(
    run_dir: str,
    solver: str = typer.Option("auto", "--solver", help="Solver to scan: auto|cplex|gurobi|both"),
    gams_path: str | None = typer.Option(
        None, "--gams-path", help="Path to gams.exe (defaults to 'gams' in PATH)"
    ),
    threads: int = typer.Option(
        7, help="Number of threads for solver to use (overrides threads setting in .opt files)"
    ),
    llm: str = typer.Option(
        "none", help="LLM provider for optional analysis: auto|openai|anthropic|amp|none"
    ),
    parallel: bool = typer.Option(
        False, "--parallel", help="Run profiles in parallel (faster but uses more resources)"
    ),
    max_workers: int | None = typer.Option(
        None,
        "--max-workers",
        envvar="TD_MAX_WORKERS",
        help="Max concurrent profile runs (default: auto by CPU/solver threads)",
    ),
    timeout_seconds: int | None = typer.Option(
        None,
        "--timeout-seconds",
        envvar="TD_TIMEOUT_SECONDS",
        help="Per-profile timeout in seconds (0=no timeout)",
    ),
) -> None:
    """
    Test multiple solver configurations to find best approach.

    Scans the run directory for LLM-generated solver .opt files in _td_opt_files/<solver>/
    and runs your model with each configuration to compare behavior.

    Use 'times-doctor review-solver-options --solver <solver>' first to generate configurations.

    By default, runs configurations sequentially. Use --parallel to run all
    configurations simultaneously (faster but uses more CPU/memory).

    Results summarized in CSV for easy comparison.

    \b
    Example:
      times-doctor review-solver-options data/065Nov25-annualupto2045/parscen --solver cplex
      times-doctor scan data/065Nov25-annualupto2045/parscen --solver cplex
      times-doctor scan data/... --solver gurobi --parallel  # Run all configs at once
      times-doctor scan data/... --solver both  # Test both CPLEX and GUROBI

    \b
    Created directories:
      <run_dir>/_td_scan/<solver>/<config_name>/
      <run_dir>/_td_scan/scan_report.csv
    """
    rd = Path(run_dir).resolve()
    solver = solver.lower()

    # Discover solver profiles
    profiles: list[tuple[str, Path, str]] = []  # List of (solver_name, opt_file_path, config_name)

    opt_files_base = rd / "_td_opt_files"
    if not opt_files_base.exists():
        console.print("[red]Error: No _td_opt_files/ directory found.[/red]")
        console.print(
            "\n[yellow]Run 'times-doctor review-solver-options --solver <solver>' first to generate solver configurations.[/yellow]"
        )
        raise typer.Exit(1)

    # Discover based on --solver flag
    if solver in ("auto", "cplex", "both"):
        cplex_dir = opt_files_base / "cplex"
        if cplex_dir.exists():
            cplex_files = sorted(cplex_dir.glob("*.opt"))
            profiles.extend(("cplex", f, f.stem) for f in cplex_files)

    if solver in ("auto", "gurobi", "both"):
        gurobi_dir = opt_files_base / "gurobi"
        if gurobi_dir.exists():
            gurobi_files = sorted(gurobi_dir.glob("*.opt"))
            profiles.extend(("gurobi", f, f.stem) for f in gurobi_files)

    # Fallback: Check for flat structure (backward compatibility)
    if not profiles and solver == "auto":
        flat_files = sorted(opt_files_base.glob("*.opt"))
        if flat_files:
            console.print("[yellow]Found .opt files in flat structure, treating as CPLEX[/yellow]")
            profiles.extend(("cplex", f, f.stem) for f in flat_files)

    if not profiles:
        console.print(f"[red]Error: No .opt files found for solver(s): {solver}[/red]")
        console.print(
            "\n[yellow]Run 'times-doctor review-solver-options --solver <solver>' first to generate configurations.[/yellow]"
        )
        raise typer.Exit(1)

    # Show discovered profiles
    console.print(f"\n[bold]Found {len(profiles)} solver configuration(s):[/bold]")
    for i, (slvr, _opt_path, cfg_name) in enumerate(profiles, 1):
        console.print(f"  {i:>2}. [{slvr.upper()}] {cfg_name}")

    if not typer.confirm(f"\nRun scan with these {len(profiles)} configurations?", default=True):
        console.print("[yellow]Scan cancelled.[/yellow]")
        raise typer.Exit(0)

    gams_cmd = gams_path if gams_path else "gams"

    # Detect TIMES version from existing run
    lst = latest_lst(rd)
    times_version = detect_times_version(lst)
    times_src = get_times_source(version=times_version)

    dd_dir = rd.parent
    restart_file = times_src / "_times.g00"
    pick_driver_gms(rd)
    scan_root = rd / "_td_scan"
    scan_root.mkdir(exist_ok=True)

    # Compute worker limit based on CPU and solver threads
    def compute_workers(n_profiles: int, solver_threads: int, override: int | None) -> int:
        """Calculate max concurrent workers based on CPU cores and solver threads."""
        cpus = os.cpu_count() or 1
        if override is not None and override > 0:
            return max(1, min(override, n_profiles))
        # Auto: allocate workers to avoid CPU oversubscription, cap at 6
        per = max(1, cpus // max(1, solver_threads))
        return max(1, min(per, n_profiles, 6))

    workers = compute_workers(len(profiles), threads, max_workers)

    # Show setup summary BEFORE starting the work
    console.print(f"\n[bold]Preparing {len(profiles)} run directories under:[/bold] {scan_root}")
    console.print(
        f"[dim]Mode: {'parallel' if parallel else 'sequential'}, "
        f"max_workers={workers}, solver threads/run={threads}[/dim]"
    )
    console.print(f"[dim]GAMS: {gams_cmd}, TIMES: {times_src}[/dim]")

    console.print("\n[bold]Jobs to launch:[/bold]")
    for i, (slvr, _opt_path, cfg_name) in enumerate(profiles, 1):
        run_path = scan_root / slvr / cfg_name
        console.print(f"  {i:>2}. [{slvr.upper()}] {cfg_name}  ->  {run_path}")

    # Set up run directories and copy .opt files with progress feedback
    setup_start = time.monotonic()
    console.print(
        f"\n[yellow]Setting up run directories (copying base run {len(profiles)} times - may take several minutes)...[/yellow]"
    )

    run_dirs: dict[str, tuple[str, Path]] = {}  # profile_name -> (solver, wdir)
    total = len(profiles)
    with console.status("[bold green]Preparing run directories...") as status:
        for idx, (slvr, opt_path, cfg_name) in enumerate(profiles, 1):
            wdir = scan_root / slvr / cfg_name
            profile_key = f"{slvr}:{cfg_name}"  # Unique key for this profile
            status.update(f"[bold green]({idx}/{total}) Preparing [{slvr.upper()}] {cfg_name}...")

            if wdir.exists():
                ok = remove_tree_robust(wdir)
                if not ok:
                    # Final fallback: use fresh timestamped subdir for this run
                    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                    wdir = scan_root / slvr / cfg_name / ts
                    wdir.parent.mkdir(parents=True, exist_ok=True)
                    console.print(
                        f"[yellow]Using alternate directory due to locks: {wdir}[/yellow]"
                    )
                elif wdir.exists():
                    # Directory still exists after remove_tree_robust (rare race condition)
                    # Use timestamped alternative
                    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                    wdir = scan_root / slvr / cfg_name / ts
                    wdir.parent.mkdir(parents=True, exist_ok=True)
                    console.print(
                        f"[yellow]Directory still locked after cleanup, using: {wdir}[/yellow]"
                    )

            wdir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(
                rd,
                wdir,
                ignore=shutil.ignore_patterns(
                    "times_doctor_out", "_td_opt_files", "_td_scan", "_td_rerun"
                ),
                dirs_exist_ok=True,
            )

            # Copy the corresponding .opt file with correct solver name
            dst_opt = wdir / f"{slvr}.opt"
            shutil.copy2(opt_path, dst_opt)

            # Override threads setting if specified
            ensure_threads_option(dst_opt, slvr, threads)

            # Ensure bardisplay/simdisplay for CPLEX to enable progress tracking
            if slvr == "cplex":
                ensure_display_options(dst_opt)

            run_dirs[profile_key] = (slvr, wdir)

    setup_elapsed = time.monotonic() - setup_start
    console.print(f"[green]Setup complete in {setup_elapsed:.1f}s[/green]")
    console.print("\n[bold]Launching jobs...[/bold]")

    # Helper function to run a single profile
    def run_profile(
        profile_name: str,
        profile_solver: str,
        monitor: MultiRunProgressMonitor,
        results: dict[str, dict[str, str]],
        timeout_sec: int | None = None,
        cancel_event: threading.Event | None = None,
        registry: "JobRegistry | None" = None,
    ) -> None:
        """Run a single profile and store results."""
        # Check if already cancelled before spawning
        if cancel_event and cancel_event.is_set():
            monitor.update_status(profile_name, RunStatus.CANCELLED)
            if registry:
                registry.set_status(profile_name, "cancelled")
            results[profile_name] = {
                "profile": profile_name,
                "solver": profile_solver.upper(),
                "model_status": "CANCELLED",
                "solver_status": "CANCELLED",
                "lp_status": "",
                "objective": "",
                "runtime": "",
                "runtime_seconds": "",
                "matrix_min": "",
                "matrix_max": "",
                "dir": str(run_dirs[profile_name][1]),
                "lst": "",
            }
            return

        try:
            slvr, wdir = run_dirs[profile_name]
            wdir_driver = pick_driver_gms(wdir)
            wdir_gdx_dir = wdir / "GAMSSAVE"
            wdir_gdx_file = wdir_gdx_dir / f"{wdir.name}.gdx"

            # Choose parser and tracker based on solver
            if slvr == "gurobi":
                parse_func = gurobi_progress.parse_gurobi_line
                tracker: Any = gurobi_progress.BarrierProgressTracker()
            else:
                parse_func = cplex_progress.parse_cplex_line
                tracker = cplex_progress.BarrierProgressTracker()

            run_gams_with_progress(
                [
                    gams_cmd,
                    wdir_driver.name,
                    f"r={restart_file}",
                    f"idir1={wdir}",
                    f"idir2={times_src}",
                    f"idir3={dd_dir}",
                    f"gdx={wdir_gdx_file}",
                    "gdxcompress=1",
                    f"LP={slvr.upper()}",
                    "OPTFILE=1",
                    "LOGOPTION=2",
                    f"logfile={wdir.name}_run_log.txt",
                    f"--GDXPATH={wdir_gdx_dir}/",
                    "--ERR_ABORT=NO",
                ],
                cwd=str(wdir),
                monitor=monitor,
                run_name=profile_name,
                timeout_seconds=timeout_sec,
                cancel_event=cancel_event,
                job_registry=registry,
                parse_line_func=parse_func,
                progress_tracker=tracker,
            )

            # Parse results
            lst = latest_lst(wdir)
            text = read_text(lst) if lst else ""
            st = parse_statuses(text)
            rng = parse_range_stats(text)

            # Update termination result in monitor
            monitor.update_result(
                profile_name, st.get("lp_status_text") or st.get("solver_status") or "–"
            )

            # Get elapsed time from monitor
            elapsed_time = monitor.runs[profile_name].get_elapsed_time()
            time_str = monitor.runs[profile_name].format_elapsed() if elapsed_time else ""

            # Store in thread-safe dict
            results[profile_name] = {
                "profile": profile_name,
                "solver": slvr.upper(),
                "model_status": st.get("model_status", ""),
                "solver_status": st.get("solver_status", ""),
                "lp_status": st.get("lp_status_text", ""),
                "objective": st.get("objective", ""),
                "runtime": time_str,
                "runtime_seconds": f"{elapsed_time:.1f}" if elapsed_time else "",
                "matrix_min": f"{rng.get('matrix', (None, None))[0]:.3e}"
                if rng.get("matrix")
                else "",
                "matrix_max": f"{rng.get('matrix', (None, None))[1]:.3e}"
                if rng.get("matrix")
                else "",
                "dir": str(wdir),
                "lst": str(lst) if lst else "",
            }
        except TimeoutError:
            slvr, wdir = run_dirs[profile_name]
            elapsed_time = monitor.runs[profile_name].get_elapsed_time()
            time_str = monitor.runs[profile_name].format_elapsed() if elapsed_time else ""
            results[profile_name] = {
                "profile": profile_name,
                "solver": slvr.upper(),
                "model_status": "ERROR",
                "solver_status": "TIMEOUT",
                "lp_status": "",
                "objective": "",
                "runtime": time_str,
                "runtime_seconds": f"{elapsed_time:.1f}" if elapsed_time else "",
                "matrix_min": "",
                "matrix_max": "",
                "dir": str(wdir),
                "lst": "",
            }
        except KeyboardInterrupt:
            slvr, wdir = run_dirs[profile_name]
            elapsed_time = monitor.runs[profile_name].get_elapsed_time()
            time_str = monitor.runs[profile_name].format_elapsed() if elapsed_time else ""
            results[profile_name] = {
                "profile": profile_name,
                "solver": slvr.upper(),
                "model_status": "CANCELLED",
                "solver_status": "CANCELLED",
                "lp_status": "",
                "objective": "",
                "runtime": time_str,
                "runtime_seconds": f"{elapsed_time:.1f}" if elapsed_time else "",
                "matrix_min": "",
                "matrix_max": "",
                "dir": str(wdir),
                "lst": "",
            }
        except Exception as e:
            slvr, wdir = run_dirs[profile_name]
            console.print(f"[red]Error running profile {profile_name}: {e}[/red]")
            elapsed_time = monitor.runs[profile_name].get_elapsed_time()
            time_str = monitor.runs[profile_name].format_elapsed() if elapsed_time else ""
            results[profile_name] = {
                "profile": profile_name,
                "solver": slvr.upper(),
                "model_status": "ERROR",
                "solver_status": str(e),
                "lp_status": "",
                "objective": "",
                "runtime": time_str,
                "runtime_seconds": f"{elapsed_time:.1f}" if elapsed_time else "",
                "matrix_min": "",
                "matrix_max": "",
                "dir": str(wdir),
                "lst": "",
            }

    # Create job registry for cancellation support
    job_registry = JobRegistry(state_path=scan_root / "scan_state.json")

    # Create monitor for tracking all profiles
    mode_str = "parallel" if parallel else "sequential"
    worker_info = f" (max {workers} concurrent)" if parallel else ""
    console.print(
        f"[cyan]Running {len(profiles)} profiles in {mode_str} mode{worker_info}...[/cyan]"
    )

    # Build profile keys for monitoring
    profile_keys = [f"{slvr}:{cfg_name}" for slvr, _, cfg_name in profiles]

    with MultiRunProgressMonitor(profile_keys, title="Scan Progress") as monitor:
        from times_doctor.interactive_cancel import InteractiveCancelController

        results: dict[str, dict[str, str]] = {}  # Thread-safe dict to collect results

        # Register all jobs
        cancel_events = {p: job_registry.register(p) for p in profile_keys}

        # Install interactive cancel controller
        controller = InteractiveCancelController(job_registry, monitor, console).install()

        try:
            # Use ThreadPoolExecutor for both parallel and sequential (max_workers=1 for sequential)
            actual_workers = workers if parallel else 1
            with ThreadPoolExecutor(
                max_workers=actual_workers, thread_name_prefix="scan"
            ) as executor:
                # Submit all jobs
                future_map = {
                    executor.submit(
                        run_profile,
                        profile_key,
                        slvr,
                        monitor,
                        results,
                        timeout_seconds,
                        cancel_events[profile_key],
                        job_registry,
                    ): profile_key
                    for slvr, _, cfg_name in profiles
                    for profile_key in [f"{slvr}:{cfg_name}"]
                }

                pending = set(future_map.keys())

                # Control loop with timeout-based polling
                while pending:
                    done, pending = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)

                    # Update display
                    monitor.update_display()

                    # Check if user pressed Ctrl-C and show menu if needed
                    if controller.should_open_menu():
                        controller.show_menu()

                    # Process completed futures
                    for future in done:
                        profile_name = future_map[future]
                        try:
                            future.result()
                        except KeyboardInterrupt:
                            # Job was cancelled
                            pass
                        except Exception as e:
                            console.print(f"[red]Worker failed for {profile_name}: {e}[/red]")

        finally:
            # Restore signal handler
            controller.uninstall()

        # Convert results dict to rows list (preserve profile order)
        rows = [results[p] for p in profile_keys if p in results]

    # Show summary table
    t = Table(title="Scan Summary")
    for col in [
        "profile",
        "solver",
        "model_status",
        "solver_status",
        "lp_status",
        "objective",
        "runtime",
        "matrix_min",
        "matrix_max",
    ]:
        t.add_column(col)
    for r in rows:
        t.add_row(
            *[
                r[c]
                for c in [
                    "profile",
                    "solver",
                    "model_status",
                    "solver_status",
                    "lp_status",
                    "objective",
                    "runtime",
                    "matrix_min",
                    "matrix_max",
                ]
            ]
        )
    console.print(t)

    csvp = scan_root / "scan_report.csv"
    with csvp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    if llm.lower() != "none":
        diag = {
            "status": {"mode": "scan"},
            "ranges": {},
            "mixed_currency_files": [],
            "used_barrier_noXO": False,
        }
        res = llm_mod.summarize(diag, provider=llm)
        if res.used:
            (scan_root / "scan_llm_advice.md").write_text(res.text, encoding="utf-8")

    print(f"[green]Wrote[/green] {csvp}")


def find_run_directories(base_dir: Path) -> list[tuple[Path, str, float]]:
    """Find all available run directories with timestamps.

    Returns list of (path, label, mtime) tuples sorted by recency.
    """
    candidates: list[tuple[Path, str, float]] = []

    # Add the main directory
    lst = latest_lst(base_dir)
    if lst:
        mtime = lst.stat().st_mtime
        candidates.append((base_dir, f"{base_dir.name} (original run)", mtime))

    # Look for _td_datacheck and other _td_* subdirectories
    for subdir in base_dir.glob("_td_*"):
        if subdir.is_dir():
            lst = latest_lst(subdir)
            if lst:
                mtime = lst.stat().st_mtime
                import datetime

                dt = datetime.datetime.fromtimestamp(mtime)
                time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                candidates.append((subdir, f"{subdir.name} ({time_str})", mtime))

    # Sort by most recent first
    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates


@app.command()
def review(
    run_dir: str,
    llm: str = typer.Option("auto", help="LLM provider: auto|openai|anthropic|amp|none"),
    model: str = typer.Option("", help="Specific model to use (will prompt if not specified)"),
    reasoning_level: str = typer.Option(
        "high",
        "--reasoning-level",
        help="Reasoning effort: minimal|low|medium|high (default: high)",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show cost estimate without making API calls"
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompts"),
) -> None:
    """
    Review TIMES run files using LLM for human-readable diagnostics.

    ⭐ START HERE - Primary command for diagnosing TIMES model issues.

    Analyzes QA_CHECK.LOG, run log, and .lst files from your failed run.
    Provides clear explanations of what went wrong and actionable steps
    to fix your model.

    If multiple runs exist (original + datacheck), you'll be prompted
    to select which one to review.

    Identifies: infeasibilities, numerical issues, matrix coefficient
    problems, solver configuration recommendations.

    After review, LLM may suggest running 'datacheck' for deeper analysis.

    \b
    Example:
      times-doctor review data/065Nov25-annualupto2045/parscen
      times-doctor review data/... --dry-run  # Show cost estimate first
      times-doctor review data/... --yes      # Skip prompts

    \b
    Output:
      <run_dir>/times_doctor_out/llm_review.md  ← Read this!
      <run_dir>/_llm_calls/                     ← API call logs
    """
    rd = Path(run_dir).resolve()

    # Check for multiple run directories
    run_dirs = find_run_directories(rd)

    if len(run_dirs) > 1:
        print(f"\n[bold]Found {len(run_dirs)} run directories:[/bold]")
        for i, (_path, label, _mtime) in enumerate(run_dirs, 1):
            print(f"  {i}. {label}")

        choice = typer.prompt(f"\nSelect run to review (1-{len(run_dirs)})", type=int, default=1)
        if 1 <= choice <= len(run_dirs):
            rd = run_dirs[choice - 1][0]
            print(f"[green]Selected: {rd}[/green]")
        else:
            print(f"[yellow]Invalid choice, using: {run_dirs[0][0]}[/yellow]")
            rd = run_dirs[0][0]

    api_keys = llm_mod.check_api_keys()
    has_any_key = any(api_keys.values())

    if llm.lower() != "none" and not has_any_key:
        print(
            "[yellow]No API keys found. Please configure one of the following in a .env file:[/yellow]"
        )
        print("  OPENAI_API_KEY=sk-...")
        print("  ANTHROPIC_API_KEY=sk-ant-...")
        print("  AMP_API_KEY=... (or ensure 'amp' CLI is available)")
        print("")
        print("[dim]Create a .env file in the current directory with one of these keys.[/dim]")
        raise typer.Exit(1)

    qa_check_path = rd / "QA_CHECK.LOG"
    qa_check = read_text(qa_check_path) if qa_check_path.exists() else ""

    run_log_path = None
    for f in rd.glob("*_run_log.txt"):
        run_log_path = f
        break
    run_log = read_text(run_log_path) if run_log_path else ""

    lst = latest_lst(rd)
    lst_text = read_text(lst) if lst else ""

    if not qa_check and not run_log and not lst_text:
        print(f"[red]No QA_CHECK.LOG, *_run_log.txt, or .lst files found in {rd}[/red]")
        raise typer.Exit(1)

    print("[yellow]Found files:[/yellow]")
    if qa_check:
        print(f"  ✓ QA_CHECK.LOG ({len(qa_check)} chars)")
    if run_log and run_log_path:
        print(f"  ✓ {run_log_path.name} ({len(run_log)} chars)")
    if lst_text and lst:
        print(f"  ✓ {lst.name} ({len(lst_text)} chars)")

    # Extract useful sections first
    llm_log_dir = rd / "_llm_calls"

    # Determine which fast model will be used
    api_keys = llm_mod.check_api_keys()
    fast_model = (
        "gpt-5-nano"
        if api_keys["openai"]
        else ("claude-3-5-haiku-20241022" if api_keys["anthropic"] else "unknown")
    )

    # DRY RUN: Show cost estimate and exit
    if dry_run:
        print("\n[bold yellow]DRY RUN - Cost Estimation[/bold yellow]")
        print("\n[cyan]Files to analyze:[/cyan]")

        total_chars = 0
        if qa_check:
            chars = len(qa_check)
            tokens = estimate_tokens(qa_check)
            total_chars += chars
            print(f"  • QA_CHECK.LOG: {chars:,} chars (~{tokens:,} tokens)")

        if run_log and run_log_path:
            chars = len(run_log)
            tokens = estimate_tokens(run_log)
            total_chars += chars
            print(f"  • {run_log_path.name}: {chars:,} chars (~{tokens:,} tokens)")

        if lst_text and lst:
            chars = len(lst_text)
            tokens = estimate_tokens(lst_text)
            total_chars += chars
            print(f"  • {lst.name}: {chars:,} chars (~{tokens:,} tokens)")

        print(f"\n[cyan]Total input size:[/cyan] {total_chars:,} chars")

        # Estimate condensing costs
        print(f"\n[cyan]Step 1: Condensing with fast model ({fast_model})[/cyan]")
        condense_input_tokens = estimate_tokens(qa_check + run_log + lst_text)
        # Assume condensing reduces by ~70%
        condense_output_tokens = int(condense_input_tokens * 0.3)
        _, _, condense_cost = estimate_cost(
            qa_check + run_log + lst_text, " " * (condense_output_tokens * 4), fast_model
        )
        print(f"  Input tokens: ~{condense_input_tokens:,}")
        print(f"  Output tokens: ~{condense_output_tokens:,} (estimated)")
        print(f"  Cost: ${condense_cost:.4f}")

        # Determine reasoning model
        reasoning_model = (
            "gpt-5 (high effort)"
            if api_keys["openai"]
            else ("claude-3-5-sonnet-20241022" if api_keys["anthropic"] else "unknown")
        )
        reasoning_model_name = (
            "gpt-5"
            if api_keys["openai"]
            else ("claude-3-5-sonnet-20241022" if api_keys["anthropic"] else "gpt-5")
        )

        print(f"\n[cyan]Step 2: Review with reasoning model ({reasoning_model})[/cyan]")
        review_input_tokens = condense_output_tokens
        # Assume review generates ~2000 token response
        review_output_tokens = 2000
        _, _, review_cost = estimate_cost(
            " " * (review_input_tokens * 4), " " * (review_output_tokens * 4), reasoning_model_name
        )
        print(f"  Input tokens: ~{review_input_tokens:,}")
        print(f"  Output tokens: ~{review_output_tokens:,} (estimated)")
        print(f"  Cost: ${review_cost:.4f}")

        total_cost = condense_cost + review_cost
        print(f"\n[bold green]Estimated total cost: ${total_cost:.4f} USD[/bold green]")
        print("[dim]Note: Actual costs may vary based on content complexity[/dim]")
        print("\n[yellow]To proceed with actual analysis, run without --dry-run flag[/yellow]")

        return

    print("\n[bold yellow]Condensing files...[/bold yellow]")
    print(f"[dim](LLM calls logged to {llm_log_dir})[/dim]")

    condensed_qa_check = ""
    condensed_run_log = ""
    condensed_lst = ""

    try:
        if qa_check:
            print("[dim]  Condensing QA_CHECK.LOG...[/dim]")

            def qa_progress(current: int, total: int, message: str) -> None:
                if current == 0:
                    print(f"[dim]    {message}[/dim]")
                else:
                    print(f"[dim]    {message}[/dim]")

            condensed_qa_check = llm_mod.condense_qa_check(qa_check, progress_callback=qa_progress)
            condensed_qa_check_path = rd / "QA_CHECK.condensed.LOG"
            condensed_qa_check_path.write_text(condensed_qa_check, encoding="utf-8")
            print(f"[green]  ✓ Saved {condensed_qa_check_path}[/green]")

        if run_log and run_log_path:
            print(f"[dim]  Condensing {run_log_path.name}...[/dim]")

            def runlog_progress(current: int, total: int, message: str) -> None:
                if current == 0:
                    print(f"[dim]    {message}[/dim]")
                else:
                    print(f"[dim]    {message}[/dim]")

            sections = llm_mod.extract_condensed_sections(
                run_log, "run_log", log_dir=llm_log_dir, progress_callback=runlog_progress
            )

            # For run_log, we get filtered_content directly
            if "filtered_content" in sections and sections["filtered_content"]:
                condensed_run_log = (
                    f"# Run Log - Filtered\n\n```\n{sections['filtered_content']}\n```\n"
                )
            else:
                condensed_run_log = llm_mod.create_condensed_markdown(
                    run_log.split("\n"), sections["sections"], "run_log"
                )

            condensed_run_log_path = rd / f"{run_log_path.stem}.condensed.txt"
            condensed_run_log_path.write_text(condensed_run_log, encoding="utf-8")
            print(f"[green]  ✓ Saved {condensed_run_log_path}[/green]")

        if lst_text and lst:
            print(f"[dim]  Condensing {lst.name}...[/dim]")

            def lst_progress(current: int, total: int, message: str) -> None:
                if current == 0:
                    print(f"[dim]    {message}[/dim]")
                else:
                    print(f"[dim]    {message}[/dim]")

            sections = llm_mod.extract_condensed_sections(
                lst_text, "lst", log_dir=llm_log_dir, progress_callback=lst_progress
            )
            condensed_lst = llm_mod.create_condensed_markdown(
                lst_text.split("\n"), sections["sections"], "lst"
            )
            condensed_lst_path = rd / f"{lst.stem}.condensed.lst"
            condensed_lst_path.write_text(condensed_lst, encoding="utf-8")
            print(f"[green]  ✓ Saved {condensed_lst_path}[/green]")
    except Exception as e:
        print(f"[red]Error during extraction: {e}[/red]")
        print(f"[yellow]Check {llm_log_dir} for detailed logs[/yellow]")
        raise typer.Exit(1)

    # Determine reasoning model
    reasoning_model = (
        "gpt-5 (high effort)"
        if api_keys["openai"]
        else ("claude-3-5-sonnet-20241022" if api_keys["anthropic"] else "unknown")
    )

    print(f"\n[bold cyan]Sending condensed files to reasoning LLM ({reasoning_model}):[/bold cyan]")
    if condensed_qa_check:
        print("  • QA_CHECK.condensed.LOG")
    if condensed_run_log and run_log_path:
        print(f"  • {run_log_path.stem}.condensed.txt")
    if condensed_lst and lst:
        print(f"  • {lst.stem}.condensed.lst")

    print("\n[bold green]Reviewing...[/bold green]\n")

    # Create streaming callback to display output as it comes
    def stream_output(chunk: str) -> None:
        print(chunk, end="", flush=True)

    result = llm_mod.review_files(
        condensed_qa_check,
        condensed_run_log,
        condensed_lst,
        provider=llm,
        model=model,
        reasoning_effort=reasoning_level,
        stream_callback=stream_output,
        log_dir=llm_log_dir,
    )

    if not result.used:
        print("[red]Failed to get LLM response. Check API keys and connectivity.[/red]")
        raise typer.Exit(1)

    # If streaming wasn't used (e.g., fallback to responses API), print the result
    if not result.text or "\n" not in result.text:
        print(result.text)

    print("\n")  # New line after output

    print(f"\n[bold cyan]LLM Provider:[/bold cyan] {result.provider}")
    if result.model:
        print(f"[bold cyan]Model:[/bold cyan] {result.model}")
    if result.input_tokens > 0:
        print(
            f"[bold cyan]Tokens:[/bold cyan] {result.input_tokens:,} in + {result.output_tokens:,} out = {result.input_tokens + result.output_tokens:,} total"
        )
    if result.cost_usd > 0:
        print(f"[bold cyan]Cost:[/bold cyan] ${result.cost_usd:.4f} USD")

    out = ensure_out(rd)
    review_path = out / "llm_review.md"
    review_path.write_text(result.text, encoding="utf-8")

    print(f"\n[green]Saved to {review_path}[/green]")


@app.command()
def explain_infeasibility(
    run_dir: str,
    llm: str = typer.Option("auto", help="LLM provider: auto|openai|anthropic|amp|none"),
    model: str = typer.Option("", help="Specific model to use (default: gpt-5)"),
    reasoning_level: str = typer.Option(
        "medium",
        "--reasoning-level",
        help="Reasoning effort: minimal|low|medium|high (default: medium)",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show cost estimate without making API calls"
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompts"),
) -> None:
    """
    Diagnose WHY a TIMES model is infeasible and EXACTLY how to fix it.

    ⭐ Use this when your TIMES run returns INFEASIBLE status.

    Analyzes QA_CHECK.LOG, run log, and .lst files to identify:
    - The specific constraints/variables causing infeasibility
    - The mechanism of the contradiction
    - Minimal conflicting set (MCS) of equations
    - Step-by-step remediation plan with exact Veda table edits

    Provides concrete fix instructions referencing specific processes,
    commodities, regions, years, and timeslices.

    \b
    Example:
      times-doctor explain-infeasibility data/065Nov25-annualupto2045/parscen
      times-doctor explain-infeasibility data/... --dry-run  # Cost estimate
      times-doctor explain-infeasibility data/... --yes      # Skip prompts

    \b
    Output:
      <run_dir>/times_doctor_out/infeasibility_diagnosis.md  ← Read this!
      <run_dir>/_llm_calls/                                  ← API call logs
    """
    rd = Path(run_dir).resolve()

    # Check for multiple run directories
    run_dirs = find_run_directories(rd)

    if len(run_dirs) > 1:
        print(f"\n[bold]Found {len(run_dirs)} run directories:[/bold]")
        for i, (_path, label, _mtime) in enumerate(run_dirs, 1):
            print(f"  {i}. {label}")

        choice = typer.prompt(f"\nSelect run to analyze (1-{len(run_dirs)})", type=int, default=1)
        if 1 <= choice <= len(run_dirs):
            rd = run_dirs[choice - 1][0]
            print(f"[green]Selected: {rd}[/green]")
        else:
            print(f"[yellow]Invalid choice, using: {run_dirs[0][0]}[/yellow]")
            rd = run_dirs[0][0]

    api_keys = llm_mod.check_api_keys()
    has_any_key = any(api_keys.values())

    if llm.lower() != "none" and not has_any_key:
        print("[yellow]No API keys found. Please configure OPENAI_API_KEY in a .env file:[/yellow]")
        print("  OPENAI_API_KEY=sk-...")
        print("")
        print("[dim]explain-infeasibility requires OpenAI API for reasoning[/dim]")
        raise typer.Exit(1)

    if llm.lower() not in ("auto", "openai", "none"):
        print("[yellow]explain-infeasibility only supports OpenAI provider with reasoning[/yellow]")
        print("  Use --llm openai or --llm auto (default)")
        raise typer.Exit(1)

    qa_check_path = rd / "QA_CHECK.LOG"
    qa_check = read_text(qa_check_path) if qa_check_path.exists() else ""

    run_log_path = None
    for f in rd.glob("*_run_log.txt"):
        run_log_path = f
        break
    run_log = read_text(run_log_path) if run_log_path else ""

    lst = latest_lst(rd)
    lst_text = read_text(lst) if lst else ""

    if not qa_check and not run_log and not lst_text:
        print(f"[red]No QA_CHECK.LOG, *_run_log.txt, or .lst files found in {rd}[/red]")
        raise typer.Exit(1)

    print("[yellow]Found files:[/yellow]")
    if qa_check:
        print(f"  ✓ QA_CHECK.LOG ({len(qa_check)} chars)")
    if run_log and run_log_path:
        print(f"  ✓ {run_log_path.name} ({len(run_log)} chars)")
    if lst_text and lst:
        print(f"  ✓ {lst.name} ({len(lst_text)} chars)")

    # Check for IIS/Conflict Refiner output
    solver_used = detect_solver_from_files(run_log, lst_text)
    iis_found, iis_reason = detect_iis_present(solver_used, run_log, lst_text)

    if not iis_found:
        print("\n[yellow]⚠ No IIS/Conflict Refiner output detected in this run.[/yellow]")
        print("[dim]IIS analysis helps identify the minimal conflicting set of constraints.[/dim]")
        if solver_used:
            print("\n[cyan]To enable IIS, rerun with:[/cyan]")
            print(f"  times-doctor rerun {rd} --solver {solver_used} --with-iis")
        else:
            print("\n[cyan]To enable IIS, rerun with one of:[/cyan]")
            print(f"  times-doctor rerun {rd} --solver cplex --with-iis")
            print(f"  times-doctor rerun {rd} --solver gurobi --with-iis")

        if not yes and not typer.confirm("\nProceed with analysis without IIS?", default=True):
            print("[dim]Exiting. Please rerun with IIS enabled for better analysis.[/dim]")
            raise typer.Exit(0)
        print("")
    else:
        print(f"[green]✓ IIS output detected ({iis_reason})[/green]")

    # Extract useful sections first
    llm_log_dir = rd / "_llm_calls"

    # Determine which fast model will be used
    api_keys = llm_mod.check_api_keys()
    fast_model = (
        "gpt-5-nano"
        if api_keys["openai"]
        else ("claude-3-5-haiku-20241022" if api_keys["anthropic"] else "unknown")
    )

    # DRY RUN: Show cost estimate and exit
    if dry_run:
        print("\n[bold yellow]DRY RUN - Cost Estimation[/bold yellow]")
        print("\n[cyan]Files to analyze:[/cyan]")

        total_chars = 0
        if qa_check:
            chars = len(qa_check)
            tokens = estimate_tokens(qa_check)
            total_chars += chars
            print(f"  • QA_CHECK.LOG: {chars:,} chars (~{tokens:,} tokens)")

        if run_log and run_log_path:
            chars = len(run_log)
            tokens = estimate_tokens(run_log)
            total_chars += chars
            print(f"  • {run_log_path.name}: {chars:,} chars (~{tokens:,} tokens)")

        if lst_text and lst:
            chars = len(lst_text)
            tokens = estimate_tokens(lst_text)
            total_chars += chars
            print(f"  • {lst.name}: {chars:,} chars (~{tokens:,} tokens)")

        print(f"\n[cyan]Total input size:[/cyan] {total_chars:,} chars")

        # Estimate condensing costs
        print(f"\n[cyan]Step 1: Condensing with fast model ({fast_model})[/cyan]")
        condense_input_tokens = estimate_tokens(qa_check + run_log + lst_text)
        condense_output_tokens = int(condense_input_tokens * 0.3)
        _, _, condense_cost = estimate_cost(
            qa_check + run_log + lst_text, " " * (condense_output_tokens * 4), fast_model
        )
        print(f"  Input tokens: ~{condense_input_tokens:,}")
        print(f"  Output tokens: ~{condense_output_tokens:,} (estimated)")
        print(f"  Cost: ${condense_cost:.4f}")

        # Estimate reasoning cost
        reasoning_model_name = "gpt-5"
        print(
            f"\n[cyan]Step 2: Infeasibility diagnosis with reasoning model (gpt-5 {reasoning_level})[/cyan]"
        )
        review_input_tokens = condense_output_tokens
        review_output_tokens = 3000  # Infeasibility diagnosis typically longer
        _, _, review_cost = estimate_cost(
            " " * (review_input_tokens * 4),
            " " * (review_output_tokens * 4),
            reasoning_model_name,
        )
        print(f"  Input tokens: ~{review_input_tokens:,}")
        print(f"  Output tokens: ~{review_output_tokens:,} (estimated)")
        print(f"  Cost: ${review_cost:.4f}")

        total_cost = condense_cost + review_cost
        print(f"\n[bold green]Estimated total cost: ${total_cost:.4f} USD[/bold green]")
        print("[dim]Note: Actual costs may vary based on content complexity[/dim]")
        print("\n[yellow]To proceed with actual analysis, run without --dry-run flag[/yellow]")

        return

    print("\n[bold yellow]Condensing files...[/bold yellow]")
    print(f"[dim](LLM calls logged to {llm_log_dir})[/dim]")

    condensed_qa_check = ""
    condensed_run_log = ""
    condensed_lst = ""

    try:
        if qa_check:
            print("[dim]  Condensing QA_CHECK.LOG...[/dim]")

            def qa_progress(current: int, total: int, message: str) -> None:
                print(f"[dim]    {message}[/dim]")

            condensed_qa_check = llm_mod.condense_qa_check(qa_check, progress_callback=qa_progress)
            condensed_qa_check_path = rd / "QA_CHECK.condensed.LOG"
            condensed_qa_check_path.write_text(condensed_qa_check, encoding="utf-8")
            print(f"[green]  ✓ Saved {condensed_qa_check_path}[/green]")

        if run_log and run_log_path:
            print(f"[dim]  Condensing {run_log_path.name}...[/dim]")

            def runlog_progress(current: int, total: int, message: str) -> None:
                print(f"[dim]    {message}[/dim]")

            sections = llm_mod.extract_condensed_sections(
                run_log, "run_log", log_dir=llm_log_dir, progress_callback=runlog_progress
            )

            if "filtered_content" in sections and sections["filtered_content"]:
                condensed_run_log = (
                    f"# Run Log - Filtered\n\n```\n{sections['filtered_content']}\n```\n"
                )
            else:
                condensed_run_log = llm_mod.create_condensed_markdown(
                    run_log.split("\n"), sections["sections"], "run_log"
                )

            condensed_run_log_path = rd / f"{run_log_path.stem}.condensed.txt"
            condensed_run_log_path.write_text(condensed_run_log, encoding="utf-8")
            print(f"[green]  ✓ Saved {condensed_run_log_path}[/green]")

        if lst_text and lst:
            print(f"[dim]  Condensing {lst.name}...[/dim]")

            def lst_progress(current: int, total: int, message: str) -> None:
                print(f"[dim]    {message}[/dim]")

            sections = llm_mod.extract_condensed_sections(
                lst_text, "lst", log_dir=llm_log_dir, progress_callback=lst_progress
            )
            condensed_lst = llm_mod.create_condensed_markdown(
                lst_text.split("\n"), sections["sections"], "lst"
            )
            condensed_lst_path = rd / f"{lst.stem}.condensed.lst"
            condensed_lst_path.write_text(condensed_lst, encoding="utf-8")
            print(f"[green]  ✓ Saved {condensed_lst_path}[/green]")
    except Exception as e:
        print(f"[red]Error during extraction: {e}[/red]")
        print(f"[yellow]Check {llm_log_dir} for detailed logs[/yellow]")
        raise typer.Exit(1)

    print(
        f"\n[bold cyan]Analyzing infeasibility with reasoning LLM (gpt-5 {reasoning_level}):[/bold cyan]"
    )
    if condensed_qa_check:
        print("  • QA_CHECK.condensed.LOG")
    if condensed_run_log and run_log_path:
        print(f"  • {run_log_path.stem}.condensed.txt")
    if condensed_lst and lst:
        print(f"  • {lst.stem}.condensed.lst")

    print("\n[bold green]Diagnosing infeasibility...[/bold green]\n")

    # Create streaming callback to display output as it comes
    def stream_output(chunk: str) -> None:
        print(chunk, end="", flush=True)

    result = llm_mod.explain_infeasibility(
        condensed_qa_check,
        condensed_run_log,
        condensed_lst,
        provider=llm,
        model=model,
        reasoning_effort=reasoning_level,
        stream_callback=stream_output,
        log_dir=llm_log_dir,
    )

    if not result.used:
        print("[red]Failed to get LLM response. Check API keys and connectivity.[/red]")
        raise typer.Exit(1)

    # If streaming wasn't used, print the result
    if not result.text or "\n" not in result.text:
        print(result.text)

    print("\n")  # New line after output

    print(f"\n[bold cyan]LLM Provider:[/bold cyan] {result.provider}")
    if result.model:
        print(f"[bold cyan]Model:[/bold cyan] {result.model}")
    if result.input_tokens > 0:
        print(
            f"[bold cyan]Tokens:[/bold cyan] {result.input_tokens:,} in + {result.output_tokens:,} out = {result.input_tokens + result.output_tokens:,} total"
        )
    if result.cost_usd > 0:
        print(f"[bold cyan]Cost:[/bold cyan] ${result.cost_usd:.4f} USD")

    out = ensure_out(rd)
    diagnosis_path = out / "infeasibility_diagnosis.md"
    diagnosis_path.write_text(result.text, encoding="utf-8")

    print(f"\n[green]Saved to {diagnosis_path}[/green]")


@app.command()
def review_qa_check(
    run_dir: str,
    llm: str = typer.Option("auto", help="LLM provider: auto|openai|anthropic|amp|none"),
    model: str = typer.Option("", help="Specific model to use (will prompt if not specified)"),
    reasoning_level: str = typer.Option(
        "medium",
        "--reasoning-level",
        help="Reasoning effort: minimal|low|medium|high (default: medium)",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show cost estimate without making API calls"
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompts"),
) -> None:
    """
    Get actionable fix recommendations for QA_CHECK.LOG issues using Oracle.

    Specialized command focused SOLELY on providing concrete remediation steps
    for each issue identified in QA_CHECK.LOG. Uses oracle (reasoning LLM) to
    analyze all files but returns only actionable "where to look" and "how to fix"
    instructions.

    Unlike 'review' which provides general diagnostics, this command:
    - Parses QA_CHECK.LOG into structured issues
    - Provides specific file/table locations for each issue
    - Gives step-by-step fix instructions with examples
    - Includes validation steps to confirm fixes

    \b
    Example:
      times-doctor review-qa-check data/065Nov25-annualupto2045/parscen
      times-doctor review-qa-check data/... --dry-run  # Cost estimate first

    \b
    Output:
      <run_dir>/times_doctor_out/qa_check_fixes.md  ← Fix instructions!
      <run_dir>/_llm_calls/                          ← API call logs
    """
    import json

    from .core import qa_check_parser
    from .core.prompts import build_review_qa_fixes_prompt

    rd = Path(run_dir).resolve()

    # Check for multiple run directories (same as review command)
    run_dirs = find_run_directories(rd)
    if len(run_dirs) > 1:
        print(f"\n[bold]Found {len(run_dirs)} run directories:[/bold]")
        for i, (_path, label, _mtime) in enumerate(run_dirs, 1):
            print(f"  {i}. {label}")

        choice = typer.prompt(f"\nSelect run to review (1-{len(run_dirs)})", type=int, default=1)
        if 1 <= choice <= len(run_dirs):
            rd = run_dirs[choice - 1][0]
            print(f"[green]Selected: {rd}[/green]")
        else:
            print(f"[yellow]Invalid choice, using: {run_dirs[0][0]}[/yellow]")
            rd = run_dirs[0][0]

    api_keys = llm_mod.check_api_keys()
    has_any_key = any(api_keys.values())

    if llm.lower() != "none" and not has_any_key:
        print(
            "[yellow]No API keys found. Please configure one of the following in a .env file:[/yellow]"
        )
        print("  OPENAI_API_KEY=sk-...")
        print("  ANTHROPIC_API_KEY=sk-ant-...")
        print("  AMP_API_KEY=... (or ensure 'amp' CLI is available)")
        print("")
        print("[dim]Create a .env file in the current directory with one of these keys.[/dim]")
        raise typer.Exit(1)

    # Find files
    qa_check_path = rd / "QA_CHECK.LOG"
    if not qa_check_path.exists():
        print(f"[red]No QA_CHECK.LOG found in {rd}[/red]")
        raise typer.Exit(1)

    run_log_path = None
    for f in rd.glob("*_run_log.txt"):
        run_log_path = f
        break
    run_log = read_text(run_log_path) if run_log_path else ""

    lst = latest_lst(rd)
    lst_text = read_text(lst) if lst else ""

    print("[yellow]Found files:[/yellow]")
    print("  ✓ QA_CHECK.LOG")
    if run_log and run_log_path:
        print(f"  ✓ {run_log_path.name}")
    if lst_text and lst:
        print(f"  ✓ {lst.name}")

    llm_log_dir = rd / "_llm_calls"

    # Determine fast model for condensing
    fast_model = (
        "gpt-5-nano"
        if api_keys["openai"]
        else ("claude-3-5-haiku-20241022" if api_keys["anthropic"] else "unknown")
    )

    # DRY RUN: Show cost estimate
    if dry_run:
        print("\n[bold yellow]DRY RUN - Cost Estimation[/bold yellow]")
        qa_text = read_text(qa_check_path)

        # Estimate condensing costs (run_log and lst only)
        print(f"\n[cyan]Step 1: Condensing run_log and lst with fast model ({fast_model})[/cyan]")
        condense_input_tokens = estimate_tokens(run_log + lst_text)
        condense_output_tokens = int(condense_input_tokens * 0.3)
        _, _, condense_cost = estimate_cost(
            run_log + lst_text, " " * (condense_output_tokens * 4), fast_model
        )
        print(f"  Input tokens: ~{condense_input_tokens:,}")
        print(f"  Output tokens: ~{condense_output_tokens:,} (estimated)")
        print(f"  Cost: ${condense_cost:.4f}")

        # Estimate oracle cost
        reasoning_model = (
            "gpt-5 (medium effort)"
            if api_keys["openai"]
            else ("claude-3-5-sonnet-20241022" if api_keys["anthropic"] else "unknown")
        )
        reasoning_model_name = "gpt-5" if api_keys["openai"] else "claude-3-5-sonnet-20241022"

        print(f"\n[cyan]Step 2: Oracle analysis with reasoning model ({reasoning_model})[/cyan]")
        # Include QA_CHECK size (rule-based condensed) + condensed excerpts
        oracle_input_tokens = estimate_tokens(qa_text) // 3 + condense_output_tokens
        oracle_output_tokens = 3000  # More detailed fix recommendations
        _, _, oracle_cost = estimate_cost(
            " " * (oracle_input_tokens * 4),
            " " * (oracle_output_tokens * 4),
            reasoning_model_name,
        )
        print(f"  Input tokens: ~{oracle_input_tokens:,}")
        print(f"  Output tokens: ~{oracle_output_tokens:,} (estimated)")
        print(f"  Cost: ${oracle_cost:.4f}")

        total_cost = condense_cost + oracle_cost
        print(f"\n[bold green]Estimated total cost: ${total_cost:.4f} USD[/bold green]")
        print("[dim]Note: Actual costs may vary based on content complexity[/dim]")
        print("\n[yellow]To proceed with actual analysis, run without --dry-run flag[/yellow]")
        return

    # Step 1: Condense run_log and lst (same as review command)
    print("\n[bold yellow]Condensing files...[/bold yellow]")
    print(f"[dim](LLM calls logged to {llm_log_dir})[/dim]")

    condensed_run_log = ""
    condensed_lst = ""

    try:
        if run_log and run_log_path:
            print(f"[dim]  Condensing {run_log_path.name}...[/dim]")

            def runlog_progress(current: int, total: int, message: str) -> None:
                print(f"[dim]    {message}[/dim]")

            sections = llm_mod.extract_condensed_sections(
                run_log, "run_log", log_dir=llm_log_dir, progress_callback=runlog_progress
            )

            if "filtered_content" in sections and sections["filtered_content"]:
                condensed_run_log = (
                    f"# Run Log - Filtered\n\n```\n{sections['filtered_content']}\n```\n"
                )
            else:
                condensed_run_log = llm_mod.create_condensed_markdown(
                    run_log.split("\n"), sections["sections"], "run_log"
                )

        if lst_text and lst:
            print(f"[dim]  Condensing {lst.name}...[/dim]")

            def lst_progress(current: int, total: int, message: str) -> None:
                print(f"[dim]    {message}[/dim]")

            sections = llm_mod.extract_condensed_sections(
                lst_text, "lst", log_dir=llm_log_dir, progress_callback=lst_progress
            )
            condensed_lst = llm_mod.create_condensed_markdown(
                lst_text.split("\n"), sections["sections"], "lst"
            )
    except Exception as e:
        print(f"[red]Error during extraction: {e}[/red]")
        print(f"[yellow]Check {llm_log_dir} for detailed logs[/yellow]")
        raise typer.Exit(1)

    # Step 2: Parse QA_CHECK.LOG into structured issues (rule-based)
    print("[dim]  Parsing QA_CHECK.LOG into structured issues...[/dim]")

    summary_rows, msg_counts, all_keys = qa_check_parser.condense_log_to_rows(
        qa_check_path, index_allow=["R", "P", "V", "T", "CG", "COM"], min_severity="WARNING"
    )

    rolled = qa_check_parser.rollup_summary_rows(summary_rows, None, sample_limit=3)
    qa_rollup_text = qa_check_parser.format_condensed_output(summary_rows, msg_counts, all_keys)

    # Build structured JSON for oracle
    qa_struct = {
        "source": str(qa_check_path),
        "severity_order": qa_check_parser.SEVERITY_ORDER,
        "issues": [
            {
                "severity": r["severity"],
                "message": r["message"],
                "occurrences": int(r["occurrences"]),
                "aggregates": r.get("aggregates", ""),
                "samples": r.get("samples", ""),
            }
            for r in rolled
        ],
        "message_counts": [
            {"severity": m["severity"], "message": m["message"], "count": int(m["events"])}
            for m in msg_counts
        ],
    }
    qa_json = json.dumps(qa_struct, ensure_ascii=False, indent=2)

    # Step 3: Build oracle prompt
    prompt = build_review_qa_fixes_prompt(
        qa_struct_json=qa_json,
        qa_rollup_text=qa_rollup_text,
        run_log_excerpts=condensed_run_log,
        lst_excerpts=condensed_lst,
    )

    # Determine reasoning model
    reasoning_model = (
        "gpt-5 (medium effort)"
        if api_keys["openai"]
        else ("claude-3-5-sonnet-20241022" if api_keys["anthropic"] else "unknown")
    )

    print(
        f"\n[bold cyan]Sending to Oracle ({reasoning_model}) for fix recommendations:[/bold cyan]"
    )
    print("  • QA_CHECK.LOG (structured, rule-based)")
    if condensed_run_log:
        print("  • Run log (condensed)")
    if condensed_lst:
        print("  • LST file (condensed)")

    print("\n[bold green]Consulting Oracle...[/bold green]\n")

    # Create streaming callback
    def stream_output(chunk: str) -> None:
        print(chunk, end="", flush=True)

    # Step 4: Call oracle
    result = llm_mod.review_qa_check_fixes(
        prompt,
        provider=llm,
        model=model,
        reasoning_effort=reasoning_level,
        stream_callback=stream_output,
        log_dir=llm_log_dir,
    )

    if not result.used:
        print("[red]Failed to get LLM response. Check API keys and connectivity.[/red]")
        raise typer.Exit(1)

    # If streaming wasn't used, print the result
    if not result.text or "\n" not in result.text:
        print(result.text)

    print("\n")  # New line after output

    print(f"\n[bold cyan]LLM Provider:[/bold cyan] {result.provider}")
    if result.model:
        print(f"[bold cyan]Model:[/bold cyan] {result.model}")
    if result.input_tokens > 0:
        print(
            f"[bold cyan]Tokens:[/bold cyan] {result.input_tokens:,} in + {result.output_tokens:,} out = {result.input_tokens + result.output_tokens:,} total"
        )
    if result.cost_usd > 0:
        print(f"[bold cyan]Cost:[/bold cyan] ${result.cost_usd:.4f} USD")

    # Save output
    out = ensure_out(rd)
    fixes_path = out / "qa_check_fixes.md"
    fixes_path.write_text(result.text, encoding="utf-8")

    print(f"\n[green]✓ Saved actionable fix instructions to {fixes_path}[/green]")
    print("[dim]Review the file for concrete remediation steps for each QA issue[/dim]")


@app.command()
def review_solver_options(
    run_dir: str,
    solver: str = typer.Option("auto", "--solver", help="Solver type: auto|cplex|gurobi"),
    llm: str = typer.Option("auto", help="LLM provider: auto|openai|anthropic|amp|none"),
    model: str = typer.Option("", help="Specific model to use (will prompt if not specified)"),
    reasoning_level: str = typer.Option(
        "high",
        "--reasoning-level",
        help="Reasoning effort: minimal|low|medium|high (default: high)",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show cost estimate without making API calls"
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompts"),
) -> None:
    """
    Review solver options for feasible-but-not-optimal solutions.

    Use this command when your TIMES model has returned a FEASIBLE but NOT PROVEN OPTIMAL solution.
    The LLM will analyze your run files and solver .opt configuration to suggest specific parameter
    tuning (tolerances, etc.) to improve the chances of reaching proven optimal status.

    This command assumes you're using barrier method without crossover (the standard for large TIMES
    models) and will NOT suggest changing the solver algorithm - only tuning parameters within it.

    \b
    Example:
      times-doctor review-solver-options data/065Nov25-annualupto2045/parscen --solver cplex
      times-doctor review-solver-options data/... --solver gurobi
      times-doctor review-solver-options data/... --dry-run  # Show cost estimate first
      times-doctor review-solver-options data/... --yes      # Skip prompts

    \b
    Output:
      <run_dir>/times_doctor_out/solver_options_review.md  ← Read this!
      <run_dir>/_td_opt_files/<solver>/                    ← Generated .opt files
      <run_dir>/_llm_calls/                                ← API call logs
    """
    rd = Path(run_dir).resolve()

    # Check for multiple run directories
    run_dirs = find_run_directories(rd)

    if len(run_dirs) > 1:
        print(f"\n[bold]Found {len(run_dirs)} run directories:[/bold]")
        for i, (_path, label, _mtime) in enumerate(run_dirs, 1):
            print(f"  {i}. {label}")

        choice = typer.prompt(f"\nSelect run to review (1-{len(run_dirs)})", type=int, default=1)
        if 1 <= choice <= len(run_dirs):
            rd = run_dirs[choice - 1][0]
            print(f"[green]Selected: {rd}[/green]")
        else:
            print(f"[yellow]Invalid choice, using: {run_dirs[0][0]}[/yellow]")
            rd = run_dirs[0][0]

    api_keys = llm_mod.check_api_keys()
    has_any_key = any(api_keys.values())

    if llm.lower() != "none" and not has_any_key:
        print(
            "[yellow]No API keys found. Please configure one of the following in a .env file:[/yellow]"
        )
        print("  OPENAI_API_KEY=sk-...")
        print("  ANTHROPIC_API_KEY=sk-ant-...")
        print("  AMP_API_KEY=... (or ensure 'amp' CLI is available)")
        print("")
        print("[dim]Create a .env file in the current directory with one of these keys.[/dim]")
        raise typer.Exit(1)

    # Read input files
    qa_check_path = rd / "QA_CHECK.LOG"
    qa_check = read_text(qa_check_path) if qa_check_path.exists() else ""

    run_log_path = None
    for f in rd.glob("*_run_log.txt"):
        run_log_path = f
        break
    run_log = read_text(run_log_path) if run_log_path else ""

    lst = latest_lst(rd)
    lst_text = read_text(lst) if lst else ""

    # Detect solver if auto
    detected_solver = solver.lower()
    if detected_solver == "auto":
        # Strategy: Check which .opt file exists in the current run directory
        # This tells us which solver was actually used for this run

        if (rd / "gurobi.opt").exists() and not (rd / "cplex.opt").exists():
            detected_solver = "gurobi"
            print("[dim]Auto-detected solver: GUROBI (found gurobi.opt)[/dim]")
        elif (rd / "cplex.opt").exists() and not (rd / "gurobi.opt").exists():
            detected_solver = "cplex"
            print("[dim]Auto-detected solver: CPLEX (found cplex.opt)[/dim]")
        elif (rd / "gurobi.opt").exists() and (rd / "cplex.opt").exists():
            # Both exist - check which is newer
            gurobi_mtime = (rd / "gurobi.opt").stat().st_mtime
            cplex_mtime = (rd / "cplex.opt").stat().st_mtime
            if gurobi_mtime > cplex_mtime:
                detected_solver = "gurobi"
                print("[dim]Auto-detected solver: GUROBI (gurobi.opt is newer)[/dim]")
            else:
                detected_solver = "cplex"
                print("[dim]Auto-detected solver: CPLEX (cplex.opt is newer)[/dim]")
        else:
            # No .opt files found - default to CPLEX
            detected_solver = "cplex"
            print("[yellow]No solver .opt file found, defaulting to CPLEX[/yellow]")
            print(
                "[yellow]Tip: Use --solver cplex or --solver gurobi to specify explicitly[/yellow]"
            )
    else:
        print(f"[dim]Using specified solver: {detected_solver.upper()}[/dim]")

    # Read solver .opt file
    opt_path = rd / f"{detected_solver}.opt"
    opt_content = read_text(opt_path) if opt_path.exists() else ""

    if not opt_content:
        print(f"[yellow]Warning: No {detected_solver}.opt file found in {rd}[/yellow]")

    if not qa_check and not run_log and not lst_text:
        print(f"[red]No QA_CHECK.LOG, *_run_log.txt, or .lst files found in {rd}[/red]")
        raise typer.Exit(1)

    print("[yellow]Found files:[/yellow]")
    if opt_content:
        print(f"  ✓ {detected_solver}.opt ({len(opt_content)} chars)")
    if qa_check:
        print(f"  ✓ QA_CHECK.LOG ({len(qa_check)} chars)")
    if run_log and run_log_path:
        print(f"  ✓ {run_log_path.name} ({len(run_log)} chars)")
    if lst_text and lst:
        print(f"  ✓ {lst.name} ({len(lst_text)} chars)")

    # Extract useful sections first
    llm_log_dir = rd / "_llm_calls"

    # Determine which fast model will be used
    api_keys = llm_mod.check_api_keys()
    fast_model = (
        "gpt-5-nano"
        if api_keys["openai"]
        else ("claude-3-5-haiku-20241022" if api_keys["anthropic"] else "unknown")
    )

    # DRY RUN: Show cost estimate and exit
    if dry_run:
        print("\n[bold yellow]DRY RUN - Cost Estimation[/bold yellow]")
        print("\n[cyan]Files to analyze:[/cyan]")

        total_chars = len(opt_content)
        if opt_content:
            chars = len(opt_content)
            tokens = estimate_tokens(opt_content)
            total_chars += chars
            print(f"  • {detected_solver}.opt: {chars:,} chars (~{tokens:,} tokens)")

        if qa_check:
            chars = len(qa_check)
            tokens = estimate_tokens(qa_check)
            total_chars += chars
            print(f"  • QA_CHECK.LOG: {chars:,} chars (~{tokens:,} tokens)")

        if run_log and run_log_path:
            chars = len(run_log)
            tokens = estimate_tokens(run_log)
            total_chars += chars
            print(f"  • {run_log_path.name}: {chars:,} chars (~{tokens:,} tokens)")

        if lst_text and lst:
            chars = len(lst_text)
            tokens = estimate_tokens(lst_text)
            total_chars += chars
            print(f"  • {lst.name}: {chars:,} chars (~{tokens:,} tokens)")

        print(f"\n[cyan]Total input size:[/cyan] {total_chars:,} chars")

        # Estimate condensing costs
        print(f"\n[cyan]Step 1: Condensing with fast model ({fast_model})[/cyan]")
        condense_input_tokens = estimate_tokens(qa_check + run_log + lst_text)
        # Assume condensing reduces by ~70%
        condense_output_tokens = int(condense_input_tokens * 0.3)
        _, _, condense_cost = estimate_cost(
            qa_check + run_log + lst_text, " " * (condense_output_tokens * 4), fast_model
        )
        print(f"  Input tokens: ~{condense_input_tokens:,}")
        print(f"  Output tokens: ~{condense_output_tokens:,} (estimated)")
        print(f"  Cost: ${condense_cost:.4f}")

        # Determine reasoning model
        reasoning_model = (
            "gpt-5 (high effort)"
            if api_keys["openai"]
            else ("claude-3-5-sonnet-20241022" if api_keys["anthropic"] else "unknown")
        )
        reasoning_model_name = (
            "gpt-5"
            if api_keys["openai"]
            else ("claude-3-5-sonnet-20241022" if api_keys["anthropic"] else "gpt-5")
        )

        print(f"\n[cyan]Step 2: Review with reasoning model ({reasoning_model})[/cyan]")
        review_input_tokens = condense_output_tokens + estimate_tokens(opt_content)
        # Assume review generates ~2000 token response
        review_output_tokens = 2000
        _, _, review_cost = estimate_cost(
            " " * (review_input_tokens * 4), " " * (review_output_tokens * 4), reasoning_model_name
        )
        print(f"  Input tokens: ~{review_input_tokens:,}")
        print(f"  Output tokens: ~{review_output_tokens:,} (estimated)")
        print(f"  Cost: ${review_cost:.4f}")

        total_cost = condense_cost + review_cost
        print(f"\n[bold green]Estimated total cost: ${total_cost:.4f} USD[/bold green]")
        print("[dim]Note: Actual costs may vary based on content complexity[/dim]")
        print("\n[yellow]To proceed with actual analysis, run without --dry-run flag[/yellow]")

        return

    print("\n[bold yellow]Condensing files...[/bold yellow]")
    print(f"[dim](LLM calls logged to {llm_log_dir})[/dim]")

    condensed_qa_check = ""
    condensed_run_log = ""
    condensed_lst = ""

    try:
        if qa_check:
            print("[dim]  Condensing QA_CHECK.LOG...[/dim]")

            def qa_progress(current: int, total: int, message: str) -> None:
                if current == 0:
                    print(f"[dim]    {message}[/dim]")
                else:
                    print(f"[dim]    {message}[/dim]")

            condensed_qa_check = llm_mod.condense_qa_check(qa_check, progress_callback=qa_progress)
            condensed_qa_check_path = rd / "QA_CHECK.condensed.LOG"
            condensed_qa_check_path.write_text(condensed_qa_check, encoding="utf-8")
            print(f"[green]  ✓ Saved {condensed_qa_check_path}[/green]")

        if run_log and run_log_path:
            print(f"[dim]  Condensing {run_log_path.name}...[/dim]")

            def runlog_progress(current: int, total: int, message: str) -> None:
                if current == 0:
                    print(f"[dim]    {message}[/dim]")
                else:
                    print(f"[dim]    {message}[/dim]")

            sections = llm_mod.extract_condensed_sections(
                run_log, "run_log", log_dir=llm_log_dir, progress_callback=runlog_progress
            )

            # For run_log, we get filtered_content directly
            if "filtered_content" in sections and sections["filtered_content"]:
                condensed_run_log = (
                    f"# Run Log - Filtered\n\n```\n{sections['filtered_content']}\n```\n"
                )
            else:
                condensed_run_log = llm_mod.create_condensed_markdown(
                    run_log.split("\n"), sections["sections"], "run_log"
                )

            condensed_run_log_path = rd / f"{run_log_path.stem}.condensed.txt"
            condensed_run_log_path.write_text(condensed_run_log, encoding="utf-8")
            print(f"[green]  ✓ Saved {condensed_run_log_path}[/green]")

        if lst_text and lst:
            print(f"[dim]  Condensing {lst.name}...[/dim]")

            def lst_progress(current: int, total: int, message: str) -> None:
                if current == 0:
                    print(f"[dim]    {message}[/dim]")
                else:
                    print(f"[dim]    {message}[/dim]")

            sections = llm_mod.extract_condensed_sections(
                lst_text, "lst", log_dir=llm_log_dir, progress_callback=lst_progress
            )
            condensed_lst = llm_mod.create_condensed_markdown(
                lst_text.split("\n"), sections["sections"], "lst"
            )
            condensed_lst_path = rd / f"{lst.stem}.condensed.lst"
            condensed_lst_path.write_text(condensed_lst, encoding="utf-8")
            print(f"[green]  ✓ Saved {condensed_lst_path}[/green]")
    except Exception as e:
        print(f"[red]Error during extraction: {e}[/red]")
        print(f"[yellow]Check {llm_log_dir} for detailed logs[/yellow]")
        raise typer.Exit(1)

    # Determine reasoning model
    reasoning_model = (
        "gpt-5 (high effort)"
        if api_keys["openai"]
        else ("claude-3-5-sonnet-20241022" if api_keys["anthropic"] else "unknown")
    )

    print(f"\n[bold cyan]Sending files to reasoning LLM ({reasoning_model}):[/bold cyan]")
    if opt_content:
        print(f"  • {detected_solver}.opt")
    if condensed_qa_check:
        print("  • QA_CHECK.condensed.LOG")
    if condensed_run_log and run_log_path:
        print(f"  • {run_log_path.stem}.condensed.txt")
    if condensed_lst and lst:
        print(f"  • {lst.stem}.condensed.lst")

    print("\n[bold green]Reviewing solver options...[/bold green]\n")

    # For structured output, don't stream - use a spinner instead
    from rich.console import Console

    console = Console()

    with console.status("[cyan]Analyzing solver results with LLM...[/cyan]", spinner="dots"):
        result = llm_mod.review_solver_options(
            condensed_qa_check,
            condensed_run_log,
            condensed_lst,
            opt_content,
            solver=detected_solver,
            provider=llm,
            model=model,
            reasoning_effort=reasoning_level,
            stream_callback=None,  # No streaming for structured output
            log_dir=llm_log_dir,
        )

    if not result.used:
        print("[red]Failed to get LLM response. Check API keys and connectivity.[/red]")
        raise typer.Exit(1)

    out = ensure_out(rd)
    review_path = out / "solver_options_review.md"

    # Parse structured output (result.text is the Pydantic model instance)
    from times_doctor.core.solver_models import SolverDiagnosis
    from times_doctor.core.solver_validation import normalize_opt_config, validate_solver_diagnosis

    # Result.text is the Pydantic model instance when using structured output
    if isinstance(result.text, SolverDiagnosis):
        diagnosis = result.text
    else:
        # Fallback: parse JSON if for some reason we got a string
        import json

        data = json.loads(result.text)
        diagnosis = SolverDiagnosis(**data)

    # Validate and normalize CPLEX options
    if detected_solver == "cplex":
        print("\n[dim]Validating CPLEX options...[/dim]")
        is_valid, error_messages = validate_solver_diagnosis(diagnosis, solver=detected_solver)

        if not is_valid:
            print("[yellow]Warning: Some CPLEX options are invalid:[/yellow]")
            for msg in error_messages:
                print(f"[yellow]{msg}[/yellow]")
            print("\n[cyan]Normalizing options (removing invalid ones)...[/cyan]")

        # Normalize all configurations
        diagnosis.opt_configurations = [
            normalize_opt_config(config, solver=detected_solver)
            for config in diagnosis.opt_configurations
        ]
        print("[green]✓ Options validated and normalized[/green]")

    # Create markdown from structured output
    md_lines = []
    md_lines.append("# Solver Options Review\n")
    md_lines.append("## Diagnosis\n")
    md_lines.append(diagnosis.summary + "\n")
    md_lines.append("\n## Generated Configurations\n")
    md_lines.append(f"Generated {len(diagnosis.opt_configurations)} solver configurations:\n")
    md_lines.extend(
        f"- `{config.filename}`: {config.description}" for config in diagnosis.opt_configurations
    )
    md_lines.append("\n## Action Plan\n")
    md_lines.extend(f"{i}. {item}" for i, item in enumerate(diagnosis.action_plan, 1))

    review_text = "\n".join(md_lines)
    review_path.write_text(review_text, encoding="utf-8")

    # Extract solver algorithm settings from original solver .opt
    from times_doctor.core.opt_renderer import extract_solver_algorithm, render_opt_lines

    base_algorithm = extract_solver_algorithm(opt_content)
    if detected_solver == "cplex":
        if not base_algorithm.get("lpmethod"):
            print("[yellow]Warning: Could not extract lpmethod from original cplex.opt[/yellow]")
        if not base_algorithm.get("solutiontype"):
            print(
                "[yellow]Warning: Could not extract solutiontype from original cplex.opt[/yellow]"
            )

    # Extract .opt files from structured output into solver-specific subdirectory
    opt_dir = rd / "_td_opt_files" / detected_solver
    opt_dir.mkdir(parents=True, exist_ok=True)

    created_files = []
    for _i, config in enumerate(diagnosis.opt_configurations, start=1):
        # Render .opt file lines with enforced algorithm settings
        opt_lines = render_opt_lines(config, base_algorithm, warn_on_override=True)
        opt_content = "\n".join(opt_lines)

        # Use filename as-is (no numeric prefix - scan will handle ordering)
        opt_path = opt_dir / config.filename
        opt_path.write_text(opt_content, encoding="utf-8")
        created_files.append(opt_path)

    # Render clean terminal output
    _render_solver_diagnosis(diagnosis, created_files, review_path)

    # Show token/cost stats
    print(f"\n[bold cyan]LLM:[/bold cyan] {result.provider}/{result.model}")
    if result.input_tokens > 0:
        print(
            f"[bold cyan]Tokens:[/bold cyan] {result.input_tokens:,} in → {result.output_tokens:,} out = {result.input_tokens + result.output_tokens:,} total"
        )
    if result.cost_usd > 0:
        print(f"[bold cyan]Cost:[/bold cyan] ${result.cost_usd:.4f} USD")

    print(f"\n[dim]Next: times-doctor scan {rd} --solver {detected_solver}[/dim]")


@app.command()
def update() -> None:
    """
    Show instructions to update times-doctor.

    \b
    Example:
      times-doctor update
    """
    print("[bold]To update times-doctor:[/bold]\n")

    print("[cyan]If installed with uv tool:[/cyan]")
    print("  uv tool upgrade times-doctor\n")

    print("[cyan]If installed with pip:[/cyan]")
    print("  pip install --upgrade times-doctor\n")

    print("[cyan]If using uvx (no install):[/cyan]")
    print("  uvx automatically uses the latest version\n")

    print(f"[dim]Current version:[/dim] {__version__}")


# Run utility commands for condensing log files
run_utility_app = typer.Typer(help="Utility commands for condensing and processing log files")
app.add_typer(run_utility_app, name="run-utility")


@run_utility_app.command("condense-qa-check")
def condense_qa_check_cmd(
    input_file: Path = typer.Argument(..., help="Path to QA_CHECK.LOG file"),
    output_file: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path (default: QA_CHECK.condensed.LOG in same directory)",
    ),
) -> None:
    """Condense a QA_CHECK.LOG file into a compact summary."""
    if not input_file.exists():
        print(f"[red]Error: File not found: {input_file}[/red]")
        raise typer.Exit(1)

    print(f"[dim]Reading {input_file}...[/dim]")
    content = read_text(input_file)

    print(f"[dim]Condensing QA_CHECK.LOG ({len(content)} chars)...[/dim]")

    def progress_callback(current: int, total: int, message: str) -> None:
        print(f"[dim]  {message}[/dim]")

    condensed = llm_mod.condense_qa_check(content, progress_callback=progress_callback)

    if output_file is None:
        output_file = input_file.parent / "QA_CHECK.condensed.LOG"

    output_file.write_text(condensed, encoding="utf-8")
    print(f"[green]✓ Saved condensed output to {output_file}[/green]")
    print(f"[dim]  Original: {len(content):,} chars → Condensed: {len(condensed):,} chars[/dim]")


@run_utility_app.command("condense-lst")
def condense_lst_cmd(
    input_file: Path = typer.Argument(..., help="Path to .lst file"),
    output_file: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path (default: <input>.condensed.lst in same directory)",
    ),
) -> None:
    """Extract useful pages from a .lst file."""
    if not input_file.exists():
        print(f"[red]Error: File not found: {input_file}[/red]")
        raise typer.Exit(1)

    print(f"[dim]Reading {input_file}...[/dim]")
    content = read_text(input_file)

    print(f"[dim]Extracting useful pages from LST ({len(content)} chars)...[/dim]")

    def progress_callback(current: int, total: int, message: str) -> None:
        print(f"[dim]  {message}[/dim]")

    result = llm_mod.extract_condensed_sections(content, "lst", progress_callback=progress_callback)
    extracted = result.get("extracted_text", "")

    if output_file is None:
        output_file = input_file.parent / f"{input_file.stem}.condensed.lst"

    output_file.write_text(extracted, encoding="utf-8")
    print(f"[green]✓ Saved extracted pages to {output_file}[/green]")
    print(f"[dim]  Original: {len(content):,} chars → Extracted: {len(extracted):,} chars[/dim]")


@run_utility_app.command("condense-run-log")
def condense_run_log_cmd(
    input_file: Path = typer.Argument(..., help="Path to run log file (*_run_log.txt)"),
    output_file: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path (default: <input>.condensed.txt in same directory)",
    ),
) -> None:
    """Filter a run log file to show only useful diagnostic information."""
    if not input_file.exists():
        print(f"[red]Error: File not found: {input_file}[/red]")
        raise typer.Exit(1)

    print(f"[dim]Reading {input_file}...[/dim]")
    content = read_text(input_file)

    print(f"[dim]Filtering run log ({len(content)} chars)...[/dim]")

    def progress_callback(current: int, total: int, message: str) -> None:
        print(f"[dim]  {message}[/dim]")

    result = llm_mod.extract_condensed_sections(
        content, "run_log", progress_callback=progress_callback
    )
    filtered = result.get("filtered_content", "")

    if output_file is None:
        output_file = input_file.parent / f"{input_file.stem}.condensed.txt"

    output_file.write_text(filtered, encoding="utf-8")
    print(f"[green]✓ Saved filtered output to {output_file}[/green]")
    print(f"[dim]  Original: {len(content):,} chars → Filtered: {len(filtered):,} chars[/dim]")
