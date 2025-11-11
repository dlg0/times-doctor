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
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich import print
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text

from . import __version__, cplex_progress
from . import logger as log
from .core import llm as llm_mod
from .core.cost_estimator import estimate_cost, estimate_tokens
from .multi_run_progress import MultiRunProgressMonitor, RunStatus

if TYPE_CHECKING:
    from .core.solver_models import SolverDiagnosis
    from .job_control import JobRegistry

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

        if not use_monitor:
            console.print(f"[dim]GAMS process started (PID: {proc.pid})[/dim]")
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

    # CPLEX progress tracking
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
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            console.print("[red]Process did not terminate, killing...[/red]")
            proc.kill()
        raise KeyboardInterrupt()

    # Only register signal handler from main thread (worker threads can't do this on Windows)
    old_handler = None
    signal_handler_installed = False
    if threading.current_thread() is threading.main_thread():
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
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        proc.kill()
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
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        proc.kill()
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

                                    # Check for CPLEX progress in new lines
                                    for new_line in new_content[old_len:]:
                                        parsed = cplex_progress.parse_cplex_line(new_line)
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

                        # Add CPLEX progress line at the top if we have one
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

                time.sleep(0.5)

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
def scan(
    run_dir: str,
    gams_path: str | None = typer.Option(
        None, "--gams-path", help="Path to gams.exe (defaults to 'gams' in PATH)"
    ),
    threads: int = typer.Option(7, help="Number of threads for CPLEX to use"),
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
        help="Max concurrent profile runs (default: auto by CPU/CPLEX threads)",
    ),
    timeout_seconds: int | None = typer.Option(
        None,
        "--timeout-seconds",
        envvar="TD_TIMEOUT_SECONDS",
        help="Per-profile timeout in seconds (0=no timeout)",
    ),
) -> None:
    """
    Test multiple CPLEX solver configurations to find best approach.

    Scans the run directory for LLM-generated solver .opt files in _td_opt_files/
    and runs your model with each configuration to compare behavior.

    Use 'times-doctor review-solver-options' first to generate configurations.

    By default, runs configurations sequentially. Use --parallel to run all
    configurations simultaneously (faster but uses more CPU/memory).

    Results summarized in CSV for easy comparison.

    \b
    Example:
      times-doctor review-solver-options data/065Nov25-annualupto2045/parscen
      times-doctor scan data/065Nov25-annualupto2045/parscen
      times-doctor scan data/... --parallel  # Run all configs at once

    \b
    Created directories:
      <run_dir>/times_doctor_out/scan_runs/<config_name>/
      <run_dir>/times_doctor_out/scan_report.csv
    """
    rd = Path(run_dir).resolve()

    # Check for _td_opt_files directory
    opt_files_dir = rd / "_td_opt_files"
    if not opt_files_dir.exists() or not opt_files_dir.is_dir():
        print("[red]Error: No _td_opt_files/ directory found.[/red]")
        print(
            "\n[yellow]Run 'times-doctor review-solver-options' first to generate solver configurations.[/yellow]"
        )
        raise typer.Exit(1)

    # Find all .opt files in the directory
    opt_files = sorted(opt_files_dir.glob("*.opt"))
    if not opt_files:
        print("[red]Error: No .opt files found in _td_opt_files/[/red]")
        print(
            "\n[yellow]Run 'times-doctor review-solver-options' first to generate solver configurations.[/yellow]"
        )
        raise typer.Exit(1)

    # Extract config names (filename without .opt extension)
    config_names = [f.stem for f in opt_files]

    # Prompt user to confirm
    print(f"\n[bold]Found {len(opt_files)} solver configurations in _td_opt_files/:[/bold]")
    for i, name in enumerate(config_names, 1):
        print(f"  {i}. {name}")

    if not typer.confirm(f"\nRun scan with these {len(opt_files)} configurations?", default=True):
        print("[yellow]Scan cancelled.[/yellow]")
        raise typer.Exit(0)

    # Now continue with the scan using these config names
    profiles = config_names
    rd = Path(run_dir).resolve()

    gams_cmd = gams_path if gams_path else "gams"

    # Detect TIMES version from existing run
    lst = latest_lst(rd)
    times_version = detect_times_version(lst)
    times_src = get_times_source(version=times_version)

    dd_dir = rd.parent
    restart_file = times_src / "_times.g00"
    out = ensure_out(rd)
    pick_driver_gms(rd)
    scan_root = out / "scan_runs"
    scan_root.mkdir(exist_ok=True)

    # Compute worker limit based on CPU and CPLEX threads
    def compute_workers(n_profiles: int, cplex_threads: int, override: int | None) -> int:
        """Calculate max concurrent workers based on CPU cores and CPLEX threads."""
        cpus = os.cpu_count() or 1
        if override is not None and override > 0:
            return max(1, min(override, n_profiles))
        # Auto: allocate workers to avoid CPU oversubscription
        per = max(1, cpus // max(1, cplex_threads))
        return max(1, min(per, n_profiles))

    workers = compute_workers(len(profiles), threads, max_workers)

    # Map config names to their .opt file paths
    opt_file_map = {f.stem: f for f in opt_files}

    # Show setup summary BEFORE starting the work
    console.print(f"\n[bold]Preparing {len(profiles)} run directories under:[/bold] {scan_root}")
    console.print(
        f"[dim]Mode: {'parallel' if parallel else 'sequential'}, "
        f"max_workers={workers}, CPLEX threads/run={threads}[/dim]"
    )
    console.print(f"[dim]GAMS: {gams_cmd}, TIMES: {times_src}[/dim]")

    console.print("\n[bold]Jobs to launch:[/bold]")
    for i, p in enumerate(profiles, 1):
        console.print(f"  {i:>2}. {p}  ->  {scan_root / p}")

    # Set up run directories and copy .opt files with progress feedback
    setup_start = time.monotonic()
    console.print(
        f"\n[yellow]Setting up run directories (copying base run {len(profiles)} times - may take several minutes)...[/yellow]"
    )

    run_dirs: dict[str, Path] = {}
    total = len(profiles)
    with console.status("[bold green]Preparing run directories...") as status:
        for idx, p in enumerate(profiles, 1):
            wdir = scan_root / p
            status.update(f"[bold green]({idx}/{total}) Preparing {p}...")

            if wdir.exists():
                ok = remove_tree_robust(wdir)
                if not ok:
                    # Final fallback: use fresh timestamped subdir for this run
                    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                    wdir = scan_root / p / ts
                    wdir.parent.mkdir(parents=True, exist_ok=True)
                    console.print(
                        f"[yellow]Using alternate directory due to locks: {wdir}[/yellow]"
                    )
                elif wdir.exists():
                    # Directory still exists after remove_tree_robust (rare race condition)
                    # Use timestamped alternative
                    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                    wdir = scan_root / p / ts
                    wdir.parent.mkdir(parents=True, exist_ok=True)
                    console.print(
                        f"[yellow]Directory still locked after cleanup, using: {wdir}[/yellow]"
                    )

            shutil.copytree(
                rd,
                wdir,
                ignore=shutil.ignore_patterns("times_doctor_out", "_td_opt_files"),
                dirs_exist_ok=True,
            )

            # Copy the corresponding .opt file as cplex.opt
            src_opt = opt_file_map[p]
            dst_opt = wdir / "cplex.opt"
            shutil.copy2(src_opt, dst_opt)

            run_dirs[p] = wdir

    setup_elapsed = time.monotonic() - setup_start
    console.print(f"[green]Setup complete in {setup_elapsed:.1f}s[/green]")
    console.print("\n[bold]Launching jobs...[/bold]")

    # Helper function to run a single profile
    def run_profile(
        profile_name: str,
        monitor: MultiRunProgressMonitor,
        results: dict[str, dict[str, str]],
        timeout_sec: int | None = None,
        cancel_event: threading.Event | None = None,
        registry: "JobRegistry | None" = None,
    ) -> None:
        """Run a single profile and store results."""
        try:
            wdir = run_dirs[profile_name]
            wdir_driver = pick_driver_gms(wdir)
            wdir_gdx_dir = wdir / "GAMSSAVE"
            wdir_gdx_file = wdir_gdx_dir / f"{wdir.name}.gdx"

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
                    "LP=CPLEX",
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
            )

            # Parse results
            lst = latest_lst(wdir)
            text = read_text(lst) if lst else ""
            st = parse_statuses(text)
            rng = parse_range_stats(text)

            # Get elapsed time from monitor
            elapsed_time = monitor.runs[profile_name].get_elapsed_time()
            time_str = monitor.runs[profile_name].format_elapsed() if elapsed_time else ""

            # Store in thread-safe dict
            results[profile_name] = {
                "profile": profile_name,
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
            elapsed_time = monitor.runs[profile_name].get_elapsed_time()
            time_str = monitor.runs[profile_name].format_elapsed() if elapsed_time else ""
            results[profile_name] = {
                "profile": profile_name,
                "model_status": "ERROR",
                "solver_status": "TIMEOUT",
                "lp_status": "",
                "objective": "",
                "runtime": time_str,
                "runtime_seconds": f"{elapsed_time:.1f}" if elapsed_time else "",
                "matrix_min": "",
                "matrix_max": "",
                "dir": str(run_dirs[profile_name]),
                "lst": "",
            }
        except KeyboardInterrupt:
            elapsed_time = monitor.runs[profile_name].get_elapsed_time()
            time_str = monitor.runs[profile_name].format_elapsed() if elapsed_time else ""
            results[profile_name] = {
                "profile": profile_name,
                "model_status": "CANCELLED",
                "solver_status": "CANCELLED",
                "lp_status": "",
                "objective": "",
                "runtime": time_str,
                "runtime_seconds": f"{elapsed_time:.1f}" if elapsed_time else "",
                "matrix_min": "",
                "matrix_max": "",
                "dir": str(run_dirs[profile_name]),
                "lst": "",
            }
        except Exception as e:
            console.print(f"[red]Error running profile {profile_name}: {e}[/red]")
            elapsed_time = monitor.runs[profile_name].get_elapsed_time()
            time_str = monitor.runs[profile_name].format_elapsed() if elapsed_time else ""
            results[profile_name] = {
                "profile": profile_name,
                "model_status": "ERROR",
                "solver_status": str(e),
                "lp_status": "",
                "objective": "",
                "runtime": time_str,
                "runtime_seconds": f"{elapsed_time:.1f}" if elapsed_time else "",
                "matrix_min": "",
                "matrix_max": "",
                "dir": str(run_dirs[profile_name]),
                "lst": "",
            }

    # Create job registry for cancellation support
    from .job_control import JobRegistry

    job_registry = JobRegistry(state_path=scan_root / "scan_state.json")

    # Create monitor for tracking all profiles
    mode_str = "parallel" if parallel else "sequential"
    worker_info = f" (max {workers} concurrent)" if parallel else ""
    console.print(
        f"[cyan]Running {len(profiles)} profiles in {mode_str} mode{worker_info}...[/cyan]"
    )

    # Install SIGINT handler for interactive cancellation
    prev_handler = signal.getsignal(signal.SIGINT)
    menu_open = threading.Event()

    def show_cancel_menu(signum, frame):
        """Handle Ctrl+C to show cancellation menu."""
        if menu_open.is_set():
            # Double Ctrl+C: cancel all and abort
            console.print("\n[bold red]Double Ctrl+C detected - cancelling all jobs...[/bold red]")
            job_registry.cancel_all()
            raise typer.Exit(130)

        menu_open.set()
        try:
            with monitor.pause_display():
                console.print("\n[bold yellow]═══ Cancel Menu ═══[/bold yellow]")
                console.print("1) Cancel specific runs (enter names/numbers)")
                console.print("2) Cancel all running jobs")
                console.print("3) Abort scan (cancel all and exit)")
                console.print("4) Resume (continue)")
                console.print("\n[dim]Press Ctrl+C again to immediately cancel all and abort[/dim]")

                choice = typer.prompt("Select option", default="4", show_default=False)

                if choice == "1":
                    # Show list of jobs
                    running = job_registry.get_running_jobs()
                    if not running:
                        console.print("[yellow]No jobs currently running[/yellow]")
                    else:
                        console.print("\n[bold]Running jobs:[/bold]")
                        for i, name in enumerate(running, 1):
                            console.print(f"  {i}. {name}")
                        selection = typer.prompt(
                            "Enter job names or numbers (comma-separated)", default=""
                        )
                        if selection.strip():
                            # Parse selection
                            targets = []
                            for item in selection.split(","):
                                item = item.strip()
                                if item.isdigit():
                                    idx = int(item) - 1
                                    if 0 <= idx < len(running):
                                        targets.append(running[idx])
                                else:
                                    if item in running:
                                        targets.append(item)
                            # Cancel selected
                            for name in targets:
                                console.print(f"[yellow]Cancelling {name}...[/yellow]")
                                job_registry.cancel(name)
                                monitor.update_status(name, RunStatus.CANCELLED)

                elif choice == "2":
                    console.print("[yellow]Cancelling all running jobs...[/yellow]")
                    job_registry.cancel_all()
                    for name in profiles:
                        if job_registry.get_cancel_event(name).is_set():
                            monitor.update_status(name, RunStatus.CANCELLED)

                elif choice == "3":
                    console.print("[yellow]Aborting scan...[/yellow]")
                    job_registry.cancel_all()
                    for name in profiles:
                        monitor.update_status(name, RunStatus.CANCELLED)
                    raise typer.Exit(130)

        finally:
            menu_open.clear()

    signal.signal(signal.SIGINT, show_cancel_menu)

    with MultiRunProgressMonitor(list(profiles), title="Scan Progress") as monitor:
        results: dict[str, dict[str, str]] = {}  # Thread-safe dict to collect results

        # Register all jobs
        cancel_events = {p: job_registry.register(p) for p in profiles}

        try:
            if parallel:
                # Parallel execution with ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="scan") as executor:
                    future_map = {
                        executor.submit(
                            run_profile,
                            p,
                            monitor,
                            results,
                            timeout_seconds,
                            cancel_events[p],
                            job_registry,
                        ): p
                        for p in profiles
                    }
                    for future in as_completed(future_map):
                        profile_name = future_map[future]
                        try:
                            future.result()
                        except KeyboardInterrupt:
                            # Job was cancelled
                            pass
                        except Exception as e:
                            console.print(f"[red]Worker failed for {profile_name}: {e}[/red]")
            else:
                # Sequential execution
                for p in profiles:
                    # Check if cancelled before starting
                    if cancel_events[p].is_set():
                        monitor.update_status(p, RunStatus.CANCELLED)
                        results[p] = {
                            "profile": p,
                            "model_status": "CANCELLED",
                            "solver_status": "CANCELLED",
                            "lp_status": "",
                            "objective": "",
                            "runtime": "",
                            "runtime_seconds": "",
                            "matrix_min": "",
                            "matrix_max": "",
                            "dir": str(run_dirs[p]),
                            "lst": "",
                        }
                        continue
                    run_profile(
                        p, monitor, results, timeout_seconds, cancel_events[p], job_registry
                    )
        finally:
            # Restore signal handler
            signal.signal(signal.SIGINT, prev_handler)

        # Convert results dict to rows list (preserve profile order)
        rows = [results[p] for p in profiles if p in results]

    # Show summary table
    t = Table(title="Scan Summary")
    for col in [
        "profile",
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

    csvp = out / "scan_report.csv"
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
            (out / "scan_llm_advice.md").write_text(res.text, encoding="utf-8")

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
def clear_cache(
    run_dir: str = typer.Argument(".", help="Run directory containing _llm_calls/cache"),
) -> None:
    """
    Clear cached LLM responses to force fresh API calls.

    Removes all cached responses from the _llm_calls/cache directory.
    Use this when you want to regenerate responses with different models
    or when cached responses may be stale.

    \b
    Example:
      times-doctor clear-cache data/065Nov25-annualupto2045/parscen
      times-doctor clear-cache .  # Clear cache in current directory
    """
    from pathlib import Path

    from .core import llm_cache

    rd = Path(run_dir).resolve()
    cache_dir = rd / "_llm_calls" / "cache"

    if not cache_dir.exists():
        print(f"[yellow]No cache directory found at {cache_dir}[/yellow]")
        return

    count = llm_cache.clear_cache(cache_dir)
    print(f"[green]Cleared {count} cached LLM response(s) from {cache_dir}[/green]")


@app.command()
def review(
    run_dir: str,
    llm: str = typer.Option("auto", help="LLM provider: auto|openai|anthropic|amp|none"),
    model: str = typer.Option("", help="Specific model to use (will prompt if not specified)"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show cost estimate without making API calls"
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompts"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Bypass LLM response cache"),
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

    print(f"\n[bold yellow]Condensing files with fast LLM ({fast_model})...[/bold yellow]")
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
            print(f"[dim]  Extracting from {run_log_path.name}...[/dim]")

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
            print(f"[dim]  Extracting from {lst.name}...[/dim]")

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
        stream_callback=stream_output,
        log_dir=llm_log_dir,
        use_cache=not no_cache,
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
def review_solver_options(
    run_dir: str,
    llm: str = typer.Option("auto", help="LLM provider: auto|openai|anthropic|amp|none"),
    model: str = typer.Option("", help="Specific model to use (will prompt if not specified)"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show cost estimate without making API calls"
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompts"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Bypass LLM response cache"),
) -> None:
    """
    Review solver options for feasible-but-not-optimal solutions.

    Use this command when your TIMES model has returned a FEASIBLE but NOT PROVEN OPTIMAL solution.
    The LLM will analyze your run files and cplex.opt configuration to suggest specific parameter
    tuning (tolerances, etc.) to improve the chances of reaching proven optimal status.

    This command assumes you're using barrier method without crossover (the standard for large TIMES
    models) and will NOT suggest changing the solver algorithm - only tuning parameters within it.

    \b
    Example:
      times-doctor review-solver-options data/065Nov25-annualupto2045/parscen
      times-doctor review-solver-options data/... --dry-run  # Show cost estimate first
      times-doctor review-solver-options data/... --yes      # Skip prompts

    \b
    Output:
      <run_dir>/times_doctor_out/solver_options_review.md  ← Read this!
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

    # Read cplex.opt file
    cplex_opt_path = rd / "cplex.opt"
    cplex_opt = read_text(cplex_opt_path) if cplex_opt_path.exists() else ""

    if not cplex_opt:
        print(f"[yellow]Warning: No cplex.opt file found in {rd}[/yellow]")

    if not qa_check and not run_log and not lst_text:
        print(f"[red]No QA_CHECK.LOG, *_run_log.txt, or .lst files found in {rd}[/red]")
        raise typer.Exit(1)

    print("[yellow]Found files:[/yellow]")
    if cplex_opt:
        print(f"  ✓ cplex.opt ({len(cplex_opt)} chars)")
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

        total_chars = len(cplex_opt)
        if cplex_opt:
            chars = len(cplex_opt)
            tokens = estimate_tokens(cplex_opt)
            total_chars += chars
            print(f"  • cplex.opt: {chars:,} chars (~{tokens:,} tokens)")

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
        review_input_tokens = condense_output_tokens + estimate_tokens(cplex_opt)
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

    print(f"\n[bold yellow]Condensing files with fast LLM ({fast_model})...[/bold yellow]")
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
            print(f"[dim]  Extracting from {run_log_path.name}...[/dim]")

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
            print(f"[dim]  Extracting from {lst.name}...[/dim]")

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
    if cplex_opt:
        print("  • cplex.opt")
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
            cplex_opt,
            provider=llm,
            model=model,
            stream_callback=None,  # No streaming for structured output
            log_dir=llm_log_dir,
            use_cache=not no_cache,
        )

    if not result.used:
        print("[red]Failed to get LLM response. Check API keys and connectivity.[/red]")
        raise typer.Exit(1)

    out = ensure_out(rd)
    review_path = out / "solver_options_review.md"

    # Parse structured output (result.text is the Pydantic model instance)
    from times_doctor.core.solver_models import SolverDiagnosis

    # Result.text is the Pydantic model instance when using structured output
    if isinstance(result.text, SolverDiagnosis):
        diagnosis = result.text
    else:
        # Fallback: parse JSON if for some reason we got a string
        import json

        data = json.loads(result.text)
        diagnosis = SolverDiagnosis(**data)

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

    # Extract .opt files from structured output
    opt_dir = rd / "_td_opt_files"
    opt_dir.mkdir(exist_ok=True)

    created_files = []
    for config in diagnosis.opt_configurations:
        # Build .opt file content
        opt_lines = [f"* {config.description}"]
        opt_lines.extend(
            f"{param.name} {param.value}  $ {param.reason}" for param in config.parameters
        )

        opt_content = "\n".join(opt_lines)
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

    print(f"\n[dim]Next: times-doctor scan {rd}[/dim]")


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
