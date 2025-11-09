import os, re, shutil, subprocess, csv, sys
from pathlib import Path
from typing import Optional
import typer
from rich import print
from rich.table import Table
from rich.console import Console
from rich.live import Live
from rich.text import Text
from .core import llm as llm_mod
from . import __version__
from . import cplex_progress
from .multi_run_progress import MultiRunProgressMonitor, RunStatus

app = typer.Typer(add_completion=False)
console = Console()

def version_callback(value: bool):
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
        is_eager=True
    )
):
    pass

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
    if version:
        times_src = cache_dir / f"TIMES_model-{version}"
    else:
        times_src = cache_dir / "TIMES_model"
    
    # Check if already downloaded
    if times_src.exists() and (times_src / "initsys.mod").exists():
        return times_src
    
    # Download with version tag if specified
    print(f"[yellow]Downloading TIMES source code{f' v{version}' if version else ''}...[/yellow]")
    
    try:
        cmd = ["git", "clone", "--depth=1"]
        if version:
            cmd.extend(["--branch", f"v{version}"])
        cmd.extend(["https://github.com/etsap-TIMES/TIMES_model.git", str(times_src)])
        
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"[green]Downloaded TIMES source to {times_src}[/green]")
    except subprocess.CalledProcessError as e:
        err_msg = e.stderr.decode('utf-8', errors='ignore') if e.stderr else ""
        if version and "not found" in err_msg.lower():
            print(f"[yellow]Version v{version} not found, trying latest 4.x...[/yellow]")
            # Fallback to latest if specific version not found
            return get_times_source(version=None)
        raise RuntimeError(f"Failed to download TIMES source. Install git or download manually from https://github.com/etsap-TIMES/TIMES_model\nError: {err_msg}")
    
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

def run_gams_with_progress(
    cmd: list[str], 
    cwd: str, 
    max_lines: int = 30,
    monitor: Optional[MultiRunProgressMonitor] = None,
    run_name: Optional[str] = None
) -> int:
    """
    Run GAMS and show live tail of output.
    
    Args:
        cmd: GAMS command and arguments
        cwd: Working directory
        max_lines: Max lines to show in standalone display
        monitor: Optional MultiRunProgressMonitor for coordinated display
        run_name: Name of this run (required if monitor is provided)
    """
    import threading
    import signal
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
        monitor.update_status(run_name, RunStatus.STARTING)
    
    # Find the _run_log.txt file that will be created
    cwd_path = Path(cwd)
    log_files_before = set(cwd_path.glob("*_run_log.txt"))
    
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
            bufsize=0
        )
        if not use_monitor:
            console.print(f"[dim]GAMS process started (PID: {proc.pid})[/dim]")
    except FileNotFoundError as e:
        error_msg = f"GAMS executable not found: {cmd[0]}"
        if use_monitor:
            monitor.update_status(run_name, RunStatus.FAILED, error_msg)
        else:
            console.print(f"[red]Error: {error_msg}[/red]")
            console.print(f"[yellow]Make sure GAMS is installed and in PATH, or use --gams-path[/yellow]")
        raise
    except Exception as e:
        error_msg = f"Error starting GAMS: {e}"
        if use_monitor:
            monitor.update_status(run_name, RunStatus.FAILED, error_msg)
        else:
            console.print(f"[red]{error_msg}[/red]")
        raise
    
    lines = []
    log_file_lines = []
    current_log_file = None
    display_text = Text("Starting GAMS...", style="dim")
    
    # CPLEX progress tracking
    progress_tracker = cplex_progress.BarrierProgressTracker()
    
    def read_output():
        for line in iter(proc.stdout.readline, b''):
            try:
                line_str = line.decode('utf-8', errors='ignore').rstrip()
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
    
    old_handler = signal.signal(signal.SIGINT, handle_interrupt)
    
    try:
        import time
        try:
            import psutil
            has_psutil = True
        except ImportError:
            has_psutil = False
        
        iterations = 0
        last_log_check = 0
        
        # Standalone mode variables
        if not use_monitor:
            nonlocal_progress_line = [None]
            live_display = Live(display_text, console=console, refresh_per_second=2)
            live_display.start()
        
        try:
            while proc.poll() is None:
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
                            with open(current_log_file, 'r', encoding='utf-8', errors='ignore') as f:
                                new_content = f.readlines()
                                if len(new_content) > len(log_file_lines):
                                    log_file_lines = new_content
                                    # Show last N non-empty lines
                                    lines = [l.rstrip() for l in new_content if l.strip()]
                                    
                                    # Check for CPLEX progress in new lines
                                    for new_line in new_content[len(log_file_lines):]:
                                        parsed = cplex_progress.parse_cplex_line(new_line)
                                        if parsed:
                                            if use_monitor:
                                                # Report to monitor
                                                monitor.update_cplex_progress(run_name, parsed)
                                            else:
                                                # Update standalone display
                                                formatted = cplex_progress.format_progress_line(parsed, tracker=progress_tracker)
                                                nonlocal_progress_line[0] = formatted
                        except Exception as e:
                            pass
                
                # Update standalone display
                if not use_monitor:
                    if lines:
                        display_lines = lines[-max_lines:]
                        
                        # Add CPLEX progress line at the top if we have one
                        if nonlocal_progress_line[0]:
                            display_lines = [f"[bold cyan]{nonlocal_progress_line[0]}[/bold cyan]", ""] + display_lines
                        
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
                                    cpu_percent = p.cpu_percent(interval=0.1)
                                    mem_mb = p.memory_info().rss / 1024 / 1024
                                    
                                    # Check for child processes (GAMS spawns gmsgennx.exe on Windows)
                                    children = p.children(recursive=True)
                                    child_info = ""
                                    if children:
                                        child_cpu = sum(c.cpu_percent(interval=0.1) for c in children)
                                        child_mem = sum(c.memory_info().rss for c in children) / 1024 / 1024
                                        # Calculate approximate core usage (CPU% / 100)
                                        cores_used = child_cpu / 100
                                        child_info = f" + {len(children)} worker(s) using ~{cores_used:.1f} cores, {child_mem:.0f}MB"
                                    
                                    display_text = Text(
                                        f"GAMS running... ({elapsed:.0f}s{child_info})",
                                        style="dim yellow"
                                    )
                                except:
                                    display_text = Text(f"Waiting for GAMS output... ({elapsed:.0f}s, {len(lines)} lines)", style="dim yellow")
                            else:
                                display_text = Text(f"Waiting for GAMS output... ({elapsed:.0f}s, {len(lines)} lines)", style="dim yellow")
                            live_display.update(display_text)
                
                # Update monitor display if in monitor mode
                if use_monitor:
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
        signal.signal(signal.SIGINT, old_handler)
        
        # Update final status in monitor mode
        if use_monitor:
            if proc.returncode == 0:
                monitor.update_status(run_name, RunStatus.COMPLETED)
            else:
                monitor.update_status(run_name, RunStatus.FAILED, f"Exit code: {proc.returncode}")
    
    reader.join(timeout=1)
    
    # Only print diagnostics in standalone mode
    if not use_monitor:
        console.print(f"[dim]GAMS process exited with code: {proc.returncode}[/dim]")
        console.print(f"[dim]Total output lines captured: {len(lines)}[/dim]")
    
    return proc.returncode

def parse_range_stats(text: str) -> dict:
    sec = re.search(r"RANGE STATISTICS.*?(RHS.*?)(?:\n\n|\r\n\r\n)", text, flags=re.S|re.I)
    if not sec:
        return {}
    block = sec.group(0)
    def grab(name):
        m = re.search(fr"{name}.*?\[\s*([\-\d\.Ee\+]+)\s*,\s*([\-\d\.Ee\+]+)\s*\]", block)
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
        "objective": (obj.group(1) if obj else "")
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

def suggest_fixes(status: dict, ranges: dict, mixed_cur_files: list[str], used_barrier_noXO: bool) -> list[str]:
    tips = []
    if status.get("lp_status_text","").lower().startswith("non-optimal") or status.get("lp_status_code") == "6":
        if used_barrier_noXO:
            tips.append("Barrier without crossover returned status 6. Re-run with dual simplex (lpmethod 2, solutiontype 1) to certify OPTIMAL.")
        tips.append("Run a short diagnostic with 'datacheck 2' to print range statistics and identify tiny coefficients/ill-conditioning.")
    matrix = ranges.get("matrix")
    if matrix:
        mn, mx = matrix
        if mn != 0.0 and abs(mn) < 1e-12:
            tips.append(f"Matrix min coefficient is {mn:.2e}. Rescale inputs: unify currencies, fix unit conversions, or drop near-zero coefficients.")
        if mx != 0.0 and abs(mx) > 1e8:
            tips.append(f"Matrix max coefficient is {mx:.2e}. Large range may cause numerical issues. Normalize units if possible.")
    if mixed_cur_files:
        tips.append("Mixed currencies detected (e.g., AUD14 and AUD25). Standardise to AUD25 and ensure G_CUREX/syssettings include it; remove the older currency rows.")
    if not tips:
        tips.append("No obvious red flags found. If still unstable, try 'lpmethod 5' (sifting) or tighten tolerances to 1e-7 and re-test.")
    return tips

@app.command()
def datacheck(
    run_dir: str,
    gams_path: str | None = typer.Option(None, "--gams-path", help="Path to gams.exe (defaults to 'gams' in PATH)"),
    threads: int = typer.Option(7, help="Number of threads for CPLEX to use during datacheck")
):
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
    
    driver = pick_driver_gms(rd)
    tmp = rd / "_td_datacheck"
    
    if tmp.exists():
        print(f"[yellow]Removing existing datacheck directory: {tmp}[/yellow]")
        shutil.rmtree(tmp)
    
    print(f"[yellow]Creating datacheck run directory: {tmp}[/yellow]")
    shutil.copytree(rd, tmp)
    
    # Write CPLEX options file with datacheck enabled
    write_opt(tmp / "cplex.opt", [
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
        "bardisplay 2"
    ])
    
    tmp_driver = pick_driver_gms(tmp)
    dd_dir = rd.parent
    restart_file = times_src / "_times.g00"
    gdx_dir = tmp / "GAMSSAVE"
    gdx_file = gdx_dir / f"{tmp.name}.gdx"
    
    print(f"[yellow]Running GAMS datacheck with {threads} threads (this may take several minutes)...[/yellow]")
    returncode = run_gams_with_progress([
        gams_cmd, tmp_driver.name,
        f"r={restart_file}",
        f"idir1={tmp}",
        f"idir2={times_src}",
        f"idir3={dd_dir}",
        f"gdx={gdx_file}",
        f"gdxcompress=1",
        "LP=CPLEX",
        "OPTFILE=1",
        "LOGOPTION=2",
        f"logfile={tmp.name}_run_log.txt",
        f"--GDXPATH={gdx_dir}/",
        "--ERR_ABORT=NO"
    ], cwd=str(tmp))
    
    print(f"\n[green]✓ GAMS datacheck complete[/green]")
    print(f"[green]✓ Datacheck output saved to: {tmp}[/green]")
    
    # Parse and display basic status
    lst = latest_lst(tmp)
    if lst:
        lst_text = read_text(lst)
        ranges = parse_range_stats(lst_text)
        status = parse_statuses(lst_text)
        
        if ranges:
            print(f"\n[bold cyan]Range Statistics:[/bold cyan]")
            for k, (mn, mx) in ranges.items():
                print(f"  {k:8s}: min {mn:.3e}, max {mx:.3e}")
        
        if status:
            print(f"\n[bold cyan]Status:[/bold cyan]")
            print(f"  Model Status:  {status.get('model_status', 'N/A')}")
            print(f"  Solver Status: {status.get('solver_status', 'N/A')}")
            print(f"  LP Status:     {status.get('lp_status_text', 'N/A')}")
    
    print(f"\n[bold yellow]Next step:[/bold yellow]")
    print(f"  Run 'times-doctor review {run_dir}' to analyze the datacheck results with LLM assistance.")
    print(f"  The review command will prompt you to select between the original run and this datacheck run.")

@app.command()
def scan(
    run_dir: str,
    gams_path: str | None = typer.Option(None, "--gams-path", help="Path to gams.exe (defaults to 'gams' in PATH)"),
    profiles: list[str] = typer.Option(["dual","sift","bar_nox"], help="Solver profiles to test (dual|sift|bar_nox)"),
    threads: int = typer.Option(7, help="Number of threads for CPLEX to use"),
    llm: str = typer.Option("none", help="LLM provider for optional analysis: auto|openai|anthropic|amp|none"),
    parallel: bool = typer.Option(False, "--parallel", help="Run profiles in parallel (faster but uses more resources)")
):
    """
    Test multiple CPLEX solver configurations to find best approach.
    
    Runs your model with different CPLEX algorithms to compare behavior.
    Each profile uses different methods that may handle numerical issues
    differently. Helps identify which solver config works best.
    
    Available profiles:
      dual    - Dual simplex (lpmethod 2) - Most robust
      sift    - Sifting (lpmethod 5) - Good for huge sparse LPs
      bar_nox - Barrier without crossover (lpmethod 4) - Fast
    
    By default, runs profiles sequentially. Use --parallel to run all
    profiles simultaneously (faster but uses more CPU/memory).
    
    Results summarized in CSV for easy comparison.
    
    \b
    Example:
      times-doctor scan data/065Nov25-annualupto2045/parscen
      times-doctor scan data/... --profiles dual sift
      times-doctor scan data/... --parallel  # Run all 3 profiles at once
    
    \b
    Created directories:
      <run_dir>/times_doctor_out/scan_runs/dual/
      <run_dir>/times_doctor_out/scan_runs/sift/
      <run_dir>/times_doctor_out/scan_runs/bar_nox/
      <run_dir>/times_doctor_out/scan_report.csv
    """
    rd = Path(run_dir).resolve()
    
    gams_cmd = gams_path if gams_path else "gams"
    
    # Detect TIMES version from existing run
    lst = latest_lst(rd)
    times_version = detect_times_version(lst)
    times_src = get_times_source(version=times_version)
    
    dd_dir = rd.parent
    restart_file = times_src / "_times.g00"
    out = ensure_out(rd)
    driver = pick_driver_gms(rd)
    scan_root = out / "scan_runs"
    scan_root.mkdir(exist_ok=True)

    def opt_lines(name: str) -> list[str]:
        if name == "dual":
            return ["lpmethod 2","solutiontype 1","aggind 1","scaind -1",f"threads {threads}","eprhs 1e-06","epopt 1e-06","numericalemphasis 1","names no","advind 0","rerun yes","iis 0","simdisplay 2","bardisplay 2"]
        if name == "sift":
            return ["lpmethod 5","solutiontype 1","aggind 1","scaind -1",f"threads {threads}","eprhs 1e-06","epopt 1e-06","numericalemphasis 1","names no","advind 0","rerun yes","iis 0","simdisplay 2","bardisplay 2"]
        if name == "bar_nox":
            return ["lpmethod 4","baralg 1","barorder 1","solutiontype 2","aggind 1","scaind -1",f"threads {threads}","eprhs 1e-06","epopt 1e-06","numericalemphasis 1","names no","advind 0","rerun yes","iis 0","simdisplay 2","bardisplay 2"]
        raise ValueError(f"Unknown profile {name}")

    # Set up run directories
    run_dirs = {}
    for p in profiles:
        wdir = scan_root / p
        if wdir.exists():
            shutil.rmtree(wdir)
        shutil.copytree(rd, wdir)
        (wdir / "cplex.opt").write_text("\n".join(opt_lines(p)) + "\n", encoding="utf-8")
        run_dirs[p] = wdir
    
    # Helper function to run a single profile
    def run_profile(profile_name: str, monitor: MultiRunProgressMonitor, results: dict):
        """Run a single profile and store results."""
        try:
            wdir = run_dirs[profile_name]
            wdir_driver = pick_driver_gms(wdir)
            wdir_gdx_dir = wdir / "GAMSSAVE"
            wdir_gdx_file = wdir_gdx_dir / f"{wdir.name}.gdx"
            
            run_gams_with_progress([
                gams_cmd, wdir_driver.name,
                f"r={restart_file}",
                f"idir1={wdir}",
                f"idir2={times_src}",
                f"idir3={dd_dir}",
                f"gdx={wdir_gdx_file}",
                f"gdxcompress=1",
                "LP=CPLEX",
                "OPTFILE=1",
                "LOGOPTION=2",
                f"logfile={wdir.name}_run_log.txt",
                f"--GDXPATH={wdir_gdx_dir}/",
                "--ERR_ABORT=NO"
            ], cwd=str(wdir), monitor=monitor, run_name=profile_name)

            # Parse results
            lst = latest_lst(wdir)
            text = read_text(lst) if lst else ""
            st = parse_statuses(text)
            rng = parse_range_stats(text)
            
            # Store in thread-safe dict
            results[profile_name] = {
                "profile": profile_name,
                "model_status": st.get("model_status",""),
                "solver_status": st.get("solver_status",""),
                "lp_status": st.get("lp_status_text",""),
                "objective": st.get("objective",""),
                "matrix_min": f"{rng.get('matrix',(None,None))[0]:.3e}" if rng.get("matrix") else "",
                "matrix_max": f"{rng.get('matrix',(None,None))[1]:.3e}" if rng.get("matrix") else "",
                "dir": str(wdir),
                "lst": str(lst) if lst else ""
            }
        except Exception as e:
            console.print(f"[red]Error running profile {profile_name}: {e}[/red]")
            results[profile_name] = {
                "profile": profile_name,
                "model_status": "ERROR",
                "solver_status": str(e),
                "lp_status": "",
                "objective": "",
                "matrix_min": "",
                "matrix_max": "",
                "dir": str(run_dirs[profile_name]),
                "lst": ""
            }
    
    # Create monitor for tracking all profiles
    mode_str = "parallel" if parallel else "sequential"
    console.print(f"[cyan]Running {len(profiles)} profiles in {mode_str} mode...[/cyan]")
    
    with MultiRunProgressMonitor(list(profiles), title="Scan Progress") as monitor:
        results = {}  # Thread-safe dict to collect results
        
        if parallel:
            # Parallel execution
            import threading
            threads = []
            
            for p in profiles:
                t = threading.Thread(
                    target=run_profile,
                    args=(p, monitor, results),
                    name=f"scan-{p}"
                )
                t.start()
                threads.append(t)
            
            # Wait for all threads to complete
            for t in threads:
                t.join()
        else:
            # Sequential execution
            for p in profiles:
                run_profile(p, monitor, results)
        
        # Convert results dict to rows list (preserve profile order)
        rows = [results[p] for p in profiles if p in results]
    
    # Show summary table
    t = Table(title="Scan Summary")
    for col in ["profile","model_status","solver_status","lp_status","objective","matrix_min","matrix_max"]:
        t.add_column(col)
    for r in rows:
        t.add_row(*[r[c] for c in ["profile","model_status","solver_status","lp_status","objective","matrix_min","matrix_max"]])
    console.print(t)

    csvp = out / "scan_report.csv"
    with csvp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    if llm.lower() != "none":
        diag = {"status": {"mode":"scan"}, "ranges": {}, "mixed_currency_files": [], "used_barrier_noXO": False}
        res = llm_mod.summarize(diag, provider=llm)
        if res.used:
            (out / "scan_llm_advice.md").write_text(res.text, encoding="utf-8")

    print(f"[green]Wrote[/green] {csvp}")

def find_run_directories(base_dir: Path) -> list[tuple[Path, str, float]]:
    """Find all available run directories with timestamps.
    
    Returns list of (path, label, mtime) tuples sorted by recency.
    """
    candidates = []
    
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
    model: str = typer.Option("", help="Specific model to use (will prompt if not specified)")
):
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
        for i, (path, label, mtime) in enumerate(run_dirs, 1):
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
        print("[yellow]No API keys found. Please configure one of the following in a .env file:[/yellow]")
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
    
    print(f"[yellow]Found files:[/yellow]")
    if qa_check: print(f"  ✓ QA_CHECK.LOG ({len(qa_check)} chars)")
    if run_log: print(f"  ✓ {run_log_path.name} ({len(run_log)} chars)")
    if lst_text: print(f"  ✓ {lst.name} ({len(lst_text)} chars)")
    
    # Extract useful sections first
    llm_log_dir = rd / "_llm_calls"
    
    # Determine which fast model will be used
    api_keys = llm_mod.check_api_keys()
    fast_model = "gpt-5-nano" if api_keys["openai"] else ("claude-3-5-haiku-20241022" if api_keys["anthropic"] else "unknown")
    
    print(f"\n[bold yellow]Condensing files with fast LLM ({fast_model})...[/bold yellow]")
    print(f"[dim](LLM calls logged to {llm_log_dir})[/dim]")
    
    condensed_qa_check = ""
    condensed_run_log = ""
    condensed_lst = ""
    
    try:
        if qa_check:
            print(f"[dim]  Condensing QA_CHECK.LOG...[/dim]")
            
            def qa_progress(current, total, message):
                if current == 0:
                    print(f"[dim]    {message}[/dim]")
                else:
                    print(f"[dim]    {message}[/dim]")
            
            condensed_qa_check = llm_mod.condense_qa_check(qa_check, progress_callback=qa_progress)
            condensed_qa_check_path = rd / "QA_CHECK.condensed.LOG"
            condensed_qa_check_path.write_text(condensed_qa_check, encoding="utf-8")
            print(f"[green]  ✓ Saved {condensed_qa_check_path}[/green]")
        
        if run_log:
            print(f"[dim]  Extracting from {run_log_path.name}...[/dim]")
            
            def runlog_progress(current, total, message):
                if current == 0:
                    print(f"[dim]    {message}[/dim]")
                else:
                    print(f"[dim]    {message}[/dim]")
            
            sections = llm_mod.extract_condensed_sections(run_log, "run_log", log_dir=llm_log_dir, progress_callback=runlog_progress)
            
            # For run_log, we get filtered_content directly
            if "filtered_content" in sections and sections["filtered_content"]:
                condensed_run_log = f"# Run Log - Filtered\n\n```\n{sections['filtered_content']}\n```\n"
            else:
                condensed_run_log = llm_mod.create_condensed_markdown(run_log.split('\n'), sections["sections"], "run_log")
            
            condensed_run_log_path = rd / f"{run_log_path.stem}.condensed.txt"
            condensed_run_log_path.write_text(condensed_run_log, encoding="utf-8")
            print(f"[green]  ✓ Saved {condensed_run_log_path}[/green]")
        
        if lst_text:
            print(f"[dim]  Extracting from {lst.name}...[/dim]")
            
            def lst_progress(current, total, message):
                if current == 0:
                    print(f"[dim]    {message}[/dim]")
                else:
                    print(f"[dim]    {message}[/dim]")
            
            sections = llm_mod.extract_condensed_sections(lst_text, "lst", log_dir=llm_log_dir, progress_callback=lst_progress)
            condensed_lst = llm_mod.create_condensed_markdown(lst_text.split('\n'), sections["sections"], "lst")
            condensed_lst_path = rd / f"{lst.stem}.condensed.lst"
            condensed_lst_path.write_text(condensed_lst, encoding="utf-8")
            print(f"[green]  ✓ Saved {condensed_lst_path}[/green]")
    except Exception as e:
        print(f"[red]Error during extraction: {e}[/red]")
        print(f"[yellow]Check {llm_log_dir} for detailed logs[/yellow]")
        raise typer.Exit(1)
    
    # Determine reasoning model
    reasoning_model = "gpt-5 (high effort)" if api_keys["openai"] else ("claude-3-5-sonnet-20241022" if api_keys["anthropic"] else "unknown")
    
    print(f"\n[bold cyan]Sending condensed files to reasoning LLM ({reasoning_model}):[/bold cyan]")
    if condensed_qa_check:
        print(f"  • QA_CHECK.condensed.LOG")
    if condensed_run_log:
        print(f"  • {run_log_path.stem}.condensed.txt")
    if condensed_lst:
        print(f"  • {lst.stem}.condensed.lst")
    
    print(f"\n[bold green]Reviewing...[/bold green]\n")
    
    # Create streaming callback to display output as it comes
    def stream_output(chunk: str):
        print(chunk, end='', flush=True)
    
    result = llm_mod.review_files(condensed_qa_check, condensed_run_log, condensed_lst, provider=llm, model=model, stream_callback=stream_output, log_dir=llm_log_dir)
    
    if not result.used:
        print(f"[red]Failed to get LLM response. Check API keys and connectivity.[/red]")
        raise typer.Exit(1)
    
    # If streaming wasn't used (e.g., fallback to responses API), print the result
    if not result.text or '\n' not in result.text:
        print(result.text)
    
    print("\n")  # New line after output
    
    print(f"\n[bold cyan]LLM Provider:[/bold cyan] {result.provider}")
    if result.model:
        print(f"[bold cyan]Model:[/bold cyan] {result.model}")
    if result.input_tokens > 0:
        print(f"[bold cyan]Tokens:[/bold cyan] {result.input_tokens:,} in + {result.output_tokens:,} out = {result.input_tokens + result.output_tokens:,} total")
    if result.cost_usd > 0:
        print(f"[bold cyan]Cost:[/bold cyan] ${result.cost_usd:.4f} USD")
    
    out = ensure_out(rd)
    review_path = out / "llm_review.md"
    review_path.write_text(result.text, encoding="utf-8")
    
    print(f"\n[green]Saved to {review_path}[/green]")

@app.command()
def update():
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
    output_file: Path = typer.Option(None, "--output", "-o", help="Output file path (default: QA_CHECK.condensed.LOG in same directory)"),
):
    """Condense a QA_CHECK.LOG file into a compact summary."""
    if not input_file.exists():
        print(f"[red]Error: File not found: {input_file}[/red]")
        raise typer.Exit(1)
    
    print(f"[dim]Reading {input_file}...[/dim]")
    content = read_text(input_file)
    
    print(f"[dim]Condensing QA_CHECK.LOG ({len(content)} chars)...[/dim]")
    
    def progress_callback(current, total, message):
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
    output_file: Path = typer.Option(None, "--output", "-o", help="Output file path (default: <input>.condensed.lst in same directory)"),
):
    """Extract useful pages from a .lst file."""
    if not input_file.exists():
        print(f"[red]Error: File not found: {input_file}[/red]")
        raise typer.Exit(1)
    
    print(f"[dim]Reading {input_file}...[/dim]")
    content = read_text(input_file)
    
    print(f"[dim]Extracting useful pages from LST ({len(content)} chars)...[/dim]")
    
    def progress_callback(current, total, message):
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
    output_file: Path = typer.Option(None, "--output", "-o", help="Output file path (default: <input>.condensed.txt in same directory)"),
):
    """Filter a run log file to show only useful diagnostic information."""
    if not input_file.exists():
        print(f"[red]Error: File not found: {input_file}[/red]")
        raise typer.Exit(1)
    
    print(f"[dim]Reading {input_file}...[/dim]")
    content = read_text(input_file)
    
    print(f"[dim]Filtering run log ({len(content)} chars)...[/dim]")
    
    def progress_callback(current, total, message):
        print(f"[dim]  {message}[/dim]")
    
    result = llm_mod.extract_condensed_sections(content, "run_log", progress_callback=progress_callback)
    filtered = result.get("filtered_content", "")
    
    if output_file is None:
        output_file = input_file.parent / f"{input_file.stem}.condensed.txt"
    
    output_file.write_text(filtered, encoding="utf-8")
    print(f"[green]✓ Saved filtered output to {output_file}[/green]")
    print(f"[dim]  Original: {len(content):,} chars → Filtered: {len(filtered):,} chars[/dim]")
