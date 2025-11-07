import os, re, shutil, subprocess, csv, sys
from pathlib import Path
import typer
from rich import print
from rich.table import Table
from rich.console import Console
from rich.live import Live
from rich.text import Text
from . import llm as llm_mod
from . import __version__

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
    lsts = sorted(run_dir.glob("*.lst"), key=lambda p: p.stat().st_mtime, reverse=True)
    return lsts[0] if lsts else None

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def write_opt(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def ensure_out(run_dir: Path) -> Path:
    out = run_dir / "times_doctor_out"
    out.mkdir(exist_ok=True)
    return out

def run_gams_with_progress(cmd: list[str], cwd: str, max_lines: int = 4) -> int:
    """Run GAMS and show live tail of output."""
    import threading
    
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=False,
        bufsize=0
    )
    
    lines = []
    display_text = Text("Starting GAMS...", style="dim")
    
    def read_output():
        for line in iter(proc.stdout.readline, b''):
            try:
                line_str = line.decode('utf-8', errors='ignore').rstrip()
                if line_str:
                    lines.append(line_str)
            except Exception:
                pass
    
    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    
    with Live(display_text, console=console, refresh_per_second=2) as live:
        while proc.poll() is None:
            if lines:
                display_lines = lines[-max_lines:]
                display_text = Text("\n".join(display_lines))
                live.update(display_text)
            import time
            time.sleep(0.5)
        
        # One final update after process ends
        if lines:
            display_lines = lines[-max_lines:]
            display_text = Text("\n".join(display_lines))
            live.update(display_text)
    
    reader.join(timeout=1)
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
def diagnose(
    run_dir: str,
    gams_path: str | None = typer.Option(None, "--gams-path", help="Path to gams.exe (defaults to 'gams' in PATH)"),
    datacheck: bool = typer.Option(False, help="Run a short CPLEX 'datacheck 2' pass"),
    threads: int = typer.Option(7, help="Threads for diagnostics rerun"),
    llm: str = typer.Option("none", help="LLM provider: auto|openai|anthropic|amp|none")
):
    rd = Path(run_dir).resolve()
    out = ensure_out(rd)

    lst = latest_lst(rd)
    lst_text = read_text(lst) if lst else ""

    used_barrier_noXO = bool(re.search(r"lpmethod\s*4", lst_text, re.I) and re.search(r"solutiontype\s*2", lst_text, re.I))

    if datacheck:
        gams_cmd = gams_path if gams_path else "gams"
        times_version = detect_times_version(lst)
        times_src = get_times_source(version=times_version)
        driver = pick_driver_gms(rd)
        tmp = rd / "_td_datacheck"
        if tmp.exists():
            shutil.rmtree(tmp)
        shutil.copytree(rd, tmp)
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
            "numericalemphasis 1"
        ])
        
        tmp_driver = pick_driver_gms(tmp)
        dd_dir = rd.parent
        restart_file = times_src / "_times.g00"
        gdx_dir = tmp / "GAMSSAVE"
        gdx_file = gdx_dir / f"{tmp.name}.gdx"
        
        print(f"[yellow]Running GAMS datacheck with {threads} threads (this may take several minutes)...[/yellow]")
        run_gams_with_progress([
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
            f"--GDXPATH={gdx_dir}/",
            "--ERR_ABORT=NO"
        ], cwd=str(tmp))
        
        print("[green]GAMS datacheck complete[/green]")
        lst = latest_lst(tmp)
        lst_text = read_text(lst) if lst else lst_text

    ranges = parse_range_stats(lst_text) if lst_text else {}
    status = parse_statuses(lst_text) if lst_text else {}
    mixed = grep_mixed_currencies(rd)

    tips = suggest_fixes(status, ranges, mixed, used_barrier_noXO)

    # optional LLM advice
    llm_text = ""
    if llm.lower() != "none":
        diag = {
            "status": status,
            "ranges": ranges,
            "mixed_currency_files": mixed,
            "used_barrier_noXO": used_barrier_noXO
        }
        res = llm_mod.summarize(diag, provider=llm)
        if res.used:
            llm_text = res.text

    table = Table(title="TIMES Doctor — Diagnose")
    for k,v in status.items():
        table.add_row(k, str(v))
    console.print(table)

    if ranges:
        print(f"[bold]Range statistics[/bold]: {ranges}")
    if mixed:
        print(f"[bold red]Mixed currencies suspected in[/bold red]: {mixed}")
    if llm_text:
        print("[bold]LLM advice[/bold]:\n" + llm_text)

    md = out / "diagnose_report.md"
    with md.open("w", encoding="utf-8") as f:
        f.write("# TIMES Doctor — Diagnose Report\n\n")
        f.write(f"Run dir: `{rd}`\n\n")
        if status:
            f.write("## Status\n")
            for k,v in status.items():
                f.write(f"- **{k}**: {v}\n")
            f.write("\n")
        if ranges:
            f.write("## Range statistics\n")
            for k,(mn,mx) in ranges.items():
                f.write(f"- **{k}**: min {mn:.3e}, max {mx:.3e}\n")
            f.write("\n")
        if mixed:
            f.write("## Mixed currencies\n")
            for p in mixed:
                f.write(f"- {p}\n")
            f.write("\n")
        f.write("## Suggestions\n")
        for t in tips:
            f.write(f"- {t}\n")
        f.write("\n")
        if llm_text:
            f.write("## LLM Advice\n")
            f.write(llm_text + "\n")
    print(f"[green]Wrote[/green] {md}")

@app.command()
def scan(
    run_dir: str,
    gams_path: str | None = typer.Option(None, "--gams-path", help="Path to gams.exe (defaults to 'gams' in PATH)"),
    profiles: list[str] = typer.Option(["dual","sift","bar_nox"], help="Profiles: dual|sift|bar_nox"),
    threads: int = typer.Option(7),
    llm: str = typer.Option("none", help="LLM provider: auto|openai|anthropic|amp|none")
):
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
            return ["lpmethod 2","solutiontype 1","aggind 1","scaind -1",f"threads {threads}","eprhs 1e-06","epopt 1e-06","numericalemphasis 1","names no","advind 0","rerun yes","iis 0"]
        if name == "sift":
            return ["lpmethod 5","solutiontype 1","aggind 1","scaind -1",f"threads {threads}","eprhs 1e-06","epopt 1e-06","numericalemphasis 1","names no","advind 0","rerun yes","iis 0"]
        if name == "bar_nox":
            return ["lpmethod 4","baralg 1","barorder 1","solutiontype 2","aggind 1","scaind -1",f"threads {threads}","eprhs 1e-06","epopt 1e-06","numericalemphasis 1","names no","advind 0","rerun yes","iis 0"]
        raise ValueError(f"Unknown profile {name}")

    rows = []
    for p in profiles:
        wdir = scan_root / p
        if wdir.exists():
            shutil.rmtree(wdir)
        shutil.copytree(rd, wdir)
        (wdir / "cplex.opt").write_text("\n".join(opt_lines(p)) + "\n", encoding="utf-8")
        
        wdir_driver = pick_driver_gms(wdir)
        wdir_gdx_dir = wdir / "GAMSSAVE"
        wdir_gdx_file = wdir_gdx_dir / f"{wdir.name}.gdx"
        
        print(f"[yellow]Running profile '{p}' (this may take several minutes)...[/yellow]")
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
            f"--GDXPATH={wdir_gdx_dir}/",
            "--ERR_ABORT=NO"
        ], cwd=str(wdir))

        lst = latest_lst(wdir)
        text = read_text(lst) if lst else ""
        st = parse_statuses(text)
        rng = parse_range_stats(text)
        rows.append({
            "profile": p,
            "model_status": st.get("model_status",""),
            "solver_status": st.get("solver_status",""),
            "lp_status": st.get("lp_status_text",""),
            "objective": st.get("objective",""),
            "matrix_min": f"{rng.get('matrix',(None,None))[0]:.3e}" if rng.get("matrix") else "",
            "matrix_max": f"{rng.get('matrix',(None,None))[1]:.3e}" if rng.get("matrix") else "",
            "dir": str(wdir),
            "lst": str(lst) if lst else ""
        })

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

@app.command()
def review(
    run_dir: str,
    llm: str = typer.Option("auto", help="LLM provider: auto|openai|anthropic|amp|none"),
    model: str = typer.Option("", help="Specific model to use (will prompt if not specified)")
):
    """Review QA_CHECK.LOG, run log, and LST files using LLM for human-readable diagnostics."""
    rd = Path(run_dir).resolve()
    
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
    
    api_keys = llm_mod.check_api_keys()
    provider_status = []
    if api_keys["openai"]: provider_status.append("OpenAI")
    if api_keys["anthropic"]: provider_status.append("Anthropic")
    if api_keys["amp"]: provider_status.append("Amp")
    
    print(f"\n[dim]Available providers: {', '.join(provider_status) if provider_status else 'none'}[/dim]")
    
    # Interactive model selection if not specified
    selected_model = model
    if not selected_model and llm.lower() in ("openai", "anthropic"):
        models = []
        if llm.lower() == "openai" and api_keys["openai"]:
            models = llm_mod.list_openai_models()
        elif llm.lower() == "anthropic" and api_keys["anthropic"]:
            models = llm_mod.list_anthropic_models()
        
        if models:
            print(f"\n[bold]Available {llm.upper()} models:[/bold]")
            for i, m in enumerate(models, 1):
                print(f"  {i}. {m}")
            
            choice = typer.prompt(f"\nSelect model (1-{len(models)})", type=int, default=1)
            if 1 <= choice <= len(models):
                selected_model = models[choice - 1]
                print(f"[green]Selected: {selected_model}[/green]")
            else:
                selected_model = models[0]
                print(f"[yellow]Invalid choice, using default: {selected_model}[/yellow]")
    
    print(f"\n[bold green]LLM Review:[/bold green]")
    
    # Stream the response and display it live
    from rich.console import Console
    from rich.markdown import Markdown
    console_out = Console()
    accumulated_text = ""
    
    def stream_handler(chunk: str):
        nonlocal accumulated_text
        accumulated_text += chunk
        # Print chunk directly for live streaming
        print(chunk, end="", flush=True)
    
    result = llm_mod.review_files(qa_check, run_log, lst_text, provider=llm, model=selected_model, stream_callback=stream_handler)
    
    if not result.used:
        print(f"[red]Failed to get LLM response. Check API keys and connectivity.[/red]")
        raise typer.Exit(1)
    
    print("\n")  # New line after streaming output
    
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
    """Update times-doctor to the latest version using uv tool upgrade."""
    if sys.platform == "win32":
        print("[yellow]On Windows, times-doctor cannot update itself while running.[/yellow]")
        print("[yellow]Please run this command in PowerShell instead:[/yellow]")
        print("[bold cyan]uv tool upgrade times-doctor[/bold cyan]")
        raise typer.Exit(0)
    
    print("[yellow]Updating times-doctor to the latest version...[/yellow]")
    try:
        subprocess.run(
            ["uv", "tool", "upgrade", "times-doctor"],
            check=True
        )
        print("[green]times-doctor updated successfully[/green]")
    except subprocess.CalledProcessError as e:
        print(f"[red]Failed to update times-doctor: {e}[/red]")
        raise typer.Exit(1)
    except FileNotFoundError:
        print("[red]Error: 'uv' command not found. Please install uv first.[/red]")
        print("Visit: https://docs.astral.sh/uv/getting-started/installation/")
        raise typer.Exit(1)
