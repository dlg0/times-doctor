# TIMES Doctor

**TIMES Doctor** is a tiny `uv/uvx`-first CLI that helps you diagnose and fix failed **TIMES/Veda** LP runs.
It parses a Veda/GAMS run folder, extracts key diagnostics (model/solver status, CPLEX range statistics, etc.),
optionally launches a quick **CPLEX `datacheck 2`** pass, and can scan a few robust solver profiles
(**dual simplex**, **sifting**, **barrier without crossover**) to suggest the most reliable path to *OPTIMAL*.

> Default behaviour is fully **offline** and deterministic. You can optionally enable an LLM “assistant”
(OpenAI, Anthropic, or a generic AMP/Claude CLI) to turn the raw diagnostics into concise, human-readable
next steps for your team.

## Install / Run

### Quick Start (Direct from GitHub)

No installation needed! Run directly:

**Windows:**
```cmd
uvx --from git+https://github.com/dlg0/times-doctor times-doctor diagnose "D:\path\to\run" --gams "C:\GAMS\win64\49\gams.exe" --datacheck
```

**macOS/Linux:**
```bash
uvx --from git+https://github.com/dlg0/times-doctor times-doctor diagnose /path/to/run --gams gams --datacheck
```

### Install as Tool

```bash
uv tool install git+https://github.com/dlg0/times-doctor
times-doctor --help
```

### From Source

```bash
git clone https://github.com/dlg0/times-doctor.git
cd times-doctor
uv run times-doctor --help
```

Or install in editable mode:
```bash
uv pip install -e .
times-doctor --help
```

See [INSTALL.md](INSTALL.md) for detailed installation instructions.

## Workflows

### Diagnose a failed run (parse existing LST)
```bash
times-doctor diagnose "D:\Veda\Gams_Wrk\...\msm_ref~0011"
```

### Diagnose + short datacheck pass
```bash
times-doctor diagnose "D:\...\msm_ref~0011" --gams "C:\GAMS\win64\49\gams.exe" --datacheck
```

### Review run files with LLM analysis
```bash
times-doctor review "D:\...\msm_ref~0011" --llm auto
```
This reads QA_CHECK.LOG, the run log, and .lst file, then sends them to an LLM for a human-readable summary and recommendations.

### Options scan (no crossover variants)
```bash
times-doctor scan "D:\...\msm_ref~0011" --gams "C:\GAMS\win64\49\gams.exe" --profiles dual sift bar_nox --threads 7
```

Artifacts under `<run>/times_doctor_out/`:
- `diagnose_report.md` — statuses, range stats, currency hints, suggestions.
- `scan_runs/<profile>/...` — copies of each run and their `.lst` files.
- `scan_report.csv` + `scan_report.md` — side-by-side results.

## LLM assistant (optional)

Create a `.env` file in your project directory or set environment variables with one of:

```bash
OPENAI_API_KEY=sk-...          # OpenAI HTTP (uses httpx) or 'openai' CLI fallback
ANTHROPIC_API_KEY=sk-ant-...   # With 'claude' CLI (or set ANTHROPIC_CLI)
AMP_CLI=amp                    # Any CLI that reads prompt on STDIN and prints completion
```

Example .env file:
```
OPENAI_API_KEY=sk-proj-abc123...
```

Example commands:
```bash
times-doctor diagnose "D:\...\msm_ref~0011" --llm auto
times-doctor review "D:\...\msm_ref~0011" --llm auto
```

## Why these solver profiles?

- **dual simplex (`lpmethod 2`)** — robust, certifies OPTIMAL without crossover.
- **sifting (`lpmethod 5`)** — can be fast on huge, sparse LPs where variables ≫ constraints.
- **barrier no crossover (`lpmethod 4`, `solutiontype 2`)** — quick interior solution for triage; often returns status 6.

## Structure

```
times-doctor/
  docs/PRD.md
  src/times_doctor/
    cli.py
    llm.py
    prompts.py
  pyproject.toml
  README.md
```
