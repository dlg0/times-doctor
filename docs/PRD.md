# PRD — TIMES Doctor

**Owner:** David Green / CSIRO Energy  
**Goal:** Small, deterministic CLI to *diagnose* and optionally *scan* a failed TIMES/Veda LP run and output a clear action plan.
**LLM:** Optional adapters (OpenAI / Anthropic / AMP CLI). Offline by default.

## Objectives
1. Parse .lst to surface MODEL/SOLVER/LP status and objective.
2. Extract CPLEX Range statistics and flag ill-conditioning.
3. Detect mixed currencies (AUD14 vs AUD25) in parscen/syssettings.
4. Datacheck mode: run CPLEX `datacheck 2` and parse output.
5. Scan mode: run dual, sifting, barrier-noXO profiles; summarise results.
6. Produce Markdown + CSV reports.
7. Optional LLM advice wrapper.

## CLI

### Diagnose
```
times-doctor diagnose <run_dir> [--gams <path>] [--datacheck] [--threads N] [--llm auto|openai|anthropic|amp|none]
```

### Scan
```
times-doctor scan <run_dir> --gams <path> [--profiles dual sift bar_nox] [--threads N] [--llm ...]
```

## Heuristics
- LP status (6) with lpmethod=4 & solutiontype=2 ⇒ recommend dual simplex.
- Matrix min < 1e-12 ⇒ advise currency/unit normalisation.
- AUD14 & AUD25 co-exist ⇒ standardise to AUD25 and update G_CUREX/syssettings.
- If still unstable ⇒ try sifting; optionally tighten tolerances to 1e-7.

## Architecture
- `cli.py`: Typer app, GAMS invocation, parsing, reporting.
- `llm.py`: provider discovery; OpenAI HTTP or CLI; Anthropic CLI; AMP CLI.
- `prompts.py`: builds compact prompt from diagnostics.

## Milestones
- M1: CLI + parsers + reports + scan.
- M2: LLM adapters.
- M3: Extended heuristics, IIS parsing, Gurobi support.

## Acceptance Criteria
- Diagnose prints statuses and range stats from existing LST.
- Datacheck writes `cplex.opt` (`datacheck 2`) and report includes range values.
- Scan writes CSV with at least the three profiles and statuses.
- Runs via `uvx` on Windows/macOS with Python ≥3.9.
