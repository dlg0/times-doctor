# Installation & Usage

## Quick Start (Windows)

### Prerequisites
1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Have GAMS with CPLEX installed
3. Set up an OpenAI or Anthropic API key

### Direct from GitHub (Recommended)

Run without installing:
```cmd
uvx --from git+https://github.com/dlg0/times-doctor times-doctor review "D:\path\to\run" --llm auto
```

Or install locally:
```cmd
uv tool install git+https://github.com/dlg0/times-doctor
times-doctor --help
```

### From Source

Clone and install in development mode:
```cmd
git clone https://github.com/dlg0/times-doctor.git
cd times-doctor
uv pip install -e .
times-doctor --help
```

## Usage Examples

### Review a Failed Run (Start Here!)
```cmd
REM Set your API key
set OPENAI_API_KEY=sk-proj-...

REM Review the run
times-doctor review "D:\Veda\Gams_Wrk\run~0011"
```

### Run Datacheck for Numerical Diagnostics
```cmd
times-doctor datacheck "D:\Veda\Gams_Wrk\run~0011" ^
  --gams "C:\GAMS\win64\49\gams.exe" ^
  --threads 7
```

After datacheck completes:
```cmd
REM Review again, selecting the datacheck run
times-doctor review "D:\Veda\Gams_Wrk\run~0011"
```

### Test Multiple Solver Profiles
```cmd
times-doctor scan "D:\Veda\Gams_Wrk\run~0011" ^
  --gams "C:\GAMS\win64\49\gams.exe" ^
  --profiles dual sift bar_nox ^
  --threads 7
```

## macOS/Linux

Same commands, just use forward slashes and adjust GAMS path:
```bash
# Set API key
export OPENAI_API_KEY=sk-proj-...

# Review a run
uvx --from git+https://github.com/dlg0/times-doctor times-doctor review \
  data/065Nov25-annualupto2045/parscen

# Datacheck
times-doctor datacheck data/065Nov25-annualupto2045/parscen

# Scan
times-doctor scan data/065Nov25-annualupto2045/parscen
```

## How It Works

1. **Auto-downloads TIMES source** matching your model version
2. **Analyzes with LLM** (GPT-5 or Claude Sonnet) for comprehensive diagnostics
3. **Provides actionable recommendations** in plain English
4. **Supports datacheck mode** for detailed numerical diagnostics
5. **Tests solver profiles** to find the most robust configuration

## Outputs

### Review Command
All outputs go to `<run_dir>/`:
- `_llm_calls/` - Detailed logs of LLM API calls
- `times_doctor_out/llm_review.md` - **Main output: Full LLM analysis**

### Datacheck Command
- `_td_datacheck/` - Complete datacheck run directory
- `_td_datacheck/*.lst` - Listing file with range statistics

### Scan Command
- `times_doctor_out/scan_runs/<profile>/` - Each solver profile run
- `times_doctor_out/scan_report.csv` - Comparison table

## LLM Configuration

Create a `.env` file in your project directory:

```bash
# OpenAI (recommended - supports GPT-5 reasoning)
OPENAI_API_KEY=sk-proj-...

# OR Anthropic
ANTHROPIC_API_KEY=sk-ant-...
```

Or set as environment variable:

**Windows:**
```cmd
set OPENAI_API_KEY=sk-proj-...
```

**macOS/Linux:**
```bash
export OPENAI_API_KEY=sk-proj-...
```

## Troubleshooting

### "No API keys found"
Make sure you have a `.env` file with your API key or have set the environment variable.

### "No .run or .gms driver found"
Make sure you're pointing to a VEDA/GAMS run directory (contains `.run` or `.gms` file).

### "GAMS executable not found"
Use `--gams-path` to specify the full path to GAMS:
```cmd
times-doctor datacheck "D:\..." --gams "C:\GAMS\win64\49\gams.exe"
```

### "Failed to download TIMES source"
Install git: `winget install Git.Git` (Windows) or use your package manager.

### Re-run doesn't reach solve
- Check GAMS license is valid
- Ensure CPLEX is available
- Demo license has size limits (~5000 equations)
