# Installation & Usage

## Quick Start (Windows)

### Prerequisites
1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Have GAMS with CPLEX installed

### Direct from GitHub (Recommended)

Run without installing:
```cmd
uvx --from git+https://github.com/dlg0/times-doctor times-doctor diagnose "D:\path\to\run" --gams "C:\GAMS\win64\49\gams.exe" --datacheck
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

### Basic Diagnosis (Parse existing .lst file)
```cmd
times-doctor diagnose "D:\Veda\Gams_Wrk\run~0011"
```

### With Datacheck (Re-run with dual simplex)
```cmd
times-doctor diagnose "D:\Veda\Gams_Wrk\run~0011" ^
  --gams "C:\GAMS\win64\49\gams.exe" ^
  --datacheck ^
  --threads 7
```

### Solver Profile Scan
```cmd
times-doctor scan "D:\Veda\Gams_Wrk\run~0011" ^
  --gams "C:\GAMS\win64\49\gams.exe" ^
  --profiles dual sift bar_nox ^
  --threads 7
```

### With LLM Advice
```cmd
set OPENAI_API_KEY=sk-...
times-doctor diagnose "D:\Veda\Gams_Wrk\run~0011" --llm openai
```

## macOS/Linux

Same commands, just use forward slashes and adjust GAMS path:
```bash
uvx --from git+https://github.com/dlg0/times-doctor times-doctor diagnose \
  data/run~0011 \
  --gams /usr/local/bin/gams \
  --datacheck
```

## How It Works

1. **Auto-downloads TIMES source** matching your model version
2. **Detects issues**: LP status, ill-conditioning, mixed currencies
3. **Re-runs with different solver options** to find what works
4. **Generates reports**: Markdown summaries + CSV data

## Outputs

All outputs go to `<run_dir>/times_doctor_out/`:
- `diagnose_report.md` - Diagnostic summary with suggestions
- `scan_runs/<profile>/` - Each solver profile run
- `scan_report.csv` - Comparison table
- `scan_report.md` - Summary report

## Troubleshooting

### "No .run or .gms driver found"
Make sure you're pointing to a VEDA/GAMS run directory (contains `.run` file).

### "Failed to download TIMES source"
Install git: `winget install Git.Git`

### Re-run doesn't reach solve
- Check GAMS license is valid
- Ensure CPLEX is available
- Demo license has size limits (~5000 equations)
