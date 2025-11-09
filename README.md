# TIMES Doctor

**TIMES Doctor** is a CLI tool that helps you diagnose and fix failed **TIMES/Veda** LP runs using AI-powered analysis.

It reads your GAMS run files (QA_CHECK.LOG, run logs, .lst files), condenses them intelligently, and sends them to an LLM (OpenAI or Anthropic) for comprehensive diagnostics and actionable recommendations.

> **Start with the `review` command** - it will guide you through the diagnostic process and suggest next steps like running `datacheck` or `scan` if needed.

## Quick Start

No installation needed! Run directly with `uvx`:

**macOS/Linux:**
```bash
# Review a failed run (start here!)
uvx --from git+https://github.com/dlg0/times-doctor times-doctor review data/065Nov25-annualupto2045/parscen
```

**Windows:**
```cmd
uvx --from git+https://github.com/dlg0/times-doctor times-doctor review D:\path\to\run
```

### Install as Tool

For repeated use, install once:

```bash
uv tool install git+https://github.com/dlg0/times-doctor
times-doctor --help
```

See [INSTALL.md](INSTALL.md) for detailed installation instructions.

## Commands

### 🔍 `review` - Analyze run files with LLM (START HERE)

**What it does:**
- Reads QA_CHECK.LOG, run log, and .lst files from your failed run
- Analyzes them with an LLM (GPT-4o or Claude 3.5 Sonnet)
- Provides human-readable explanations and actionable recommendations

**When to use it:**
- ⭐ **Start here** - This is your first step when a TIMES run fails
- After running `datacheck` to get deeper analysis of numerical issues
- Any time you need to understand what went wrong with a run

**Examples:**

```bash
# Review a failed run (you'll be prompted to select if multiple runs exist)
times-doctor review data/065Nov25-annualupto2045/parscen

# On Windows with full path
times-doctor review "D:\Veda\Gams_Wrk\YourModel\msm_ref~0011"

# Use a specific model
times-doctor review data/065Nov25-annualupto2045/parscen --model gpt-4o
```

**What gets created:**
```
<run_dir>/
├── _llm_calls/                         # Detailed logs of all LLM API calls
└── times_doctor_out/
    └── llm_review.md                   # ⭐ Full LLM analysis (read this!)
```

**Run selection:**
If you've run `datacheck` before, the review command will show you all available runs:
```
Found 2 run directories:
  1. parscen (original run)
  2. _td_datacheck (2025-11-09 14:23:45)

Select run to review (1-2): 
```

---

### 🔬 `datacheck` - Rerun with CPLEX diagnostics

**What it does:**
- Creates a new directory `_td_datacheck` with a copy of your run
- Reruns GAMS with CPLEX's `datacheck 2` mode enabled
- Generates detailed range statistics without solving the full model
- Identifies numerical conditioning issues (tiny coefficients, huge ranges)

**When to use it:**
- When the `review` command suggests numerical issues
- When you see "non-optimal" or status code 6
- To get matrix coefficient ranges and identify ill-conditioning

**Examples:**

```bash
# Run datacheck with default settings (7 threads)
times-doctor datacheck data/065Nov25-annualupto2045/parscen --gams-path gams

# Windows with specific GAMS path
times-doctor datacheck "D:\Veda\Gams_Wrk\YourModel\msm_ref~0011" --gams-path "C:\GAMS\win64\49\gams.exe"

# Use more threads for faster execution
times-doctor datacheck data/065Nov25-annualupto2045/parscen --threads 12
```

**What gets created:**
```
<run_dir>/
└── _td_datacheck/                      # New datacheck run directory
    ├── cplex.opt                       # CPLEX options with datacheck enabled
    ├── <name>_run_log.txt              # Run log with range statistics
    ├── <name>.lst                      # Listing file with diagnostics
    └── GAMSSAVE/
        └── <name>.gdx                  # Solution database
```

**Output example:**
```
Range Statistics:
  rhs     : min 1.000e+00, max 1.234e+06
  bound   : min 0.000e+00, max 1.000e+20
  matrix  : min 1.234e-08, max 5.678e+09

Status:
  Model Status:  1 Optimal
  Solver Status: 1 Normal Completion
  LP Status:     optimal

Next step:
  Run 'times-doctor review data/...' to analyze the datacheck results with LLM assistance.
```

---

### 🧪 `scan` - Test multiple solver configurations

**What it does:**
- Runs your model with different CPLEX solver algorithms
- Compares results to find the most robust configuration
- Tests dual simplex, sifting, and barrier methods

**When to use it:**
- When your model is unstable or gives different results with different settings
- To find which solver configuration works best for your specific model
- When experimenting with solver tuning

**Available profiles:**
- `dual` - Dual simplex (lpmethod 2) - Most robust, certifies optimality
- `sift` - Sifting (lpmethod 5) - Good for huge sparse models
- `bar_nox` - Barrier without crossover (lpmethod 4) - Fast interior solution

**Examples:**

```bash
# Test all default profiles (sequential)
times-doctor scan data/065Nov25-annualupto2045/parscen --gams-path gams

# Test only specific profiles
times-doctor scan data/065Nov25-annualupto2045/parscen --profiles dual --profiles sift

# Run profiles in parallel (faster!)
times-doctor scan data/065Nov25-annualupto2045/parscen --parallel

# Windows with more threads
times-doctor scan "D:\...\msm_ref~0011" --gams-path "C:\GAMS\win64\49\gams.exe" --threads 12
```

**What gets created:**
```
<run_dir>/
└── times_doctor_out/
    ├── scan_runs/
    │   ├── dual/                       # Dual simplex run
    │   │   ├── cplex.opt
    │   │   └── <name>.lst
    │   ├── sift/                       # Sifting run
    │   │   └── <name>.lst
    │   └── bar_nox/                    # Barrier run
    │       └── <name>.lst
    ├── scan_report.csv                 # ⭐ Comparison table
    └── scan_llm_advice.md              # Optional LLM analysis (if --llm used)
```

---

### 🔄 `update` - Update times-doctor

**What it does:**
- Updates times-doctor to the latest version from GitHub

**Examples:**

```bash
times-doctor update
```

On Windows, you'll be prompted to run the update manually:
```powershell
uv tool upgrade times-doctor
```

---

## LLM Configuration

TIMES Doctor uses LLMs for intelligent analysis. You need to configure API access.

**Configuration priority (highest to lowest):**
1. CLI arguments (`--model`, etc.)
2. Environment variables
3. `config.toml` file
4. Default values

### Option 1: config.toml (Recommended)

Create a `config.toml` file in your working directory:

```toml
[llm]
openai_api_key = "sk-proj-..."
# OR
# anthropic_api_key = "sk-ant-..."

# Optional: customize models and temperature
openai_model = "gpt-4o-mini"
openai_temperature = 0.2

[paths]
log_dir = "_llm_calls"
output_dir = "times_doctor_out"

[gams]
# gams_path = "/opt/gams/gams"
# cplex_threads = 4
```

Copy `config.toml.example` to get started.

### Option 2: Environment Variables

Create a `.env` file in your project directory:

```bash
# OpenAI (recommended)
OPENAI_API_KEY=sk-proj-...

# OR Anthropic
ANTHROPIC_API_KEY=sk-ant-...

# Optional: customize behavior
OPENAI_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=0.2
```

### Option 3: System Environment

**macOS/Linux:**
```bash
export OPENAI_API_KEY=sk-proj-...
```

**Windows:**
```powershell
$env:OPENAI_API_KEY="sk-proj-..."
```

### LLM Provider Selection

The `--llm` flag controls which provider to use:

- `--llm auto` - Automatically selects based on available API keys (default for `review`)
- `--llm openai` - Force OpenAI (uses gpt-4o-mini for condensing, gpt-4o for reasoning)
- `--llm anthropic` - Force Anthropic (uses Claude 3.5 Haiku + Sonnet)
- `--llm none` - Disable LLM features (only for `scan` command)

### Cost Estimates

Typical costs for reviewing a failed TIMES run:
- **Condensing phase** (fast model): $0.001 - $0.01
- **Review phase** (reasoning model): $0.05 - $0.50
- **Total per review**: $0.10 - $1.00 depending on file sizes

## Typical Workflow

### Scenario 1: First-time diagnosis

```bash
# 1. Start by reviewing your failed run
times-doctor review data/065Nov25-annualupto2045/parscen

# The LLM might suggest running datacheck if it suspects numerical issues

# 2. Run datacheck if recommended
times-doctor datacheck data/065Nov25-annualupto2045/parscen

# 3. Review the datacheck results
times-doctor review data/065Nov25-annualupto2045/parscen
# (You'll be prompted to select the datacheck run)

# 4. Read the analysis in times_doctor_out/llm_review.md
# and follow the recommendations to fix your model
```

### Scenario 2: Solver configuration testing

```bash
# 1. Review first to understand the problem
times-doctor review data/065Nov25-annualupto2045/parscen

# 2. Test different solver configurations
times-doctor scan data/065Nov25-annualupto2045/parscen

# 3. Check scan_report.csv to see which profile worked best
# 4. Update your model's CPLEX options based on the results
```

## Understanding the Output

### Review Output Structure

The `llm_review.md` file contains:

1. **Executive Summary** - Quick overview of the main issues
2. **Detailed Analysis** - Deep dive into each error/warning
3. **Root Causes** - Why the model is failing
4. **Recommendations** - Step-by-step fixes prioritized by impact
5. **Next Steps** - What commands to run next

### Common Issues Identified

- **Infeasibilities** - Conflicting constraints, missing data
- **Numerical issues** - Tiny coefficients, mixed currencies, unit mismatches
- **Solver configuration** - Wrong algorithm, insufficient tolerances
- **Data errors** - Missing capacity bounds, invalid parameters

## Real-World Example Paths

The tool works with any TIMES/Veda run directory that contains:
- `.lst` file (GAMS listing)
- `QA_CHECK.LOG` (optional but helpful)
- `*_run_log.txt` (optional but helpful)

Example directory structures:

**Veda on Windows:**
```
D:\Veda\Gams_Wrk\MyModel\msm_ref~0011\
├── QA_CHECK.LOG
├── msm_ref_run_log.txt
├── msm_ref.lst
├── msm_ref.gms
└── cplex.opt
```

**Direct GAMS run:**
```
/Users/you/models/times-model/parscen/
├── parscen.lst
├── parscen_run_log.txt
├── parscen.run
└── cplex.opt
```

You can run times-doctor from anywhere - just provide the full or relative path to the run directory.

## Advanced Usage

### Run Utility Commands

TIMES Doctor includes utility commands for manual file processing:

```bash
# Condense a QA_CHECK.LOG file
times-doctor run-utility condense-qa-check QA_CHECK.LOG

# Condense a .lst file
times-doctor run-utility condense-lst model.lst

# Condense a run log
times-doctor run-utility condense-run-log model_run_log.txt
```

### Custom Model Selection

```bash
# Use a specific reasoning model
times-doctor review data/065Nov25-annualupto2045/parscen --model gpt-4o

# Or for Anthropic
times-doctor review data/065Nov25-annualupto2045/parscen --llm anthropic --model claude-3-5-sonnet-20241022
```

## Troubleshooting

### "No API keys found"

Make sure you have a `.env` file with your API key or have set the environment variable.

### "GAMS executable not found"

Use the `--gams-path` flag to specify the full path to your GAMS installation:

```bash
times-doctor datacheck path/to/run --gams-path "C:\GAMS\win64\49\gams.exe"
```

### Large file warnings

If your .lst or log files are very large (>10MB), the condensing phase might take longer and cost more. The tool automatically filters out irrelevant sections to minimize costs.

## Why TIMES Doctor?

**Traditional approach:**
1. Open 50,000-line .lst file
2. Scroll through GAMS output looking for errors
3. Search for "ERROR", "INFES", etc.
4. Try to interpret cryptic CPLEX messages
5. Guess at fixes and rerun (hours per iteration)

**With TIMES Doctor:**
1. Run `times-doctor review path/to/run`
2. Get clear English explanation of all issues
3. Get prioritized, actionable recommendations
4. Fix issues based on expert guidance
5. Iterate faster with datacheck (minutes per iteration)

## Structure

```
times-doctor/
├── src/times_doctor/
│   ├── cli.py              # Command-line interface
│   ├── llm.py              # LLM integration and analysis
│   ├── prompts.py          # LLM prompts for different tasks
│   └── qa_check_parser.py  # QA_CHECK.LOG parsing
├── data/                   # Example run data for testing
├── docs/                   # Additional documentation
├── tests/                  # Test suite
├── pyproject.toml          # Project configuration
└── README.md               # This file
```

## Contributing

See [DEPLOYMENT.md](DEPLOYMENT.md) for development setup and publishing instructions.

## Known Limitations

- **GAMS Required**: For `datacheck` and `scan` commands, you need GAMS installed
- **Large Files**: Very large .lst files (>50MB) may take longer to process and cost more
- **Windows Paths**: Spaces in paths are supported but may require quoting in some edge cases
- **LLM Costs**: Review operations cost $0.10-$1.00 per run depending on file size

## Privacy & Telemetry

**TIMES Doctor does NOT collect any telemetry or usage data.** All operations are local except:
- LLM API calls to OpenAI or Anthropic (only when using `--llm` flag)
- Git clone operations to download TIMES source code

API keys are redacted from all log files for your security.

## License

MIT
