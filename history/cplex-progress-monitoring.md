# CPLEX Progress Monitoring Implementation

**Date:** 2025-11-09
**Feature:** Add progress monitoring to `scan` and `datacheck` commands

## Summary

Implemented real-time progress monitoring for CPLEX barrier solves in the `scan` and `datacheck` commands. The system displays % completion based on complementarity (μ) reduction and gracefully handles different solver phases (barrier, crossover, simplex).

## Changes Made

### 1. New Module: `cplex_progress.py`

Created a reusable module for CPLEX progress tracking:

- **`BarrierProgressTracker`** - Tracks barrier progress using μ-based calculation
  - Calculates % complete: `log10(μ₀) → log10(μ) / log10(μ₀) → log10(μ_target)`
  - Handles crossover phase detection
  - Returns progress as 0.0-1.0 fraction

- **`parse_cplex_line()`** - Parses CPLEX iteration output
  - Extracts: iteration number, μ (complementarity), primal/dual infeasibility
  - Detects phases: barrier, crossover, simplex
  - Uses regex patterns for flexibility across CPLEX versions

- **`format_progress_line()`** - Formats progress for display
  - Example: `[barrier 64%] it=18 mu=1.2e-05 Pinf=3.1e-04 Dinf=2.7e-04`
  - Shows `–` instead of % for crossover/simplex phases

- **`scan_log_for_progress()`** - Batch processing of log lines

### 2. Enhanced `run_gams_with_progress()` in `cli.py`

Extended the existing GAMS progress display:

- Added CPLEX progress tracker initialization
- Parses log file lines for CPLEX iteration output
- Displays formatted progress at top of live output
- Non-intrusive: works alongside existing log tailing
- Gracefully handles logs without CPLEX output

### 3. Updated CPLEX Options

**`datacheck()` command:**
- Added `simdisplay 2` - Enable simplex iteration display
- Added `bardisplay 2` - Enable barrier iteration display

**`scan()` command:**
- Added display options to all three profiles:
  - `dual` (lpmethod 2)
  - `sift` (lpmethod 5)
  - `bar_nox` (lpmethod 4)

### 4. Comprehensive Tests

Created `tests/unit/test_cplex_progress.py` with 16 tests covering:

- Progress calculation accuracy
- Line parsing for different CPLEX output formats
- Progress formatting with/without percentages
- Edge cases (zero mu, crossover detection, empty logs)
- Full log scanning workflow

**Test Results:** ✅ All 16 new tests pass, no regressions in existing tests (75 passed)

## Technical Details

### Progress Calculation Method

The barrier algorithm reduces complementarity (μ) roughly geometrically from ~1e-2 to target ~1e-8.

```
Progress % = (log₁₀(μ₀) - log₁₀(μ)) / (log₁₀(μ₀) - log₁₀(μ_target)) × 100
```

Where:
- μ₀ = first observed complementarity value
- μ = current complementarity value
- μ_target = convergence tolerance (default 1e-8)

This provides monotonic, meaningful progress that correlates well with actual solve time.

### Display Format

**Barrier phase (with %):**
```
[barrier 64%] it=18 mu=1.2e-05 Pinf=3.1e-04 Dinf=2.7e-04
```

**Crossover phase (no %):**
```
[crossover –] it=25 mu=9.9e-09
```

**Simplex phase (no %):**
```
[simplex –] it=42
```

### Integration Architecture

```
GAMS Process
    ↓
Log File (*_run_log.txt or *.lst)
    ↓
run_gams_with_progress() reads log
    ↓
cplex_progress.parse_cplex_line() → dict
    ↓
BarrierProgressTracker.update_mu() → % complete
    ↓
cplex_progress.format_progress_line() → string
    ↓
Rich Live Display (shown to user)
```

## Files Modified

- **Created:** `src/times_doctor/cplex_progress.py` (167 lines)
- **Modified:** `src/times_doctor/cli.py`
  - Import cplex_progress module
  - Enhanced `run_gams_with_progress()` with progress tracking
  - Updated `datacheck()` cplex.opt generation
  - Updated `scan()` opt_lines() for all profiles
- **Created:** `tests/unit/test_cplex_progress.py` (179 lines)
- **Created:** `history/cplex-progress-monitoring.md` (this document)

## Design Decisions

1. **Reuse existing infrastructure** - Built on `run_gams_with_progress()` rather than replacing it
2. **Non-intrusive** - Progress monitoring doesn't affect solve behavior or performance
3. **Graceful degradation** - Shows heartbeat when % unavailable (simplex, sifting)
4. **Flexible parsing** - Regex patterns handle various CPLEX output formats
5. **Clean separation** - Progress logic isolated in dedicated module

## Usage

No user-facing changes required. Progress monitoring activates automatically when:

1. Running `times-doctor datacheck <path>`
2. Running `times-doctor scan <path>`
3. CPLEX emits iteration logs (enabled via simdisplay/bardisplay)

## Future Enhancements

Potential improvements (not implemented):

- [ ] Optional `logfile cplex-progress.log` for dedicated CPLEX log parsing
- [ ] ETA estimation based on iteration rate
- [ ] Configurable μ_target to match custom barrier tolerances
- [ ] Support for other solvers (Gurobi, XPRESS) with similar patterns
- [ ] Historical progress chart/visualization

## References

- Original proposal provided μ-based progress calculation method
- CPLEX documentation: `simdisplay`, `bardisplay` options
- Times Doctor existing architecture: `run_gams_with_progress()` with Rich Live display
