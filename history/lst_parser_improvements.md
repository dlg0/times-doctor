# LST Parser Improvements - Solution Report & Title Extraction

**Date**: 2025-11-17

## Summary

Enhanced the LST parser to:
1. **Extract critical solver status** (INFEASIBLE, iterations, resource usage)
2. **Improve title detection** with generic letter-spacing normalization
3. **Fix content alignment** using dynamic offsets instead of fixed header skips

## Changes Made

### 1. Dynamic Title Extraction (`_extract_section_title`)

**Before**:
- Fixed 4-line lookahead
- Hardcoded `start_line + 3` for content
- Only normalized "Compilation" and "Execution"

**After**:
- 10-line lookahead for titles
- Returns `(title, offset)` tuple
- Dynamic content start: `start_line + offset + 1`
- Skips separator lines (`----`, blank, symbols)

**Impact**: Solution Report sections now correctly captured with full content.

### 2. Generic Letter-Spacing Normalization (`_normalize_title`)

**Before**: Only "C o m p i l a t i o n" → "Compilation"

**After**: Any letter-spaced heading normalized:
- "S o l u t i o n Report" → "Solution Report"
- "E x e c u t i o n" → "Execution"
- Generic algorithm: joins consecutive single-letter tokens

**Impact**: All section types recognized regardless of letter-spacing.

### 3. Solution Report Processor (`SolutionReportProcessor`)

**New processor** extracts:

- ✅ **Solver type** (LP, MIP, RMIP, NLP, etc.)
- ✅ **Status code and text** (e.g., code 3: infeasible)
- ✅ **Infeasibility flag** (detects "infeasible" in multiple places)
- ✅ **Solution availability** ("No solution returned")
- ✅ **Resource usage** (time used vs. limit)
- ✅ **Iteration counts** (actual vs. limit)
- ✅ **Execution errors** (EXECERROR messages)

**Patterns matched**:
```python
# Solver status line (handles "---" prefix)
r'^\s*(?:---\s*)?(LP|MIP|...)\s+status\s*\((\d+)\):\s*(.+)$'

# Resource usage
r'^\s*RESOURCE USAGE, LIMIT\s+([\d.]+)\s+([\d.]+)'

# Iteration count
r'^\s*ITERATION COUNT, LIMIT\s+(\d+)\s+(\d+)'
```

**Output summary example**:
```
Solver: LP
Status: infeasible. (code 3)
⚠️ MODEL IS INFEASIBLE
⚠️ No solution returned
Resource usage: 28.97s / 50000s limit
Iterations: 0 / 999,999 limit
```

## Test Results

### Test File: `test.lst` (11 MB)
- 26,365+ domain violation errors
- Solver status: **LP status (3): infeasible**
- Model proven infeasible, no solution returned

### Condensed Output: 1.9 KB (99.98% reduction)

**Critical information preserved**:
- ✅ Error 170: 26,365 occurrences aggregated by pattern
- ✅ Solver: LP, Status code 3 (infeasible)
- ✅ Resource usage: 28.97s / 50,000s
- ✅ Iterations: 0
- ✅ Clear warnings: "⚠️ MODEL IS INFEASIBLE"

### All Tests Passing
- ✅ 6 existing LST parser tests
- ✅ 4 new robustness tests
- ✅ New solver status extraction test

## Files Modified

1. **`src/times_doctor/core/lst_parser.py`**
   - Updated `_extract_section_title()` to return `(title, offset)`
   - Added `_normalize_title()` for generic de-spacing
   - Updated `_find_section_starts()` to track title offset
   - Updated `_extract_sections()` to use dynamic `content_start`
   - Added `SolutionReportProcessor` class
   - Routed "Solution Report" sections to new processor

2. **`tests/test_lst_parser.py`**
   - Updated `test_section_name_normalization` for tuple return

3. **`tests/test_lst_condense_robustness.py`**
   - Added `test_solver_status_extraction` test

## Benefits

### For Users
1. **Never miss critical diagnostics** - Infeasibility immediately visible
2. **Faster diagnosis** - Solver status in summary, not buried in 11MB file
3. **Better error context** - Resource limits, iteration counts shown

### For Developers
4. **More robust parsing** - Handles varied LST file formats
5. **Extensible** - Easy to add new processors for other sections
6. **Well-tested** - Comprehensive test coverage with real-world file

## Oracle Guidance Applied

The oracle identified the core issues:
- ❌ Fixed header offset dropped critical lines
- ❌ No Solution Report processor
- ❌ Title lookahead too short
- ❌ No generic de-spacing

All issues addressed in this implementation.

## Example Output

**Before**: Solution Report section empty or missing

**After**:
```markdown
## Solution Report SOLVE TIMES Using LP From line 2073416

### Solver Status
- **Solver**: LP
- **Status Code**: 3
- **Status**: infeasible.
- **Infeasible**: ❌ YES
- **Has Solution**: ❌ NO
- **Resource Usage**: 28.968s / 50000.0s limit
- **Iterations**: 0
```

## Future Enhancements (Optional)

As suggested by the oracle:
- Add Range Statistics processor (matrix ranges for numerical issues)
- Add Model Statistics processor (equation/variable counts)
- Global fallback extraction for solver status if section missed
- Cross-page subsection splitting for complex reports
