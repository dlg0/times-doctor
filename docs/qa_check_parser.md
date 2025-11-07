# QA_CHECK.LOG Parser

## Overview

The QA_CHECK.LOG parser provides rule-based, deterministic parsing and condensing of TIMES/VEDA QA_CHECK.LOG files. It replaces the previous LLM-based compression with a structured approach similar to the LST and run_log.txt processing.

## Key Features

- **No LLM required**: Pure Python standard library implementation
- **Streaming**: Memory-efficient processing of large files
- **Deterministic**: Reproducible results with structured deduplication
- **Comprehensive**: Handles complex cases like composite keys, values with spaces, and auto-relaxed entries

## Architecture

### Core Components

1. **Event Parsing** (`iter_events`)
   - Parses section headers (`*** ...`)
   - Extracts events (`*NN SEVERITY - body`)
   - Handles severity filtering and normalization

2. **Key-Value Extraction** (`parse_kv_fields`)
   - Parses index fields (R=, P=, V=, T=, CG=, COM=, etc.)
   - Expands composite keys (e.g., `R.T.P=A.B.C` → `{R:A, T:B, P:C}`)
   - Filters values with allow-list
   - Ignores SUM fields and "(Auto-relaxed)" suffixes

3. **Event Condensation** (`condense_events`)
   - Deduplicates by (severity, message, exact index-set)
   - Counts occurrences
   - Provides message-level summaries

4. **Output Formatting** (`format_condensed_output`)
   - Human-readable text report
   - Grouped by severity
   - Shows occurrence counts and indices

## Usage

### Basic Usage

```python
from times_doctor.qa_check_parser import condense_log_to_rows

# Parse and condense a QA_CHECK.LOG file
summary_rows, message_counts, all_keys = condense_log_to_rows(
    "/path/to/QA_CHECK.LOG",
    index_allow=["R", "P", "V", "T", "CG", "COM"],  # Optional filter
    min_severity="WARNING"  # Optional severity filter
)
```

### Integration with CLI

The `times-doctor review` command automatically uses the parser:

```bash
times-doctor review /path/to/run/directory
```

Output is saved to `QA_CHECK_condensed.md` in the run directory.

## Examples

### Input (QA_CHECK.LOG excerpt)

```
*** Inconsistent CAP_BND(UP/LO/FX) defined for process capacity
*01 WARNING - Lower bound set equal to upper bound,   R.T.P= SWNSW.2030.ENPS168-SNOWY2
*01 WARNING - Lower bound set equal to upper bound,   R.T.P= SWNSW.2030.ENPS169-SNOWY2

*** FLO_SHARE violations
*01 WARNING - FLO_SHARE auto-relaxed R=NSW P=Coal V=ELEC CG=Power SUM=100.5 (Auto-relaxed)
*02 WARNING - FLO_SHARE auto-relaxed R=NSW P=Coal V=ELEC CG=Power SUM=101.2 (Auto-relaxed)
*03 WARNING - FLO_SHARE auto-relaxed R=VIC P=Gas V=ELEC CG=Power SUM=98.3 (Auto-relaxed)
```

### Output (Condensed)

```
================================================================================
QA_CHECK.LOG SUMMARY (Rule-based)
================================================================================

OVERVIEW BY SEVERITY
--------------------------------------------------------------------------------
  WARNING         :     5 events

DETAILED BREAKDOWN
--------------------------------------------------------------------------------

[WARNING]
  Inconsistent CAP_BND(UP/LO/FX) defined for process capacity :: Lower bound set equal to upper bound
    Occurrences: 2
    Indices: P=ENPS168-SNOWY2, R=SWNSW, T=2030
  
  FLO_SHARE violations :: FLO_SHARE auto-relaxed
    Occurrences: 2
    Indices: CG=Power, P=Coal, R=NSW, V=ELEC
  
  FLO_SHARE violations :: FLO_SHARE auto-relaxed
    Occurrences: 1
    Indices: CG=Power, P=Gas, R=VIC, V=ELEC

================================================================================
See QA_CHECK.LOG for full detail
================================================================================
```

## Key Parsing Rules

### Severity Normalization
- `SEVERE WARNING` → `WARNING`
- Standard severities: `SEVERE ERROR`, `ERROR`, `WARNING`, `NOTE`, `INFO`
- Ranking: SEVERE ERROR (0) > ERROR (1) > WARNING (2) > NOTE (3) > INFO (4)

### Message Construction
- Format: `<section> :: <base message>`
- Base message = text before first KEY=VALUE
- If no section seen yet: `(no-section)`

### Index Parsing
- Standard keys: `R`, `P`, `V`, `T`, `CG`, `COM`, etc.
- Composite keys: `R.T.P=A.B.C` expands to `{R:A, T:B, P:C}`
- Values with spaces are preserved
- `SUM=...` fields are ignored
- `(Auto-relaxed)` suffix is stripped

### Deduplication
- Events are grouped by exact match of (severity, message, index-set)
- Occurrences are counted for each unique combination
- Results sorted deterministically by severity rank, then message, then indices

## Testing

Comprehensive test suite in `tests/unit/test_qa_check_parser.py`:

```bash
# Run parser tests
uv run pytest tests/unit/test_qa_check_parser.py -v

# Run all tests
uv run pytest tests/ -v
```

Test coverage includes:
- Severity normalization and ranking
- Composite key expansion
- KEY=VALUE parsing with spaces
- SUM field filtering
- Event iteration and filtering
- Deduplication and condensation
- Output formatting
- Real-world examples (FLO_SHARE, CAP_BND)

## Performance

- **Streaming**: Processes files line-by-line (no full-file load)
- **Memory**: Proportional to unique event patterns, not file size
- **Target**: 50-200MB logs on typical workstation
- **Speed**: Significantly faster than LLM-based compression (no API calls)

## Comparison with Previous Approach

| Aspect | LLM-based (Old) | Rule-based (New) |
|--------|----------------|------------------|
| Speed | Slow (API calls) | Fast (local) |
| Cost | API costs | Free |
| Determinism | Variable | 100% deterministic |
| Accuracy | Good for summaries | Perfect for structure |
| Dependencies | OpenAI/Anthropic | Python stdlib only |
| Memory | High (chunking) | Low (streaming) |
| Offline support | No | Yes |

## Future Enhancements

Potential improvements (from PRD):
- Configurable composite key map and delimiter
- Pluggable exporters (CSV/JSONL/Parquet)
- Schema validators
- Message templating for further normalization
- HTML/PDF reports
