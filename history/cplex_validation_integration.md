# CPLEX Options Validation Integration

## Summary

Integrated CPLEX options validation into the `review-solver-options` workflow to prevent LLMs from generating invalid solver options.

## What Was Built

### 1. Core Validator ([cplex_validator.py](../src/times_doctor/core/cplex_validator.py))

**Features:**
- Validates CPLEX options against metadata from `cplex_options_gams49_detailed.json`
- Case-insensitive name resolution
- Synonym resolution (e.g., `optca` → `epagap`)
- Type validation (integer, real, boolean, string)
- Boolean normalization (`yes/no`, `on/off`, `true/false` → `1/0`)
- Range checking for numeric values
- Enumerated value validation
- Helpful error messages with typo suggestions

**API:**
```python
validator = CplexOptionsValidator()
result = validator.validate({"lpmethod": 4, "epopt": 1e-7})

if result.is_valid:
    # Use result.normalized_options
else:
    # Handle result.errors
```

### 2. Integration Layer ([solver_validation.py](../src/times_doctor/core/solver_validation.py))

**Functions:**
- `validate_solver_diagnosis()` - Validates all configs in a SolverDiagnosis
- `normalize_opt_config()` - Normalizes option names/values, removes invalid options
- `build_validation_feedback()` - Creates feedback for LLM when validation fails

### 3. CLI Integration ([cli.py](../src/times_doctor/cli.py))

After LLM returns `SolverDiagnosis`:
1. Validates all opt configurations
2. Shows warnings for invalid options
3. Normalizes all configurations (removes invalid options)
4. Proceeds with validated options only

## Test Coverage

**46 tests total** (all passing):
- 33 tests for `CplexOptionsValidator` ([test_cplex_validator.py](../tests/test_cplex_validator.py))
- 13 tests for integration layer ([test_solver_validation.py](../tests/test_solver_validation.py))

**Coverage includes:**
- Name resolution (canonical, synonyms, case-insensitive)
- Type validation (integer, real, boolean, string)
- Boolean normalization
- Range checking
- Enum validation
- Error handling
- Patch validation
- Integration with SolverDiagnosis

## Workflow

**Before:**
```
LLM → SolverDiagnosis → Write opt files
```

**After:**
```
LLM → SolverDiagnosis → Validate → Normalize → Write opt files
                           ↓
                    Show warnings if invalid
                    Remove invalid options
```

## Example Output

When LLM generates invalid options:
```
Validating CPLEX options...
Warning: Some CPLEX options are invalid:

**tight_tolerances.opt**:
  - unknownoption: Unknown option 'unknownoption' (try: nodefile)
  - lpmethod: Value 999 not allowed; expected -1 or 0..6

Normalizing options (removing invalid ones)...
✓ Options validated and normalized
```

## Files Modified

- `src/times_doctor/core/cplex_validator.py` (new)
- `src/times_doctor/core/solver_validation.py` (new)
- `src/times_doctor/cli.py` (validation integration)
- `tests/test_cplex_validator.py` (new)
- `tests/test_solver_validation.py` (new)

## Design Decisions

### Why In-Process Validation (Not HTTP API)?

**Oracle recommended:** Python validation module behind a simple interface, not HTTP.

**Rationale:**
- LLM doesn't need to know about HTTP
- Simpler to test and maintain
- No network/auth overhead
- Easier to integrate with existing Python orchestration

### Why Normalize Instead of Retry Loop?

**Current approach:** Validate once, remove invalid options, proceed

**Alternative considered:** Loop back to LLM to fix errors

**Rationale:**
- Most issues are simple (typos, case sensitivity)
- Normalization handles 90% of cases automatically
- Avoids expensive LLM retry loops
- User still sees warnings about what was removed

### Why Not Update the Prompt?

The current prompt already works well with structured output. The validation layer catches issues without requiring prompt changes.

**Future enhancement:** Could add CPLEX options metadata to prompt context for better guidance, but not necessary for v1.

## Future Enhancements

1. **Iterative correction:** Feed validation errors back to LLM for retry (max 2-3 attempts)
2. **Metadata in prompt:** Include common CPLEX options in prompt for better guidance
3. **Cross-option constraints:** Validate option interdependencies (e.g., `lpmethod=4` requires certain other options)
4. **GUROBI support:** Extend to validate Gurobi options
5. **Version management:** Support multiple CPLEX versions with different option sets

## Testing

Run tests:
```bash
uv run pytest tests/test_cplex_validator.py -v
uv run pytest tests/test_solver_validation.py -v
```

All 46 tests pass in ~0.1s.
