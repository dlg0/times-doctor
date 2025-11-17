# Condensed LST Output

**Source**: tests/fixtures/sample_files/test.lst

**Sections**: 11

**GAMS Version**: 49.6.1

---

## Compilation_1

**Summary**:
```
No errors found
```

---

## Compilation_2

**Summary**:
```
No errors found
```

---

## Compilation_3

**Summary**:
```
Error 170: 19998 occurrences
  - ACT'.YEAR.'H2prd_elec_AE'.'AUD25: 41
  - ACT'.YEAR.'H2prd_elec_PEM'.'AUD25: 41
  - ADE'.YEAR.'EN_Battery_Util1'.'AUD25: 41
  - ADE'.YEAR.'EN_Battery_Util12'.'AUD25: 41
  - ADE'.YEAR.'EN_Battery_Util2'.'AUD25: 41
```

---

## Compilation_4

**Summary**:
```
Error 170: 6367 occurrences
  - SWIS'.YEAR.'EN_Battery_Util1'.'AUD25: 41
  - SWIS'.YEAR.'EN_Battery_Util12'.'AUD25: 41
  - SWIS'.YEAR.'EN_Battery_Util2'.'AUD25: 41
  - SWIS'.YEAR.'EN_Battery_Util24'.'AUD25: 41
  - SWIS'.YEAR.'EN_Battery_Util4'.'AUD25: 41
Error 0: 1 occurrences
```

---

## Execution_1

**Summary**:
```
{'total_time_secs': 33.0, 'peak_memory_mb': 1081, 'major_operations_count': 5}
```

---

## Execution_2

**Summary**:
```
{'total_time_secs': 5.047, 'peak_memory_mb': 1222, 'major_operations_count': 1}
```

---

## Include File Summary

---

## Model Analysis SOLVE TIMES Using LP From line 2073416

**Summary**:
```
{'total_equation_count': 3081380, 'equation_types': 108, 'total_generation_time': 74.688}
```

---

## Model Statistics SOLVE TIMES Using LP From line 2073416

---

## Range Statistics SOLVE TIMES Using LP From line 2073416

---

## Solution Report SOLVE TIMES Using LP From line 2073416

**Summary**:
```
Solver: LP
Status: infeasible. (code 3)
⚠️ MODEL IS INFEASIBLE
⚠️ No solution returned
Resource usage: 28.97s / 50000s limit
Iterations: 0 / 999,999 limit
```

### Solver Status

- **Solver**: LP
- **Status Code**: 3
- **Status**: infeasible.
- **Infeasible**: ❌ YES
- **Has Solution**: ❌ NO
- **Resource Usage**: 28.968s / 50000.0s limit
- **Iterations**: 0

---
