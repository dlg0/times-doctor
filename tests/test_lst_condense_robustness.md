# LST Condensation Robustness Testing

## Overview

This document describes the robustness testing for the LST condensation algorithm using real-world problematic files.

## Test File: test.lst

**Location**: `tests/fixtures/sample_files/test.lst`

**Characteristics**:
- **Size**: ~10.7 MB (10,758 KB)
- **Issue**: Contains 26,365+ domain violation errors (error 170)
- **Pattern**: Repetitive errors for the element 'AUD25' across multiple regions and years

**Sample Error Pattern**:
```
1833710  'ACT'.2021.'H2prd_elec_AE'.'AUD25' 2872
****                                      $170
**** LINE 400703 BATINCLUDE  D:\Veda\...\austimes_0017.dd
**** LINE    104 INPUT       D:\Veda\...\ref-case~0017.RUN
**** 170  Domain violation for element
```

## What Makes This File Challenging

1. **Volume**: Thousands of repetitive errors that could overwhelm output
2. **Multi-token elements**: Elements like `'ACT'.2021.'H2prd_elec_AE'.'AUD25'` have multiple components
3. **Year variation**: Same errors repeat across many years (2015-2045)
4. **Element position**: The problematic element ('AUD25') is the last token, not the first

## Expected Useful Output

A useful condensed output should:

1. **Aggregate errors**: Report total count (~26K) rather than listing each one
2. **Identify the problem element**: Clearly show 'AUD25' is the issue
3. **Generalize years**: Show patterns like `'ACT'.YEAR.'H2prd_elec_AE'.'AUD25'` instead of listing each year separately
4. **Preserve context**: Keep enough information to identify affected technologies and regions

## Current Algorithm Performance

The current LST condensation algorithm successfully handles this file:

### Compilation Section 3
- **Error 170 count**: 19,998 occurrences
- **Top patterns**:
  - `ACT'.YEAR.'H2prd_elec_AE'.'AUD25`: 41 occurrences
  - `ACT'.YEAR.'H2prd_elec_PEM'.'AUD25'`: 41 occurrences
- **Aggregated elements**: Shows patterns for ACT, ADE, CAN, CQ, CVIC, etc.

### Compilation Section 4
- **Error 170 count**: 6,367 occurrences
- **Top patterns**:
  - `SWIS'.YEAR.'EN_Battery_Util1'.'AUD25`: 41 occurrences

### Key Success Factors

✅ **Year generalization**: Replaces specific years (2021, 2022, etc.) with 'YEAR'
✅ **Element extraction**: Correctly identifies 'AUD25' as the problematic element
✅ **Pattern aggregation**: Groups similar errors and counts occurrences
✅ **Summary generation**: Creates concise summary with error codes and top patterns

## Test Coverage

The test suite (`test_lst_condense_robustness.py`) verifies:

1. **`test_aud25_domain_violations_are_condensed_usefully`**
   - Detects error 170 with high occurrence counts (>20)
   - Element patterns include 'AUD25'
   - Year generalization occurs (patterns contain 'YEAR')
   - Samples include context or element information

2. **`test_element_extraction_direction`**
   - Elements are correctly extracted (element line precedes error marker)
   - Sample contexts include the element with 'AUD25'

3. **`test_full_element_pattern_capture`**
   - Full element descriptors are captured (not just first token)
   - Patterns include technology names and multiple components

## Conclusion

The LST condensation algorithm successfully condenses this 10.7 MB file with 26K+ repetitive errors into a useful summary that:
- Reduces output by >99% while preserving critical information
- Identifies the root cause (AUD25 domain violation)
- Shows affected regions, technologies, and counts
- Enables quick diagnosis without reading thousands of error lines

This validates the algorithm's robustness for real-world GAMS/TIMES error logs.
