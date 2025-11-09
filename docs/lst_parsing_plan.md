# LST File Parsing Plan

## Problem
GAMS LST files have inconsistent page numbering across runs. We need to extract semantic sections instead of numbered pages, and intelligently aggregate/summarize content.

## LST File Structure

### Page Headers
Each "page" starts with:
```
GAMS 49.6.1  ... Page N
[TIMES -- VERSION ...]
Section Title
```

### Section Types Observed
1. **C o m p i l a t i o n** - Domain violations and compilation errors
2. **Include File Summary** - List of included files
3. **E x e c u t i o n** - Execution trace with timing info
4. **Model Analysis** - Equation generation statistics
5. **Range Statistics** - Matrix statistics
6. **Model Statistics** - Model size summary
7. **Solution Report** - Solver results

## Parsing Strategy

### 1. Section Extraction
- Split file by lines starting with `^GAMS` or `^TIMES -- VERSION`
- Extract section title from 2-3 lines after header
- Normalize section titles (remove extra spaces: "C o m p i l a t i o n" → "Compilation")
- Use normalized section title as the page "name" key

### 2. Section-Specific Processing

#### Compilation Section
**Issue**: Contains thousands of repetitive domain violation errors
**Goal**: Aggregate by error type and element pattern

**Processing**:
- Parse domain violation errors (similar to QA check aggregation)
- Pattern: `**** 170  Domain violation for element`
- Extract:
  - Error code (170)
  - Element pattern (e.g., 'ACT'.2015.'AUD25')
  - File/line context
- **Aggregate**:
  - Count by error code
  - Count by element type/pattern
  - Sample first N occurrences with full context
  - Summary: "170 domain violations: 500 for 'X'.YEAR.'AUD25', 300 for 'Y'.YEAR.'AUD25', ..."

**Output Structure**:
```python
{
    "section": "Compilation",
    "error_summary": {
        "170": {
            "total_count": 1000,
            "patterns": {
                "'X'.YEAR.'AUD25'": 500,
                "'Y'.YEAR.'AUD25'": 300,
                ...
            },
            "samples": [
                {"line": 40, "element": "'ACT'.2015.'AUD25'", "context": "..."},
                ...
            ]
        }
    },
    "full_text": "..." # For reference, but not for embedding
}
```

#### Include File Summary
**Issue**: Long table, but useful for understanding model structure
**Processing**: Keep as-is (relatively small)

#### Execution Section
**Issue**: Extremely long execution trace (thousands of lines)
**Goal**: Keep timing summary, drop individual assignment lines

**Processing**:
- Keep header info
- **Drop** lines like: `----   1286 Assignment Z             0.000     0.031 SECS    118 MB      1`
- **Keep** summary statistics:
  - Total execution time
  - Memory usage peaks
  - Major phase transitions
- Extract timing for major operations only (>1 second?)

**Output Structure**:
```python
{
    "section": "Execution",
    "summary": {
        "total_time_secs": 289.063,
        "peak_memory_mb": 6569,
        "major_operations": [
            {"line": 2101186, "operation": "Loop", "time": 0.204, "count": ...},
            ...
        ]
    },
    "full_text": "..." # Original, but not embedded
}
```

#### Model Analysis
**Issue**: Very long equation list
**Processing**:
- Keep summary count (e.g., "1035406 equations of type EQ_ACTFLO")
- Drop individual equation generation lines
- Keep total counts

#### Range Statistics
**Processing**: Keep as-is (small, important)

#### Model Statistics
**Processing**: Keep as-is (small, critical)

#### Solution Report
**Processing**: Keep entire section (critical for understanding results)

## Implementation Plan

### Phase 1: Section Parser
```python
def parse_lst_sections(lst_path: Path) -> List[LSTSection]:
    """Extract sections from LST file"""
    # Split by page headers
    # Extract section titles
    # Return list of sections with metadata
```

### Phase 2: Section Processors
```python
def process_compilation_section(content: str) -> Dict:
    """Aggregate domain violations"""
    # Parse error patterns
    # Count by type and pattern
    # Return summary + samples

def process_execution_section(content: str) -> Dict:
    """Extract timing summary, drop verbose trace"""
    # Keep major operations only
    # Compute summary stats

def process_model_analysis_section(content: str) -> Dict:
    """Summarize equation counts"""
    # Extract equation type counts
    # Drop verbose generation lines
```

### Phase 3: Storage Strategy
```python
{
    "sections": {
        "Compilation_1": {
            "summary": {...},  # Embedded
            "full_text_hash": "sha256..."  # Link to full text in blob storage
        },
        "Compilation_2": {...},
        "Execution": {...},
        "Model_Analysis": {...},
        ...
    },
    "metadata": {
        "gams_version": "49.6.1",
        "times_version": "4.8.3",
        "run_date": "11/06/25 13:17:19"
    }
}
```

## Benefits
1. **Consistent keys**: Section names instead of page numbers
2. **Reduced size**: Aggregation eliminates repetition
3. **Better retrieval**: Can query "show me compilation errors" instead of "page 3-4"
4. **Semantic search**: Summaries are more meaningful for embedding

## Edge Cases
- Multiple Compilation sections (number them: Compilation_1, Compilation_2)
- Missing sections (handle gracefully)
- Very large individual sections (chunk if needed)
- Non-standard section titles (fallback to page number)
