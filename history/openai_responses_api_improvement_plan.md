# OpenAI Responses API Improvement Plan

**Date**: November 10, 2025
**Thread**: T-c9644674-dcc6-4abe-a0f7-9e28c03c3acc

## Current Issues

### 1. Incorrect Field Usage in API Calls

**Current implementation** (`src/times_doctor/core/llm.py:271-278`):
```python
payload = {
    "model": model,
    "input": [{"role": "user", "content": prompt}],  # ❌ WRONG
    "text": {"format": {"type": "text"}, "verbosity": "medium"},
    "reasoning": {"effort": reasoning_effort, "summary": "auto"},
    "store": True,
    "stream": bool(stream_callback),
}
```

**Problem**: The `input` field is being used incorrectly. According to the API docs:
- `input` should be a **string** or use the proper `ResponseInput` format
- For system-level instructions, use the `instructions` field
- The current usage mimics Chat Completions API message format, which is incorrect

**Correct approach** (from docs):
```python
# Simple text input
response = client.responses.create(
    model="gpt-4o",
    instructions="You are a coding assistant that talks like a pirate.",  # System prompt
    input="How do I check if a Python object is an instance of a class?",  # User input
)
```

### 2. Missing Separation of System Prompts and User Input

**Current**: Everything goes into `input` as a single blob
- Template from `prompts/solver_options_review/v1.txt` (system instructions)
- Diagnostic data (QA_CHECK.LOG, run_log, LST, cplex.opt)

**Should be**:
- `instructions`: The template/system prompt defining the expert role and task
- `input`: The actual diagnostic data to analyze

### 3. No Structured Output Support

**Current approach** (`src/times_doctor/cli.py:138-154`):
```python
def _extract_opt_files(text: str) -> dict[str, str]:
    """Extract .opt files from LLM response using ===OPT_FILE: / ===END_OPT_FILE delimiters."""
    pattern = r"===OPT_FILE:\s*(\S+\.opt)\s*\n(.*?)===END_OPT_FILE"
    matches = re.finditer(pattern, text, re.DOTALL | re.MULTILINE)
    # ... brittle string parsing
```

**Problem**:
- Fragile regex-based extraction
- LLM might not follow exact format
- No validation of generated content
- No type safety

**Better approach** (from API docs):
```python
from pydantic import BaseModel

class OptFile(BaseModel):
    name: str
    content: str
    description: str

class SolverDiagnosis(BaseModel):
    summary: str
    opt_files: list[OptFile]
    action_plan: list[str]

# Use structured output
response = client.responses.create(
    model="gpt-4o-2024-08-06",
    input="...",
    text_format=SolverDiagnosis,  # Returns typed Pydantic model
)

# Access parsed content
diagnosis = response.output[0].content[0].parsed
for opt_file in diagnosis.opt_files:
    print(opt_file.name, opt_file.content)
```

### 4. No Tools/Function Calling Integration

The API supports tools parameter for structured interactions:

```python
from pydantic import BaseModel

class CreateOptFileArgs(BaseModel):
    filename: str
    parameters: dict[str, str]
    description: str

response = client.responses.create(
    model="gpt-4o-2024-08-06",
    input="Generate solver configurations...",
    tools=[openai.pydantic_function_tool(CreateOptFileArgs)],
)

# LLM can call the tool multiple times with structured args
for tool_call in response.tool_calls:
    args = tool_call.function.parsed_arguments  # Typed Pydantic instance
```

## Improvement Plan

### Phase 1: Fix Basic Field Usage (High Priority)

**Goal**: Use `instructions` and `input` fields correctly

1. **Refactor `_call_openai_responses_api`**:
   - Add `instructions` parameter
   - Change `input` to accept simple string
   - Update all callers to separate system prompts from user data

2. **Update `build_solver_options_review_prompt`**:
   - Return tuple: `(instructions, input_data)`
   - Instructions = template from prompts/
   - Input = formatted diagnostic data

**Files to modify**:
- `src/times_doctor/core/llm.py` (lines 214-500)
- `src/times_doctor/core/prompts.py` (lines 303-340)

### Phase 2: Implement Structured Outputs (High Priority)

**Goal**: Replace regex parsing with Pydantic models

1. **Create Pydantic models**:
```python
# In new file: src/times_doctor/core/solver_models.py
from pydantic import BaseModel, Field

class OptParameter(BaseModel):
    name: str
    value: str
    reason: str = Field(description="Why this parameter is set")

class OptFileConfig(BaseModel):
    filename: str = Field(pattern=r"^[a-z_]+\.opt$")
    description: str
    parameters: list[OptParameter]

class SolverDiagnosis(BaseModel):
    summary: str = Field(description="Why solver stopped at feasible not optimal")
    opt_configurations: list[OptFileConfig] = Field(
        min_items=10, max_items=15,
        description="Different cplex.opt configurations to test"
    )
    action_plan: list[str] = Field(description="Ranked action items")
```

2. **Update API call to use `text_format`**:
```python
response = client.responses.create(
    model="gpt-5",
    instructions=template,
    input=diagnostic_data,
    text_format=SolverDiagnosis,
)

# Access typed data
diagnosis = response.output_text  # or response.output[0].content[0].parsed
```

3. **Remove regex extraction**, use typed models directly

**Files to modify**:
- Create: `src/times_doctor/core/solver_models.py`
- Update: `src/times_doctor/core/llm.py` (add `text_format` support)
- Update: `src/times_doctor/cli.py` (remove `_extract_opt_files`, use typed response)

### Phase 3: Add Tools Support (Medium Priority)

**Goal**: Use function calling for multi-step interactions

This is optional but would allow:
- LLM to call `create_opt_file(name, params, desc)` multiple times
- Better control over generation process
- Validation at each step

**Implementation**:
```python
tools = [openai.pydantic_function_tool(CreateOptFileArgs)]

response = client.responses.create(
    model="gpt-5",
    instructions="...",
    input="...",
    tools=tools,
)

# Process tool calls
opt_files = []
for tool_call in response.tool_calls or []:
    if tool_call.function.name == "create_opt_file":
        args = tool_call.function.parsed_arguments
        opt_files.append((args.filename, args.parameters))
```

### Phase 4: Add Comprehensive Tests (High Priority)

**Missing tests**:
1. Test structured output parsing for solver diagnosis
2. Test that generated .opt files are valid
3. Test field separation (instructions vs input)
4. Integration test for full workflow

**Files to create**:
- `tests/unit/test_solver_diagnosis.py`
- `tests/integration/test_opt_generation.py`

**Test cases needed**:
```python
def test_solver_diagnosis_structured_output():
    """Test that we get typed SolverDiagnosis from API."""

def test_opt_file_generation_count():
    """Test that we generate 10-15 opt configs."""

def test_opt_file_validation():
    """Test that generated .opt files have valid CPLEX parameters."""

def test_instructions_vs_input_separation():
    """Test that system prompt goes to instructions, data goes to input."""
```

## Benefits of These Changes

1. **Correctness**: Using API as designed, not misusing fields
2. **Reliability**: Structured outputs eliminate regex parsing failures
3. **Type Safety**: Pydantic models provide validation and IDE support
4. **Maintainability**: Clearer separation of concerns
5. **Testability**: Can validate schema, not just string formats
6. **Cost Efficiency**: Proper field usage may reduce token costs

## Implementation Order

1. ✅ Document issues (this file)
2. **Next**: Fix field usage (instructions vs input)
3. **Then**: Add Pydantic models for structured output
4. **Then**: Update API calls to use `text_format`
5. **Then**: Add tests
6. **Optional**: Add tools/function calling support

## References

- OpenAI Responses API Docs: `/openai/openai-python` (Context7)
- Current implementation: `src/times_doctor/core/llm.py:214-500`
- Prompt templates: `prompts/solver_options_review/v1.txt`
- Extraction code: `src/times_doctor/cli.py:138-154`
