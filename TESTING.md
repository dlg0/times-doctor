# Testing Strategy for times-doctor

## Test Structure

```
tests/
├── unit/
│   ├── test_llm_api.py          # Test LLM API calls (OpenAI, Anthropic)
│   ├── test_compression.py       # Test compression logic
│   ├── test_extraction.py        # Test section extraction
│   └── test_parsing.py           # Test GAMS output parsing
├── integration/
│   ├── test_review_command.py    # Test full review workflow
│   ├── test_diagnose_command.py  # Test diagnose workflow
│   └── test_scan_command.py      # Test scan workflow
├── fixtures/
│   ├── api_responses/
│   │   ├── openai_gpt5_response.json
│   │   ├── anthropic_claude_response.json
│   │   └── error_responses.json
│   └── sample_files/
│       ├── small_qa_check.log    # <100k for testing single-call
│       ├── large_qa_check.log    # >300k for testing chunking
│       ├── sample_run_log.txt
│       └── sample.lst
└── conftest.py                    # Pytest fixtures and configuration

```

## Testing Framework

Use **pytest** with the following dependencies:
- `pytest` - Test runner
- `pytest-mock` - Mocking support
- `pytest-cov` - Coverage reporting
- `responses` or `httpx-mock` - HTTP mocking for API calls
- `pytest-env` - Environment variable management

## Priority Tests

### 1. LLM API Response Parsing (CRITICAL)
Test that we correctly extract text from API responses.

### 2. Chunking Logic
Test that files are chunked correctly at 300k boundaries.

### 3. Cost Calculation
Test that token counts and costs are accurate.

### 4. Error Handling
Test API failures, empty responses, timeout handling.

## Test Commands

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src/times_doctor --cov-report=html

# Run specific test file
uv run pytest tests/unit/test_llm_api.py

# Run with verbose output
uv run pytest -v

# Run only LLM tests
uv run pytest -k llm
```
