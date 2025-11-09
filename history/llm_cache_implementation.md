# LLM Response Caching Implementation Summary

**Issue:** times-doctor-rqb
**Status:** Closed
**Date:** 2025-11-09

## Problem

Re-running analyses on the same inputs called LLM APIs repeatedly, wasting time and money. Running `times-doctor review` on the same run directory multiple times would make identical API calls each time, incurring unnecessary costs.

## Solution

Implemented a comprehensive LLM response caching system that stores API responses keyed by prompt + model + parameters.

### Architecture

1. **Cache Key Generation** (`llm_cache.py:compute_cache_key`)
   - SHA256 hash of: prompt + model + all parameters (temperature, reasoning_effort, etc.)
   - Stable across runs for identical inputs
   - Parameters sorted for consistency

2. **Cache Storage**
   - Location: `<run_dir>/_llm_calls/cache/cache_<hash>.json`
   - Directory-specific (different run dirs have separate caches)
   - Each file contains: response, metadata, timestamp, model, parameters

3. **Cache Integration**
   - Modified `_call_openai_responses_api()` to check cache before API call
   - Modified `_call_anthropic_api()` to check cache before API call
   - Cache write after successful API responses (both streaming and non-streaming)
   - Graceful fallback on cache read/write failures

### CLI Interface

**Bypass cache (force fresh API calls):**
```bash
times-doctor review <run_dir> --no-cache
```

**Clear cached responses:**
```bash
times-doctor clear-cache <run_dir>
times-doctor clear-cache .  # Current directory
```

### User Feedback

- Cache hits show: `$0.0000 (cache hit)` in output
- Clear indication that no API call was made
- Same token counts displayed as original call

### Testing

Comprehensive test suite (`tests/test_llm_cache.py`):
- ✅ Cache key stability
- ✅ Cache key differentiation (prompts, models, params)
- ✅ Cache read/write operations
- ✅ Cache file structure validation
- ✅ Cache clearing
- ✅ Unicode handling
- ✅ Corrupted file handling
- ✅ Multiple parameter combinations

**Test Coverage:** 100% of llm_cache module (15 tests, all passing)

## Implementation Details

### Files Modified

1. **src/times_doctor/core/llm_cache.py** (new)
   - `compute_cache_key()`: Generate stable hash from inputs
   - `read_cache()`: Check for cached response
   - `write_cache()`: Store response after API call
   - `clear_cache()`: Delete all cache files in directory

2. **src/times_doctor/core/llm.py**
   - Added `use_cache` parameter to `_call_openai_responses_api()`
   - Added `use_cache` parameter to `_call_anthropic_api()`
   - Added `use_cache` parameter to `review_files()`
   - Cache check before API calls (non-streaming only)
   - Cache write after successful responses (streaming + non-streaming)

3. **src/times_doctor/cli.py**
   - Added `--no-cache` flag to `review` command
   - Added `clear_cache` command
   - Pass `use_cache=not no_cache` to `review_files()`

4. **AGENTS.md**
   - Documented caching behavior
   - CLI usage examples
   - When to clear cache
   - Cache storage location

5. **tests/test_llm_cache.py** (new)
   - 15 comprehensive tests
   - 100% coverage of cache module

### Key Design Decisions

1. **Cache key includes all parameters** - Ensures different model configurations get separate cache entries
2. **Skip streaming for cache** - Streaming responses can't be checked synchronously; cached after completion
3. **Graceful degradation** - Cache failures don't break main functionality
4. **Directory-specific cache** - Different run directories maintain separate caches
5. **No automatic expiration** - User controls cache clearing (predictable behavior)

## Benefits

✅ **Cost savings** - Identical API calls cost $0 on cache hit
✅ **Performance** - Instant responses for cached prompts
✅ **Debugging** - Re-run with same inputs without new API costs
✅ **Testing** - Faster iteration when developing prompts
✅ **User control** - `--no-cache` flag for fresh results
✅ **Transparency** - Clear indication of cache hits in output

## Future Enhancements

Potential improvements (not implemented):
- Cache expiration policy (e.g., 30-day TTL)
- Global cache (share across run directories)
- Cache statistics command
- Selective cache clearing by model/date
- Cache compression for large responses

## Commit

```
feat: implement LLM response caching to reduce costs

- Add llm_cache module with cache key computation, read/write operations
- Cache responses in _llm_calls/cache/ directory keyed by prompt+model+params
- Integrate caching into _call_openai_responses_api and _call_anthropic_api
- Add --no-cache flag to review command to bypass cache
- Add clear-cache CLI command to delete cached responses
- Cache hits show $0.0000 (cache hit) in output
- Write comprehensive tests with 100% coverage of llm_cache module
- Update AGENTS.md with caching documentation

Closes times-doctor-rqb
```

## Testing Performed

```bash
# Run all cache tests
uv run pytest tests/test_llm_cache.py -v
# All 15 tests passed

# Type checking
uv run mypy src/times_doctor/core/llm_cache.py --strict
# Success: no issues found

# Integration test (manual)
# 1. Run review on sample data
# 2. Check _llm_calls/cache/ for cache files
# 3. Re-run review - should see cache hits
# 4. Run clear-cache command
# 5. Verify cache files deleted
```
