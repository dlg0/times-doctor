# Job Cancellation Feature for times-doctor scan

## Overview

Added interactive job cancellation support to the `times-doctor scan` command, allowing users to cancel running GAMS solver jobs during execution.

## Implementation Date

2025-11-11

## Key Features

### 1. Interactive Cancellation via Ctrl+C

- **First Ctrl+C**: Opens an interactive cancel menu
- **Second Ctrl+C**: Immediately cancels all jobs and aborts

### 2. Cancel Menu Options

When you press Ctrl+C during a scan:
```
═══ Cancel Menu ═══
1) Cancel specific runs (enter names/numbers)
2) Cancel all running jobs
3) Abort scan (cancel all and exit)
4) Resume (continue)

Press Ctrl+C again to immediately cancel all and abort
```

### 3. Clean Process Termination

- **POSIX systems**: Uses process groups with SIGTERM → SIGKILL escalation
- **Windows**: Uses taskkill with tree termination or CTRL_BREAK fallback
- No orphaned GAMS processes left behind

### 4. Status Tracking

New `CANCELLED` status added to job states:
- Displays in progress table with bold yellow color
- Counted separately in summary statistics
- Recorded in scan results CSV

### 5. Job State File

Optional `scan_state.json` file written to `times_doctor_out/scan_runs/`:
- Tracks PIDs and process groups
- Enables future external cancellation tools
- Updated in real-time as jobs start/stop

## Technical Components

### 1. New Module: `job_control.py`

Core cancellation infrastructure:
- `JobRegistry`: Thread-safe registry for tracking jobs
- `JobHandle`: Stores PID, PGID, cancel event for each job
- `_terminate_process_group()`: Cross-platform process termination
- `create_process_group_kwargs()`: Helper for subprocess creation

### 2. Updated: `multi_run_progress.py`

Progress tracking enhancements:
- Added `RunStatus.CANCELLED` enum value
- Added `pause_display()` context manager for interactive menus
- Updated `all_completed()` to treat CANCELLED as terminal
- Added "cancelled" to summary statistics

### 3. Updated: `cli.py`

Scan command enhancements:
- `run_gams_with_progress()`: Accepts `cancel_event` and `job_registry`
- SIGINT handler with interactive menu
- Process groups for all GAMS subprocesses
- Cancelled job result handling in worker function

## Usage Example

```bash
# Start a scan with multiple configurations
times-doctor scan data/my-run --parallel --max-workers 4

# While running, press Ctrl+C to open cancel menu:
# - Select option 1 to cancel specific jobs
# - Select option 2 to cancel all running jobs
# - Select option 3 to abort the entire scan
# - Select option 4 to resume

# Press Ctrl+C twice quickly to immediately abort
```

## Architecture

```
User presses Ctrl+C
     ↓
SIGINT handler triggered
     ↓
Display paused, menu shown
     ↓
User selects jobs to cancel
     ↓
JobRegistry sets cancel_event
     ↓
Worker threads check cancel_event
     ↓
GAMS process terminated (SIGTERM → SIGKILL)
     ↓
Status updated to CANCELLED
     ↓
Results recorded with CANCELLED state
```

## Cross-Platform Support

### POSIX (Linux, macOS)
- Process groups created via `os.setsid()`
- Termination via `os.killpg(pgid, SIGTERM)`
- Force kill via `os.killpg(pgid, SIGKILL)` after grace period

### Windows
- Process groups created via `CREATE_NEW_PROCESS_GROUP`
- Termination via `taskkill /T /F /PID <pid>`
- Fallback to CTRL_BREAK and ctypes TerminateProcess

## Benefits

✅ **User Control**: Cancel long-running jobs without killing entire scan
✅ **Clean Shutdown**: No orphaned GAMS processes
✅ **Partial Results**: Keep results from completed jobs
✅ **Thread-Safe**: Works correctly with parallel execution
✅ **Cross-Platform**: Reliable on Unix and Windows
✅ **Future-Proof**: State file enables external tools

## Future Enhancements

Potential additions:
1. External `times-doctor cancel` command to cancel from another terminal
2. Pause/resume functionality (currently only cancel)
3. Session management for multi-scan coordination
4. Web-based dashboard for monitoring and cancellation

## Testing

Basic validation completed:
- ✅ Syntax and type checking pass
- ✅ CLI help works correctly
- ✅ Imports resolve properly
- ⏸️ End-to-end testing pending (requires GAMS installation)

## References

- Oracle consultation thread: T-8777ea79-0c6b-4995-90f0-61d787574368
- Implementation based on oracle's comprehensive design
- Cross-platform process termination patterns from best practices
