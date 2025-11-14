"""Job control for managing and cancelling GAMS runs."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

try:
    import psutil

    _HAS_PSUTIL = True
except Exception:
    _HAS_PSUTIL = False

CTRL_BREAK = getattr(signal, "CTRL_BREAK_EVENT", None) if sys.platform == "win32" else None


@dataclass
class JobHandle:
    """Handle for tracking a single job."""

    name: str
    cancel_event: threading.Event
    pid: int | None = None  # parent pid (kept for compatibility)
    pgid: int | None = None
    status: str = "waiting"
    parent_pid: int | None = None  # explicit alias for clarity
    child_pids: list[int] = field(default_factory=list)  # gamscmex.exe etc.


class JobRegistry:
    """
    Thread-safe registry for tracking and controlling GAMS jobs.

    Manages:
    - Cancel events for signaling jobs to stop
    - Process IDs and group IDs for termination
    - Job status tracking
    - Optional state file for external tools
    """

    def __init__(self, state_path: Path | None = None):
        """
        Initialize the job registry.

        Args:
            state_path: Optional path to write job state JSON for external monitoring
        """
        self._lock = threading.Lock()
        self._jobs: dict[str, JobHandle] = {}
        self._state_path = state_path

    def register(self, name: str) -> threading.Event:
        """
        Register a new job and return its cancel event.

        Args:
            name: Unique job identifier

        Returns:
            threading.Event that will be set when job should cancel
        """
        with self._lock:
            ev = threading.Event()
            self._jobs[name] = JobHandle(name=name, cancel_event=ev)
            self._write_state_locked()
            return ev

    def attach_popen(self, name: str, popen: subprocess.Popen) -> None:
        """
        Attach process information to a job.

        Args:
            name: Job identifier
            popen: Subprocess handle
        """
        with self._lock:
            if name not in self._jobs:
                return
            j = self._jobs[name]
            j.pid = popen.pid
            j.parent_pid = popen.pid
            try:
                if os.name == "posix":
                    j.pgid = os.getpgid(popen.pid)
                else:
                    j.pgid = None
            except Exception:
                j.pgid = None
            self._write_state_locked()

    def set_status(self, name: str, status: str) -> None:
        """
        Update job status.

        Args:
            name: Job identifier
            status: New status (waiting|starting|running|completed|failed|cancelled)
        """
        with self._lock:
            if name in self._jobs:
                self._jobs[name].status = status
                self._write_state_locked()

    def update_child_pids(self, name: str, child_pids: list[int]) -> None:
        """
        Update child process IDs for a job.

        Args:
            name: Job identifier
            child_pids: List of child process IDs (e.g., gamscmex.exe)
        """
        with self._lock:
            j = self._jobs.get(name)
            if not j:
                return
            j.child_pids = child_pids
            self._write_state_locked()

    def get_cancel_event(self, name: str) -> threading.Event:
        """
        Get the cancel event for a job.

        Args:
            name: Job identifier

        Returns:
            Cancel event for the job
        """
        return self._jobs[name].cancel_event

    def get_running_jobs(self) -> list[str]:
        """
        Get list of job names that are currently running or starting.

        Returns:
            List of job names
        """
        with self._lock:
            return [
                name
                for name, job in self._jobs.items()
                if job.status in ("starting", "running") and job.pid is not None
            ]

    def cancel(self, name: str, grace_seconds: float = 10.0) -> bool:
        """
        Cancel a job by terminating its process.

        Sends SIGTERM (or CTRL_BREAK on Windows), waits for graceful exit,
        then sends SIGKILL if needed.

        Args:
            name: Job identifier
            grace_seconds: Time to wait for graceful shutdown before force kill

        Returns:
            True if job was cancelled, False if not found
        """
        with self._lock:
            j = self._jobs.get(name)
            if not j:
                return False
            j.cancel_event.set()
            pid, pgid = j.pid, j.pgid

        if pid is None:
            self.set_status(name, "cancelled")
            return True

        # Terminate process group
        _terminate_process_group(pid, pgid, grace_seconds)
        self.set_status(name, "cancelled")
        return True

    def cancel_all(self, grace_seconds: float = 10.0) -> None:
        """
        Cancel all jobs.

        Args:
            grace_seconds: Time to wait for graceful shutdown before force kill
        """
        with self._lock:
            names = list(self._jobs.keys())
        for name in names:
            self.cancel(name, grace_seconds)

    def _write_state_locked(self) -> None:
        """Write job state to file (must be called with lock held)."""
        if not self._state_path:
            return
        try:
            # Convert events to serializable format
            state = {}
            for name, j in self._jobs.items():
                state[name] = {
                    "name": j.name,
                    "pid": j.pid,
                    "pgid": j.pgid,
                    "parent_pid": j.parent_pid,
                    "child_pids": j.child_pids,
                    "status": j.status,
                }
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            self._state_path.write_text(json.dumps(state, indent=2))
        except Exception:
            pass


def _list_children_pids(pid: int) -> list[int]:
    """
    List child process IDs using psutil if available.

    Args:
        pid: Parent process ID

    Returns:
        List of child PIDs, empty if psutil unavailable or error
    """
    if not _HAS_PSUTIL:
        return []
    try:
        p = psutil.Process(pid)
        return [c.pid for c in p.children(recursive=True)]
    except Exception:
        return []


def _terminate_process_group(pid: int, pgid: int | None, grace_seconds: float) -> None:
    """
    Terminate a process and its children.

    Args:
        pid: Process ID
        pgid: Process group ID (POSIX only)
        grace_seconds: Time to wait before force kill
    """
    if os.name == "posix":
        # POSIX: Use process group kill
        try:
            if pgid is not None:
                os.killpg(pgid, signal.SIGTERM)
            else:
                os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return  # Already dead

        # Wait for graceful exit
        t0 = time.time()
        while time.time() - t0 < grace_seconds:
            try:
                os.kill(pid, 0)  # Check if alive
            except OSError:
                return  # Exited successfully
            time.sleep(0.2)

        # Force kill if still alive
        try:
            if pgid is not None:
                os.killpg(pgid, signal.SIGKILL)
            else:
                os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    else:
        # Windows: Kill parent first, then remaining children
        # (Killing child first causes gams.exe to respawn it)

        # 1) Gentle: CTRL_BREAK to parent if possible
        try:
            if CTRL_BREAK is not None:
                os.kill(pid, CTRL_BREAK)
        except Exception:
            pass

        # 2) Give GAMS a moment to terminate/cleanup child
        time.sleep(1.0)

        # 3) If still alive, force terminate parent (not the whole tree yet)
        try:
            if shutil.which("taskkill"):
                subprocess.run(  # noqa: S603
                    ["taskkill", "/F", "/PID", str(pid)],  # noqa: S607
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=max(1.0, grace_seconds / 2),
                )
            else:
                # Fallback to TerminateProcess
                import ctypes

                kernel32 = ctypes.windll.kernel32  # type: ignore
                handle = kernel32.OpenProcess(1, False, pid)
                if handle:
                    kernel32.TerminateProcess(handle, 1)
                    kernel32.CloseHandle(handle)
        except Exception:
            pass

        # 4) Kill any surviving children (if psutil available)
        if _HAS_PSUTIL:
            try:
                for cpid in _list_children_pids(pid):
                    try:
                        if shutil.which("taskkill"):
                            subprocess.run(  # noqa: S603
                                ["taskkill", "/F", "/PID", str(cpid)],  # noqa: S607
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                                timeout=2,
                            )
                        else:
                            import ctypes

                            kernel32 = ctypes.windll.kernel32  # type: ignore
                            handle = kernel32.OpenProcess(1, False, cpid)
                            if handle:
                                kernel32.TerminateProcess(handle, 1)
                                kernel32.CloseHandle(handle)
                    except Exception:
                        pass
            except Exception:
                pass

        # 5) Final wait loop up to grace_seconds to ensure gone
        t0 = time.time()
        while time.time() - t0 < grace_seconds:
            try:
                os.kill(pid, 0)
                time.sleep(0.2)
            except OSError:
                break  # parent is gone


def terminate_process_tree(proc: subprocess.Popen, grace_seconds: float = 5.0) -> None:
    """
    Terminate a process and all its children.

    Uses process group termination on POSIX and taskkill on Windows
    to ensure child processes (CPLEX, gmsgennx.exe, etc.) are killed.

    Args:
        proc: Process to terminate
        grace_seconds: Time to wait for graceful shutdown before force kill
    """
    if proc.poll() is not None:
        return  # Already dead

    pid = proc.pid

    if os.name == "posix":
        # POSIX: Use process group kill
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            return  # Already dead

        # Wait for graceful exit
        t0 = time.time()
        while time.time() - t0 < grace_seconds:
            try:
                os.kill(pid, 0)  # Check if alive
            except OSError:
                return  # Exited successfully
            time.sleep(0.2)

        # Force kill if still alive
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass

    else:
        # Windows: Kill parent first, then remaining children
        # (Killing child first causes gams.exe to respawn it)

        # 1) Gentle: CTRL_BREAK to parent if possible
        try:
            if CTRL_BREAK is not None:
                os.kill(pid, CTRL_BREAK)
        except Exception:
            pass

        # 2) Give GAMS a moment to terminate/cleanup child
        time.sleep(1.0)

        # 3) If still alive, force terminate parent (not the whole tree yet)
        try:
            if shutil.which("taskkill"):
                subprocess.run(  # noqa: S603
                    ["taskkill", "/F", "/PID", str(pid)],  # noqa: S607
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=max(1.0, grace_seconds / 2),
                )
            else:
                # Fallback to TerminateProcess
                import ctypes

                kernel32 = ctypes.windll.kernel32  # type: ignore
                handle = kernel32.OpenProcess(1, False, pid)
                if handle:
                    kernel32.TerminateProcess(handle, 1)
                    kernel32.CloseHandle(handle)
        except Exception:
            pass

        # 4) Kill any surviving children (if psutil available)
        if _HAS_PSUTIL:
            try:
                for cpid in _list_children_pids(pid):
                    try:
                        if shutil.which("taskkill"):
                            subprocess.run(  # noqa: S603
                                ["taskkill", "/F", "/PID", str(cpid)],  # noqa: S607
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                                timeout=2,
                            )
                        else:
                            import ctypes

                            kernel32 = ctypes.windll.kernel32  # type: ignore
                            handle = kernel32.OpenProcess(1, False, cpid)
                            if handle:
                                kernel32.TerminateProcess(handle, 1)
                                kernel32.CloseHandle(handle)
                    except Exception:
                        pass
            except Exception:
                pass

        # 5) Final wait loop up to grace_seconds to ensure gone
        t0 = time.time()
        while time.time() - t0 < grace_seconds:
            try:
                os.kill(pid, 0)
                time.sleep(0.2)
            except OSError:
                break  # parent is gone


def create_process_group_kwargs() -> dict:
    """
    Get kwargs for subprocess.Popen to create a new process group.

    Returns:
        Dict with preexec_fn (POSIX) or creationflags (Windows)
    """
    if os.name == "posix":
        return {"preexec_fn": os.setsid}
    else:
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
