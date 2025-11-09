"""Tests for multi-run progress monitoring."""

import pytest
from times_doctor.multi_run_progress import (
    MultiRunProgressMonitor,
    RunProgress,
    RunStatus,
)


class TestRunProgress:
    """Test RunProgress dataclass."""
    
    def test_initial_state(self):
        run = RunProgress(name="test")
        assert run.name == "test"
        assert run.status == RunStatus.WAITING
        assert run.phase == "–"
        assert run.progress_pct is None
    
    def test_format_progress_with_percentage(self):
        run = RunProgress(name="test", phase="barrier", progress_pct=0.64)
        assert run.format_progress() == "64%"
    
    def test_format_progress_without_percentage(self):
        run = RunProgress(name="test", phase="simplex")
        assert run.format_progress() == "–"
    
    def test_format_iteration(self):
        run = RunProgress(name="test", iteration="42")
        assert run.format_iteration() == "it=42"
    
    def test_format_details(self):
        run = RunProgress(
            name="test",
            mu=1.2e-4,
            primal_infeas=2.3e-2,
            dual_infeas=3.4e-3
        )
        details = run.format_details()
        assert "μ=1.20e-04" in details
        assert "P=2.30e-02" in details
        assert "D=3.40e-03" in details


class TestMultiRunProgressMonitor:
    """Test MultiRunProgressMonitor."""
    
    def test_initialization(self):
        monitor = MultiRunProgressMonitor(["run1", "run2", "run3"])
        assert len(monitor.runs) == 3
        assert "run1" in monitor.runs
        assert "run2" in monitor.runs
        assert "run3" in monitor.runs
    
    def test_update_status(self):
        monitor = MultiRunProgressMonitor(["run1"])
        monitor.update_status("run1", RunStatus.RUNNING)
        assert monitor.runs["run1"].status == RunStatus.RUNNING
    
    def test_update_status_with_error(self):
        monitor = MultiRunProgressMonitor(["run1"])
        monitor.update_status("run1", RunStatus.FAILED, "Test error")
        assert monitor.runs["run1"].status == RunStatus.FAILED
        assert monitor.runs["run1"].error_msg == "Test error"
    
    def test_update_cplex_progress(self):
        monitor = MultiRunProgressMonitor(["run1"])
        
        parsed = {
            'phase': 'barrier',
            'iteration': '10',
            'mu': 1e-4,
            'primal_infeas': 2.3e-2,
            'dual_infeas': 3.4e-3
        }
        
        monitor.update_cplex_progress("run1", parsed)
        
        run = monitor.runs["run1"]
        assert run.status == RunStatus.RUNNING
        assert run.phase == "barrier"
        assert run.iteration == "10"
        assert run.mu == 1e-4
        assert run.primal_infeas == 2.3e-2
        assert run.dual_infeas == 3.4e-3
    
    def test_update_cplex_progress_crossover(self):
        monitor = MultiRunProgressMonitor(["run1"])
        
        # First update with barrier
        monitor.update_cplex_progress("run1", {'phase': 'barrier', 'mu': 1e-2})
        
        # Then crossover
        monitor.update_cplex_progress("run1", {'phase': 'crossover', 'mu': 1e-8})
        
        run = monitor.runs["run1"]
        assert run.phase == "crossover"
        assert run.tracker.in_crossover
    
    def test_get_table(self):
        monitor = MultiRunProgressMonitor(["run1", "run2"])
        monitor.update_status("run1", RunStatus.RUNNING)
        monitor.update_status("run2", RunStatus.WAITING)
        
        table = monitor.get_table()
        assert table is not None
        assert table.title == "Progress"
    
    def test_all_completed(self):
        monitor = MultiRunProgressMonitor(["run1", "run2"])
        assert not monitor.all_completed()
        
        monitor.update_status("run1", RunStatus.COMPLETED)
        assert not monitor.all_completed()
        
        monitor.update_status("run2", RunStatus.COMPLETED)
        assert monitor.all_completed()
    
    def test_all_completed_with_failures(self):
        monitor = MultiRunProgressMonitor(["run1", "run2"])
        monitor.update_status("run1", RunStatus.COMPLETED)
        monitor.update_status("run2", RunStatus.FAILED)
        assert monitor.all_completed()
    
    def test_get_summary(self):
        monitor = MultiRunProgressMonitor(["run1", "run2", "run3"])
        monitor.update_status("run1", RunStatus.RUNNING)
        monitor.update_status("run2", RunStatus.COMPLETED)
        # run3 stays WAITING
        
        summary = monitor.get_summary()
        assert summary["total"] == 3
        assert summary["running"] == 1
        assert summary["completed"] == 1
        assert summary["waiting"] == 1
        assert summary["failed"] == 0
    
    def test_context_manager(self):
        with MultiRunProgressMonitor(["run1"]) as monitor:
            assert monitor.live is not None
            monitor.update_status("run1", RunStatus.RUNNING)
        
        # Live display should be stopped after context exit
        assert monitor.live is None
    
    def test_thread_safety(self):
        """Test that updates from multiple threads work safely."""
        import threading
        
        monitor = MultiRunProgressMonitor(["run1", "run2"])
        
        def update_run(run_name, iterations=100):
            for i in range(iterations):
                monitor.update_cplex_progress(run_name, {
                    'phase': 'barrier',
                    'iteration': str(i),
                    'mu': 1e-2 / (i + 1)
                })
        
        t1 = threading.Thread(target=update_run, args=("run1",))
        t2 = threading.Thread(target=update_run, args=("run2",))
        
        t1.start()
        t2.start()
        
        t1.join()
        t2.join()
        
        # Should complete without errors
        assert monitor.runs["run1"].iteration is not None
        assert monitor.runs["run2"].iteration is not None
