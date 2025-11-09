"""Tests for CPLEX progress monitoring."""

import pytest
from times_doctor.cplex_progress import (
    BarrierProgressTracker,
    parse_cplex_line,
    format_progress_line,
    scan_log_for_progress,
)


class TestBarrierProgressTracker:
    """Test the barrier progress tracker."""
    
    def test_initial_mu_sets_baseline(self):
        tracker = BarrierProgressTracker(mu_target=1e-8)
        pct = tracker.update_mu(1e-2)
        assert pct is not None
        assert 0.0 <= pct <= 1.0
    
    def test_progress_increases_as_mu_decreases(self):
        tracker = BarrierProgressTracker(mu_target=1e-8)
        
        pct1 = tracker.update_mu(1e-2)
        pct2 = tracker.update_mu(1e-4)
        pct3 = tracker.update_mu(1e-6)
        
        assert pct1 < pct2 < pct3
    
    def test_progress_clamped_to_0_1(self):
        tracker = BarrierProgressTracker(mu_target=1e-8)
        
        # Even if mu goes beyond target
        tracker.update_mu(1e-2)
        pct = tracker.update_mu(1e-10)
        
        assert pct <= 1.0
    
    def test_handles_zero_mu(self):
        tracker = BarrierProgressTracker()
        pct = tracker.update_mu(0.0)
        assert pct is None
    
    def test_crossover_flag(self):
        tracker = BarrierProgressTracker()
        assert not tracker.in_crossover
        
        tracker.in_crossover = True
        assert tracker.in_crossover


class TestParseCplexLine:
    """Test parsing CPLEX log lines."""
    
    def test_parse_barrier_line(self):
        line = "Barrier iteration 10: mu = 1.23e-04, primal infeas = 2.34e-02, dual infeas = 3.45e-03"
        parsed = parse_cplex_line(line)
        
        assert parsed is not None
        assert parsed['phase'] == 'barrier'
        assert parsed['iteration'] == '10'
        assert parsed['mu'] == 1.23e-04
        assert parsed['primal_infeas'] == 2.34e-02
        assert parsed['dual_infeas'] == 3.45e-03
    
    def test_parse_crossover_line(self):
        line = "Starting crossover, iteration 25, mu = 9.99e-09"
        parsed = parse_cplex_line(line)
        
        assert parsed is not None
        assert parsed['phase'] == 'crossover'
        assert parsed['mu'] == 9.99e-09
    
    def test_parse_simplex_line(self):
        line = "Dual simplex iteration 42"
        parsed = parse_cplex_line(line)
        
        assert parsed is not None
        assert parsed['phase'] == 'simplex'
        assert parsed['iteration'] == '42'
    
    def test_parse_non_iteration_line(self):
        line = "CPLEX starting, initializing..."
        parsed = parse_cplex_line(line)
        
        # Should return None or empty dict
        assert parsed is None or not parsed
    
    def test_parse_real_cplex_format(self):
        # Actual CPLEX output format
        line = "    10    1.2345e-04  2.3456e-02  3.4567e-03"
        # This might not match since CPLEX format varies
        # Just ensure it doesn't crash
        parsed = parse_cplex_line(line)
        assert parsed is None or isinstance(parsed, dict)


class TestFormatProgressLine:
    """Test formatting progress lines."""
    
    def test_format_barrier_with_percentage(self):
        tracker = BarrierProgressTracker()
        tracker.update_mu(1e-2)
        
        parsed = {
            'phase': 'barrier',
            'iteration': '10',
            'mu': 1e-4,
            'primal_infeas': 2.34e-02,
            'dual_infeas': 3.45e-03
        }
        
        formatted = format_progress_line(parsed, tracker=tracker)
        
        assert 'barrier' in formatted
        assert 'it=10' in formatted
        assert 'mu=' in formatted
        assert 'Pinf=' in formatted
        assert 'Dinf=' in formatted
        assert '%' in formatted
    
    def test_format_crossover_no_percentage(self):
        tracker = BarrierProgressTracker()
        tracker.in_crossover = True
        
        parsed = {
            'phase': 'crossover',
            'iteration': '25',
            'mu': 9.99e-09
        }
        
        formatted = format_progress_line(parsed, tracker=tracker)
        
        assert 'crossover' in formatted
        assert '–' in formatted  # No percentage for crossover
    
    def test_format_without_tracker(self):
        parsed = {
            'phase': 'barrier',
            'iteration': '10',
            'mu': 1e-4
        }
        
        formatted = format_progress_line(parsed)
        
        assert 'barrier' in formatted
        assert 'it=10' in formatted


class TestScanLogForProgress:
    """Test scanning log files for progress."""
    
    def test_scan_multiple_lines(self):
        lines = [
            "CPLEX starting...",
            "Barrier iteration 10: mu = 1e-2, primal infeas = 1e-1, dual infeas = 1e-1",
            "Barrier iteration 20: mu = 1e-4, primal infeas = 1e-3, dual infeas = 1e-3",
            "Some other output",
            "Barrier iteration 30: mu = 1e-6, primal infeas = 1e-5, dual infeas = 1e-5",
            "Starting crossover",
        ]
        
        progress_lines = scan_log_for_progress(lines)
        
        # Should have found at least the barrier lines
        assert len(progress_lines) >= 3
        assert any('barrier' in line for line in progress_lines)
    
    def test_scan_empty_log(self):
        lines = []
        progress_lines = scan_log_for_progress(lines)
        assert progress_lines == []
    
    def test_scan_no_cplex_output(self):
        lines = [
            "GAMS starting...",
            "Compiling model...",
            "Model compiled successfully"
        ]
        
        progress_lines = scan_log_for_progress(lines)
        assert progress_lines == []
