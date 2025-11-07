"""Unit tests for run_log filtering."""

import pytest
from pathlib import Path


class TestRunLogFiltering:
    """Test run_log.txt filtering."""
    
    @pytest.fixture
    def sample_run_log_file(self):
        """Path to sample run_log file in data directory."""
        data_path = Path(__file__).parent.parent.parent / "data"
        run_log_file = data_path / "065Nov25-annualupto2045" / "parscen" / "parscen~0011" / "parscen~0011_run_log.txt"
        
        if not run_log_file.exists():
            pytest.skip(f"Sample run_log file not found at {run_log_file}")
        
        return run_log_file
    
    def test_filter_run_log_basic(self, sample_run_log_file):
        """Test that _filter_run_log returns filtered content."""
        from times_doctor.llm import _filter_run_log
        
        content = sample_run_log_file.read_text(encoding='utf-8', errors='ignore')
        
        result = _filter_run_log(content)
        
        # Should return dict with filtered_content
        assert "filtered_content" in result
        assert isinstance(result["filtered_content"], str)
        assert len(result["filtered_content"]) > 0
        
        # Should have empty sections
        assert "sections" in result
        assert result["sections"] == []
    
    def test_filter_run_log_skips_before_execution(self, sample_run_log_file):
        """Test that content before execution start is skipped."""
        from times_doctor.llm import _filter_run_log
        
        content = sample_run_log_file.read_text(encoding='utf-8', errors='ignore')
        original_lines = content.split('\n')
        
        result = _filter_run_log(content)
        filtered_content = result["filtered_content"]
        
        # Find where "starting execution" or "Restarting execution" appears
        exec_start_line = None
        for i, line in enumerate(original_lines):
            if "starting execution" in line.lower() or "Restarting execution" in line:
                exec_start_line = i
                break
        
        # Filtered content should not include lines before execution
        if exec_start_line and exec_start_line > 0:
            early_line = original_lines[exec_start_line - 10] if exec_start_line > 10 else original_lines[0]
            assert early_line not in filtered_content, \
                "Content before execution start should be filtered out"
    
    def test_filter_run_log_removes_noise(self):
        """Test that DMoves, PMoves, Iteration, and Elapsed time lines are removed."""
        from times_doctor.llm import _filter_run_log
        
        test_content = """--- Restarting execution
Important line 1
DMoves update: 12345
PMoves update: 67890
Important line 2
Iteration: 100
Elapsed time = 5.2 seconds
Important line 3
Another DMoves line here
Final important line
"""
        
        result = _filter_run_log(test_content)
        filtered = result["filtered_content"]
        
        # Should keep important lines
        assert "Important line 1" in filtered
        assert "Important line 2" in filtered
        assert "Important line 3" in filtered
        assert "Final important line" in filtered
        
        # Should remove noise lines
        assert "DMoves" not in filtered
        assert "PMoves" not in filtered
        assert "Iteration:" not in filtered
        assert "Elapsed time =" not in filtered
    
    def test_filter_run_log_with_progress_callback(self, sample_run_log_file):
        """Test that progress callback is called."""
        from times_doctor.llm import _filter_run_log
        
        content = sample_run_log_file.read_text(encoding='utf-8', errors='ignore')
        
        callback_calls = []
        def progress_callback(current, total, message):
            callback_calls.append((current, total, message))
        
        result = _filter_run_log(content, progress_callback=progress_callback)
        
        # Should have called the callback
        assert len(callback_calls) > 0
        
        # Message should mention filtering
        first_call = callback_calls[0]
        assert "filter" in first_call[2].lower() or "lines" in first_call[2].lower()
    
    def test_extract_useful_sections_uses_filtering(self, sample_run_log_file):
        """Test that extract_useful_sections uses filtering for run_log files."""
        from times_doctor.llm import extract_useful_sections
        
        content = sample_run_log_file.read_text(encoding='utf-8', errors='ignore')
        
        result = extract_useful_sections(content, "run_log")
        
        # Should return filtered_content
        assert "filtered_content" in result
        assert len(result["filtered_content"]) > 0
        
        # Filtered content should be shorter than original
        assert len(result["filtered_content"]) < len(content)


class TestRunLogFilteringEdgeCases:
    """Test edge cases in run_log filtering."""
    
    def test_filter_run_log_no_execution_marker(self):
        """Test filtering when no execution marker is found."""
        from times_doctor.llm import _filter_run_log
        
        content = """Line 1
DMoves here
Line 2
PMoves here
Line 3
"""
        
        result = _filter_run_log(content)
        filtered = result["filtered_content"]
        
        # Should still filter DMoves/PMoves even without execution marker
        assert "DMoves" not in filtered
        assert "PMoves" not in filtered
        assert "Line 1" in filtered
        assert "Line 2" in filtered
        assert "Line 3" in filtered
    
    def test_filter_run_log_case_insensitive_execution(self):
        """Test that execution marker search is case-insensitive."""
        from times_doctor.llm import _filter_run_log
        
        content = """Before line
--- STARTING EXECUTION
After line
"""
        
        result = _filter_run_log(content)
        filtered = result["filtered_content"]
        
        # Should skip "Before line"
        assert "Before line" not in filtered
        assert "After line" in filtered
    
    def test_filter_run_log_empty_content(self):
        """Test filtering empty content."""
        from times_doctor.llm import _filter_run_log
        
        result = _filter_run_log("")
        
        assert "filtered_content" in result
        assert result["filtered_content"] == ""
