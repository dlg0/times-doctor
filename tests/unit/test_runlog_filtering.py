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
Important line A
DMoves update: 12345
PMoves update: 67890
Important line B
Iteration: 100
Elapsed time = 5.2 seconds
Important line C
Another DMoves line here
Final important line D
"""
        
        result = _filter_run_log(test_content)
        filtered = result["filtered_content"]
        
        # Should keep important lines (note: may be condensed if they have similar patterns)
        assert "Important line A" in filtered or "Important" in filtered
        assert "Final important line D" in filtered or "Final" in filtered
        
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
    
    def test_extract_condensed_sections_uses_filtering(self, sample_run_log_file):
        """Test that extract_condensed_sections uses filtering for run_log files."""
        from times_doctor.llm import extract_condensed_sections
        
        content = sample_run_log_file.read_text(encoding='utf-8', errors='ignore')
        
        result = extract_condensed_sections(content, "run_log")
        
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
        
        content = """First line here
DMoves here
Second line here
PMoves here
Third line here
"""
        
        result = _filter_run_log(content)
        filtered = result["filtered_content"]
        
        # Should still filter DMoves/PMoves even without execution marker
        assert "DMoves" not in filtered
        assert "PMoves" not in filtered
        # At least one of the important lines should be present (may be condensed)
        assert "First line" in filtered or "Second line" in filtered or "Third line" in filtered
    
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


class TestRunLogCondensing:
    """Test run_log condensing functionality."""
    
    def test_condense_pre_execution_errors(self):
        """Test that errors before execution start are captured and condensed."""
        from times_doctor.llm import _filter_run_log
        
        content = """*** Error 170 in D:\\Veda\\syssettings.dd
    Domain violation for element
*** Error 170 in D:\\Veda\\syssettings.dd
    Domain violation for element
*** Error 170 in D:\\Veda\\syssettings.dd
    Domain violation for element
*** Error 170 in D:\\Veda\\syssettings.dd
    Domain violation for element
--- Some other line
--- Starting execution: elapsed 0:00:34.104
--- parscen~0011.RUN(124) 118 Mb
--- parscen~0011.RUN(2101186) 118 Mb
"""
        
        result = _filter_run_log(content)
        filtered = result["filtered_content"]
        
        # Should contain the error
        assert "*** Error 170" in filtered
        assert "Domain violation for element" in filtered
        
        # Should be condensed (repeated 4 times)
        assert "repeated 4 times" in filtered
        
        # Should have separator
        assert "[Pre-execution errors above]" in filtered
        
        # Should include execution content after separator
        assert "Starting execution" in filtered
    
    def test_condense_multiline_errors(self):
        """Test that multi-line error patterns are condensed."""
        from times_doctor.llm import _condense_multiline_errors
        
        # Same file, same error, same description = should condense
        lines = [
            "*** Error 170 in syssettings.dd",
            "    Domain violation for element",
            "*** Error 170 in syssettings.dd",
            "    Domain violation for element",
            "*** Error 170 in syssettings.dd",
            "    Domain violation for element",
            "*** Error 170 in syssettings.dd",
            "    Domain violation for element",
            "*** Error 180 in other.dd",
            "    Different error message",
        ]
        
        result = _condense_multiline_errors(lines)
        condensed_text = '\n'.join(result)
        
        # Error 170 should be condensed (appears 4 times consecutively)
        assert "repeated" in condensed_text and "times" in condensed_text
        
        # Error 180 should not be condensed (only appears once)
        assert "*** Error 180 in other.dd" in condensed_text
        assert "Different error message" in condensed_text
        
        # Result should be shorter than input
        assert len(result) < len(lines)
    
    def test_condense_repetitive_lines(self):
        """Test that repetitive lines differing only in numbers are condensed."""
        from times_doctor.llm import _condense_repetitive_lines
        
        lines = [
            "--- parscen~0011.RUN(124) 118 Mb",
            "--- parscen~0011.RUN(2101186) 118 Mb",
            "--- parscen~0011.RUN(2101211) 118 Mb",
            "--- parscen~0011.RUN(2101214) 118 Mb",
            "--- GDX File operation complete",
            "--- parscen~0011.RUN(2101712) 129 Mb",
            "--- parscen~0011.RUN(2101765) 135 Mb",
        ]
        
        result = _condense_repetitive_lines(lines)
        result_text = '\n'.join(result)
        
        # Should condense first group (4 similar lines)
        assert "repeated" in result_text
        
        # Should keep GDX line (different pattern)
        assert "GDX File operation complete" in result_text
        
        # Result should be shorter
        assert len(result) < len(lines)
    
    def test_condense_repetitive_lines_shows_sample_values(self):
        """Test that condensed output shows sample varying values."""
        from times_doctor.llm import _condense_repetitive_lines
        
        lines = [
            "--- parscen~0011.RUN(100) 50 Mb",
            "--- parscen~0011.RUN(200) 60 Mb",
            "--- parscen~0011.RUN(300) 70 Mb",
            "--- parscen~0011.RUN(400) 80 Mb",
        ]
        
        result = _condense_repetitive_lines(lines)
        
        # Should show example values
        summary_line = [line for line in result if "repeated" in line][0]
        assert "100" in summary_line or "200" in summary_line or "300" in summary_line
        assert "values:" in summary_line.lower()
    
    def test_format_group_small_groups_unchanged(self):
        """Test that small groups (1-2 items) are not condensed."""
        from times_doctor.llm import _format_group
        
        # Single item
        result = _format_group(["line 1"], "pattern")
        assert result == ["line 1"]
        
        # Two items
        result = _format_group(["line 1", "line 2"], "pattern")
        assert result == ["line 1", "line 2"]
    
    def test_format_group_large_groups_condensed(self):
        """Test that groups of 3+ items are condensed."""
        from times_doctor.llm import _format_group
        
        group = [
            "--- file.RUN(100) 50 Mb",
            "--- file.RUN(200) 60 Mb",
            "--- file.RUN(300) 70 Mb",
        ]
        
        result = _format_group(group, "--- file.RUN({N}) {N} Mb")
        
        # Should return first line + summary
        assert len(result) == 2
        assert result[0] == group[0]
        assert "repeated 3 times" in result[1]
    
    def test_end_to_end_condensing_with_real_patterns(self, sample_run_log_file=None):
        """Test end-to-end condensing with realistic run_log patterns."""
        from times_doctor.llm import _filter_run_log
        
        content = """*** Error 170 in syssettings.dd
    Domain violation for element
*** Error 170 in syssettings.dd
    Domain violation for element
*** Error 170 in syssettings.dd
    Domain violation for element
--- Starting execution: elapsed 0:00:34.104
--- parscen~0011.RUN(124) 118 Mb
--- parscen~0011.RUN(2101186) 118 Mb
--- parscen~0011.RUN(2101211) 118 Mb
--- parscen~0011.RUN(2101214) 118 Mb
DMoves update here
PMoves update here
--- GDX File (execute_unload) D:\\file.gdx
--- parscen~0011.RUN(2134842) 2059 Mb
Iteration: 100
--- parscen~0011.RUN(2134842) 2062 Mb
--- parscen~0011.RUN(2134842) 2080 Mb
--- Generating LP model TIMES
"""
        
        result = _filter_run_log(content)
        filtered = result["filtered_content"]
        
        # Verify pre-execution errors are captured
        assert "*** Error 170" in filtered
        assert "Domain violation" in filtered
        assert "repeated 3 times" in filtered
        
        # Verify separator exists
        assert "[Pre-execution errors above]" in filtered
        
        # Verify DMoves/PMoves filtered out
        assert "DMoves" not in filtered
        assert "PMoves" not in filtered
        assert "Iteration:" not in filtered
        
        # Verify important milestones kept
        assert "Starting execution" in filtered
        assert "GDX File" in filtered
        assert "Generating LP model TIMES" in filtered
        
        # Verify repetitive lines condensed
        assert "repeated" in filtered
        
        # Verify output is significantly shorter
        original_lines = len(content.split('\n'))
        filtered_lines = len(filtered.split('\n'))
        # With condensing, we should have significant reduction
        # Original had ~20 lines, expect condensed to be much smaller
        assert filtered_lines < 18  # Should be well under original
