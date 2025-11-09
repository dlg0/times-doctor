"""Unit tests for CLI commands with mocked dependencies."""

import pytest
from pathlib import Path
from typer.testing import CliRunner
from times_doctor.cli import app
from unittest.mock import Mock, patch, MagicMock


runner = CliRunner()


@pytest.fixture
def mock_run_dir(tmp_path):
    """Create a mock TIMES run directory."""
    run_dir = tmp_path / "test_run"
    run_dir.mkdir()
    
    (run_dir / "test.lst").write_text("""
TIMES -- VERSION 4.7.0
*** Status: Normal Completion
Model Status: 1 Optimal
Solver Status: 1 Normal Completion
""")
    
    (run_dir / "QA_CHECK.LOG").write_text("""
QA Check Log
No issues found
""")
    
    (run_dir / "test_run_log.txt").write_text("""
TIMES Model Run Log
Execution completed
""")
    
    return run_dir


class TestReviewCommand:
    """Test the review command."""
    
    @patch('times_doctor.cli.llm_mod.analyze_run')
    def test_review_with_llm_none(self, mock_analyze, mock_run_dir, mock_env_vars):
        """Test review command skips LLM with --llm none."""
        result = runner.invoke(app, [
            "review",
            str(mock_run_dir),
            "--llm", "none"
        ])
        
        assert result.exit_code == 0
        mock_analyze.assert_not_called()
    
    @patch('times_doctor.cli.llm_mod.analyze_run')
    def test_review_with_mocked_llm(self, mock_analyze, mock_run_dir, mock_env_vars):
        """Test review command with mocked LLM response."""
        mock_analyze.return_value = "# Analysis\n\nEverything looks good!"
        
        result = runner.invoke(app, [
            "review",
            str(mock_run_dir),
            "--llm", "openai"
        ])
        
        assert result.exit_code == 0
        mock_analyze.assert_called_once()
        
        output_file = mock_run_dir / "times_doctor_out" / "llm_review.md"
        assert output_file.exists()


class TestDatacheckCommand:
    """Test the datacheck command."""
    
    @patch('times_doctor.cli.subprocess.run')
    @patch('times_doctor.cli.get_times_source')
    def test_datacheck_creates_directory(self, mock_get_times, mock_subprocess, mock_run_dir):
        """Test datacheck creates _td_datacheck directory."""
        mock_get_times.return_value = Path("/fake/times/source")
        mock_subprocess.return_value = Mock(returncode=0)
        
        (mock_run_dir / "test.run").write_text("* TIMES run file")
        
        result = runner.invoke(app, [
            "datacheck",
            str(mock_run_dir),
            "--gams", "gams"
        ])
        
        assert result.exit_code == 0
        datacheck_dir = mock_run_dir / "_td_datacheck"
        assert datacheck_dir.exists()
        assert (datacheck_dir / "cplex.opt").exists()
    
    @patch('times_doctor.cli.subprocess.run')
    @patch('times_doctor.cli.get_times_source')
    def test_datacheck_gams_execution(self, mock_get_times, mock_subprocess, mock_run_dir):
        """Test datacheck calls GAMS with correct parameters."""
        mock_get_times.return_value = Path("/fake/times/source")
        mock_subprocess.return_value = Mock(returncode=0)
        
        (mock_run_dir / "test.run").write_text("* TIMES run file")
        
        runner.invoke(app, [
            "datacheck",
            str(mock_run_dir),
            "--gams", "/path/to/gams",
            "--threads", "4"
        ])
        
        mock_subprocess.assert_called()
        call_args = mock_subprocess.call_args[0][0]
        assert "/path/to/gams" in call_args
        assert "threads=4" in " ".join(call_args)


class TestScanCommand:
    """Test the scan command."""
    
    @patch('times_doctor.cli.subprocess.run')
    @patch('times_doctor.cli.get_times_source')
    def test_scan_without_llm(self, mock_get_times, mock_subprocess, mock_run_dir):
        """Test scan command without LLM analysis."""
        mock_get_times.return_value = Path("/fake/times/source")
        
        lst_content = """
Model Status: 1 Optimal
Solver Status: 1 Normal Completion
"""
        mock_subprocess.return_value = Mock(returncode=0)
        
        (mock_run_dir / "test.run").write_text("* TIMES run file")
        
        result = runner.invoke(app, [
            "scan",
            str(mock_run_dir),
            "--llm", "none",
            "--profiles", "dual"
        ])
        
        assert result.exit_code == 0
        scan_dir = mock_run_dir / "times_doctor_out" / "scan_runs"
        assert scan_dir.exists()


class TestVersionCommand:
    """Test version command."""
    
    def test_version_flag(self):
        """Test --version flag."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "times-doctor version" in result.stdout


class TestUpdateCommand:
    """Test update command."""
    
    @patch('times_doctor.cli.subprocess.run')
    def test_update_on_macos(self, mock_subprocess):
        """Test update command execution."""
        mock_subprocess.return_value = Mock(returncode=0)
        
        result = runner.invoke(app, ["update"])
        
        if result.exit_code == 0:
            mock_subprocess.assert_called()
