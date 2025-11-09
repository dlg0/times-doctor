"""Unit tests for CLI commands with mocked dependencies."""

from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from times_doctor.cli import app

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

    @patch("times_doctor.cli.llm_mod.condense_qa_check")
    @patch("times_doctor.cli.llm_mod.extract_condensed_sections")
    @patch("times_doctor.cli.llm_mod.create_condensed_markdown")
    @patch("times_doctor.cli.llm_mod.review_files")
    @patch("times_doctor.cli.llm_mod.check_api_keys")
    def test_review_with_llm_none(
        self,
        mock_check_keys,
        mock_review,
        mock_create,
        mock_extract,
        mock_condense,
        mock_run_dir,
        mock_env_vars,
    ):
        """Test review command skips LLM with --llm none."""
        from times_doctor.core.llm import LLMResult

        mock_check_keys.return_value = {"openai": False, "anthropic": False, "amp": False}
        mock_condense.return_value = "condensed qa check"
        mock_extract.return_value = {"sections": [], "filtered_content": "filtered run log"}
        mock_create.return_value = "# Condensed LST"
        mock_review.return_value = LLMResult(
            text="# Review", used=True, provider="none", model="none"
        )

        result = runner.invoke(app, ["review", str(mock_run_dir), "--llm", "none"])

        assert result.exit_code == 0
        mock_review.assert_called_once()

    @patch("times_doctor.cli.llm_mod.condense_qa_check")
    @patch("times_doctor.cli.llm_mod.extract_condensed_sections")
    @patch("times_doctor.cli.llm_mod.create_condensed_markdown")
    @patch("times_doctor.cli.llm_mod.review_files")
    @patch("times_doctor.cli.llm_mod.check_api_keys")
    def test_review_with_mocked_llm(
        self,
        mock_check_keys,
        mock_review,
        mock_create,
        mock_extract,
        mock_condense,
        mock_run_dir,
        mock_env_vars,
    ):
        """Test review command with mocked LLM response."""
        from times_doctor.core.llm import LLMResult

        mock_check_keys.return_value = {"openai": True, "anthropic": False, "amp": False}
        mock_condense.return_value = "condensed qa check"
        mock_extract.return_value = {"sections": [], "filtered_content": "filtered run log"}
        mock_create.return_value = "# Condensed LST"
        mock_review.return_value = LLMResult(
            text="# Analysis\n\nEverything looks good!", used=True, provider="openai", model="gpt-5"
        )

        result = runner.invoke(app, ["review", str(mock_run_dir), "--llm", "openai"])

        assert result.exit_code == 0
        mock_review.assert_called_once()

        output_file = mock_run_dir / "times_doctor_out" / "llm_review.md"
        assert output_file.exists()


class TestDatacheckCommand:
    """Test the datacheck command."""

    @patch("times_doctor.cli.run_gams_with_progress")
    @patch("times_doctor.cli.get_times_source")
    def test_datacheck_creates_directory(
        self, mock_get_times, mock_run_gams, mock_run_dir, tmp_path
    ):
        """Test datacheck creates _td_datacheck directory."""
        fake_times_dir = tmp_path / "fake_times_source"
        fake_times_dir.mkdir()
        (fake_times_dir / "_times.g00").touch()
        mock_get_times.return_value = fake_times_dir
        mock_run_gams.return_value = 0

        (mock_run_dir / "test.run").write_text("* TIMES run file")

        result = runner.invoke(app, ["datacheck", str(mock_run_dir), "--gams-path", "gams"])

        assert result.exit_code == 0
        datacheck_dir = mock_run_dir / "_td_datacheck"
        assert datacheck_dir.exists()
        assert (datacheck_dir / "cplex.opt").exists()

    @patch("times_doctor.cli.run_gams_with_progress")
    @patch("times_doctor.cli.get_times_source")
    def test_datacheck_gams_execution(self, mock_get_times, mock_run_gams, mock_run_dir, tmp_path):
        """Test datacheck calls GAMS with correct parameters."""
        fake_times_dir = tmp_path / "fake_times_source"
        fake_times_dir.mkdir()
        (fake_times_dir / "_times.g00").touch()
        mock_get_times.return_value = fake_times_dir
        mock_run_gams.return_value = 0

        (mock_run_dir / "test.run").write_text("* TIMES run file")

        runner.invoke(
            app, ["datacheck", str(mock_run_dir), "--gams-path", "/path/to/gams", "--threads", "4"]
        )

        mock_run_gams.assert_called()
        call_args = mock_run_gams.call_args[0][0]
        assert "/path/to/gams" in call_args

        datacheck_dir = mock_run_dir / "_td_datacheck"
        cplex_opt = datacheck_dir / "cplex.opt"
        assert cplex_opt.exists()
        opt_content = cplex_opt.read_text()
        assert "threads 4" in opt_content


class TestScanCommand:
    """Test the scan command."""

    @patch("times_doctor.cli.shutil.copytree")
    @patch("times_doctor.cli.run_gams_with_progress")
    @patch("times_doctor.cli.get_times_source")
    def test_scan_without_llm(
        self, mock_get_times, mock_run_gams, mock_copytree, mock_run_dir, tmp_path
    ):
        """Test scan command without LLM analysis."""
        fake_times_dir = tmp_path / "fake_times_source"
        fake_times_dir.mkdir()
        (fake_times_dir / "_times.g00").touch()
        mock_get_times.return_value = fake_times_dir
        mock_run_gams.return_value = 0

        (mock_run_dir / "test.run").write_text("* TIMES run file")

        # Mock copytree to just create the target directory
        def copytree_side_effect(src, dst, *args, **kwargs):
            Path(dst).mkdir(parents=True, exist_ok=True)
            (Path(dst) / "test.run").write_text("* TIMES run file")
            return dst

        mock_copytree.side_effect = copytree_side_effect

        # Create a mock _td_opt_files directory with a test .opt file
        opt_files_dir = mock_run_dir / "_td_opt_files"
        opt_files_dir.mkdir(parents=True, exist_ok=True)
        (opt_files_dir / "test_config.opt").write_text("epopt 1e-7\n")

        # Answer 'y' to confirmation prompt
        result = runner.invoke(app, ["scan", str(mock_run_dir), "--llm", "none"], input="y\n")

        if result.exit_code != 0:
            print(result.stdout)
            if result.exception:
                import traceback

                traceback.print_exception(
                    type(result.exception), result.exception, result.exception.__traceback__
                )

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

    def test_update_on_macos(self):
        """Test update command shows instructions."""
        result = runner.invoke(app, ["update"])

        assert result.exit_code == 0
        assert "update times-doctor" in result.stdout.lower()
        assert "version" in result.stdout.lower()
