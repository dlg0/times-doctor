"""Unit tests for ensure_threads_option function."""

from pathlib import Path

from times_doctor.cli import ensure_threads_option


class TestEnsureThreadsOption:
    """Test the ensure_threads_option helper function."""

    def test_create_cplex_opt_if_missing(self, tmp_path: Path):
        """Should create minimal CPLEX opt file if it doesn't exist."""
        opt_file = tmp_path / "cplex.opt"
        ensure_threads_option(opt_file, "cplex", 4)

        content = opt_file.read_text()
        assert "threads 4" in content

    def test_create_gurobi_opt_if_missing(self, tmp_path: Path):
        """Should create minimal GUROBI opt file if it doesn't exist."""
        opt_file = tmp_path / "gurobi.opt"
        ensure_threads_option(opt_file, "gurobi", 8)

        content = opt_file.read_text()
        assert "Threads 8" in content

    def test_update_existing_cplex_threads(self, tmp_path: Path):
        """Should update existing threads setting in CPLEX opt file."""
        opt_file = tmp_path / "cplex.opt"
        opt_file.write_text("lpmethod 4\nthreads 2\nsimdisplay 2\n")

        ensure_threads_option(opt_file, "cplex", 16)

        content = opt_file.read_text()
        assert "threads 16" in content
        assert "lpmethod 4" in content
        assert "simdisplay 2" in content

    def test_update_existing_gurobi_threads(self, tmp_path: Path):
        """Should update existing Threads setting in GUROBI opt file."""
        opt_file = tmp_path / "gurobi.opt"
        opt_file.write_text("Method 2\nThreads 4\nBarConvTol 1e-8\n")

        ensure_threads_option(opt_file, "gurobi", 12)

        content = opt_file.read_text()
        assert "Threads 12" in content
        assert "Method 2" in content
        assert "BarConvTol 1e-8" in content

    def test_case_insensitive_match(self, tmp_path: Path):
        """Should match threads parameter case-insensitively."""
        opt_file = tmp_path / "cplex.opt"
        opt_file.write_text("THREADS 4\n")

        ensure_threads_option(opt_file, "cplex", 8)

        content = opt_file.read_text()
        assert "THREADS 8" in content
        assert "threads 4" not in content.lower().replace("threads 8", "")

    def test_preserve_equals_separator(self, tmp_path: Path):
        """Should preserve '=' separator style if present."""
        opt_file = tmp_path / "gurobi.opt"
        opt_file.write_text("Method=2\nThreads=4\n")

        ensure_threads_option(opt_file, "gurobi", 10)

        content = opt_file.read_text()
        assert "Threads=10" in content
        assert "Method=2" in content

    def test_preserve_space_separator(self, tmp_path: Path):
        """Should preserve space separator style if present."""
        opt_file = tmp_path / "cplex.opt"
        opt_file.write_text("lpmethod 4\nthreads 2\n")

        ensure_threads_option(opt_file, "cplex", 6)

        content = opt_file.read_text()
        assert "threads 6" in content
        assert "lpmethod 4" in content

    def test_preserve_inline_comments(self, tmp_path: Path):
        """Should preserve inline comments after parameter values."""
        opt_file = tmp_path / "gurobi.opt"
        opt_file.write_text("Threads 4 # Use 4 threads for testing\nMethod 2\n")

        ensure_threads_option(opt_file, "gurobi", 8)

        content = opt_file.read_text()
        assert "Threads 8 # Use 4 threads for testing" in content

    def test_preserve_full_line_comments(self, tmp_path: Path):
        """Should preserve full-line comments (*, #, $)."""
        opt_file = tmp_path / "cplex.opt"
        opt_file.write_text("* CPLEX options\n# Comment line\nthreads 4\n$ Another comment\n")

        ensure_threads_option(opt_file, "cplex", 12)

        content = opt_file.read_text()
        assert "* CPLEX options" in content
        assert "# Comment line" in content
        assert "$ Another comment" in content
        assert "threads 12" in content

    def test_append_if_not_present(self, tmp_path: Path):
        """Should append threads setting if not present in file."""
        opt_file = tmp_path / "cplex.opt"
        opt_file.write_text("lpmethod 4\nsimdisplay 2\n")

        ensure_threads_option(opt_file, "cplex", 8)

        content = opt_file.read_text()
        assert "threads 8" in content
        assert "lpmethod 4" in content
        assert "simdisplay 2" in content

    def test_update_multiple_threads_occurrences(self, tmp_path: Path):
        """Should update all occurrences of threads parameter."""
        opt_file = tmp_path / "cplex.opt"
        opt_file.write_text("threads 2\nlpmethod 4\nthreads 4\n")

        ensure_threads_option(opt_file, "cplex", 16)

        content = opt_file.read_text()
        assert content.count("threads 16") == 2
        assert "threads 2" not in content
        assert "threads 4" not in content

    def test_preserve_whitespace_formatting(self, tmp_path: Path):
        """Should preserve leading whitespace in formatted files."""
        opt_file = tmp_path / "gurobi.opt"
        opt_file.write_text("  Method 2\n  Threads 4\n  BarConvTol 1e-8\n")

        ensure_threads_option(opt_file, "gurobi", 6)

        content = opt_file.read_text()
        assert "  Threads 6" in content
        assert "  Method 2" in content

    def test_handle_empty_file(self, tmp_path: Path):
        """Should handle empty opt file gracefully."""
        opt_file = tmp_path / "cplex.opt"
        opt_file.write_text("")

        ensure_threads_option(opt_file, "cplex", 4)

        content = opt_file.read_text()
        assert "threads 4" in content

    def test_cplex_dollar_comment(self, tmp_path: Path):
        """Should handle CPLEX-style $ inline comments."""
        opt_file = tmp_path / "cplex.opt"
        opt_file.write_text("threads 2 $ Use 2 threads\nlpmethod 4\n")

        ensure_threads_option(opt_file, "cplex", 8)

        content = opt_file.read_text()
        assert "threads 8 $ Use 2 threads" in content

    def test_gurobi_hash_comment(self, tmp_path: Path):
        """Should handle GUROBI-style # inline comments."""
        opt_file = tmp_path / "gurobi.opt"
        opt_file.write_text("Threads 4 # Number of threads\nMethod 2\n")

        ensure_threads_option(opt_file, "gurobi", 10)

        content = opt_file.read_text()
        assert "Threads 10 # Number of threads" in content
