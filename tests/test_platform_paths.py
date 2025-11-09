"""Tests for cross-platform path handling and line ending compatibility."""

import sys

import pytest

from times_doctor.core.lst_parser import (
    CompilationProcessor,
    LSTParser,
    LSTSection,
    process_lst_file,
)


class TestCrossPlatformPaths:
    """Test that path handling works across Windows, macOS, and Linux."""

    def test_windows_style_paths_in_lst(self, tmp_path):
        """Test parsing LST files with Windows-style paths (backslashes, drive letters)."""
        content = r"""GAMS 49.6.1  55d34574 May 28, 2025          WEX-WEI x86 64bit/MS Windows - 11/06/25 13:17:19 Page 3
TIMES -- VERSION 4.8.3 -- Restart (v4.8)
C o m p i l a t i o n


556722  'AUD25'.'AUD25' 1
****          $170    $170
**** LINE  10808 BATINCLUDE  C:\Users\John\Veda\model\file.dd
**** LINE     88 INPUT       C:\Users\John\Veda\model\file.RUN
**** 170  Domain violation for element
556741  'ACT'.2015.'AUD25' 0.07
****                     $170
**** LINE  10827 BATINCLUDE  C:\Users\John\Veda\model\file.dd
**** LINE     88 INPUT       C:\Users\John\Veda\model\file.RUN
**** 170  Domain violation for element
"""
        lst_file = tmp_path / "test_windows.lst"
        lst_file.write_text(content)

        result = process_lst_file(lst_file)
        assert result is not None
        assert "metadata" in result
        assert result["metadata"]["file"] == str(lst_file)

    def test_unix_style_paths_in_lst(self, tmp_path):
        """Test parsing LST files with Unix-style paths (forward slashes)."""
        content = """GAMS 49.6.1  55d34574 May 28, 2025          LEX-LEG x86 64bit/Linux - 11/06/25 13:17:19 Page 3
TIMES -- VERSION 4.8.3 -- Restart (v4.8)
C o m p i l a t i o n


556722  'AUD25'.'AUD25' 1
****          $170    $170
**** LINE  10808 BATINCLUDE  /home/user/veda/model/file.dd
**** LINE     88 INPUT       /home/user/veda/model/file.RUN
**** 170  Domain violation for element
"""
        lst_file = tmp_path / "test_unix.lst"
        lst_file.write_text(content)

        result = process_lst_file(lst_file)
        assert result is not None
        assert "metadata" in result

    def test_pathlib_path_handling(self, tmp_path):
        """Test that pathlib.Path objects work correctly across platforms."""
        # Create nested directory structure
        nested_dir = tmp_path / "level1" / "level2"
        nested_dir.mkdir(parents=True, exist_ok=True)

        lst_file = nested_dir / "test.lst"
        content = """GAMS 49.6.1  55d34574 May 28, 2025          WEX-WEI x86 64bit/MS Windows - 11/06/25 13:17:19 Page 1
TIMES -- VERSION 4.8.3
E x e c u t i o n

----    461 Other                    0.000     0.031 SECS    118 MB
"""
        lst_file.write_text(content)

        # Test with Path object
        result = process_lst_file(lst_file)
        assert result is not None

        # Test with string path
        result_str = process_lst_file(str(lst_file))
        assert result_str is not None

    def test_path_with_spaces(self, tmp_path):
        """Test handling paths with spaces (common on Windows)."""
        dir_with_spaces = tmp_path / "My Documents" / "GAMS Files"
        dir_with_spaces.mkdir(parents=True, exist_ok=True)

        lst_file = dir_with_spaces / "model output.lst"
        content = """GAMS 49.6.1  55d34574 May 28, 2025          WEX-WEI x86 64bit/MS Windows - 11/06/25 13:17:19 Page 1
TIMES -- VERSION 4.8.3
E x e c u t i o n

----    461 Other                    0.000     0.031 SECS    118 MB
"""
        lst_file.write_text(content)

        result = process_lst_file(lst_file)
        assert result is not None
        assert (
            "My Documents" in result["metadata"]["file"]
            or "My%20Documents" in result["metadata"]["file"]
        )

    def test_mixed_path_separators_in_content(self, tmp_path):
        """Test that content with mixed path separators is handled correctly."""
        # Some GAMS output might have inconsistent separators
        content = r"""GAMS 49.6.1  55d34574 May 28, 2025          WEX-WEI x86 64bit/MS Windows - 11/06/25 13:17:19 Page 3
TIMES -- VERSION 4.8.3
C o m p i l a t i o n


**** LINE  10808 BATINCLUDE  C:\Users\John/Veda\model/file.dd
**** LINE     88 INPUT       C:/Users/John\Veda/model\file.RUN
**** 170  Domain violation for element
"""
        section = LSTSection(
            name="Compilation",
            page_number=3,
            start_line=1,
            end_line=10,
            header="GAMS header",
            content=content,
        )

        # Should not crash with mixed separators
        result = CompilationProcessor.process(section)
        assert result is not None


class TestLineEndings:
    """Test that different line ending styles (LF, CRLF) are handled correctly."""

    def test_unix_line_endings(self, tmp_path):
        """Test parsing LST files with Unix line endings (LF)."""
        content = "GAMS 49.6.1\nTIMES\nC o m p i l a t i o n\n\nLine 1\nLine 2\n"
        lst_file = tmp_path / "test_lf.lst"
        lst_file.write_text(content, newline="\n")

        parser = LSTParser(lst_file)
        sections = parser.parse()
        assert len(sections) >= 0  # Parser should handle LF without crashing

    def test_windows_line_endings(self, tmp_path):
        """Test parsing LST files with Windows line endings (CRLF)."""
        content = "GAMS 49.6.1\r\nTIMES\r\nC o m p i l a t i o n\r\n\r\nLine 1\r\nLine 2\r\n"
        lst_file = tmp_path / "test_crlf.lst"
        lst_file.write_text(content, newline="\r\n")

        parser = LSTParser(lst_file)
        sections = parser.parse()
        assert len(sections) >= 0  # Parser should handle CRLF without crashing

    def test_mixed_line_endings(self, tmp_path):
        """Test parsing LST files with mixed line endings (uncommon but possible)."""
        content = "GAMS 49.6.1\nTIMES\r\nC o m p i l a t i o n\r\n\nLine 1\r\nLine 2\n"
        lst_file = tmp_path / "test_mixed.lst"
        # Write in binary mode to preserve exact line endings
        lst_file.write_bytes(content.encode("utf-8"))

        parser = LSTParser(lst_file)
        sections = parser.parse()
        assert len(sections) >= 0  # Parser should handle mixed endings

    def test_line_ending_normalization(self, tmp_path):
        """Test that line endings are normalized correctly during parsing."""
        # Create same content with different line endings
        base_content = """GAMS 49.6.1  55d34574 May 28, 2025          WEX-WEI x86 64bit/MS Windows - 11/06/25 13:17:19 Page 3
TIMES -- VERSION 4.8.3
C o m p i l a t i o n


556722  'AUD25'.'AUD25' 1
****          $170    $170
**** 170  Domain violation for element
"""

        # Test with LF
        lst_lf = tmp_path / "test_lf.lst"
        lst_lf.write_text(base_content, newline="\n")

        # Test with CRLF
        lst_crlf = tmp_path / "test_crlf.lst"
        lst_crlf.write_text(base_content, newline="\r\n")

        # Both should produce same results
        result_lf = process_lst_file(lst_lf)
        result_crlf = process_lst_file(lst_crlf)

        assert result_lf is not None
        assert result_crlf is not None
        # Section counts should match
        assert result_lf["metadata"]["section_count"] == result_crlf["metadata"]["section_count"]


class TestPlatformSpecificEdgeCases:
    """Test platform-specific edge cases."""

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific test")
    def test_windows_drive_letter_handling(self, tmp_path):
        """Test handling of Windows drive letters (C:, D:, etc.)."""
        content = r"""GAMS 49.6.1  55d34574 May 28, 2025          WEX-WEI x86 64bit/MS Windows - 11/06/25 13:17:19 Page 1
TIMES -- VERSION 4.8.3
C o m p i l a t i o n

**** LINE  10808 BATINCLUDE  D:\Models\file.dd
**** LINE     88 INPUT       E:\Data\file.RUN
"""
        lst_file = tmp_path / "test_drives.lst"
        lst_file.write_text(content)

        result = process_lst_file(lst_file)
        assert result is not None

    def test_unc_paths_in_content(self, tmp_path):
        """Test handling of UNC paths (\\\\server\\share) in Windows environments."""
        content = r"""GAMS 49.6.1  55d34574 May 28, 2025          WEX-WEI x86 64bit/MS Windows - 11/06/25 13:17:19 Page 1
TIMES -- VERSION 4.8.3
C o m p i l a t i o n

**** LINE  10808 BATINCLUDE  \\server\share\models\file.dd
**** LINE     88 INPUT       \\nas-01\data\file.RUN
"""
        lst_file = tmp_path / "test_unc.lst"
        lst_file.write_text(content)

        result = process_lst_file(lst_file)
        assert result is not None

    def test_long_paths_windows(self, tmp_path):
        """Test handling of long paths (Windows has 260 char limit without special handling)."""
        # Create a moderately long path
        long_name = "a" * 50
        nested = tmp_path / long_name / long_name / long_name
        nested.mkdir(parents=True, exist_ok=True)

        lst_file = nested / "test.lst"
        content = """GAMS 49.6.1  55d34574 May 28, 2025          WEX-WEI x86 64bit/MS Windows - 11/06/25 13:17:19 Page 1
TIMES -- VERSION 4.8.3
E x e c u t i o n

----    461 Other                    0.000     0.031 SECS    118 MB
"""
        try:
            lst_file.write_text(content)
            result = process_lst_file(lst_file)
            assert result is not None
        except OSError:
            # Skip if path is too long for the platform
            pytest.skip("Path too long for this platform")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
