"""Unit tests for .lst file section extraction."""

from pathlib import Path

import pytest


class TestLstPageExtraction:
    """Test GAMS .lst file semantic section extraction."""

    @pytest.fixture
    def sample_lst_file(self):
        """Path to sample .lst file in data directory."""
        data_path = Path(__file__).parent.parent.parent / "data"
        lst_file = (
            data_path / "065Nov25-annualupto2045" / "parscen" / "parscen~0011" / "parscen~0011.lst"
        )

        if not lst_file.exists():
            pytest.skip(f"Sample .lst file not found at {lst_file}")

        return lst_file

    def test_extract_lst_pages(self, sample_lst_file):
        """Test that _extract_lst_pages correctly identifies and extracts semantic sections."""
        from times_doctor.core.llm import _extract_lst_pages

        # Read the file
        content = sample_lst_file.read_text(encoding="utf-8", errors="ignore")

        # Extract sections (now semantic, not page-based)
        result = _extract_lst_pages(content)

        # Should return a dict with 'sections' key
        assert "sections" in result
        assert isinstance(result["sections"], list)
        assert len(result["sections"]) > 0

        # Should also have extracted_text
        assert "extracted_text" in result
        assert len(result["extracted_text"]) > 0

        # Check that sections have required keys (for backward compatibility)
        for section in result["sections"]:
            assert "name" in section
            assert "start_line" in section
            assert "end_line" in section

        # Should extract compilation, execution, and model analysis sections
        section_names = [s["name"] for s in result["sections"]]
        " ".join(section_names).lower()

        # Check for expected section types
        assert any(
            "compilation" in name.lower() or "o m p i l a t i o n" in name for name in section_names
        ), "Should extract compilation sections"

    def test_extract_lst_pages_line_numbers(self, sample_lst_file):
        """Test that extracted sections have valid line numbers (even if placeholder)."""
        from times_doctor.core.llm import _extract_lst_pages

        content = sample_lst_file.read_text(encoding="utf-8", errors="ignore")
        result = _extract_lst_pages(content)

        for section in result["sections"]:
            start = section["start_line"]
            end = section["end_line"]

            # Line numbers should be positive integers
            assert isinstance(start, int)
            assert isinstance(end, int)
            assert start > 0
            assert end > 0

    def test_extract_lst_pages_with_progress_callback(self, sample_lst_file):
        """Test that progress callback is called during extraction."""
        from times_doctor.core.llm import _extract_lst_pages

        content = sample_lst_file.read_text(encoding="utf-8", errors="ignore")

        calls = []

        def progress_callback(current, total, message):
            calls.append((current, total, message))

        _extract_lst_pages(content, progress_callback=progress_callback)

        # Should have called progress callback
        assert len(calls) > 0

        # First call should mention parsing or sections
        first_call = calls[0]
        assert "parsing" in first_call[2].lower() or "section" in first_call[2].lower(), (
            f"Expected parsing/section message, got: {first_call[2]}"
        )

    def test_extract_condensed_sections_uses_page_extraction(self, sample_lst_file):
        """Test that extract_condensed_sections correctly handles LST files."""
        from times_doctor.core.llm import extract_condensed_sections

        content = sample_lst_file.read_text(encoding="utf-8", errors="ignore")
        result = extract_condensed_sections(content, "lst")

        # Should have sections
        assert "sections" in result
        assert len(result["sections"]) > 0

        # Should have extracted text
        assert "extracted_text" in result
        assert len(result["extracted_text"]) > 0

        # Check that extraction worked - should have meaningful section names
        for section in result["sections"]:
            assert section["name"], "Section should have a non-empty name"


class TestLstContentExtraction:
    """Test that LST content extraction produces useful output."""

    @pytest.fixture
    def sample_lst_file(self):
        """Path to sample .lst file."""
        data_path = Path(__file__).parent.parent.parent / "data"
        lst_file = (
            data_path / "065Nov25-annualupto2045" / "parscen" / "parscen~0011" / "parscen~0011.lst"
        )

        if not lst_file.exists():
            pytest.skip(f"Sample .lst file not found at {lst_file}")

        return lst_file

    def test_extracted_pages_contain_useful_content(self, sample_lst_file):
        """Test that extracted sections contain substantial diagnostic content."""
        from times_doctor.core.llm import _extract_lst_pages

        content = sample_lst_file.read_text(encoding="utf-8", errors="ignore")
        result = _extract_lst_pages(content)

        # Check that extracted text has content
        extracted_text = result["extracted_text"]
        assert len(extracted_text) > 1000, "Extracted text should be substantial (>1000 chars)"

        # Check for expected content types
        extracted_lower = extracted_text.lower()

        # Should mention compilation, execution, or model analysis
        has_compilation = "compilation" in extracted_lower or "error" in extracted_lower
        has_execution = "execution" in extracted_lower or "time" in extracted_lower
        has_model = "model" in extracted_lower or "equation" in extracted_lower

        assert has_compilation or has_execution or has_model, (
            "Extracted text should contain compilation, execution, or model information"
        )
