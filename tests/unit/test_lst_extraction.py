"""Unit tests for .lst file page extraction."""

import pytest
from pathlib import Path


class TestLstPageExtraction:
    """Test GAMS .lst file page-based extraction."""
    
    @pytest.fixture
    def sample_lst_file(self):
        """Path to sample .lst file in data directory."""
        data_path = Path(__file__).parent.parent.parent / "data"
        lst_file = data_path / "065Nov25-annualupto2045" / "parscen" / "parscen~0011" / "parscen~0011.lst"
        
        if not lst_file.exists():
            pytest.skip(f"Sample .lst file not found at {lst_file}")
        
        return lst_file
    
    def test_extract_lst_pages(self, sample_lst_file):
        """Test that _extract_lst_pages correctly identifies and extracts pages 1, 2, 5, 6, 7, 8."""
        from times_doctor.llm import _extract_lst_pages
        
        # Read the file
        content = sample_lst_file.read_text(encoding='utf-8', errors='ignore')
        
        # Extract pages
        result = _extract_lst_pages(content)
        
        # Should return a dict with 'sections' key
        assert "sections" in result
        assert isinstance(result["sections"], list)
        
        # Extract page numbers from section names
        page_numbers = set()
        for section in result["sections"]:
            assert "name" in section
            assert "start_line" in section
            assert "end_line" in section
            
            # Parse page number from name like "Page 5"
            if section["name"].startswith("Page "):
                page_num = int(section["name"].split()[1])
                page_numbers.add(page_num)
        
        # Should extract pages 1, 2, 5, 6, 7, 8
        # (but only if they exist in the file)
        expected_pages = {1, 2, 5, 6, 7, 8}
        assert page_numbers.issubset(expected_pages), f"Found unexpected pages: {page_numbers - expected_pages}"
        
        # Should have at least pages 1 and 2
        assert 1 in page_numbers, "Page 1 should be extracted"
        assert 2 in page_numbers, "Page 2 should be extracted"
    
    def test_extract_lst_pages_line_numbers(self, sample_lst_file):
        """Test that extracted sections have valid line numbers."""
        from times_doctor.llm import _extract_lst_pages
        
        content = sample_lst_file.read_text(encoding='utf-8', errors='ignore')
        lines = content.split('\n')
        total_lines = len(lines)
        
        result = _extract_lst_pages(content)
        
        for section in result["sections"]:
            start = section["start_line"]
            end = section["end_line"]
            
            # Line numbers should be valid
            assert start >= 1, f"Start line {start} should be >= 1"
            assert end <= total_lines, f"End line {end} should be <= {total_lines}"
            assert start <= end, f"Start line {start} should be <= end line {end}"
            
            # The line at start_line should contain the page marker
            page_line = lines[start - 1]  # Convert to 0-indexed
            assert "GAMS" in page_line and "Page" in page_line, \
                f"Line {start} should contain GAMS page marker, got: {page_line[:100]}"
    
    def test_extract_lst_pages_with_progress_callback(self, sample_lst_file):
        """Test that progress callback is called."""
        from times_doctor.llm import _extract_lst_pages
        
        content = sample_lst_file.read_text(encoding='utf-8', errors='ignore')
        
        # Track progress callback calls
        callback_calls = []
        def progress_callback(current, total, message):
            callback_calls.append((current, total, message))
        
        result = _extract_lst_pages(content, progress_callback=progress_callback)
        
        # Should have called the callback at least once
        assert len(callback_calls) > 0, "Progress callback should be called"
        
        # First call should report found pages
        first_call = callback_calls[0]
        assert "pages" in first_call[2].lower(), f"Expected pages message, got: {first_call[2]}"
    
    def test_extract_useful_sections_uses_page_extraction(self, sample_lst_file):
        """Test that extract_useful_sections uses page extraction for .lst files."""
        from times_doctor.llm import extract_useful_sections
        
        content = sample_lst_file.read_text(encoding='utf-8', errors='ignore')
        
        # Should use page extraction (no LLM calls) for lst files
        result = extract_useful_sections(content, "lst")
        
        # Should return sections
        assert "sections" in result
        assert len(result["sections"]) > 0
        
        # All sections should be named "Page N"
        for section in result["sections"]:
            assert section["name"].startswith("Page "), \
                f"Expected page section, got: {section['name']}"


class TestLstContentExtraction:
    """Test extracting actual content from pages."""
    
    @pytest.fixture
    def sample_lst_file(self):
        """Path to sample .lst file in data directory."""
        data_path = Path(__file__).parent.parent.parent / "data"
        lst_file = data_path / "065Nov25-annualupto2045" / "parscen" / "parscen~0011" / "parscen~0011.lst"
        
        if not lst_file.exists():
            pytest.skip(f"Sample .lst file not found at {lst_file}")
        
        return lst_file
    
    def test_extracted_pages_contain_useful_content(self, sample_lst_file):
        """Test that extracted pages contain expected diagnostic content."""
        from times_doctor.llm import _extract_lst_pages
        
        content = sample_lst_file.read_text(encoding='utf-8', errors='ignore')
        lines = content.split('\n')
        
        result = _extract_lst_pages(content)
        
        # Check that each section contains non-trivial content
        for section in result["sections"]:
            start = section["start_line"]
            end = section["end_line"]
            
            # Extract the actual lines
            section_lines = lines[start-1:end]
            section_text = '\n'.join(section_lines)
            
            # Should have some content
            assert len(section_text) > 100, \
                f"{section['name']} should have substantial content"
            
            # Should contain the page header
            assert "GAMS" in section_text, \
                f"{section['name']} should contain GAMS header"
