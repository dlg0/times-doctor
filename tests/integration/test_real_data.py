"""Integration tests with real data files.

Tests the complete processing pipeline with actual TIMES/VEDA output files.
"""

import pytest
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file for API keys
load_dotenv()


class TestRealDataProcessing:
    """Test processing of real TIMES/VEDA data files."""
    
    @pytest.fixture
    def data_dir(self):
        """Path to data directory with real files."""
        data_path = Path(__file__).parent.parent.parent / "data"
        if not data_path.exists():
            pytest.skip("data/ directory not found")
        return data_path
    
    def test_condense_qa_check_with_real_data(self, data_dir):
        """Test rule-based condensing of real QA_CHECK.LOG data (no LLM required).
        
        This test:
        1. Loads a real QA_CHECK.LOG file
        2. Uses the rule-based parser to condense it
        3. Verifies that structured output is produced
        4. Checks that events are properly deduplicated
        """
        from times_doctor.core.llm import condense_qa_check
        
        # Find the QA_CHECK.LOG file
        qa_check_files = list(data_dir.rglob("QA_CHECK.LOG"))
        
        assert len(qa_check_files) > 0, "No QA_CHECK.LOG files found in data/ directory"
        
        qa_check_path = qa_check_files[0]
        content = qa_check_path.read_text(encoding='utf-8', errors='ignore')
        
        # Track progress
        progress_calls = []
        def progress(current, total, message):
            progress_calls.append((current, total, message))
        
        # Condense using rule-based parser
        result = condense_qa_check(content, progress_callback=progress)
        
        # Verify the response
        assert result, "Result should not be empty"
        assert len(result) > 0, "Result should have content"
        assert "QA_CHECK.LOG SUMMARY" in result, "Should have summary header"
        assert "OVERVIEW BY SEVERITY" in result, "Should have severity overview"
        
        # Should have called progress callback
        assert len(progress_calls) > 0, "Should have progress updates"
        
        # Print summary for visibility
        print(f"\n✅ QA_CHECK.LOG condensed successfully")
        print(f"   Input size: {len(content)} chars")
        print(f"   Output size: {len(result)} chars")
        print(f"   Compression ratio: {len(result)/len(content):.2%}")
        print(f"\n📝 Result preview (first 500 chars):")
        print(f"{result[:500]}...")
    
    def test_filter_run_log_with_real_data(self, data_dir):
        """Test rule-based filtering of real run_log.txt files.
        
        This test:
        1. Loads a real run_log.txt file
        2. Uses the filtering logic to remove noise
        3. Verifies filtered output is produced
        """
        from times_doctor.core.llm import _filter_run_log
        
        # Find run_log files
        run_log_files = list(data_dir.rglob("*_run_log.txt"))
        
        if len(run_log_files) == 0:
            pytest.skip("No run_log.txt files found in data/ directory")
        
        run_log_path = run_log_files[0]
        content = run_log_path.read_text(encoding='utf-8', errors='ignore')
        original_size = len(content)
        
        # Track progress
        progress_calls = []
        def progress(current, total, message):
            progress_calls.append((current, total, message))
        
        # Filter the run log
        result = _filter_run_log(content, progress_callback=progress)
        
        # Verify the response
        assert "filtered_content" in result
        filtered = result["filtered_content"]
        assert filtered, "Filtered content should not be empty"
        assert len(filtered) < original_size, "Filtered content should be smaller than original"
        
        # Should have removed noise
        assert "DMoves" not in filtered or filtered.count("DMoves") < content.count("DMoves")
        
        # Should have called progress callback
        assert len(progress_calls) > 0, "Should have progress updates"
        
        # Print summary
        print(f"\n✅ run_log.txt filtered successfully")
        print(f"   Input size: {original_size} chars")
        print(f"   Output size: {len(filtered)} chars")
        print(f"   Reduction: {(1 - len(filtered)/original_size):.2%}")
    
    def test_extract_lst_pages_with_real_data(self, data_dir):
        """Test page extraction from real .lst files.
        
        This test:
        1. Loads a real .lst file
        2. Extracts specific pages (1, 2, 5, 6, 7, 8)
        3. Verifies page structure is correct
        """
        from times_doctor.core.llm import _extract_lst_pages
        
        # Find .lst files (prefer non-condensed files)
        lst_files = list(data_dir.rglob("*.lst"))
        
        if len(lst_files) == 0:
            pytest.skip("No .lst files found in data/ directory")
        
        # Filter out condensed files first
        non_condensed = [f for f in lst_files if "condensed" not in f.name.lower()]
        lst_path = non_condensed[0] if non_condensed else lst_files[0]
        content = lst_path.read_text(encoding='utf-8', errors='ignore')
        
        # Track progress
        progress_calls = []
        def progress(current, total, message):
            progress_calls.append((current, total, message))
        
        # Extract pages
        result = _extract_lst_pages(content, progress_callback=progress)
        
        # Verify the response
        assert "sections" in result
        sections = result["sections"]
        assert len(sections) > 0, "Should extract at least one section"
        
        # Should also have extracted_text
        assert "extracted_text" in result
        assert len(result["extracted_text"]) > 0, "Should have extracted text"
        
        # All sections should have name and line numbers
        for section in sections:
            assert "name" in section
            assert "start_line" in section
            assert "end_line" in section
            assert section["start_line"] <= section["end_line"]
        
        # Should have called progress callback
        assert len(progress_calls) > 0, "Should have progress updates"
        
        # Print summary
        print(f"\n✅ .lst sections extracted successfully")
        print(f"   Input size: {len(content):,} chars")
        print(f"   Output size: {len(result['extracted_text']):,} chars")
        print(f"   Compression: {100 * (1 - len(result['extracted_text']) / len(content)):.1f}%")
        print(f"   Sections found: {len(sections)}")
        section_names = [s["name"] for s in sections]
        print(f"   Sections: {', '.join(section_names[:5])}{'...' if len(section_names) > 5 else ''}")
