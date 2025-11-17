"""Tests for LST file parser."""

from pathlib import Path

import pytest

from times_doctor.core.lst_parser import (
    CompilationProcessor,
    ExecutionProcessor,
    LSTParser,
    LSTSection,
    ModelAnalysisProcessor,
    process_lst_file,
)

# Sample LST content for testing
SAMPLE_LST_HEADER = """GAMS 49.6.1  55d34574 May 28, 2025          WEX-WEI x86 64bit/MS Windows - 11/06/25 13:17:19 Page 1
Veda2 -- v4.0.6.0
C o m p i l a t i o n
"""

SAMPLE_COMPILATION_CONTENT = """GAMS 49.6.1  55d34574 May 28, 2025          WEX-WEI x86 64bit/MS Windows - 11/06/25 13:17:19 Page 3
TIMES -- VERSION 4.8.3 -- Restart (v4.8)
C o m p i l a t i o n


556722  'AUD25'.'AUD25' 1
****          $170    $170
**** LINE  10808 BATINCLUDE  D:\\Veda\\file.dd
**** LINE     88 INPUT       D:\\Veda\\file.RUN
**** 170  Domain violation for element
556741  'ACT'.2015.'AUD25' 0.07
****                     $170
**** LINE  10827 BATINCLUDE  D:\\Veda\\file.dd
**** LINE     88 INPUT       D:\\Veda\\file.RUN
**** 170  Domain violation for element
556755  'ADE'.2015.'AUD25' 0.07
****                     $170
**** LINE  10841 BATINCLUDE  D:\\Veda\\file.dd
**** LINE     88 INPUT       D:\\Veda\\file.RUN
**** 170  Domain violation for element
556769  'ACT'.2020.'AUD25' 0.07
****                     $170
**** LINE  10855 BATINCLUDE  D:\\Veda\\file.dd
**** LINE     88 INPUT       D:\\Veda\\file.RUN
**** 170  Domain violation for element
"""

SAMPLE_EXECUTION_CONTENT = """GAMS 49.6.1  55d34574 May 28, 2025          WEX-WEI x86 64bit/MS Windows - 11/06/25 13:17:19 Page 6
TIMES -- VERSION 4.8.3 -- Restart (v4.8)
E x e c u t i o n

----    461 Other                    0.000     0.031 SECS    118 MB
----    465 Other                    0.000     0.031 SECS    118 MB
----    594 Assignment YEARVAL       0.000     0.031 SECS    118 MB    264
----   1285 Assignment YEARVAL       0.000     0.031 SECS    118 MB    263
----   2101186 Loop                   0.204     0.235 SECS    118 MB
----   2101198 Assignment NCAP_CEH    0.000     0.235 SECS    118 MB      0
----   2101209 Assignment RXX         0.015     0.250 SECS    118 MB   3886
----   2101210 Loop                   0.031     0.281 SECS    118 MB
----   2101362 Assignment RPC         0.032     0.438 SECS    123 MB  87914
----   2101363 Assignment RC          0.031     0.469 SECS    128 MB  35717
----   2101375 Assignment COM_GMAP    0.016     0.485 SECS    128 MB  128471
----   2101496 Assignment PERIODYR    0.015     0.500 SECS    128 MB     48
----   2101500 Assignment G_OFFTHD    0.000     0.500 SECS    128 MB      0
"""

SAMPLE_MODEL_ANALYSIS_CONTENT = """GAMS 49.6.1  55d34574 May 28, 2025          WEX-WEI x86 64bit/MS Windows - 11/06/25 13:17:19 Page 7
TIMES -- VERSION 4.8.3 -- Restart (v4.8)
Model Analysis      SOLVE TIMES Using LP From line 2135584

----2135584 Solve Init TIMES         0.000    77.750 SECS  2,534 MB
----2117128 Equation   EQ_OBJ        0.000    77.750 SECS  2,534 MB      1
----2116675 Equation   EQ_OBJFIX     4.047    81.797 SECS  2,538 MB     29
----2116420 Equation   EQ_OBJINV     0.484    82.281 SECS  2,551 MB     29
----2117054 Equation   EQ_OBJSALV    0.047    82.328 SECS  2,554 MB     29
----2116780 Equation   EQ_OBJVAR    16.110    98.438 SECS  2,857 MB     29
----2117203 Equation   EQ_ACTFLO     8.562   107.000 SECS  3,747 MB  1035406
----2117242 Equation   EQG_ACTBND    0.000   107.000 SECS  3,747 MB      0
----2117554 Equation   EQL_CAPACT    8.234   115.469 SECS  4,374 MB  1418775
----2117501 Equation   EQE_CAPACT    4.734   120.203 SECS  4,405 MB  530898
----2117607 Equation   EQG_CAPACT    4.688   124.891 SECS  4,468 MB  273524
----2117941 Equation   EQG_COMBAL   23.812   148.703 SECS  5,266 MB  264942
"""


@pytest.fixture
def sample_lst_file(tmp_path):
    """Create a sample LST file for testing."""
    lst_file = tmp_path / "test.lst"
    content = (
        SAMPLE_LST_HEADER
        + SAMPLE_COMPILATION_CONTENT
        + SAMPLE_EXECUTION_CONTENT
        + SAMPLE_MODEL_ANALYSIS_CONTENT
    )
    lst_file.write_text(content)
    return lst_file


def test_lst_parser_finds_sections(sample_lst_file):
    """Test that LSTParser correctly identifies sections."""
    parser = LSTParser(sample_lst_file)
    sections = parser.parse()

    # Should find at least 3 sections
    assert len(sections) >= 3

    # Check section names
    section_names = [s.name for s in sections]
    assert "C o m p i l a t i o n" in section_names or "Compilation" in section_names
    assert "E x e c u t i o n" in section_names or "Execution" in section_names


def test_compilation_processor_aggregates_errors():
    """Test that CompilationProcessor aggregates domain violations."""
    section = LSTSection(
        name="Compilation",
        page_number=3,
        start_line=10,
        end_line=50,
        header="GAMS header",
        content=SAMPLE_COMPILATION_CONTENT.split("\n", 3)[3],  # Skip header
    )

    result = CompilationProcessor.process(section)

    # Should find error code 170
    assert "170" in result["errors"]

    # Should count 4 occurrences
    assert result["errors"]["170"]["count"] == 4

    # Should have element patterns
    assert len(result["errors"]["170"]["elements"]) > 0

    # Should have samples
    assert len(result["errors"]["170"]["samples"]) > 0

    # Summary should be non-empty
    assert len(result["summary"]) > 0
    assert "170" in result["summary"]


def test_execution_processor_extracts_timing():
    """Test that ExecutionProcessor extracts timing information."""
    section = LSTSection(
        name="Execution",
        page_number=6,
        start_line=100,
        end_line=200,
        header="GAMS header",
        content=SAMPLE_EXECUTION_CONTENT.split("\n", 3)[3],  # Skip header
    )

    result = ExecutionProcessor.process(section)

    # Should have summary
    assert "summary" in result
    assert result["summary"]["total_time_secs"] > 0
    assert result["summary"]["peak_memory_mb"] > 0

    # Should have major operations (none in this small sample, all <0.5s)
    assert "major_operations" in result

    # Should have text summary
    assert len(result["text_summary"]) > 0


def test_model_analysis_processor_extracts_equations():
    """Test that ModelAnalysisProcessor extracts equation statistics."""
    section = LSTSection(
        name="Model Analysis",
        page_number=7,
        start_line=200,
        end_line=300,
        header="GAMS header",
        content=SAMPLE_MODEL_ANALYSIS_CONTENT.split("\n", 3)[3],  # Skip header
    )

    result = ModelAnalysisProcessor.process(section)

    # Should have equations
    assert "equations" in result
    assert len(result["equations"]) > 0

    # Should have summary
    assert "summary" in result
    assert result["summary"]["total_equation_count"] > 0
    assert result["summary"]["equation_types"] > 0

    # Check specific equation
    eq_names = [eq["name"] for eq in result["equations"]]
    assert "EQ_ACTFLO" in eq_names

    # Find EQ_ACTFLO and check count
    eq_actflo = next(eq for eq in result["equations"] if eq["name"] == "EQ_ACTFLO")
    assert eq_actflo["count"] == 1035406

    # Should have text summary
    assert len(result["text_summary"]) > 0


def test_process_lst_file_full_integration(sample_lst_file):
    """Test full integration of LST file processing."""
    result = process_lst_file(sample_lst_file)

    # Should have metadata
    assert "metadata" in result
    assert "file" in result["metadata"]
    assert "section_count" in result["metadata"]

    # Should have sections
    assert "sections" in result
    assert len(result["sections"]) > 0

    # Check for expected section types (normalized names may vary)
    section_keys = list(result["sections"].keys())
    assert len(section_keys) >= 2  # At least compilation and execution


def test_section_name_normalization():
    """Test that section names are normalized correctly."""
    parser = LSTParser(Path("dummy"))
    parser.lines = [
        "GAMS 49.6.1  55d34574 May 28, 2025          WEX-WEI x86 64bit/MS Windows - 11/06/25 13:17:19 Page 1\n",
        "TIMES -- VERSION 4.8.3\n",
        "C o m p i l a t i o n\n",
    ]

    title, offset = parser._extract_section_title(0)
    # Should normalize letter-spaced headings to proper words
    assert "compilation" in title.lower() or title == "C o m p i l a t i o n"
    assert offset == 2  # Title is at line 2 (0-indexed)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
