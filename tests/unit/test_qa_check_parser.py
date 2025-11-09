"""Unit tests for QA_CHECK.LOG parser."""

from times_doctor.core.qa_check_parser import (
    condense_events,
    expand_composite_key,
    format_condensed_output,
    iter_events,
    normalize_severity,
    parse_kv_fields,
    severity_rank,
)


class TestSeverityUtils:
    """Test severity normalization and ranking."""

    def test_normalize_severe_warning(self):
        assert normalize_severity("SEVERE WARNING") == "WARNING"

    def test_normalize_standard_severities(self):
        assert normalize_severity("SEVERE ERROR") == "SEVERE ERROR"
        assert normalize_severity("ERROR") == "ERROR"
        assert normalize_severity("WARNING") == "WARNING"
        assert normalize_severity("NOTE") == "NOTE"
        assert normalize_severity("INFO") == "INFO"

    def test_severity_rank_order(self):
        """SEVERE ERROR should rank highest (0), INFO lowest (4)."""
        assert severity_rank("SEVERE ERROR") < severity_rank("ERROR")
        assert severity_rank("ERROR") < severity_rank("WARNING")
        assert severity_rank("WARNING") < severity_rank("NOTE")
        assert severity_rank("NOTE") < severity_rank("INFO")
        assert severity_rank("UNKNOWN") == 99


class TestCompositeKeys:
    """Test composite key expansion."""

    def test_expand_composite_key_matching_parts(self):
        result = expand_composite_key("R.T.P", "SWNSW.2030.ENPS168-SNOWY2")
        assert result == {"R": "SWNSW", "T": "2030", "P": "ENPS168-SNOWY2"}

    def test_expand_composite_key_mismatched_parts(self):
        """If parts don't match, return as-is."""
        result = expand_composite_key("R.T.P", "A.B")
        assert result == {"R.T.P": "A.B"}

    def test_expand_simple_key(self):
        result = expand_composite_key("R", "NSW")
        assert result == {"R": "NSW"}


class TestKVParsing:
    """Test KEY=VALUE parsing."""

    def test_parse_single_kv(self):
        result = parse_kv_fields("R=NSW")
        assert result == {"R": "NSW"}

    def test_parse_multiple_kvs(self):
        result = parse_kv_fields("R=NSW P=EE_Solar425 T=2030")
        assert result == {"R": "NSW", "P": "EE_Solar425", "T": "2030"}

    def test_parse_value_with_spaces(self):
        result = parse_kv_fields("P=Ipc_Steam Cracking-Propane R=NSW")
        assert result == {"P": "Ipc_Steam Cracking-Propane", "R": "NSW"}

    def test_parse_composite_key(self):
        result = parse_kv_fields("R.T.P=SWNSW.2030.ENPS168-SNOWY2")
        assert result == {"R": "SWNSW", "T": "2030", "P": "ENPS168-SNOWY2"}

    def test_ignore_sum(self):
        result = parse_kv_fields("R=NSW SUM=123.45 P=EE_Solar")
        assert result == {"R": "NSW", "P": "EE_Solar"}
        assert "SUM" not in result

    def test_filter_with_allow_list(self):
        result = parse_kv_fields("R=NSW P=EE_Solar V=ELEC T=2030", index_allow=["R", "P"])
        assert result == {"R": "NSW", "P": "EE_Solar"}
        assert "V" not in result
        assert "T" not in result


class TestEventIteration:
    """Test event parsing from QA_CHECK.LOG lines."""

    def test_parse_section_header(self):
        lines = [
            "*** Inconsistent CAP_BND(UP/LO/FX) defined for process capacity",
            "*01 WARNING - Lower bound set equal to upper bound,   R.T.P= SWNSW.2030.ENPS168-SNOWY2",
        ]

        events = list(iter_events(lines))
        assert len(events) == 1

        sev, msg, idx = events[0]
        assert sev == "WARNING"
        assert "Inconsistent CAP_BND" in msg
        assert "Lower bound set equal to upper bound" in msg
        assert idx == {"R": "SWNSW", "T": "2030", "P": "ENPS168-SNOWY2"}

    def test_parse_multiple_events_same_section(self):
        lines = [
            "*** Test Section",
            "*01 ERROR - First error R=NSW P=Test1",
            "*02 ERROR - Second error R=VIC P=Test2",
        ]

        events = list(iter_events(lines))
        assert len(events) == 2

        assert events[0][0] == "ERROR"
        assert "First error" in events[0][1]
        assert events[0][2] == {"R": "NSW", "P": "Test1"}

        assert events[1][0] == "ERROR"
        assert "Second error" in events[1][1]
        assert events[1][2] == {"R": "VIC", "P": "Test2"}

    def test_severity_filter(self):
        lines = [
            "*01 SEVERE ERROR - Critical R=NSW",
            "*02 WARNING - Minor R=VIC",
            "*03 INFO - Information R=QLD",
        ]

        # Only include WARNING and above
        events = list(iter_events(lines, min_severity="WARNING"))
        assert len(events) == 2
        assert events[0][0] == "SEVERE ERROR"
        assert events[1][0] == "WARNING"

    def test_skip_non_event_lines(self):
        lines = [
            "Some random text",
            "*01 ERROR - Real error R=NSW",
            "Another random line",
        ]

        events = list(iter_events(lines))
        assert len(events) == 1
        assert events[0][0] == "ERROR"

    def test_handle_leading_whitespace(self):
        """Leading whitespace should be stripped (real QA_CHECK.LOG files have it)."""
        lines = [
            "*** Test Section",
            " *01 WARNING - With leading space R=NSW",
            "  *02 ERROR - With more leading space R=VIC",
        ]

        events = list(iter_events(lines))
        assert len(events) == 2
        assert events[0][0] == "WARNING"
        assert events[1][0] == "ERROR"


class TestCondensation:
    """Test event deduplication and condensation."""

    def test_deduplicate_identical_events(self):
        events = [
            ("ERROR", "Test :: Error message", {"R": "NSW", "P": "Test"}),
            ("ERROR", "Test :: Error message", {"R": "NSW", "P": "Test"}),
            ("ERROR", "Test :: Error message", {"R": "NSW", "P": "Test"}),
        ]

        summary_rows, message_counts, all_keys = condense_events(events)

        assert len(summary_rows) == 1
        assert summary_rows[0]["severity"] == "ERROR"
        assert summary_rows[0]["occurrences"] == "3"
        assert summary_rows[0]["R"] == "NSW"
        assert summary_rows[0]["P"] == "Test"

    def test_different_indices_create_separate_rows(self):
        events = [
            ("ERROR", "Test :: Error message", {"R": "NSW", "P": "Test1"}),
            ("ERROR", "Test :: Error message", {"R": "NSW", "P": "Test2"}),
        ]

        summary_rows, message_counts, all_keys = condense_events(events)

        assert len(summary_rows) == 2
        assert summary_rows[0]["P"] in ["Test1", "Test2"]
        assert summary_rows[1]["P"] in ["Test1", "Test2"]
        assert summary_rows[0]["P"] != summary_rows[1]["P"]

    def test_message_counts(self):
        events = [
            ("ERROR", "Test :: Error A", {"R": "NSW"}),
            ("ERROR", "Test :: Error A", {"R": "VIC"}),
            ("WARNING", "Test :: Warning B", {"R": "QLD"}),
        ]

        summary_rows, message_counts, all_keys = condense_events(events)

        assert len(message_counts) == 2

        error_count = next(m for m in message_counts if m["severity"] == "ERROR")
        assert error_count["events"] == "2"

        warning_count = next(m for m in message_counts if m["severity"] == "WARNING")
        assert warning_count["events"] == "1"

    def test_deterministic_sort_by_severity(self):
        events = [
            ("INFO", "Test :: Info", {}),
            ("SEVERE ERROR", "Test :: Severe", {}),
            ("WARNING", "Test :: Warning", {}),
            ("ERROR", "Test :: Error", {}),
        ]

        summary_rows, _, _ = condense_events(events)

        severities = [r["severity"] for r in summary_rows]
        assert severities == ["SEVERE ERROR", "ERROR", "WARNING", "INFO"]


class TestFormatting:
    """Test output formatting."""

    def test_format_condensed_output(self):
        summary_rows = [
            {
                "severity": "ERROR",
                "message": "Test :: Error message",
                "occurrences": "3",
                "R": "NSW",
                "P": "Test",
            }
        ]
        message_counts = [{"severity": "ERROR", "message": "Test :: Error message", "events": "3"}]
        all_keys = ["R", "P"]

        output = format_condensed_output(summary_rows, message_counts, all_keys)

        assert "QA_CHECK.LOG SUMMARY" in output
        assert "ERROR" in output
        assert "Test :: Error message" in output
        assert "Total occurrences: 3" in output
        assert "See QA_CHECK.LOG for full detail" in output


class TestIntegration:
    """Integration tests with realistic QA_CHECK.LOG samples."""

    def test_flo_share_example(self):
        """Test the FLO_SHARE auto-relaxed example from the PRD."""
        lines = [
            "*** FLO_SHARE violations",
            "*01 WARNING - FLO_SHARE auto-relaxed R=NSW P=Coal V=ELEC CG=Power SUM=100.5 (Auto-relaxed)",
            "*02 WARNING - FLO_SHARE auto-relaxed R=NSW P=Coal V=ELEC CG=Power SUM=101.2 (Auto-relaxed)",
            "*03 WARNING - FLO_SHARE auto-relaxed R=VIC P=Gas V=ELEC CG=Power SUM=98.3 (Auto-relaxed)",
        ]

        events = list(iter_events(lines))
        summary_rows, message_counts, all_keys = condense_events(events)

        # SUM should be ignored
        assert "SUM" not in all_keys

        # Should have 2 unique combinations (NSW+Coal vs VIC+Gas)
        assert len(summary_rows) == 2

        # All should be WARNING
        assert all(r["severity"] == "WARNING" for r in summary_rows)

        # Message count should show 3 total events
        assert message_counts[0]["events"] == "3"
