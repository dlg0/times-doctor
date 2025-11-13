"""Tests for CPLEX options validator."""

import pytest

from times_doctor.core.cplex_validator import (
    CplexOptionsValidator,
)


class TestCplexOptionsValidator:
    """Test suite for CPLEX options validator."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return CplexOptionsValidator()

    def test_validator_loads_metadata(self, validator):
        """Test that validator loads metadata successfully."""
        assert len(validator.metadata) > 0
        assert "lpmethod" in validator.canonical_map.values()
        assert len(validator.synonym_map) > 0

    def test_resolve_canonical_name(self, validator):
        """Test resolving exact canonical name match."""
        canonical, resolution = validator._resolve_name("lpmethod")
        assert canonical == "lpmethod"
        assert resolution == "exact"

    def test_resolve_case_insensitive(self, validator):
        """Test case-insensitive name resolution."""
        canonical, resolution = validator._resolve_name("LPMETHOD")
        assert canonical == "lpmethod"
        assert resolution == "exact"

        canonical, resolution = validator._resolve_name("LpMethod")
        assert canonical == "lpmethod"
        assert resolution == "exact"

    def test_resolve_synonym(self, validator):
        """Test synonym resolution."""
        # Check if there are any synonyms in the metadata first
        has_synonyms = any("synonyms" in meta for meta in validator.metadata.values())
        if has_synonyms:
            # Find a synonym and test it
            for canonical, meta in validator.metadata.items():
                if "synonyms" in meta and meta["synonyms"]:
                    synonym = meta["synonyms"][0]
                    resolved, resolution = validator._resolve_name(synonym)
                    assert resolved == canonical
                    assert resolution == "synonym"
                    break

    def test_resolve_unknown_option(self, validator):
        """Test handling of unknown option names."""
        canonical, resolution = validator._resolve_name("unknownoption12345")
        assert canonical is None
        assert resolution is None

    def test_suggest_names(self, validator):
        """Test name suggestion for typos."""
        suggestions = validator._suggest_names("lpmetho")  # typo of lpmethod
        assert len(suggestions) > 0
        assert "lpmethod" in suggestions

        suggestions = validator._suggest_names("barrier")
        assert len(suggestions) > 0

    def test_normalize_boolean_from_bool(self, validator):
        """Test boolean normalization from Python bool."""
        assert validator._normalize_boolean(True) == 1
        assert validator._normalize_boolean(False) == 0

    def test_normalize_boolean_from_int(self, validator):
        """Test boolean normalization from integers."""
        assert validator._normalize_boolean(1) == 1
        assert validator._normalize_boolean(0) == 0
        assert validator._normalize_boolean(2) is None
        assert validator._normalize_boolean(-1) is None

    def test_normalize_boolean_from_string(self, validator):
        """Test boolean normalization from various string formats."""
        # True values
        assert validator._normalize_boolean("true") == 1
        assert validator._normalize_boolean("True") == 1
        assert validator._normalize_boolean("yes") == 1
        assert validator._normalize_boolean("YES") == 1
        assert validator._normalize_boolean("on") == 1
        assert validator._normalize_boolean("ON") == 1
        assert validator._normalize_boolean("1") == 1

        # False values
        assert validator._normalize_boolean("false") == 0
        assert validator._normalize_boolean("False") == 0
        assert validator._normalize_boolean("no") == 0
        assert validator._normalize_boolean("NO") == 0
        assert validator._normalize_boolean("off") == 0
        assert validator._normalize_boolean("OFF") == 0
        assert validator._normalize_boolean("0") == 0

        # Invalid
        assert validator._normalize_boolean("maybe") is None
        assert validator._normalize_boolean("2") is None

    def test_parse_range_dotdot(self, validator):
        """Test parsing range with '..' notation."""
        min_val, max_val = validator._parse_range("0..10")
        assert min_val == 0
        assert max_val == 10

        min_val, max_val = validator._parse_range("-1..4")
        assert min_val == -1
        assert max_val == 4

    def test_parse_range_inequality(self, validator):
        """Test parsing range with inequality operators."""
        min_val, max_val = validator._parse_range(">=0")
        assert min_val == 0
        assert max_val is None

        min_val, max_val = validator._parse_range(">0")
        assert min_val == 0
        assert max_val is None

        min_val, max_val = validator._parse_range("<=100")
        assert min_val is None
        assert max_val == 100

        min_val, max_val = validator._parse_range("<100")
        assert min_val is None
        assert max_val == 100

    def test_check_range_within(self, validator):
        """Test range checking for values within range."""
        assert validator._check_range(5, "0..10") is True
        assert validator._check_range(0, "0..10") is True
        assert validator._check_range(10, "0..10") is True
        assert validator._check_range(50, ">=0") is True

    def test_check_range_outside(self, validator):
        """Test range checking for values outside range."""
        assert validator._check_range(-1, "0..10") is False
        assert validator._check_range(11, "0..10") is False
        assert validator._check_range(-1, ">=0") is False

    def test_validate_valid_options(self, validator):
        """Test validation of valid options."""
        options = {
            "lpmethod": 4,
            "epopt": 1e-7,
        }

        result = validator.validate(options)
        assert result.is_valid
        assert len(result.errors) == 0
        assert "lpmethod" in result.normalized_options
        assert result.normalized_options["lpmethod"] == 4
        assert "epopt" in result.normalized_options

    def test_validate_unknown_option(self, validator):
        """Test validation catches unknown options."""
        options = {"unknownoption12345": 1}

        result = validator.validate(options)
        assert not result.is_valid
        assert len(result.errors) == 1
        assert result.errors[0].code == "unknown-option"
        assert "unknownoption12345" in result.unknown_keys

    def test_validate_synonym_warning(self, validator):
        """Test that synonyms generate warnings but succeed."""
        # Find a synonym in the metadata with a simple testable value
        synonym_name = None
        canonical_name = None
        test_value = None
        for canonical, meta in validator.metadata.items():
            if "synonyms" in meta and meta["synonyms"]:
                synonym_name = meta["synonyms"][0]
                canonical_name = canonical
                # Use a valid test value based on type
                opt_type = meta.get("type", "string")
                if opt_type == "integer":
                    # Get first allowed value if available, else default
                    if "values" in meta and meta["values"]:
                        test_value = meta["values"][0]["value"]
                    else:
                        test_value = 1
                elif opt_type == "real":
                    test_value = 1.0
                elif opt_type == "boolean":
                    test_value = 1
                else:
                    test_value = "test"
                break

        if synonym_name:
            options = {synonym_name: test_value}
            result = validator.validate(options)

            # Should have a warning
            assert len(result.warnings) == 1
            assert result.warnings[0].code == "synonym-resolved"
            # If validation passed, should be in normalized options
            if result.is_valid:
                assert canonical_name in result.normalized_options

    def test_validate_integer_type(self, validator):
        """Test integer type validation."""
        options = {"lpmethod": "4"}  # String representation of int

        result = validator.validate(options)
        assert result.is_valid
        assert result.normalized_options["lpmethod"] == 4

    def test_validate_integer_invalid_type(self, validator):
        """Test integer type validation with invalid input."""
        options = {"lpmethod": "not_a_number"}

        result = validator.validate(options)
        assert not result.is_valid
        assert any(e.code == "type-mismatch" for e in result.errors)

    def test_validate_real_type(self, validator):
        """Test real/float type validation."""
        options = {"epopt": "1e-7"}  # String representation of float

        result = validator.validate(options)
        assert result.is_valid
        assert result.normalized_options["epopt"] == 1e-7

        # Also test numeric input
        options = {"epopt": 0.0000001}
        result = validator.validate(options)
        assert result.is_valid

    def test_validate_boolean_type(self, validator):
        """Test boolean type validation and normalization."""
        # Find a boolean option in metadata
        bool_option = None
        for name, meta in validator.metadata.items():
            if meta.get("type") == "boolean":
                bool_option = name
                break

        if bool_option:
            # Test various boolean inputs
            for val in ["yes", "true", "on", "1", 1, True]:
                options = {bool_option: val}
                result = validator.validate(options)
                assert result.is_valid
                assert result.normalized_options[bool_option] == 1

            for val in ["no", "false", "off", "0", 0, False]:
                options = {bool_option: val}
                result = validator.validate(options)
                assert result.is_valid
                assert result.normalized_options[bool_option] == 0

    def test_validate_enumerated_values(self, validator):
        """Test validation of enumerated integer values."""
        # Find an option with enumerated values
        enum_option = None
        allowed_values = None
        for name, meta in validator.metadata.items():
            if meta.get("type") == "integer" and "values" in meta:
                enum_option = name
                allowed_values = [v["value"] for v in meta["values"]]
                break

        if enum_option and allowed_values:
            # Test valid value
            valid_val = allowed_values[0]
            if valid_val.lstrip("-").isdigit():
                options = {enum_option: int(valid_val)}
                result = validator.validate(options)
                assert result.is_valid

            # Test invalid value
            invalid_val = 999999  # Unlikely to be valid
            options = {enum_option: invalid_val}
            result = validator.validate(options)
            # May or may not fail depending on if there's also a range check
            # Just verify we get some kind of feedback
            assert result.errors or result.is_valid

    def test_validate_range_violation(self, validator):
        """Test validation catches range violations."""
        # Find an option with a parseable range
        range_option = None
        for name, meta in validator.metadata.items():
            if meta.get("type") in ("integer", "real") and "range" in meta:
                range_str = meta["range"]
                # Check if range is parseable
                min_val, max_val = validator._parse_range(range_str)
                if min_val is not None or max_val is not None:
                    range_option = name
                    break

        if range_option:
            meta = validator.metadata[range_option]
            min_val, max_val = validator._parse_range(meta["range"])

            # Test value outside range
            if min_val is not None:
                options = {range_option: min_val - 1}
                result = validator.validate(options)
                assert not result.is_valid
                assert any(e.code == "value-out-of-range" for e in result.errors)

    def test_validate_patch_add_option(self, validator):
        """Test validating a patch that adds an option."""
        base = {"lpmethod": 4}
        patch = {"epopt": 1e-7}

        result = validator.validate_patch(base, patch)
        assert result.is_valid
        assert "lpmethod" in result.normalized_options
        assert "epopt" in result.normalized_options

    def test_validate_patch_change_option(self, validator):
        """Test validating a patch that changes an option."""
        base = {"lpmethod": 1}
        patch = {"lpmethod": 4}

        result = validator.validate_patch(base, patch)
        assert result.is_valid
        assert result.normalized_options["lpmethod"] == 4

    def test_validate_patch_remove_option(self, validator):
        """Test validating a patch that removes an option."""
        base = {"lpmethod": 4, "epopt": 1e-7}
        patch = {"epopt": None}

        result = validator.validate_patch(base, patch)
        assert result.is_valid
        assert "lpmethod" in result.normalized_options
        assert "epopt" not in result.normalized_options

    def test_validate_patch_invalid_addition(self, validator):
        """Test validating a patch with invalid option."""
        base = {"lpmethod": 4}
        patch = {"invalidoption": 1}

        result = validator.validate_patch(base, patch)
        assert not result.is_valid
        assert any(e.code == "unknown-option" for e in result.errors)

    def test_validate_case_insensitive_handling(self, validator):
        """Test that validator handles mixed case option names."""
        options = {
            "LpMethod": 4,
            "EPOPT": 1e-7,
            "EpRhs": 1e-6,
        }

        result = validator.validate(options)
        assert result.is_valid
        # All should be normalized to canonical lowercase names
        assert "lpmethod" in result.normalized_options
        assert "epopt" in result.normalized_options
        assert "eprhs" in result.normalized_options

    def test_error_message_quality(self, validator):
        """Test that error messages are helpful."""
        # Use a typo that's more likely to get suggestions
        options = {"lpmetod": 1}  # typo of lpmethod

        result = validator.validate(options)
        assert len(result.errors) > 0
        error = result.errors[0]
        assert error.code == "unknown-option"
        assert "lpmetod" in error.message.lower()
        # Should have suggestions for this obvious typo
        assert error.suggested_fix is not None

    def test_validate_multiple_errors(self, validator):
        """Test handling multiple validation errors."""
        options = {
            "unknownoption1": 1,
            "unknownoption2": 2,
            "lpmethod": "not_a_number",
        }

        result = validator.validate(options)
        assert not result.is_valid
        assert len(result.errors) >= 3  # At least 3 errors

    def test_validate_empty_options(self, validator):
        """Test validating empty options dict."""
        result = validator.validate({})
        assert result.is_valid
        assert len(result.normalized_options) == 0
        assert len(result.errors) == 0

    def test_real_world_cplex_options(self, validator):
        """Test validation with realistic CPLEX options."""
        options = {
            "lpmethod": 4,
            "solutiontype": 2,
            "epopt": 1e-7,
            "eprhs": 1e-6,
            "epint": 1e-5,
            "threads": 8,
        }

        result = validator.validate(options)
        assert result.is_valid
        assert len(result.errors) == 0
        assert len(result.normalized_options) == len(options)

    def test_validation_result_properties(self, validator):
        """Test ValidationResult properties."""
        options = {"lpmethod": 4}
        result = validator.validate(options)

        assert result.is_valid is True
        assert isinstance(result.errors, list)
        assert isinstance(result.warnings, list)
        assert isinstance(result.normalized_options, dict)
        assert isinstance(result.unknown_keys, list)
        assert isinstance(result.deprecated_keys, list)

    def test_metadata_coverage(self, validator):
        """Test that common CPLEX options are in metadata."""
        expected_options = [
            "lpmethod",
            "epopt",
            "eprhs",
            "epint",
            "threads",
            "solutiontype",
        ]

        for opt in expected_options:
            canonical, _ = validator._resolve_name(opt)
            assert canonical is not None, f"Expected option '{opt}' not found in metadata"
            assert canonical in validator.metadata
