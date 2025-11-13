"""CPLEX options validator with metadata-based validation."""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ValidationError:
    """A validation error for an option."""

    key: str
    code: str
    message: str
    suggested_fix: str | None = None


@dataclass
class ValidationWarning:
    """A validation warning for an option."""

    key: str
    code: str
    message: str


@dataclass
class ValidationResult:
    """Result of validating CPLEX options."""

    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationWarning] = field(default_factory=list)
    normalized_options: dict[str, str | int | float] = field(default_factory=dict)
    unknown_keys: list[str] = field(default_factory=list)
    deprecated_keys: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Check if validation passed (no errors)."""
        return len(self.errors) == 0


class CplexOptionsValidator:
    """Validates CPLEX options against metadata and provides normalization."""

    def __init__(self, metadata_path: Path | str | None = None):
        """Initialize validator with CPLEX options metadata.

        Args:
            metadata_path: Path to cplex_options_gams49_detailed.json.
                          If None, uses default location in prompts/cplex_options/
        """
        if metadata_path is None:
            metadata_path = (
                Path(__file__).parent.parent.parent.parent
                / "prompts"
                / "cplex_options"
                / "cplex_options_gams49_detailed.json"
            )
        else:
            metadata_path = Path(metadata_path)

        if not metadata_path.exists():
            raise FileNotFoundError(f"CPLEX metadata not found: {metadata_path}")

        with open(metadata_path, encoding="utf-8") as f:
            self.metadata: dict[str, dict[str, Any]] = json.load(f)

        self.canonical_map: dict[str, str] = {}
        self.synonym_map: dict[str, str] = {}

        for canonical_name, meta in self.metadata.items():
            canonical_lower = canonical_name.lower()
            self.canonical_map[canonical_lower] = canonical_name

            if "synonyms" in meta:
                for syn in meta["synonyms"]:
                    self.synonym_map[syn.lower()] = canonical_name

    def _resolve_name(self, name: str) -> tuple[str | None, str | None]:
        """Resolve option name to canonical form.

        Args:
            name: Option name to resolve (case-insensitive)

        Returns:
            (canonical_name, resolution_type) where resolution_type is:
            - "exact": exact match
            - "synonym": resolved via synonym
            - None: unknown option
        """
        name_lower = name.lower()

        if name_lower in self.canonical_map:
            return self.canonical_map[name_lower], "exact"

        if name_lower in self.synonym_map:
            return self.synonym_map[name_lower], "synonym"

        return None, None

    def _suggest_names(self, name: str, max_suggestions: int = 3) -> list[str]:
        """Suggest similar option names using Levenshtein distance.

        Args:
            name: Unknown option name
            max_suggestions: Maximum number of suggestions to return

        Returns:
            List of suggested canonical names
        """
        from difflib import get_close_matches

        all_names = list(self.canonical_map.values()) + list(self.synonym_map.keys())
        matches = get_close_matches(
            name.lower(), [n.lower() for n in all_names], n=max_suggestions, cutoff=0.6
        )

        suggestions = []
        for match in matches:
            canonical, _ = self._resolve_name(match)
            if canonical and canonical not in suggestions:
                suggestions.append(canonical)

        return suggestions[:max_suggestions]

    def _normalize_boolean(self, value: Any) -> int | None:
        """Normalize boolean-like values to 0 or 1.

        Args:
            value: Value to normalize (accepts: true/false, yes/no, on/off, 0/1)

        Returns:
            0 or 1, or None if value cannot be normalized
        """
        if isinstance(value, bool):
            return 1 if value else 0

        if isinstance(value, int | float):
            if value in (0, 1):
                return int(value)
            return None

        if isinstance(value, str):
            lower = value.lower().strip()
            if lower in ("true", "yes", "on", "1"):
                return 1
            if lower in ("false", "no", "off", "0"):
                return 0

        return None

    def _parse_range(self, range_str: str) -> tuple[float | None, float | None]:
        """Parse range string to (min, max) bounds.

        Supports formats:
        - "a..b" -> (a, b)
        - ">=a" -> (a, None)
        - "<=b" -> (None, b)
        - ">a" -> (a, None) (exclusive)
        - "<b" -> (None, b) (exclusive)

        Args:
            range_str: Range specification string

        Returns:
            (min_value, max_value) where None means unbounded
        """
        range_str = range_str.strip()

        # Range like "a..b"
        if ".." in range_str:
            parts = range_str.split("..")
            if len(parts) == 2:
                try:
                    return (float(parts[0].strip()), float(parts[1].strip()))
                except ValueError:
                    return (None, None)

        # Patterns like >=a, <=b, >a, <b
        patterns = [
            (r"^>=\s*([+-]?\d+\.?\d*)", lambda x: (float(x), None)),
            (r"^>\s*([+-]?\d+\.?\d*)", lambda x: (float(x), None)),
            (r"^<=\s*([+-]?\d+\.?\d*)", lambda x: (None, float(x))),
            (r"^<\s*([+-]?\d+\.?\d*)", lambda x: (None, float(x))),
        ]

        for pattern, parser in patterns:
            match = re.match(pattern, range_str)
            if match:
                try:
                    return parser(match.group(1))
                except ValueError:
                    pass

        return (None, None)

    def _check_range(self, value: float, range_str: str) -> bool:
        """Check if value is within specified range.

        Args:
            value: Numeric value to check
            range_str: Range specification from metadata

        Returns:
            True if value is in range or range is unparseable
        """
        min_val, max_val = self._parse_range(range_str)

        if min_val is None and max_val is None:
            return True

        if min_val is not None and value < min_val:
            return False

        return not (max_val is not None and value > max_val)

    def _validate_value(
        self, canonical_name: str, value: Any, meta: dict[str, Any]
    ) -> tuple[str | int | float, list[ValidationError]]:
        """Validate and normalize a single option value.

        Args:
            canonical_name: Canonical option name
            value: Value to validate
            meta: Metadata for this option

        Returns:
            (normalized_value, errors)
        """
        errors: list[ValidationError] = []
        opt_type = meta.get("type", "string")

        # Boolean normalization
        if opt_type == "boolean":
            normalized = self._normalize_boolean(value)
            if normalized is None:
                errors.append(
                    ValidationError(
                        key=canonical_name,
                        code="type-mismatch",
                        message=f"Expected boolean; got '{value}'. Use 1/0, true/false, yes/no, or on/off.",
                        suggested_fix="1 or 0",
                    )
                )
                return str(value), errors
            return normalized, errors

        # Integer validation
        if opt_type == "integer":
            try:
                if isinstance(value, str | int | float):
                    int_val = int(value)
                else:
                    raise ValueError
            except (ValueError, TypeError):
                errors.append(
                    ValidationError(
                        key=canonical_name,
                        code="type-mismatch",
                        message=f"Expected integer; got '{value}'",
                        suggested_fix=None,
                    )
                )
                return str(value), errors

            # Check enumerated values
            if "values" in meta:
                allowed = [v["value"] for v in meta["values"]]
                if str(int_val) not in allowed and int_val not in [
                    int(v) for v in allowed if v.lstrip("-").isdigit()
                ]:
                    errors.append(
                        ValidationError(
                            key=canonical_name,
                            code="value-out-of-range",
                            message=f"Value {int_val} not allowed; expected one of: {', '.join(allowed)}",
                            suggested_fix=allowed[0] if allowed else None,
                        )
                    )
                    return int_val, errors

            # Check range
            if "range" in meta and not self._check_range(float(int_val), meta["range"]):
                errors.append(
                    ValidationError(
                        key=canonical_name,
                        code="value-out-of-range",
                        message=f"Value {int_val} outside allowed range: {meta['range']}",
                        suggested_fix=None,
                    )
                )

            return int_val, errors

        # Real/float validation
        if opt_type == "real":
            try:
                if isinstance(value, str | int | float):
                    float_val = float(value)
                else:
                    raise ValueError
            except (ValueError, TypeError):
                errors.append(
                    ValidationError(
                        key=canonical_name,
                        code="type-mismatch",
                        message=f"Expected real number; got '{value}'",
                        suggested_fix=None,
                    )
                )
                return str(value), errors

            # Check range
            if "range" in meta and not self._check_range(float_val, meta["range"]):
                errors.append(
                    ValidationError(
                        key=canonical_name,
                        code="value-out-of-range",
                        message=f"Value {float_val} outside allowed range: {meta['range']}",
                        suggested_fix=None,
                    )
                )

            return float_val, errors

        # String type - return as-is
        return str(value), errors

    def validate(self, options: dict[str, Any]) -> ValidationResult:
        """Validate CPLEX options dictionary.

        Args:
            options: Dictionary of option_name -> value

        Returns:
            ValidationResult with errors, warnings, and normalized options
        """
        result = ValidationResult()

        for name, value in options.items():
            canonical, resolution = self._resolve_name(name)

            if canonical is None:
                result.unknown_keys.append(name)
                suggestions = self._suggest_names(name)
                result.errors.append(
                    ValidationError(
                        key=name,
                        code="unknown-option",
                        message=f"Unknown option '{name}'",
                        suggested_fix=suggestions[0] if suggestions else None,
                    )
                )
                continue

            if resolution == "synonym":
                result.warnings.append(
                    ValidationWarning(
                        key=name,
                        code="synonym-resolved",
                        message=f"Resolved synonym '{name}' to canonical name '{canonical}'",
                    )
                )

            meta = self.metadata[canonical]
            normalized_value, errors = self._validate_value(canonical, value, meta)

            result.errors.extend(errors)
            if not errors:
                result.normalized_options[canonical] = normalized_value

        return result

    def validate_patch(
        self, base_options: dict[str, Any], patch: dict[str, Any]
    ) -> ValidationResult:
        """Validate a patch applied to base options.

        Args:
            base_options: Current/base options
            patch: Options to add/change (None value removes option)

        Returns:
            ValidationResult for the merged options
        """
        merged = base_options.copy()

        for key, value in patch.items():
            if value is None:
                merged.pop(key, None)
            else:
                merged[key] = value

        return self.validate(merged)
