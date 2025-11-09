"""Parser for GAMS LST (listing) files.

Extracts and processes sections from LST files, aggregating repetitive content
and extracting key information for embedding and analysis.
"""

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LSTSection:
    """A section from an LST file."""

    name: str  # Normalized section name (e.g., "Compilation")
    page_number: int  # Original page number
    start_line: int  # Line number where section starts
    end_line: int  # Line number where section ends
    header: str  # Full header text (GAMS version, etc.)
    content: str  # Raw section content


class LSTParser:
    """Parser for GAMS LST files."""

    # Pattern to match page headers
    GAMS_HEADER_RE = re.compile(r"^GAMS\s+[\d.]+\s+.*?Page\s+(\d+)", re.MULTILINE)

    TIMES_HEADER_RE = re.compile(r"^TIMES\s+--\s+VERSION\s+[\d.]+", re.MULTILINE)

    # Pattern to normalize section titles (remove extra spacing)
    SECTION_TITLE_RE = re.compile(r"\s{2,}")

    def __init__(self, lst_path: Path):
        """Initialize parser with LST file path."""
        self.lst_path = lst_path
        self.lines: list[str] = []
        self.sections: list[LSTSection] = []

    def parse(self) -> list[LSTSection]:
        """Parse the LST file and extract sections."""
        # Read file
        with open(self.lst_path, encoding="utf-8", errors="replace") as f:
            self.lines = f.readlines()

        # Find section boundaries
        section_starts = self._find_section_starts()

        # Extract sections
        self.sections = self._extract_sections(section_starts)

        return self.sections

    def _find_section_starts(self) -> list[tuple[int, int, str, str]]:
        """Find line numbers where sections start.

        Returns:
            List of (line_num, page_num, header, section_title) tuples
        """
        section_starts = []

        for i, line in enumerate(self.lines):
            # Check for GAMS header
            gams_match = self.GAMS_HEADER_RE.match(line)
            if gams_match:
                page_num = int(gams_match.group(1))

                # Look ahead for section title (usually 2-3 lines after)
                section_title = self._extract_section_title(i)

                # Collect full header (2-3 lines)
                header_lines = []
                for j in range(i, min(i + 4, len(self.lines))):
                    header_lines.append(self.lines[j].rstrip())
                header = "\n".join(header_lines)

                section_starts.append((i, page_num, header, section_title))

        return section_starts

    def _extract_section_title(self, header_line: int) -> str:
        """Extract and normalize section title after header.

        Args:
            header_line: Line number of the GAMS/TIMES header

        Returns:
            Normalized section title
        """
        # Look in next few lines for section title
        for offset in range(1, 5):
            if header_line + offset >= len(self.lines):
                break

            line = self.lines[header_line + offset].strip()

            # Skip TIMES version lines
            if line.startswith("TIMES --") or line.startswith("Veda"):
                continue

            # Skip empty lines
            if not line:
                continue

            # Found a potential section title
            # Normalize by removing extra spaces
            normalized = self.SECTION_TITLE_RE.sub(" ", line)
            return normalized

        # Fallback to "Unknown"
        return "Unknown"

    def _extract_sections(
        self, section_starts: list[tuple[int, int, str, str]]
    ) -> list[LSTSection]:
        """Extract section content between section boundaries.

        Args:
            section_starts: List of (line_num, page_num, header, title) tuples

        Returns:
            List of LSTSection objects
        """
        sections = []

        for i, (start_line, page_num, header, title) in enumerate(section_starts):
            # Determine end line (start of next section or EOF)
            end_line = section_starts[i + 1][0] if i + 1 < len(section_starts) else len(self.lines)

            # Extract content (skip header lines)
            content_start = start_line + 3  # Skip GAMS, TIMES, title lines
            content_lines = self.lines[content_start:end_line]
            content = "".join(content_lines)

            sections.append(
                LSTSection(
                    name=title,
                    page_number=page_num,
                    start_line=start_line,
                    end_line=end_line,
                    header=header,
                    content=content,
                )
            )

        return sections


class CompilationProcessor:
    """Process Compilation section - aggregate domain violations."""

    # Pattern for domain violation errors
    ERROR_PATTERN = re.compile(r"^\*\*\*\*\s+(\d+)\s+(.+?)$", re.MULTILINE)

    # Pattern to extract element from domain violation
    ELEMENT_PATTERN = re.compile(r"^(\d+)\s+'(.+?)'\s+", re.MULTILINE)

    @staticmethod
    def process(section: LSTSection) -> dict:
        """Process compilation section and aggregate errors.

        Returns:
            Dictionary with error summary and samples
        """
        content = section.content

        # Find all errors
        errors = defaultdict(lambda: {"count": 0, "elements": defaultdict(int), "samples": []})

        lines = content.split("\n")
        i = 0

        while i < len(lines):
            line = lines[i]

            # Check for error marker
            if line.startswith("****"):
                error_match = CompilationProcessor.ERROR_PATTERN.match(line)
                if error_match:
                    error_code = error_match.group(1)
                    error_msg = error_match.group(2).strip()

                    # Look ahead for element info (usually next non-**** line)
                    element = None
                    context_lines = [line]

                    for j in range(i + 1, min(i + 10, len(lines))):
                        next_line = lines[j]
                        context_lines.append(next_line)

                        # Try to extract element
                        elem_match = CompilationProcessor.ELEMENT_PATTERN.match(next_line)
                        if elem_match:
                            element = elem_match.group(2)
                            break

                        # Stop at next error or empty line
                        if next_line.startswith("****") or not next_line.strip():
                            break

                    # Aggregate
                    errors[error_code]["count"] += 1

                    if element:
                        # Generalize element pattern (replace numbers with wildcards)
                        pattern = re.sub(r"\d{4}", "YEAR", element)
                        errors[error_code]["elements"][pattern] += 1

                    # Keep first 10 samples
                    if len(errors[error_code]["samples"]) < 10:
                        errors[error_code]["samples"].append(
                            {
                                "line_num": section.start_line + i,
                                "element": element,
                                "message": error_msg,
                                "context": "\n".join(context_lines[:5]),
                            }
                        )

            i += 1

        return {
            "section": section.name,
            "page": section.page_number,
            "errors": dict(errors),
            "summary": CompilationProcessor._create_summary(errors),
        }

    @staticmethod
    def _create_summary(errors: dict) -> str:
        """Create human-readable summary of errors."""
        if not errors:
            return "No errors found"

        lines = []
        for error_code, info in errors.items():
            total = info["count"]
            lines.append(f"Error {error_code}: {total} occurrences")

            # Top 5 element patterns
            top_patterns = sorted(info["elements"].items(), key=lambda x: x[1], reverse=True)[:5]

            for pattern, count in top_patterns:
                lines.append(f"  - {pattern}: {count}")

        return "\n".join(lines)


class ExecutionProcessor:
    """Process Execution section - extract timing summary."""

    # Pattern for execution lines
    EXEC_LINE_PATTERN = re.compile(
        r"^----\s*(\d+)\s+(\w+)\s+(\w+)?\s+([\d.]+)\s+([\d.]+)\s+SECS\s+([\d,]+)\s+MB(?:\s+([\d,]+))?"
    )

    @staticmethod
    def process(section: LSTSection) -> dict:
        """Process execution section and extract timing info.

        Returns:
            Dictionary with execution summary
        """
        content = section.content

        # Track major operations (>0.5 seconds)
        major_ops = []
        total_time = 0.0
        peak_memory = 0

        for line in content.split("\n"):
            match = ExecutionProcessor.EXEC_LINE_PATTERN.match(line)
            if match:
                line_num = int(match.group(1))
                op_type = match.group(2)
                op_name = match.group(3) or ""
                exec_time = float(match.group(4))
                cumul_time = float(match.group(5))
                memory = int(match.group(6).replace(",", ""))
                count = match.group(7)

                total_time = max(total_time, cumul_time)
                peak_memory = max(peak_memory, memory)

                # Keep major operations (>0.5 sec execution time)
                if exec_time > 0.5:
                    major_ops.append(
                        {
                            "line": line_num,
                            "type": op_type,
                            "name": op_name,
                            "time": exec_time,
                            "cumulative_time": cumul_time,
                            "memory_mb": memory,
                            "count": int(count) if count else None,
                        }
                    )

        # Sort by execution time (descending)
        major_ops.sort(key=lambda x: x["time"], reverse=True)

        return {
            "section": section.name,
            "page": section.page_number,
            "summary": {
                "total_time_secs": total_time,
                "peak_memory_mb": peak_memory,
                "major_operations_count": len(major_ops),
            },
            "major_operations": major_ops[:50],  # Top 50
            "text_summary": ExecutionProcessor._create_summary(total_time, peak_memory, major_ops),
        }

    @staticmethod
    def _create_summary(total_time: float, peak_memory: int, major_ops: list[dict]) -> str:
        """Create human-readable execution summary."""
        lines = [
            f"Total execution time: {total_time:.2f} seconds",
            f"Peak memory usage: {peak_memory} MB",
            f"Major operations (>0.5s): {len(major_ops)}",
        ]

        if major_ops:
            lines.append("\nTop 10 time-consuming operations:")
            for i, op in enumerate(major_ops[:10], 1):
                lines.append(
                    f"  {i}. Line {op['line']}: {op['type']} {op['name']} ({op['time']:.3f}s)"
                )

        return "\n".join(lines)


class ModelAnalysisProcessor:
    """Process Model Analysis section - extract equation statistics."""

    # Pattern for equation generation lines
    EQUATION_PATTERN = re.compile(
        r"^----\d+\s+Equation\s+(\S+)\s+([\d.]+)\s+([\d.]+)\s+SECS\s+([\d,]+)\s+MB\s+([\d,]+)"
    )

    @staticmethod
    def process(section: LSTSection) -> dict:
        """Process model analysis section.

        Returns:
            Dictionary with equation statistics
        """
        content = section.content

        equations = []
        total_equations = 0
        total_time = 0.0

        for line in content.split("\n"):
            match = ModelAnalysisProcessor.EQUATION_PATTERN.match(line)
            if match:
                eq_name = match.group(1)
                exec_time = float(match.group(2))
                float(match.group(3))
                memory = int(match.group(4).replace(",", ""))
                count = int(match.group(5).replace(",", ""))

                equations.append(
                    {"name": eq_name, "count": count, "time": exec_time, "memory_mb": memory}
                )

                total_equations += count
                total_time += exec_time

        # Sort by count (descending)
        equations.sort(key=lambda x: x["count"], reverse=True)

        return {
            "section": section.name,
            "page": section.page_number,
            "summary": {
                "total_equation_count": total_equations,
                "equation_types": len(equations),
                "total_generation_time": total_time,
            },
            "equations": equations,
            "text_summary": ModelAnalysisProcessor._create_summary(total_equations, equations),
        }

    @staticmethod
    def _create_summary(total_equations: int, equations: list[dict]) -> str:
        """Create human-readable model analysis summary."""
        lines = [
            f"Total equations: {total_equations:,}",
            f"Equation types: {len(equations)}",
            "\nTop 10 equation types by count:",
        ]

        for i, eq in enumerate(equations[:10], 1):
            lines.append(f"  {i}. {eq['name']}: {eq['count']:,} equations ({eq['time']:.3f}s)")

        return "\n".join(lines)


def process_lst_file(lst_path: Path) -> dict:
    """Process an LST file and return structured data.

    Args:
        lst_path: Path to LST file

    Returns:
        Dictionary with processed sections
    """
    parser = LSTParser(lst_path)
    sections = parser.parse()

    # Group sections by name (handle duplicates)
    section_groups = defaultdict(list)
    for section in sections:
        section_groups[section.name].append(section)

    # Process sections
    processed = {
        "metadata": {
            "file": str(lst_path),
            "section_count": len(sections),
        },
        "sections": {},
    }

    # Extract GAMS/TIMES version from first section
    if sections:
        header = sections[0].header
        gams_match = re.search(r"GAMS\s+([\d.]+)", header)
        times_match = re.search(r"TIMES.*?VERSION\s+([\d.]+)", header)

        if gams_match:
            processed["metadata"]["gams_version"] = gams_match.group(1)
        if times_match:
            processed["metadata"]["times_version"] = times_match.group(1)

    # Process each section type
    for section_name, section_list in section_groups.items():
        for i, section in enumerate(section_list):
            # Generate unique key
            key = f"{section_name}_{i + 1}" if len(section_list) > 1 else section_name

            # Process based on section type
            section_name_lower = section_name.lower()
            if "compilation" in section_name_lower or "o m p i l a t i o n" in section_name_lower:
                processed["sections"][key] = CompilationProcessor.process(section)
            elif "execution" in section_name_lower or "x e c u t i o n" in section_name_lower:
                processed["sections"][key] = ExecutionProcessor.process(section)
            elif "model analysis" in section_name_lower:
                processed["sections"][key] = ModelAnalysisProcessor.process(section)
            else:
                # For other sections, keep as-is (they're typically small)
                processed["sections"][key] = {
                    "section": section.name,
                    "page": section.page_number,
                    "content": section.content[:5000],  # Limit to 5K chars
                    "truncated": len(section.content) > 5000,
                }

    return processed
