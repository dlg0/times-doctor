#!/usr/bin/env python3
import json
import os
import re
from typing import Any

SOURCE_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "prompts",
        "cplex_options",
        "cplex_options_gams49_detailed.md",
    )
)
OUTPUT_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "prompts",
        "cplex_options",
        "cplex_options_gams49_detailed.json",
    )
)


HEADER_RE = re.compile(r"^([.\w]+)\s+\((integer|real|string|boolean)\):\s*(.+)$", re.IGNORECASE)


def normalize_line(line: str) -> str:
    # Drop visible "↵" markers if present and trim whitespace
    return line.replace("↵", "").strip()


def is_header(line: str) -> re.Match | None:
    return HEADER_RE.match(line)


def starts_with_any(line: str, prefixes: list[str]) -> bool:
    lower = line.lower()
    return any(lower.startswith(p) for p in prefixes)


def looks_like_value_row(line: str) -> bool:
    """
    Heuristic to detect a new "value meaning" row.
    Accept common patterns:
      - integers, signed integers
      - decimals
      - tokens like N>n>0
      - tokens like -1, 0, 1, 2, 3, etc.
    """
    if not line:
        return False
    # A token followed by at least one space and some text
    m = re.match(r"^([^\s]+)\s+.+", line)
    if not m:
        return False
    tok = m.group(1)  # noqa: S105 - Not a password, this is a text token from markdown
    if tok == "value":  # header row itself
        return False
    # common numeric tokens
    if re.fullmatch(r"[+\-]?\d+(\.\d+)?", tok):
        return True
    if tok in {"-1", "0", "1", "2", "3", "4", "5"}:
        return True
    # pattern like N>n>0 or similar symbolic bounds
    if re.fullmatch(r"[A-Za-z0-9]+[><=].*", tok):
        return True
    return True  # fallback: treat as value row if it fits tok + text pattern


def parse_value_table(lines: list[str], start_index: int) -> tuple[list[dict[str, str]], int]:
    """
    Parse rows following a 'value meaning' header.
    Returns (values_list, next_index)
    """
    values: list[dict[str, str]] = []
    i = start_index
    # Consume rows until a break condition
    while i < len(lines):
        s2 = normalize_line(lines[i])
        if not s2:
            i += 1
            continue
        # Stop if next section starts
        if is_header(s2) or starts_with_any(
            s2, ["default:", "range:", "synonym:", "value", "note:"]
        ):
            break
        if not looks_like_value_row(s2):
            break
        mval = re.match(r"^([^\s]+)\s+(.*)$", s2)
        if not mval:
            break
        val, meaning = mval.groups()
        i += 1
        # Accumulate continuation lines into meaning
        while i < len(lines):
            cont_s = normalize_line(lines[i])
            if not cont_s:
                i += 1
                continue
            if is_header(cont_s) or starts_with_any(
                cont_s, ["default:", "range:", "synonym:", "value", "note:"]
            ):
                break
            if looks_like_value_row(cont_s):
                # If this looks like a new row, stop continuation
                break
            meaning = f"{meaning} {cont_s}".strip()
            i += 1
        values.append({"value": val, "meaning": meaning})
    return values, i


def parse_options(md_lines: list[str]) -> dict[str, dict[str, Any]]:
    options: dict[str, dict[str, Any]] = {}
    i = 0
    n = len(md_lines)
    while i < n:
        s = normalize_line(md_lines[i])
        if not s:
            i += 1
            continue
        header_match = is_header(s)
        if not header_match:
            i += 1
            continue
        name, opt_type, summary = header_match.groups()
        current: dict[str, Any] = {
            "type": opt_type.lower(),
            "summary": summary,
        }
        i += 1
        desc_lines: list[str] = []
        # Walk until next header or EOF
        while i < n:
            line = md_lines[i]
            s2 = normalize_line(line)
            if not s2:
                i += 1
                continue
            if is_header(s2):
                break
            lower = s2.lower()
            if lower.startswith("default:"):
                current["default"] = s2.split(":", 1)[1].strip()
                i += 1
                continue
            if lower.startswith("range:"):
                current["range"] = s2.split(":", 1)[1].strip()
                i += 1
                continue
            if lower.startswith("synonym:"):
                syn = s2.split(":", 1)[1].strip()
                syns = [x.strip() for x in syn.split(",") if x.strip()]
                if syns:
                    current["synonyms"] = syns
                i += 1
                continue
            if "value" in lower and "meaning" in lower and lower.startswith("value"):
                i += 1
                values, i = parse_value_table(md_lines, i)
                if values:
                    current["values"] = values
                continue
            # If not a recognized directive, treat as description text
            desc_lines.append(s2)
            i += 1
        if desc_lines:
            desc = " ".join(desc_lines).strip()
            if desc:
                current["description"] = desc
        options[name] = current
    return options


def main() -> None:
    if not os.path.isfile(SOURCE_PATH):
        raise FileNotFoundError(f"Source markdown not found: {SOURCE_PATH}")
    with open(SOURCE_PATH, encoding="utf-8") as f:
        md_lines = f.readlines()
    options = parse_options(md_lines)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(options, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(options)} options to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
