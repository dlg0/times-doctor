"""QA_CHECK.LOG parser for TIMES/VEDA logs.

Provides rule-based parsing and condensing of QA_CHECK.LOG files into structured records.
Standard-library only, no LLM required.
"""

from __future__ import annotations
import re
from pathlib import Path
from collections import Counter
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

# ----------------------------
# Regexes and constants
# ----------------------------

SEVERITY_ORDER: List[str] = ["SEVERE ERROR", "ERROR", "WARNING", "NOTE", "INFO"]
_SEV_RANK = {s: i for i, s in enumerate(SEVERITY_ORDER)}

# "*** <section text>" lines
_SECTION_RE = re.compile(r"^\*{3}\s+(?P<section>.+?)\s*$")

# "*NN <SEVERITY> - <body>" event lines
_EVENT_RE = re.compile(
    r"^\*\d{2}\s+(?P<severity>SEVERE ERROR|SEVERE WARNING|ERROR|WARNING|NOTE|INFO)\s*-\s*(?P<body>.*)$"
)

# KEY=VALUE tokens; KEY may be "R", "P", "CG" ... or composite like "R.T.P".
# VALUE may include spaces; it ends before the next KEY= or end of line.
# Updated to handle composite keys (with dots) and regular keys more accurately
_KV_RE = re.compile(
    r"(?P<key>[A-Z]+(?:\.[A-Z]+)*)\s*=\s*(?P<val>.+?)(?=\s+[A-Z]+(?:\.[A-Z]+)*\s*=|$)"
)

# ----------------------------
# Utilities
# ----------------------------

def normalize_severity(severity: str) -> str:
    """
    Normalize uncommon spellings into the canonical set. We fold 'SEVERE WARNING' into 'WARNING'.
    """
    s = severity.strip().upper()
    if s == "SEVERE WARNING":
        return "WARNING"
    return s


def severity_rank(severity: str) -> int:
    """
    Rank lower is higher priority: SEVERE ERROR (0) ... INFO (4). Unknown -> 99.
    """
    return _SEV_RANK.get(severity, 99)


def _split_message_and_kvs(body: str) -> Tuple[str, str]:
    """
    Split the event body into 'base message' (before the first KEY=) and the remainder containing KV pairs.
    If no KEY= present, kv_tail = ''.
    """
    m = _KV_RE.search(body)
    if not m:
        return body.strip(" -,:;"), ""
    base = body[:m.start()].strip(" -,:;")
    tail = body[m.start():]
    return base, tail


def expand_composite_key(key: str, val: str) -> Dict[str, str]:
    """
    Expand composite keys like 'R.T.P= A.B.C' into {'R':'A','T':'B','P':'C'} when
    the number of parts matches. Otherwise, return the original key/value.
    """
    if "." in key:
        kparts = key.split(".")
        vparts = [p.strip() for p in val.split(".")]
        if len(kparts) == len(vparts):
            return dict(zip(kparts, vparts))
    return {key: val.strip()}


def parse_kv_fields(kv_text: str, index_allow: Optional[Sequence[str]] = None) -> Dict[str, str]:
    """
    Parse KEY=VALUE fields from kv_text. If index_allow is provided, only keep those keys.
    SUM=... is ignored. Trailing commas/semicolons and "(Auto-relaxed)" suffix are stripped from values.
    """
    allowed = set(index_allow) if index_allow is not None else None
    out: Dict[str, str] = {}
    for m in _KV_RE.finditer(kv_text):
        key = m.group("key").strip()
        val = m.group("val").strip().rstrip(",;")
        
        # Strip "(Auto-relaxed)" suffix if present
        if val.endswith("(Auto-relaxed)"):
            val = val[:-len("(Auto-relaxed)")].strip()
        
        if key == "SUM":
            continue
        for k, v in expand_composite_key(key, val).items():
            if (allowed is None) or (k in allowed):
                out[k] = v
    return out

# ----------------------------
# Event iterator
# ----------------------------

def iter_events(
    source: Iterable[str] | Path,
    index_allow: Optional[Sequence[str]] = None,
    min_severity: str = "INFO",
) -> Iterator[Tuple[str, str, Dict[str, str]]]:
    """
    Iterate parsed events from a QA_CHECK.LOG.

    Yields tuples: (severity, message_key, indices_dict)
      - severity: normalized severity ('SEVERE ERROR'|'ERROR'|'WARNING'|'NOTE'|'INFO')
      - message_key: '<section> :: <base>' or '<section>' if no base found; '(no-section)' if none yet
      - indices_dict: parsed index fields (e.g., {'R':'NSW','P':'EE_Solar425',...})

    Args:
        source: Path or any iterable of lines (e.g., an open file handle or list of strings).
        index_allow: keys to keep (e.g., ['R','P','V','T','CG','COM']). If None, keep all (except SUM).
        min_severity: minimum severity to include, using SEVERITY_ORDER ranking.
    """
    # Prepare the line iterator
    if isinstance(source, Path):
        fh = source.open("r", encoding="utf-8", errors="ignore")
        should_close = True
        lines = fh
    else:
        should_close = False
        lines = source

    current_section = ""
    try:
        for raw in lines:
            line = raw.strip()
            if not line:
                continue

            # Section headers
            m_sec = _SECTION_RE.match(line)
            if m_sec:
                current_section = m_sec.group("section").strip()
                continue

            # Event lines
            m_evt = _EVENT_RE.match(line)
            if not m_evt:
                continue

            sev = normalize_severity(m_evt.group("severity"))
            if severity_rank(sev) > severity_rank(normalize_severity(min_severity)):
                continue

            body = m_evt.group("body")
            base, kv_tail = _split_message_and_kvs(body)
            idx = parse_kv_fields(kv_tail, index_allow=index_allow)
            message_key = f"{current_section} :: {base}" if base else (current_section or "(no-section)")

            yield sev, message_key, idx
    finally:
        if should_close:
            fh.close()

# ----------------------------
# Condenser
# ----------------------------

def condense_events(
    events: Iterable[Tuple[str, str, Dict[str, str]]]
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[str]]:
    """
    Deduplicate events by (severity, message_key, exact index-set) and count occurrences.

    Returns:
        summary_rows: list of dicts with columns:
            - 'severity', 'message', 'occurrences', plus discovered index columns.
        message_counts: list of dicts with columns:
            - 'severity', 'message', 'events' (count of individual event lines per message)
        all_index_keys: sorted list of every discovered index key (for consumers to build stable columns)
    """
    bucket = Counter()           # (sev, msg, frozenset(idx.items())) -> count
    msg_counts = Counter()       # (sev, msg) -> event count
    all_keys: set[str] = set()

    for sev, msg, idx in events:
        keyset = frozenset(sorted(idx.items()))
        bucket[(sev, msg, keyset)] += 1
        msg_counts[(sev, msg)] += 1
        all_keys.update(k for k, _ in keyset)

    all_index_keys = sorted(all_keys)

    # Build rows
    summary_rows: List[Dict[str, str]] = []
    for (sev, msg, keyset), n in bucket.items():
        row: Dict[str, str] = {"severity": sev, "message": msg, "occurrences": str(n)}
        for k, v in keyset:
            row[k] = v
        summary_rows.append(row)

    # Deterministic sort
    def _sort_key(r: Dict[str, str]) -> Tuple[int, str, Tuple[str, ...]]:
        return (
            severity_rank(r.get("severity", "")),
            r.get("message", ""),
            tuple(r.get(k, "") for k in all_index_keys),
        )

    summary_rows.sort(key=_sort_key)

    # Message counts
    message_counts: List[Dict[str, str]] = [
        {"severity": s, "message": m, "events": str(n)}
        for (s, m), n in msg_counts.items()
    ]
    message_counts.sort(key=lambda r: (severity_rank(r["severity"]), r["message"]))

    return summary_rows, message_counts, all_index_keys

# ----------------------------
# Convenience wrapper
# ----------------------------

def condense_log_to_rows(
    path: Path | str,
    index_allow: Optional[Sequence[str]] = None,
    min_severity: str = "INFO",
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[str]]:
    """
    One-shot helper: read a log file (streaming), parse events, and condense.

    Example:
        summary_rows, message_counts, all_keys = condense_log_to_rows(
            "/path/to/QA_CHECK.LOG",
            index_allow=["R","P","V","T","CG","COM"],
            min_severity="WARNING",
        )
    """
    p = Path(path)
    events = iter_events(p, index_allow=index_allow, min_severity=min_severity)
    return condense_events(events)


def format_condensed_output(
    summary_rows: List[Dict[str, str]],
    message_counts: List[Dict[str, str]],
    all_index_keys: List[str],
) -> str:
    """
    Format condensed QA_CHECK.LOG data into a human-readable text report.
    
    Args:
        summary_rows: Detailed event occurrences with indices
        message_counts: Event counts per message
        all_index_keys: List of all discovered index keys
    
    Returns:
        Formatted text report suitable for review
    """
    lines = []
    
    # Header
    lines.append("=" * 80)
    lines.append("QA_CHECK.LOG SUMMARY (Rule-based)")
    lines.append("=" * 80)
    lines.append("")
    
    # Message counts by severity
    lines.append("OVERVIEW BY SEVERITY")
    lines.append("-" * 80)
    
    severity_totals = Counter()
    for row in message_counts:
        severity_totals[row["severity"]] += int(row["events"])
    
    for severity in SEVERITY_ORDER:
        count = severity_totals.get(severity, 0)
        if count > 0:
            lines.append(f"  {severity:15} : {count:5} events")
    
    lines.append("")
    
    # Detailed breakdown by message type
    lines.append("DETAILED BREAKDOWN")
    lines.append("-" * 80)
    
    current_severity = None
    for row in summary_rows:
        severity = row["severity"]
        message = row["message"]
        occurrences = row["occurrences"]
        
        # Group by severity
        if severity != current_severity:
            if current_severity is not None:
                lines.append("")
            lines.append(f"\n[{severity}]")
            current_severity = severity
        
        # Message line
        lines.append(f"  {message}")
        lines.append(f"    Occurrences: {occurrences}")
        
        # Show indices if present
        indices = {k: v for k, v in row.items() if k not in ["severity", "message", "occurrences"] and v}
        if indices:
            indices_str = ", ".join(f"{k}={v}" for k, v in sorted(indices.items()))
            lines.append(f"    Indices: {indices_str}")
    
    lines.append("")
    lines.append("=" * 80)
    lines.append("See QA_CHECK.LOG for full detail")
    lines.append("=" * 80)
    
    return "\n".join(lines)
