"""
Parsing helpers for laupy INFO files / bash scripts (no Qt dependency).

INFO layout produced by lautools:

    laupy project INFO file created by lautools on: ...
    Key: Value                      <- non-executable header
    ...
    -------------------
    Original INFO content follows:
    -------------------
    <bash script with comments, submitslurm calls, functions, ...>
"""
from __future__ import annotations

import re
from dataclasses import dataclass

INFO_HEADER_FIRST_LINE = "laupy project INFO file"
INFO_HEADER_END = "Original INFO content follows:"

SEPARATOR_RE = re.compile(r"^\s*[-=]{3,}\s*$")
_FUNC_START_RE = re.compile(
    r"^\s*(?:function\s+([A-Za-z_][\w:.-]*)\s*(?:\(\s*\))?"
    r"|([A-Za-z_][\w:.-]*)\s*\(\s*\))\s*(?:\{|$)"
)


@dataclass(frozen=True)
class ShellFunction:
    name: str
    start: int   # 0-based first line
    end: int     # 0-based last line (inclusive)
    code: str


def info_header_end(lines: list[str]) -> int:
    """Index of the first line after the INFO header (0 when no header)."""
    if not lines or not lines[0].startswith(INFO_HEADER_FIRST_LINE):
        return 0
    for i, line in enumerate(lines):
        if line.strip() == INFO_HEADER_END:
            j = i + 1
            if j < len(lines) and SEPARATOR_RE.match(lines[j]):
                j += 1
            return j
    return len(lines)  # header never closed: everything is header


def comment_start(line: str) -> int:
    """Index of an unquoted '#' starting a comment, or -1."""
    quote = None
    escaped = False
    for i, ch in enumerate(line):
        if escaped:
            escaped = False
            continue
        if ch == "\\" and quote != "'":
            escaped = True
            continue
        if quote:
            if ch == quote:
                quote = None
            continue
        if ch in "'\"":
            quote = ch
        elif ch == "#" and (i == 0 or line[i - 1].isspace()):
            return i
    return -1


def _code_part(line: str) -> str:
    """Line without comment and quoted strings (for brace counting)."""
    ci = comment_start(line)
    if ci >= 0:
        line = line[:ci]
    return re.sub(r'"(?:[^"\\]|\\.)*"|\'[^\']*\'', "", line)


def parse_functions(text: str) -> list[ShellFunction]:
    """Find bash function definitions by brace counting."""
    lines = text.split("\n")
    functions: list[ShellFunction] = []
    i = info_header_end(lines)
    while i < len(lines):
        match = _FUNC_START_RE.match(lines[i])
        if not match:
            i += 1
            continue
        name = match.group(1) or match.group(2)
        depth, opened, j = 0, False, i
        while j < len(lines):
            code = _code_part(lines[j])
            depth += code.count("{") - code.count("}")
            opened = opened or "{" in code
            if opened and depth <= 0:
                break
            j += 1
        if not opened or j >= len(lines):
            i += 1  # incomplete definition (still being typed)
            continue
        functions.append(
            ShellFunction(name, i, j, "\n".join(lines[i:j + 1]))
        )
        i = j + 1
    return functions


def function_at(functions: list[ShellFunction], line: int) -> ShellFunction | None:
    return next((f for f in functions if f.start <= line <= f.end), None)


def executable_text(text: str) -> str:
    """Script with header and separator lines blanked (line numbers preserved)."""
    lines = text.split("\n")
    header_end = info_header_end(lines)
    return "\n".join(
        "" if i < header_end or SEPARATOR_RE.match(line) else line
        for i, line in enumerate(lines)
    )
