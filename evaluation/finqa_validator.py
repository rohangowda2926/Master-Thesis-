"""Strict, architecture-independent validation of sequential FinQA programs.

Only syntax whitespace is normalized: surrounding whitespace and spacing around
operation names, parentheses, arguments, and step separators. Operand spellings
and row-label contents are preserved. No labels, Markdown, assignments, nested
expressions, trailing punctuation, or missing operands are repaired.

Every operation has two serialized arguments. Table operations use an exact raw
row label and the placeholder ``none``. Unlike the permissive upstream executor,
this validator rejects other placeholders and table-row references. Raw row names
containing DSL delimiters (commas, parentheses, pipes, newlines) are unsupported;
there is no invented quoting/escaping convention.

Reference reviewed: https://github.com/czyssrs/FinQA/blob/main/code/evaluate/evaluate.py
Numerical conversion accepts finite numeric const_ suffixes, not just the model's
fixed constant vocabulary. Explicit percent literals mean value / 100; no result
scale is inferred from a question or an answer. No dataset or gold data is used.
"""

from __future__ import annotations

import math
import re
from typing import TypedDict


OPERATION_ARITY = dict.fromkeys(
    ("add", "subtract", "multiply", "divide", "exp", "greater",
     "table_max", "table_min", "table_sum", "table_average"), 2
)
TABLE_OPERATIONS = frozenset(name for name in OPERATION_ARITY if name.startswith("table_"))
_NUMBER = re.compile(
    r"[+-]?(?:(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]*)?|\.[0-9]+)"
    r"(?:[eE][+-]?[0-9]+)?"
)
_REFERENCE = re.compile(r"#(?:0|[1-9][0-9]*)")
_OPERATION = re.compile(r"([A-Za-z_][A-Za-z_0-9]*)\s*\(")


class Operation(TypedDict):
    operation: str
    arguments: list[str]


class ValidationResult(TypedDict):
    valid: bool
    normalized_program: str | None
    operations: list[Operation]
    error_type: str | None
    error_message: str | None


def parse_number(token: str) -> float:
    """Parse a finite FinQA literal; raise ValueError on unsupported spelling.

    Used by both validation and execution so their accepted literals agree.
    Commas must be standard thousands grouping, not arbitrary punctuation.
    """
    if token == "const_m1":
        return -1.0
    if token.startswith("const_"):
        numeric = token[len("const_"):]
        percent = False
    else:
        percent = token.endswith("%")
        numeric = token[:-1] if percent else token
    if not _NUMBER.fullmatch(numeric):
        raise ValueError(f"Not a supported numeric literal: {token!r}.")
    value = float(numeric.replace(",", ""))
    if not math.isfinite(value):
        raise ValueError(f"Numeric literal must be finite: {token!r}.")
    return value / 100.0 if percent else value


def _failure(error_type: str, message: str) -> ValidationResult:
    # No executable partial program is exposed after a validation failure.
    return {"valid": False, "normalized_program": None, "operations": [],
            "error_type": error_type, "error_message": message}


def _scalar_token(token: str) -> bool:
    if _REFERENCE.fullmatch(token):
        return True
    try:
        parse_number(token)
        return True
    except ValueError:
        return False


def _arguments(body: str, operation: str) -> list[str]:
    parts = [part.strip() for part in body.split(",")]
    if len(parts) <= 2 or operation in TABLE_OPERATIONS:
        return parts
    # A comma inside a grouped number is not an extra operand. Prefer explicit
    # comma-whitespace separators; reject ambiguous groupings instead of guessing.
    boundaries = list(re.finditer(r",\s+", body))
    if not boundaries:
        boundaries = list(re.finditer(",", body))
    candidates = []
    for boundary in boundaries:
        pair = [body[:boundary.start()].strip(), body[boundary.start() + 1:].strip()]
        if all(_scalar_token(arg) for arg in pair):
            candidates.append(pair)
    return candidates[0] if len(candidates) == 1 else parts


def validate_program(program: str) -> ValidationResult:
    """Validate a whole string and return JSON-serializable status and operations.

    References must be canonical nonnegative integers (#0, #1, ...) strictly less
    than the current zero-based step. Table availability and arithmetic domain
    errors are checked later by the executor, which receives the financial table.
    """
    if not isinstance(program, str):
        return _failure("invalid_input", "Program must be a string.")
    text = program.strip()
    if not text:
        return _failure("empty_program", "Program is empty.")
    if "=" in text:
        return _failure("assignment_syntax", "Assignments are not valid FinQA syntax.")

    operations: list[Operation] = []
    position = 0
    while position < len(text):
        match = _OPERATION.match(text, position)
        if not match:
            return _failure("malformed_syntax", f"Expected an operation at character {position}.")
        name = match.group(1)
        if name not in OPERATION_ARITY:
            return _failure("unsupported_operation", f"Unsupported operation: {name!r}.")
        close = text.find(")", match.end())
        if close < 0:
            return _failure("malformed_syntax", f"Missing closing parenthesis for step {len(operations)}.")
        body = text[match.end():close]
        if "(" in body:
            return _failure("nested_operation", "Nested parentheses/operations are not allowed.")
        arguments = _arguments(body, name)
        if len(arguments) != OPERATION_ARITY[name] or any(not arg for arg in arguments):
            return _failure("invalid_arity", f"{name} requires exactly two nonempty arguments; comma grouping must be unambiguous.")

        index = len(operations)
        for argument in arguments:
            if "#" in argument:
                if not _REFERENCE.fullmatch(argument):
                    return _failure("invalid_reference", f"Invalid reference {argument!r} at step {index}.")
                # Compare digit strings, avoiding an unbounded int conversion.
                digits, limit = argument[1:], str(index)
                if len(digits) > len(limit) or (len(digits) == len(limit) and digits >= limit):
                    return _failure("forward_reference", f"{argument} at step {index} must refer to an earlier step.")

        if name in TABLE_OPERATIONS:
            label, placeholder = arguments
            if "#" in label or placeholder != "none" or any(c in label for c in "|\r\n`\""):
                return _failure("invalid_table_argument", "Table operations require an exact raw row label and the placeholder none; row references/escaping are unsupported.")
        else:
            for argument in arguments:
                if _REFERENCE.fullmatch(argument):
                    continue
                try:
                    parse_number(argument)
                except ValueError as exc:
                    error = "invalid_constant" if argument.startswith("const_") else "invalid_operand"
                    return _failure(error, f"Step {index}: {exc}")
        operations.append({"operation": name, "arguments": arguments})
        position = close + 1
        while position < len(text) and text[position].isspace():
            position += 1
        if position == len(text):
            break
        if text[position] != ",":
            return _failure("malformed_syntax", f"Expected a comma between operations at character {position}.")
        position += 1
        while position < len(text) and text[position].isspace():
            position += 1
        if position == len(text):
            return _failure("malformed_syntax", "Trailing comma is not allowed.")

    normalized = ", ".join(
        f"{step['operation']}({', '.join(step['arguments'])})" for step in operations
    )
    return {"valid": True, "normalized_program": normalized, "operations": operations,
            "error_type": None, "error_message": None}
