"""Deterministic execution of validated FinQA programs, with no API/dataset I/O.

Public API: execute_program(program, table=None). The entire string is validated
internally before any operation runs; callers cannot bypass validation by passing
a mutable result dictionary. No gold answers, gold programs, or questions enter
this API. Returned dictionaries can be logged as JSON.

Numerical reference reviewed:
https://github.com/czyssrs/FinQA/blob/main/code/evaluate/evaluate.py
Use Python floats, unrounded intermediate results, and round(final_result, 5).
greater returns 'yes'/'no'. Table labels match exactly and the last duplicate row
wins, as upstream. Table cells lose '$' and any suffix beginning with '(' before
numeric conversion. '(100)' is invalid, not silently changed to -100. Explicit
percent literals are divided by 100; there is no automatic final-result scaling.

Intentional stricter behavior: reject nonfinite/complex values, nonnumeric result
references, malformed inputs, noncanonical table placeholders, and table-row
references (the upstream reference branch uses an uninitialized/stale row).
Symbolic program equivalence and answer scoring are separate, future components.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Sequence
from typing import TypedDict

from evaluation.finqa_validator import TABLE_OPERATIONS, parse_number, validate_program


class ExecutionResult(TypedDict):
    executable: bool
    result: float | str | None
    intermediate_results: list[float | str]
    error_type: str | None
    error_message: str | None


class _ExecutionError(Exception):
    def __init__(self, error_type: str, message: str):
        super().__init__(message)
        self.error_type = error_type


_ARITHMETIC = {
    "add": operator.add,
    "subtract": operator.sub,
    "multiply": operator.mul,
    "divide": operator.truediv,
    "exp": operator.pow,
}


def _failure(error_type: str, message: str, intermediates: list[float | str]) -> ExecutionResult:
    return {"executable": False, "result": None, "intermediate_results": intermediates,
            "error_type": error_type, "error_message": message}


def _finite_real(value: object) -> float:
    if isinstance(value, complex):
        raise _ExecutionError("non_real_result", "Operation produced a complex result.")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _ExecutionError("non_numeric_reference", "Arithmetic requires numerical results, not yes/no values.")
    if not math.isfinite(value):
        raise _ExecutionError("numeric_overflow", "Operation produced a nonfinite result.")
    return float(value)


def _resolve(argument: str, intermediates: list[float | str]) -> float:
    if argument.startswith("#"):
        return _finite_real(intermediates[int(argument[1:])])
    return parse_number(argument)


def _table_values(table: Sequence[Sequence[str]] | None, label: str) -> list[float]:
    if table is None:
        raise _ExecutionError("missing_table", "A table is required for table operations.")
    if not isinstance(table, (list, tuple)):
        raise _ExecutionError("invalid_table", "Table must be a list/tuple of rows.")
    selected = None
    for row in table:
        if not isinstance(row, (list, tuple)) or not row or not isinstance(row[0], str):
            raise _ExecutionError("invalid_table", "Each table row must contain a string row label.")
        if row[0] == label:
            selected = row[1:]  # Last duplicate label wins, matching upstream.
    if selected is None:
        raise _ExecutionError("missing_table_row", f"No exact table row named {label!r}.")
    if not selected:
        raise _ExecutionError("invalid_table", f"Table row {label!r} contains no values.")

    numbers = []
    for column, cell in enumerate(selected, start=1):
        if not isinstance(cell, str):
            raise _ExecutionError("invalid_table_cell", f"Cell {column} of row {label!r} must be a string.")
        # This is upstream cell preprocessing, not model-program repair.
        text = cell.replace("$", "").strip().split("(", 1)[0].strip()
        try:
            numbers.append(parse_number(text))
        except ValueError as exc:
            raise _ExecutionError("invalid_table_cell", f"Cell {column} of row {label!r}: {exc}") from exc
    return numbers


def execute_program(program: str, table: Sequence[Sequence[str]] | None = None) -> ExecutionResult:
    """Validate the entire program, execute sequentially, and round only its end.

    Validation failures return no intermediate results. Execution failures retain
    completed steps for diagnostics, but result is always None on failure. Table
    data is never mutated, and invalid cells are never skipped.
    """
    validation = validate_program(program)
    if not validation["valid"]:
        return _failure(validation["error_type"], validation["error_message"], [])

    intermediates: list[float | str] = []
    for index, step in enumerate(validation["operations"]):
        operation = step["operation"]
        arg1, arg2 = step["arguments"]
        try:
            if operation in TABLE_OPERATIONS:
                values = _table_values(table, arg1)
                if operation == "table_max":
                    value = max(values)
                elif operation == "table_min":
                    value = min(values)
                elif operation == "table_sum":
                    value = sum(values)
                else:
                    value = sum(values) / len(values)
            else:
                left, right = _resolve(arg1, intermediates), _resolve(arg2, intermediates)
                if operation == "greater":
                    value = "yes" if left > right else "no"
                else:
                    value = _ARITHMETIC[operation](left, right)
            if operation != "greater":
                value = _finite_real(value)
            intermediates.append(value)
        except _ExecutionError as exc:
            return _failure(exc.error_type, f"Step {index} ({operation}): {exc}", intermediates)
        except ZeroDivisionError:
            return _failure("division_by_zero", f"Step {index} ({operation}) attempted division by zero or a negative power of zero.", intermediates)
        except OverflowError:
            return _failure("numeric_overflow", f"Step {index} ({operation}) overflowed.", intermediates)
        except (ValueError, TypeError) as exc:
            return _failure("execution_error", f"Step {index} ({operation}): {exc}", intermediates)

    final = intermediates[-1]
    if isinstance(final, float):
        final = round(final, 5)
    return {"executable": True, "result": final, "intermediate_results": intermediates,
            "error_type": None, "error_message": None}
