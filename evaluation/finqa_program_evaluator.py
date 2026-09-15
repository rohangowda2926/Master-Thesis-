"""FinQA symbolic program-accuracy evaluation.

This module implements an architecture-independent program comparison
based closely on the official FinQA evaluator's equal_program logic.

Compatibility reference reviewed 2026-09-12:
https://github.com/czyssrs/FinQA/blob/main/code/evaluate/evaluate.py
Tokenization preserves the original string, including outer whitespace. Table
symbols use the upstream raw step identity, including whitespace and the leading
pipe on steps after the first. Do not canonicalize these identities or numerically
merge literal/constant spellings: doing so changes official program accuracy.

The comparison wrapper deliberately rejects malformed predictions using the
shared validator, even where upstream equal_program permissively accepts them.
It also returns structured errors for invalid gold inputs instead of crashing.
No normalized program replaces the supplied text during symbolic comparison.

This is the symbolic component only: official evaluate_result also asserts that
a symbolically correct prediction executes to the gold answer. A final benchmark
report must check that invariant separately; symbolic equivalence alone does not
establish that every intermediate operation is executable.

It does NOT:
- call an LLM
- load gold answers automatically
- repair predicted programs
- use execution results to decide program correctness

Public functions:

    program_tokenization(program_string)

    compare_programs(
        gold_program,
        predicted_program
    )

Example:

    result = compare_programs(
        "divide(60, 243), multiply(#0, const_100)",
        "divide(60, 243), multiply(#0, const_100)",
    )

    print(result["program_correct"])
"""

from __future__ import annotations

from typing import TypedDict

from sympy import simplify

if __package__ in (None, ""):
    from finqa_validator import validate_program
else:
    from .finqa_validator import validate_program


# ---------------------------------------------------------------------
# FINQA OPERATIONS
# ---------------------------------------------------------------------

ALL_OPERATIONS = {
    "add",
    "subtract",
    "multiply",
    "divide",
    "exp",
    "greater",
    "table_max",
    "table_min",
    "table_sum",
    "table_average",
}


# ---------------------------------------------------------------------
# RESULT TYPE
# ---------------------------------------------------------------------

class ProgramComparisonResult(TypedDict):
    program_correct: bool
    gold_tokens: list[str] | None
    predicted_tokens: list[str] | None
    gold_symbolic: str | None
    predicted_symbolic: str | None
    error_type: str | None
    error_message: str | None


# ---------------------------------------------------------------------
# TOKENIZATION
# ---------------------------------------------------------------------

def program_tokenization(
    original_program: str,
) -> list[str]:
    """Tokenize a FinQA program following the official representation.

    Example:

        divide(60, 243), multiply(#0, const_100)

    becomes approximately:

        [
            "divide(",
            "60",
            "243",
            ")",
            "multiply(",
            "#0",
            "const_100",
            ")",
            "EOF",
        ]

    The official FinQA evaluator expects programs in this sequential
    token representation.
    """

    if not isinstance(original_program, str):
        raise TypeError(
            "Program must be a string."
        )

    # Match upstream exactly for strings: even empty/whitespace input is
    # tokenized as-is. compare_programs rejects invalid structure afterward.

    # FinQA programs conventionally separate steps using ", ".
    steps = original_program.split(", ")

    program: list[str] = []

    for token in steps:

        current = ""

        for character in token:

            if character == ")":

                if current:
                    program.append(current)
                    current = ""

            current += character

            if character in ("(", ")"):

                program.append(current)
                current = ""

        if current:
            program.append(current)

    program.append("EOF")

    return program


# ---------------------------------------------------------------------
# TOKEN STRUCTURE CHECK
# ---------------------------------------------------------------------

def _validate_token_structure(
    tokens: list[str],
) -> tuple[bool, str | None]:
    """Check basic structure expected by the FinQA evaluator."""

    if not isinstance(tokens, list):

        return (
            False,
            "Tokenized program must be a list.",
        )

    if not tokens:

        return (
            False,
            "Tokenized program is empty.",
        )

    if tokens[-1] != "EOF":

        return (
            False,
            "Tokenized program must end with EOF.",
        )

    body = tokens[:-1]

    if len(body) == 0:

        return (
            False,
            "Program contains no operations.",
        )

    if len(body) % 4 != 0:

        return (
            False,
            "FinQA token structure must contain four tokens per operation.",
        )

    for index, token in enumerate(body):

        position = index % 4

        # Operation token
        if position == 0:

            operation = token.strip("(")

            if operation not in ALL_OPERATIONS:

                return (
                    False,
                    f"Unsupported operation: {operation!r}.",
                )

            if not token.endswith("("):

                return (
                    False,
                    f"Malformed operation token: {token!r}.",
                )

        # Closing parenthesis
        elif position == 3:

            if token != ")":

                return (
                    False,
                    f"Expected ')' but received {token!r}.",
                )

    return True, None


# ---------------------------------------------------------------------
# STEP EXTRACTION
# ---------------------------------------------------------------------

def _tokens_to_steps(
    tokens: list[str],
) -> list[dict]:
    """Convert tokenized program into structured sequential steps."""

    body = tokens[:-1]

    steps: list[dict] = []

    # Upstream does not rebuild table identities from stripped operands. Its
    # join/split leaves a leading pipe on every step after the first.
    raw_steps = "|".join(body).split(")")[:-1]

    for index in range(
        0,
        len(body),
        4,
    ):

        operation_token = body[index]

        operation = operation_token[:-1]

        arg1 = body[index + 1].strip()

        arg2 = body[index + 2].strip()

        steps.append(
            {
                "operation": operation,
                "arg1": arg1,
                "arg2": arg2,
                "raw_step": raw_steps[index // 4].strip(),
            }
        )

    return steps


# ---------------------------------------------------------------------
# REFERENCE VALIDATION
# ---------------------------------------------------------------------

def _reference_index(
    argument: str,
) -> int | None:
    """Return integer reference index for #N, otherwise None."""

    if not argument.startswith("#"):
        return None

    reference = argument[1:]

    if not reference.isdigit():

        raise ValueError(
            f"Invalid intermediate reference: {argument!r}."
        )

    return int(reference)


def _validate_references(
    steps: list[dict],
) -> tuple[bool, str | None]:
    """Require each # reference to point to an earlier operation."""

    for step_index, step in enumerate(steps):

        for argument_name in (
            "arg1",
            "arg2",
        ):

            argument = step[argument_name]

            if not argument.startswith("#"):
                continue

            try:

                reference = _reference_index(
                    argument
                )

            except ValueError as exc:

                return (
                    False,
                    str(exc),
                )

            if reference is None:

                continue

            if reference >= step_index:

                return (
                    False,
                    (
                        f"{argument} at step {step_index} "
                        "does not refer to an earlier operation."
                    ),
                )

    return True, None


# ---------------------------------------------------------------------
# GOLD SYMBOL MAP
# ---------------------------------------------------------------------

def _build_gold_symbol_map(
    gold_steps: list[dict],
) -> dict[str, str]:
    """Build the symbol mapping used for FinQA program equivalence.

    Non-reference gold operands receive symbolic names such as:

        a0
        a1
        a2

    Table operations are represented as atomic symbolic values.
    """

    symbol_map: dict[str, str] = {}

    symbol_index = 0

    for step in gold_steps:

        operation = step["operation"]

        arg1 = step["arg1"]

        arg2 = step["arg2"]


        # FinQA treats each table operation as an atomic symbolic value.
        if operation.startswith("table_"):

            key = _table_step_key(
                step,
            )

            if key not in symbol_map:

                symbol_map[key] = (
                    f"a{symbol_index}"
                )

                symbol_index += 1

            continue


        for argument in (
            arg1,
            arg2,
        ):

            if argument.startswith("#"):
                continue

            if argument not in symbol_map:

                symbol_map[argument] = (
                    f"a{symbol_index}"
                )

                symbol_index += 1


    return symbol_map


# ---------------------------------------------------------------------
# TABLE STEP KEY
# ---------------------------------------------------------------------

def _table_step_key(
    step: dict,
) -> str:
    """Preserve the raw table-step identity used by official equal_program."""

    return step["raw_step"]


# ---------------------------------------------------------------------
# PREDICTED OPERAND COMPATIBILITY
# ---------------------------------------------------------------------

def _validate_prediction_symbols(
    predicted_steps: list[dict],
    gold_symbol_map: dict[str, str],
) -> tuple[bool, str | None]:
    """Ensure predicted operands exist in the gold symbolic vocabulary.

    This intentionally follows FinQA's symbolic program scoring behavior.

    For example, if the gold program contains:

        const_5

    but prediction contains:

        5.0

    they may produce the same execution result, but are not necessarily
    program-equivalent according to official program accuracy.
    """

    for step_index, step in enumerate(
        predicted_steps
    ):

        operation = step["operation"]

        arg1 = step["arg1"]

        arg2 = step["arg2"]


        if operation.startswith("table_"):

            key = _table_step_key(
                step,
            )

            if key not in gold_symbol_map:

                return (
                    False,
                    (
                        "Predicted table operation "
                        "does not appear in the gold symbolic program: "
                        f"{key}"
                    ),
                )

            continue


        for argument in (
            arg1,
            arg2,
        ):

            if argument.startswith("#"):

                reference = _reference_index(
                    argument
                )

                if reference is None:

                    return (
                        False,
                        f"Invalid reference: {argument}",
                    )

                if reference >= step_index:

                    return (
                        False,
                        (
                            f"{argument} at step {step_index} "
                            "does not reference an earlier step."
                        ),
                    )

            else:

                if argument not in gold_symbol_map:

                    return (
                        False,
                        (
                            f"Predicted operand {argument!r} "
                            "does not appear in the gold program."
                        ),
                    )


    return True, None


# ---------------------------------------------------------------------
# SYMBOLIC EXPANSION
# ---------------------------------------------------------------------

def _symbolic_expression(
    step_index: int,
    steps: list[dict],
    symbol_map: dict[str, str],
    active_steps: set[int] | None = None,
) -> str:
    """Recursively expand a FinQA program step into symbolic algebra."""

    if active_steps is None:

        active_steps = set()


    if step_index in active_steps:

        raise ValueError(
            "Cyclic FinQA intermediate reference detected."
        )


    if step_index < 0 or step_index >= len(steps):

        raise ValueError(
            f"Invalid step reference: {step_index}."
        )


    active_steps = set(
        active_steps
    )

    active_steps.add(
        step_index
    )


    step = steps[step_index]

    operation = step["operation"]

    arg1 = step["arg1"]

    arg2 = step["arg2"]


    # -------------------------------------------------------------
    # TABLE OPERATION
    # -------------------------------------------------------------

    if operation.startswith("table_"):

        key = _table_step_key(
            step,
        )

        if key not in symbol_map:

            raise ValueError(
                (
                    "Table operation is not represented "
                    f"in symbolic map: {key}"
                )
            )

        return symbol_map[key]


    # -------------------------------------------------------------
    # FIRST ARGUMENT
    # -------------------------------------------------------------

    if arg1.startswith("#"):

        arg1_index = _reference_index(
            arg1
        )

        if arg1_index is None:

            raise ValueError(
                f"Invalid reference: {arg1}"
            )

        left = _symbolic_expression(
            arg1_index,
            steps,
            symbol_map,
            active_steps,
        )

    else:

        if arg1 not in symbol_map:

            raise ValueError(
                (
                    "Operand missing from symbolic map: "
                    f"{arg1!r}"
                )
            )

        left = symbol_map[arg1]


    # -------------------------------------------------------------
    # SECOND ARGUMENT
    # -------------------------------------------------------------

    if arg2.startswith("#"):

        arg2_index = _reference_index(
            arg2
        )

        if arg2_index is None:

            raise ValueError(
                f"Invalid reference: {arg2}"
            )

        right = _symbolic_expression(
            arg2_index,
            steps,
            symbol_map,
            active_steps,
        )

    else:

        if arg2 not in symbol_map:

            raise ValueError(
                (
                    "Operand missing from symbolic map: "
                    f"{arg2!r}"
                )
            )

        right = symbol_map[arg2]


    # -------------------------------------------------------------
    # OPERATION → SYMBOLIC EXPRESSION
    # -------------------------------------------------------------

    if operation == "add":

        return (
            f"({left} + {right})"
        )


    if operation == "subtract":

        return (
            f"({left} - {right})"
        )


    if operation == "multiply":

        return (
            f"({left} * {right})"
        )


    if operation == "divide":

        return (
            f"({left} / {right})"
        )


    if operation == "exp":

        return (
            f"({left} ** {right})"
        )


    if operation == "greater":

        return (
            f"({left} > {right})"
        )


    raise ValueError(
        f"Unsupported symbolic operation: {operation}"
    )


# ---------------------------------------------------------------------
# PUBLIC PROGRAM COMPARISON
# ---------------------------------------------------------------------

def compare_programs(
    gold_program: str,
    predicted_program: str | None,
) -> ProgramComparisonResult:
    """Compare predicted program with gold FinQA program symbolically."""

    if predicted_program is None:

        return {
            "program_correct": False,
            "gold_tokens": None,
            "predicted_tokens": None,
            "gold_symbolic": None,
            "predicted_symbolic": None,
            "error_type": "missing_prediction",
            "error_message":
                "Predicted program is None.",
        }


    try:

        gold_tokens = program_tokenization(
            gold_program
        )

    except Exception as exc:

        return {
            "program_correct": False,
            "gold_tokens": None,
            "predicted_tokens": None,
            "gold_symbolic": None,
            "predicted_symbolic": None,
            "error_type": "gold_tokenization_error",
            "error_message": str(exc),
        }


    try:

        predicted_tokens = program_tokenization(
            predicted_program
        )

    except Exception as exc:

        return {
            "program_correct": False,
            "gold_tokens": gold_tokens,
            "predicted_tokens": None,
            "gold_symbolic": None,
            "predicted_symbolic": None,
            "error_type": "prediction_tokenization_error",
            "error_message": str(exc),
        }


    # -------------------------------------------------------------
    # STRUCTURE VALIDATION
    # -------------------------------------------------------------

    gold_valid, gold_error = _validate_token_structure(
        gold_tokens
    )

    if not gold_valid:

        return {
            "program_correct": False,
            "gold_tokens": gold_tokens,
            "predicted_tokens": predicted_tokens,
            "gold_symbolic": None,
            "predicted_symbolic": None,
            "error_type": "invalid_gold_structure",
            "error_message": gold_error,
        }


    prediction_valid, prediction_error = _validate_token_structure(
        predicted_tokens
    )

    if not prediction_valid:

        return {
            "program_correct": False,
            "gold_tokens": gold_tokens,
            "predicted_tokens": predicted_tokens,
            "gold_symbolic": None,
            "predicted_symbolic": None,
            "error_type": "invalid_prediction_structure",
            "error_message": prediction_error,
        }


    # -------------------------------------------------------------
    # STEPS
    # -------------------------------------------------------------

    gold_steps = _tokens_to_steps(
        gold_tokens
    )

    predicted_steps = _tokens_to_steps(
        predicted_tokens
    )


    # -------------------------------------------------------------
    # REFERENCES
    # -------------------------------------------------------------

    gold_refs_valid, gold_ref_error = _validate_references(
        gold_steps
    )

    if not gold_refs_valid:

        return {
            "program_correct": False,
            "gold_tokens": gold_tokens,
            "predicted_tokens": predicted_tokens,
            "gold_symbolic": None,
            "predicted_symbolic": None,
            "error_type": "invalid_gold_reference",
            "error_message": gold_ref_error,
        }


    predicted_refs_valid, predicted_ref_error = _validate_references(
        predicted_steps
    )

    if not predicted_refs_valid:

        return {
            "program_correct": False,
            "gold_tokens": gold_tokens,
            "predicted_tokens": predicted_tokens,
            "gold_symbolic": None,
            "predicted_symbolic": None,
            "error_type": "invalid_prediction_reference",
            "error_message": predicted_ref_error,
        }


    # The official token checker can accept malformed strings such as two
    # adjacent calls with no separating comma. Keep the thesis validity rule
    # explicit, without replacing the original prediction with normalized text.
    prediction_validation = validate_program(predicted_program)
    if not prediction_validation["valid"]:
        return {
            "program_correct": False,
            "gold_tokens": gold_tokens,
            "predicted_tokens": predicted_tokens,
            "gold_symbolic": None,
            "predicted_symbolic": None,
            "error_type": "invalid_prediction_program",
            "error_message": (
                f"{prediction_validation['error_type']}: "
                f"{prediction_validation['error_message']}"
            ),
        }

    # -------------------------------------------------------------
    # SYMBOL MAP
    # -------------------------------------------------------------

    gold_symbol_map = _build_gold_symbol_map(
        gold_steps
    )


    compatible, compatibility_error = _validate_prediction_symbols(
        predicted_steps,
        gold_symbol_map,
    )

    if not compatible:

        return {
            "program_correct": False,
            "gold_tokens": gold_tokens,
            "predicted_tokens": predicted_tokens,
            "gold_symbolic": None,
            "predicted_symbolic": None,
            "error_type": "symbol_mismatch",
            "error_message": compatibility_error,
        }


    # -------------------------------------------------------------
    # SYMBOLIC EXPRESSIONS
    # -------------------------------------------------------------

    try:

        gold_expression = _symbolic_expression(
            len(gold_steps) - 1,
            gold_steps,
            gold_symbol_map,
        )


        predicted_expression = _symbolic_expression(
            len(predicted_steps) - 1,
            predicted_steps,
            gold_symbol_map,
        )


        gold_symbolic = simplify(
            gold_expression,
            evaluate=False,
        )


        predicted_symbolic = simplify(
            predicted_expression,
            evaluate=False,
        )


    except Exception as exc:

        return {
            "program_correct": False,
            "gold_tokens": gold_tokens,
            "predicted_tokens": predicted_tokens,
            "gold_symbolic": None,
            "predicted_symbolic": None,
            "error_type": "symbolic_evaluation_error",
            "error_message": str(exc),
        }


    # -------------------------------------------------------------
    # FINAL PROGRAM ACCURACY
    # -------------------------------------------------------------

    program_correct = (
        gold_symbolic
        == predicted_symbolic
    )


    return {
        "program_correct":
            bool(program_correct),

        "gold_tokens":
            gold_tokens,

        "predicted_tokens":
            predicted_tokens,

        "gold_symbolic":
            str(gold_symbolic),

        "predicted_symbolic":
            str(predicted_symbolic),

        "error_type":
            None,

        "error_message":
            None,
    }


# ---------------------------------------------------------------------
# SIMPLE LOCAL SELF-TEST
# ---------------------------------------------------------------------

if __name__ == "__main__":

    examples = [

        # Exact same program
        (
            "divide(60, 243), multiply(#0, const_100)",
            "divide(60, 243), multiply(#0, const_100)",
            True,
        ),

        # Equivalent commutative addition
        (
            "add(1356, 2220)",
            "add(2220, 1356)",
            True,
        ),

        # Wrong operation
        (
            "add(1356, 2220)",
            "subtract(1356, 2220)",
            False,
        ),

        # Missing prediction
        (
            "divide(59.1, 98.0)",
            None,
            False,
        ),
    ]


    passed = 0


    for index, (
        gold,
        predicted,
        expected,
    ) in enumerate(
        examples,
        start=1,
    ):

        result = compare_programs(
            gold,
            predicted,
        )


        actual = result[
            "program_correct"
        ]


        status = (
            "PASS"
            if actual == expected
            else "FAIL"
        )


        print(
            f"Test {index}: "
            f"{status} | "
            f"expected={expected} | "
            f"actual={actual}"
        )


        if actual == expected:
            passed += 1


    print()

    print(
        f"Passed {passed}/{len(examples)} tests."
    )
