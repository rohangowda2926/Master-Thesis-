"""20-example FinQA single-agent few-shot pilot.

Run from the repository root:

    python -m single_agent.baseline

Model:
    qwen/qwen3-30b-a3b-instruct-2507

Evaluation slice:
    dev[30:50]

Few-shot demonstrations:
    Four fixed examples selected deterministically from train.json.

Pipeline:

Qwen3-30B
    -> FinQA program
    -> strict JSON parsing
    -> validator
    -> deterministic executor
    -> execution accuracy
    -> symbolic program accuracy

Gold answers and gold programs are never provided to the model.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------
# PROJECT PATH
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------
# SHARED COMPONENTS
# ---------------------------------------------------------------------

from evaluation.finqa_executor import execute_program
from evaluation.finqa_program_evaluator import compare_programs
from evaluation.finqa_validator import OPERATION_ARITY, validate_program
from single_agent.few_shot import format_few_shot_examples


# ---------------------------------------------------------------------
# EXPERIMENT CONFIGURATION
# ---------------------------------------------------------------------

MODEL = "qwen/qwen3-30b-a3b-instruct-2507"

ARCHITECTURE = "single_agent"
CONFIGURATION = "A"

DATASET_PATH = PROJECT_ROOT / "data" / "finqa" / "dev.json"
RESULTS_DIR = PROJECT_ROOT / "results"


# ---------------------------------------------------------------------
# DEVELOPMENT SLICE
# ---------------------------------------------------------------------
#
# dev[0:5]
#     used during early pipeline/prompt development
#
# dev[5:10]
#     used for first Qwen3-30B smoke test
#
# dev[10:30]
#     used for 20-example zero-shot pilot
#
# Therefore this few-shot pilot uses new development examples:
#
#     dev[30:50]
#
# ---------------------------------------------------------------------

DEV_START_INDEX = 50
NUM_EXAMPLES = 20


# ---------------------------------------------------------------------
# MODEL SETTINGS
# ---------------------------------------------------------------------

TEMPERATURE = 0

MAX_OUTPUT_TOKENS = 400

MAX_ATTEMPTS = 3

RETRY_DELAY_SECONDS = 2


# ---------------------------------------------------------------------
# FALLBACK PRICING
# ---------------------------------------------------------------------
#
# Actual OpenRouter usage.cost is preferred.
# These are used only if actual cost is unavailable.
#
# Pricing may change/provider routing may differ.
# Final RQ3 reporting should use actual returned API cost where possible.
#

FALLBACK_INPUT_PRICE_PER_MILLION = 0.04815
FALLBACK_OUTPUT_PRICE_PER_MILLION = 0.1931


# ---------------------------------------------------------------------
# OPENROUTER CLIENT
# ---------------------------------------------------------------------

def create_client():

    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(
        PROJECT_ROOT / ".env"
    )

    api_key = os.getenv(
        "OPENROUTER_API_KEY"
    )

    if not api_key:

        raise ValueError(
            "OPENROUTER_API_KEY was not found in environment/.env."
        )

    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        max_retries=0,
    )


# ---------------------------------------------------------------------
# COST
# ---------------------------------------------------------------------

def calculate_fallback_cost(
    input_tokens,
    output_tokens,
):

    if (
        input_tokens is None
        or output_tokens is None
    ):
        return None

    return (
        input_tokens
        / 1_000_000
        * FALLBACK_INPUT_PRICE_PER_MILLION
        +
        output_tokens
        / 1_000_000
        * FALLBACK_OUTPUT_PRICE_PER_MILLION
    )


def _extract_usage_cost(
    usage,
):
    """Read OpenRouter's actual cost when exposed by the SDK."""

    if usage is None:
        return None

    value = getattr(
        usage,
        "cost",
        None,
    )

    if value is None:

        model_extra = getattr(
            usage,
            "model_extra",
            None,
        )

        if isinstance(
            model_extra,
            dict,
        ):
            value = model_extra.get(
                "cost"
            )

    try:

        if value is not None:
            return float(value)

    except (
        TypeError,
        ValueError,
    ):
        pass

    return None


# ---------------------------------------------------------------------
# PROMPT
# ---------------------------------------------------------------------

def build_prompt(
    question,
    context,
):

    allowed_operations = ", ".join(
        OPERATION_ARITY
    )

    few_shot_examples = (
        format_few_shot_examples()
    )

    return f"""
You are solving a FinQA financial numerical reasoning problem.

Your ONLY task is to produce a valid sequential FinQA program.

Python will execute the program deterministically.

Return exactly one JSON object:

{{"program": "<FinQA program>"}}

Return nothing else.

Do NOT output:
- explanations
- reasoning
- final numerical answer
- Markdown
- code fences
- comments
- additional JSON keys
- text before or after the JSON object


============================================================
ALLOWED OPERATIONS
============================================================

{allowed_operations}


============================================================
ARITHMETIC OPERANDS
============================================================

Prefer explicit numerical values directly from the supplied evidence.

Arithmetic operands must be one of:

1. numeric literal

Examples:

250
193.5
0.25


2. FinQA constant

Examples:

const_1
const_2
const_5
const_100
const_m1


3. previous operation result

Examples:

#0
#1
#2


Do NOT use words such as:

Canada
revenue
sales
2020 revenue

as arithmetic operands.

If a number is visible in the table or text, use the number itself.


============================================================
SEQUENTIAL PROGRAM FORMAT
============================================================

Multi-step calculations must use sequential operations.

Correct:

{{"program": "divide(60, 243), multiply(#0, const_100)"}}

Incorrect:

{{"program": "multiply(divide(60, 243), const_100)"}}

Never nest operations.


============================================================
REFERENCES
============================================================

Operations are numbered from zero.

At step 0:

No # reference is available.

At step 1:

#0 may be used.

At step 2:

#0 and #1 may be used.

Never reference a future operation.


============================================================
NO ASSIGNMENTS
============================================================

Incorrect:

#0 = divide(60, 243)

Correct:

divide(60, 243)


============================================================
TABLE VALUES
============================================================

When the required numbers are directly visible in the table,
use those numbers directly.

Example:

Country | Sales
Canada  | 120
USA     | 200

Question:

What is the difference between USA and Canada sales?

Correct:

{{"program": "subtract(200, 120)"}}

Incorrect:

{{"program": "subtract(usa, canada)"}}


============================================================
TABLE OPERATIONS
============================================================

Use:

table_sum
table_average
table_max
table_min

ONLY when the question genuinely requires aggregation over a numeric row.

Example:

table_sum(revenue, none)

Rules:

- first argument = exact raw table row label
- second argument = none
- do not quote row labels
- do not use table operations merely to retrieve one cell
- do not invent table labels

If a row label contains parentheses, commas, quotes, or syntax-conflicting
characters, prefer using the required numerical values directly instead.


============================================================
UNSUPPORTED OPERATIONS
============================================================

Never use:

table_value
retrieve_value
lookup
get_value
extract_value

Only use the allowed FinQA operations listed above.


============================================================
PERCENTAGES AND RATIOS
============================================================

Infer the required transformation from the question.

Do NOT always multiply by 100.

Do NOT always avoid multiplying by 100.

Example requiring percentage conversion:

60 / 243 × 100

Correct:

{{"program": "divide(60, 243), multiply(#0, const_100)"}}

Example where percentage conversion is NOT required:

59.1 / 98.0

Correct:

{{"program": "divide(59.1, 98.0)"}}


============================================================
SELF-CHECK
============================================================

Before answering, check internally:

1. Did I use only allowed operations?
2. Are arithmetic operands only numbers, constants, or # references?
3. Did I avoid row/column/country names as numerical operands?
4. Did I avoid nested operations?
5. Are # references sequential and valid?
6. Did I avoid assignment syntax?
7. Did I use table operations only when genuinely needed?
8. Is percentage conversion actually required by the question?
9. Is the response exactly one JSON object containing only "program"?


============================================================
FINQA TRAINING DEMONSTRATIONS
============================================================

The following examples come ONLY from the FinQA training split.

Use them to learn the expected reasoning and program representation.

Do not copy values from these demonstrations into the new problem.

{few_shot_examples}


============================================================
QUESTION
============================================================

{question}


============================================================
FINANCIAL EVIDENCE
============================================================

{context}
""".strip()


# ---------------------------------------------------------------------
# STRICT JSON PARSER
# ---------------------------------------------------------------------

def _unique_json_object(
    pairs,
):

    result = {}

    for key, value in pairs:

        if key in result:

            raise ValueError(
                f"Duplicate JSON key: {key!r}."
            )

        result[key] = value

    return result


def _reject_json_constant(
    value,
):

    raise ValueError(
        f"Nonstandard JSON constant: {value}."
    )


def parse_program_response(
    raw_response,
):

    if (
        not isinstance(
            raw_response,
            str,
        )
        or not raw_response.strip()
    ):

        return None, {

            "error_type":
                "empty_response",

            "error_message":
                "Expected a nonempty JSON response.",
        }


    try:

        parsed = json.loads(
            raw_response,
            object_pairs_hook=
                _unique_json_object,
            parse_constant=
                _reject_json_constant,
        )

    except ValueError as exc:

        return None, {

            "error_type":
                "invalid_json",

            "error_message":
                str(exc),
        }


    if (
        not isinstance(
            parsed,
            dict,
        )
        or set(parsed) != {"program"}
    ):

        return None, {

            "error_type":
                "invalid_json_schema",

            "error_message":
                'Expected an object with exactly one key, "program".',
        }


    if not isinstance(
        parsed["program"],
        str,
    ):

        return None, {

            "error_type":
                "invalid_program_type",

            "error_message":
                'The "program" value must be a string.',
        }


    return (
        parsed["program"],
        None,
    )


# ---------------------------------------------------------------------
# TOKEN HELPERS
# ---------------------------------------------------------------------

def _token_count(
    usage,
    field,
):

    value = getattr(
        usage,
        field,
        None,
    )

    if (
        type(value) is int
        and value >= 0
    ):
        return value

    return None


def _complete_usage_sum(
    attempts,
    field,
):

    values = [
        attempt[field]
        for attempt in attempts
    ]

    if all(
        value is not None
        for value in values
    ):
        return sum(values)

    return None


# ---------------------------------------------------------------------
# MODEL CALL
# ---------------------------------------------------------------------

def call_model(
    client,
    question,
    context,
):

    prompt = build_prompt(
        question,
        context,
    )

    attempts = []

    raw_response = None

    finish_reason = None

    api_error = None

    started = time.perf_counter()


    for attempt_number in range(
        1,
        MAX_ATTEMPTS + 1,
    ):

        attempt_started = (
            time.perf_counter()
        )

        attempt = {

            "attempt_number":
                attempt_number,

            "latency":
                None,

            "input_tokens":
                None,

            "output_tokens":
                None,

            "total_tokens":
                None,

            "actual_cost_usd":
                None,

            "estimated_cost_usd":
                None,

            "api_error":
                None,
        }


        try:

            response = (
                client.chat.completions.create(

                    model=MODEL,

                    messages=[
                        {
                            "role":
                                "user",

                            "content":
                                prompt,
                        }
                    ],

                    temperature=
                        TEMPERATURE,

                    max_tokens=
                        MAX_OUTPUT_TOKENS,

                    response_format={
                        "type":
                            "json_object"
                    },
                )
            )


        except Exception as exc:

            api_error = {

                "error_type":
                    type(exc).__name__,

                "error_message":
                    str(exc),
            }

            attempt[
                "api_error"
            ] = api_error


        else:

            usage = getattr(
                response,
                "usage",
                None,
            )


            attempt[
                "input_tokens"
            ] = _token_count(
                usage,
                "prompt_tokens",
            )


            attempt[
                "output_tokens"
            ] = _token_count(
                usage,
                "completion_tokens",
            )


            reported_total = (
                _token_count(
                    usage,
                    "total_tokens",
                )
            )


            if reported_total is not None:

                attempt[
                    "total_tokens"
                ] = reported_total


            elif (
                attempt["input_tokens"]
                is not None
                and
                attempt["output_tokens"]
                is not None
            ):

                attempt[
                    "total_tokens"
                ] = (

                    attempt[
                        "input_tokens"
                    ]

                    +

                    attempt[
                        "output_tokens"
                    ]
                )


            attempt[
                "actual_cost_usd"
            ] = _extract_usage_cost(
                usage
            )


            attempt[
                "estimated_cost_usd"
            ] = calculate_fallback_cost(

                attempt[
                    "input_tokens"
                ],

                attempt[
                    "output_tokens"
                ],
            )


            choices = getattr(
                response,
                "choices",
                None,
            )


            if not choices:

                api_error = {

                    "error_type":
                        "invalid_api_response",

                    "error_message":
                        "API response contained no choices.",
                }

                attempt[
                    "api_error"
                ] = api_error


            else:

                message = getattr(
                    choices[0],
                    "message",
                    None,
                )


                raw_response = getattr(
                    message,
                    "content",
                    None,
                )


                finish_reason = getattr(
                    choices[0],
                    "finish_reason",
                    None,
                )


                api_error = None


        attempt[
            "latency"
        ] = (
            time.perf_counter()
            - attempt_started
        )


        attempts.append(
            attempt
        )


        if api_error is None:
            break


        if (
            attempt_number
            < MAX_ATTEMPTS
        ):

            time.sleep(
                RETRY_DELAY_SECONDS
            )


    input_tokens = (
        _complete_usage_sum(
            attempts,
            "input_tokens",
        )
    )


    output_tokens = (
        _complete_usage_sum(
            attempts,
            "output_tokens",
        )
    )


    total_tokens = (
        _complete_usage_sum(
            attempts,
            "total_tokens",
        )
    )


    actual_cost_usd = (
        _complete_usage_sum(
            attempts,
            "actual_cost_usd",
        )
    )


    estimated_cost_usd = (
        calculate_fallback_cost(
            input_tokens,
            output_tokens,
        )
    )


    effective_cost_usd = (

        actual_cost_usd

        if actual_cost_usd
        is not None

        else estimated_cost_usd
    )


    if actual_cost_usd is not None:

        cost_source = (
            "openrouter_usage"
        )

    elif estimated_cost_usd is not None:

        cost_source = (
            "fallback_estimate"
        )

    else:

        cost_source = None


    return {

        "raw_response":
            raw_response,

        "finish_reason":
            finish_reason,

        "api_latency":
            (
                time.perf_counter()
                - started
            ),

        "input_tokens":
            input_tokens,

        "output_tokens":
            output_tokens,

        "total_tokens":
            total_tokens,

        "actual_cost_usd":
            actual_cost_usd,

        "estimated_cost_usd":
            estimated_cost_usd,

        "effective_cost_usd":
            effective_cost_usd,

        "cost_source":
            cost_source,

        "usage_complete":
            (
                total_tokens
                is not None
            ),

        "api_error":
            api_error,

        "retry_count":
            len(attempts) - 1,

        "attempt_count":
            len(attempts),

        "attempts":
            attempts,
    }


# ---------------------------------------------------------------------
# FINQA CONTEXT
# ---------------------------------------------------------------------

def build_context(
    example,
):

    pre_text = example.get(
        "pre_text",
        [],
    )

    post_text = example.get(
        "post_text",
        [],
    )

    table = example.get(
        "table",
        [],
    )

    context_parts = []


    if pre_text:

        context_parts.append(
            "Pre-text:\n"
            + "\n".join(
                pre_text
            )
        )


    if table:

        table_text = "\n".join(

            " | ".join(
                str(cell)
                for cell in row
            )

            for row in table
        )


        context_parts.append(
            "Table:\n"
            + table_text
        )


    if post_text:

        context_parts.append(
            "Post-text:\n"
            + "\n".join(
                post_text
            )
        )


    return "\n\n".join(
        context_parts
    )


# ---------------------------------------------------------------------
# RUN ONE EXAMPLE
# ---------------------------------------------------------------------

def run_example(
    client,
    example,
    run_id,
    dataset_index,
):

    started = time.perf_counter()


    question = (
        example["qa"]["question"]
    )


    context = build_context(
        example
    )


    # -------------------------------------------------------------
    # MODEL CALL
    # -------------------------------------------------------------

    model_result = call_model(
        client,
        question,
        context,
    )


    generated_program = None

    parse_error = None

    validation = None

    execution = None


    # -------------------------------------------------------------
    # PARSE
    # -------------------------------------------------------------

    if (
        model_result["api_error"]
        is None
    ):

        (
            generated_program,
            parse_error,
        ) = parse_program_response(
            model_result[
                "raw_response"
            ]
        )


        # ---------------------------------------------------------
        # VALIDATE
        # ---------------------------------------------------------

        if parse_error is None:

            validation = (
                validate_program(
                    generated_program
                )
            )


            # -----------------------------------------------------
            # EXECUTE
            # -----------------------------------------------------

            if validation[
                "valid"
            ]:

                execution = (
                    execute_program(

                        generated_program,

                        table=
                            example.get(
                                "table"
                            ),
                    )
                )


    program_valid = (

        validation is not None

        and validation["valid"]
    )


    executable = (

        execution is not None

        and execution[
            "executable"
        ]
    )


    predicted_answer = (

        execution["result"]

        if executable

        else None
    )


    # -------------------------------------------------------------
    # GOLD
    #
    # Accessed only AFTER model generation.
    # -------------------------------------------------------------

    gold_answer = (
        example["qa"]["exe_ans"]
    )

    gold_program = (
        example["qa"]["program"]
    )


    # -------------------------------------------------------------
    # EXECUTION ACCURACY
    # -------------------------------------------------------------

    execution_correct = bool(

        executable

        and predicted_answer
        == gold_answer
    )


    # -------------------------------------------------------------
    # SYMBOLIC PROGRAM EVALUATION
    # -------------------------------------------------------------

    program_evaluation = (
        compare_programs(
            gold_program,
            generated_program,
        )
    )


    program_symbolically_correct = bool(

        program_evaluation[
            "program_correct"
        ]
    )


    # -------------------------------------------------------------
    # FINAL PROGRAM ACCURACY
    # -------------------------------------------------------------

    program_correct = bool(

        program_symbolically_correct

        and executable

        and execution_correct
    )


    # -------------------------------------------------------------
    # RECORD
    # -------------------------------------------------------------

    record = {

        "example_id":
            example["id"],

        "dataset_index":
            dataset_index,

        "architecture":
            ARCHITECTURE,

        "configuration":
            CONFIGURATION,

        "run_id":
            run_id,

        "run_number":
            1,

        "dataset_split":
            "dev",

        "development_slice":
            (
                f"dev["
                f"{DEV_START_INDEX}:"
                f"{DEV_START_INDEX + NUM_EXAMPLES}"
                f"]"
            ),

        "prompt_mode":
            "fixed_4_shot_train",

        "question":
            question,


        # Programs

        "generated_program":
            generated_program,

        "normalized_program":
            (
                validation[
                    "normalized_program"
                ]

                if validation

                else None
            ),

        "gold_program":
            gold_program,


        # Validation

        "program_valid":
            program_valid,

        "validation_attempted":
            validation is not None,

        "validation_error":
            (
                {
                    "error_type":
                        validation[
                            "error_type"
                        ],

                    "error_message":
                        validation[
                            "error_message"
                        ],
                }

                if (
                    validation
                    is not None

                    and not validation[
                        "valid"
                    ]
                )

                else None
            ),


        # Execution

        "executable":
            executable,

        "execution_attempted":
            execution is not None,

        "execution_error":
            (
                {
                    "error_type":
                        execution[
                            "error_type"
                        ],

                    "error_message":
                        execution[
                            "error_message"
                        ],
                }

                if (
                    execution
                    is not None

                    and not execution[
                        "executable"
                    ]
                )

                else None
            ),

        "intermediate_results":
            (
                execution[
                    "intermediate_results"
                ]

                if execution

                else []
            ),

        "predicted_answer":
            predicted_answer,

        "gold_answer":
            gold_answer,

        "execution_correct":
            execution_correct,


        # Program accuracy

        "program_symbolically_correct":
            program_symbolically_correct,

        "program_correct":
            program_correct,

        "program_accuracy_evaluated":
            True,

        "program_evaluation_error":
            (
                {
                    "error_type":
                        program_evaluation[
                            "error_type"
                        ],

                    "error_message":
                        program_evaluation[
                            "error_message"
                        ],
                }

                if (
                    program_evaluation[
                        "error_type"
                    ]
                    is not None
                )

                else None
            ),

        "gold_program_tokens":
            program_evaluation[
                "gold_tokens"
            ],

        "predicted_program_tokens":
            program_evaluation[
                "predicted_tokens"
            ],

        "gold_symbolic_program":
            program_evaluation[
                "gold_symbolic"
            ],

        "predicted_symbolic_program":
            program_evaluation[
                "predicted_symbolic"
            ],


        # Parsing

        "parse_error":
            parse_error,


        # Model configuration

        "model":
            MODEL,

        "temperature":
            TEMPERATURE,

        "max_output_tokens":
            MAX_OUTPUT_TOKENS,


        # API metrics

        **model_result,


        # Total latency

        "latency":
            (
                time.perf_counter()
                - started
            ),
    }


    return record


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    # -------------------------------------------------------------
    # LOAD DEV SET
    # -------------------------------------------------------------

    with DATASET_PATH.open(
        "r",
        encoding="utf-8",
    ) as dataset_file:

        dataset = json.load(
            dataset_file
        )


    examples = dataset[
        DEV_START_INDEX:
        DEV_START_INDEX
        + NUM_EXAMPLES
    ]


    if (
        len(examples)
        != NUM_EXAMPLES
    ):

        raise ValueError(
            "Not enough development examples for requested pilot slice."
        )


    # -------------------------------------------------------------
    # CLIENT
    # -------------------------------------------------------------

    client = create_client()


    # -------------------------------------------------------------
    # RUN ID
    # -------------------------------------------------------------

    run_id = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%S_%fZ"
    )


    # -------------------------------------------------------------
    # RESULT FILE
    # -------------------------------------------------------------

    results_path = (

        RESULTS_DIR

        / (
            "single_agent_qwen30b_"
            "fewshot_pilot_"
            f"{run_id}.jsonl"
        )
    )


    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    # -------------------------------------------------------------
    # COUNTERS
    # -------------------------------------------------------------

    valid_count = 0

    executable_count = 0

    execution_correct_count = 0

    symbolic_correct_count = 0

    program_correct_count = 0

    parse_error_count = 0

    validation_failure_count = 0

    execution_failure_count = 0

    api_error_count = 0


    total_input_tokens = 0

    total_output_tokens = 0

    total_effective_cost = 0.0

    total_latency = 0.0

    cost_complete = True


    # -------------------------------------------------------------
    # RUN
    # -------------------------------------------------------------

    with results_path.open(
        "x",
        encoding="utf-8",
    ) as results_file:


        print(
            "Single-agent FinQA Qwen3-30B "
            "few-shot pilot"
        )

        print(
            f"Model: {MODEL}"
        )

        print(
            f"Temperature: "
            f"{TEMPERATURE}"
        )

        print(
            "Few-shot mode: "
            "4 fixed train demonstrations"
        )

        print(
            f"Dataset slice: "
            f"dev[{DEV_START_INDEX}:"
            f"{DEV_START_INDEX + NUM_EXAMPLES}]"
        )

        print(
            f"Examples: "
            f"{len(examples)}"
        )

        print(
            f"Results: "
            f"{results_path}"
        )

        print()


        for offset, example in enumerate(
            examples
        ):

            dataset_index = (
                DEV_START_INDEX
                + offset
            )


            record = run_example(
                client,
                example,
                run_id,
                dataset_index,
            )


            # -----------------------------------------------------
            # SAVE
            # -----------------------------------------------------

            results_file.write(

                json.dumps(
                    record,
                    ensure_ascii=False,
                    allow_nan=False,
                )

                + "\n"
            )

            results_file.flush()


            # -----------------------------------------------------
            # COUNTERS
            # -----------------------------------------------------

            valid_count += int(
                record[
                    "program_valid"
                ]
            )


            executable_count += int(
                record[
                    "executable"
                ]
            )


            execution_correct_count += int(
                record[
                    "execution_correct"
                ]
            )


            symbolic_correct_count += int(
                record[
                    "program_symbolically_correct"
                ]
            )


            program_correct_count += int(
                record[
                    "program_correct"
                ]
            )


            if (
                record[
                    "parse_error"
                ]
                is not None
            ):

                parse_error_count += 1


            if (
                record[
                    "validation_error"
                ]
                is not None
            ):

                validation_failure_count += 1


            if (
                record[
                    "execution_error"
                ]
                is not None
            ):

                execution_failure_count += 1


            if (
                record[
                    "api_error"
                ]
                is not None
            ):

                api_error_count += 1


            if (
                record[
                    "input_tokens"
                ]
                is not None
            ):

                total_input_tokens += (
                    record[
                        "input_tokens"
                    ]
                )


            if (
                record[
                    "output_tokens"
                ]
                is not None
            ):

                total_output_tokens += (
                    record[
                        "output_tokens"
                    ]
                )


            if (
                record[
                    "effective_cost_usd"
                ]
                is not None
            ):

                total_effective_cost += (
                    record[
                        "effective_cost_usd"
                    ]
                )

            else:

                cost_complete = False


            total_latency += (
                record[
                    "latency"
                ]
            )


            # -----------------------------------------------------
            # TERMINAL OUTPUT
            # -----------------------------------------------------

            print(
                f"[dev index "
                f"{dataset_index}] "
                f"{record['example_id']}"
            )

            print(
                f"  Program: "
                f"{record['generated_program']}"
            )

            print(
                f"  Valid: "
                f"{record['program_valid']}"
            )

            print(
                f"  Executable: "
                f"{record['executable']}"
            )

            print(
                f"  Predicted: "
                f"{record['predicted_answer']}"
            )

            print(
                f"  Gold: "
                f"{record['gold_answer']}"
            )

            print(
                f"  Execution correct: "
                f"{record['execution_correct']}"
            )

            print(
                f"  Symbolically correct: "
                f"{record['program_symbolically_correct']}"
            )

            print(
                f"  Program correct: "
                f"{record['program_correct']}"
            )

            print(
                f"  Tokens: "
                f"in={record['input_tokens']}, "
                f"out={record['output_tokens']}"
            )

            print(
                f"  Cost: "
                f"{record['effective_cost_usd']} "
                f"({record['cost_source']})"
            )

            print(
                f"  Latency: "
                f"{record['latency']:.2f}s"
            )


            for field in (

                "api_error",
                "parse_error",
                "validation_error",
                "execution_error",
                "program_evaluation_error",

            ):

                if (
                    record[field]
                    is not None
                ):

                    print(
                        f"  {field}: "
                        f"{record[field]}"
                    )


            print()


    # -------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------

    total = len(
        examples
    )


    print(
        "----------------------------------------"
    )

    print(
        "QWEN3-30B FEW-SHOT PILOT SUMMARY"
    )

    print(
        "----------------------------------------"
    )


    print(
        f"Examples: "
        f"{total}"
    )


    print(
        f"Valid programs: "
        f"{valid_count}/{total}"
    )


    print(
        f"Valid rate: "
        f"{valid_count / total * 100:.1f}%"
    )


    print(
        f"Executable programs: "
        f"{executable_count}/{total}"
    )


    print(
        f"Executable rate: "
        f"{executable_count / total * 100:.1f}%"
    )


    print(
        f"Execution correct: "
        f"{execution_correct_count}/{total}"
    )


    print(
        f"Execution accuracy: "
        f"{execution_correct_count / total * 100:.1f}%"
    )


    print(
        f"Symbolically correct: "
        f"{symbolic_correct_count}/{total}"
    )


    print(
        f"Final program correct: "
        f"{program_correct_count}/{total}"
    )


    print(
        f"Program accuracy: "
        f"{program_correct_count / total * 100:.1f}%"
    )


    print()


    print(
        "Failure counts"
    )

    print(
        f"API errors: "
        f"{api_error_count}"
    )

    print(
        f"Parse errors: "
        f"{parse_error_count}"
    )

    print(
        f"Validation failures: "
        f"{validation_failure_count}"
    )

    print(
        f"Execution failures: "
        f"{execution_failure_count}"
    )


    print()


    print(
        f"Total input tokens: "
        f"{total_input_tokens}"
    )

    print(
        f"Average input tokens: "
        f"{total_input_tokens / total:.1f}"
    )


    print(
        f"Total output tokens: "
        f"{total_output_tokens}"
    )

    print(
        f"Average output tokens: "
        f"{total_output_tokens / total:.1f}"
    )


    print(
        f"Total latency: "
        f"{total_latency:.2f}s"
    )

    print(
        f"Average latency: "
        f"{total_latency / total:.2f}s"
    )


    if cost_complete:

        print(
            f"Total API cost: "
            f"${total_effective_cost:.6f}"
        )

        print(
            f"Average cost/question: "
            f"${total_effective_cost / total:.8f}"
        )

    else:

        print(
            "Total API cost: incomplete "
            "(one or more responses lacked cost information)"
        )


    print(
        f"Results saved to: "
        f"{results_path}"
    )


# ---------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()