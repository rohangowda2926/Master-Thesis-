"""Configuration A: Single-Agent FinQA baseline.

Architecture:

    FinQA question + financial evidence
        -> one Qwen3-30B agent
        -> FinQA program
        -> validator
        -> deterministic executor
        -> execution/program evaluation

Shared controls come from agents/common.py.

Run examples:

Offline check only:
    python -m agents.configuration_a --dry-run

Small API smoke test:
    python -m agents.configuration_a --start 70 --count 5

Larger development run:
    python -m agents.configuration_a --start 70 --count 20
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

from agents.common import (
    get_dataset_path,
    MODEL,
    RESULTS_DIR,
    TEMPERATURE,
    build_context,
    call_model,
    create_client,
    get_few_shot_examples,
    parse_json_response,
)

from evaluation.finqa_executor import execute_program
from evaluation.finqa_program_evaluator import compare_programs
from evaluation.finqa_validator import OPERATION_ARITY, validate_program


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

ARCHITECTURE = "single_agent"

CONFIGURATION = "A"

PROMPT_MODE = "fixed_4_shot_direct_program"


# ---------------------------------------------------------------------
# FROZEN CONFIGURATION-A PROMPT
# ---------------------------------------------------------------------

def build_prompt(
    question: str,
    context: str,
) -> str:
    """Build the frozen direct-program prompt for Configuration A."""

    allowed_operations = ", ".join(
        OPERATION_ARITY
    )

    few_shot_examples = (
        get_few_shot_examples()
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
characters, prefer using required numerical values directly.


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
3. Did I avoid labels as numerical operands?
4. Did I avoid nested operations?
5. Are # references sequential and valid?
6. Did I avoid assignment syntax?
7. Did I use table operations only when genuinely needed?
8. Is percentage conversion actually required?
9. Is the response exactly one JSON object containing only "program"?


============================================================
FINQA TRAINING DEMONSTRATIONS
============================================================

The following examples come only from the FinQA training split.

Use them to learn the required reasoning and program representation.

Do not copy numerical values from these demonstrations into the new problem.

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
# RUN ONE EXAMPLE
# ---------------------------------------------------------------------

def run_example(
    client,
    example,
    dataset_index,
    run_id,
    dataset_split,
):
    """Run Configuration A on one FinQA example."""

    started = time.perf_counter()

    question = example["qa"]["question"]

    context = build_context(
        example
    )

    prompt = build_prompt(
        question,
        context,
    )


    # -----------------------------------------------------------------
    # MODEL GENERATION
    # -----------------------------------------------------------------

    model_result = call_model(
        client,
        prompt,
    )


    generated_program = None

    parse_error = None

    validation = None

    execution = None


    # -----------------------------------------------------------------
    # PARSE MODEL OUTPUT
    # -----------------------------------------------------------------

    if model_result["api_error"] is None:

        parsed, parse_error = (
            parse_json_response(
                model_result["raw_response"],
                required_keys={"program"},
            )
        )

        if parse_error is None:

            generated_program = (
                parsed["program"]
            )


    # -----------------------------------------------------------------
    # VALIDATE
    # -----------------------------------------------------------------

    if generated_program is not None:

        validation = validate_program(
            generated_program
        )


    program_valid = bool(
        validation is not None
        and validation["valid"]
    )


    # -----------------------------------------------------------------
    # EXECUTE
    # -----------------------------------------------------------------

    if program_valid:

        execution = execute_program(
            generated_program,
            table=example.get("table"),
        )


    executable = bool(
        execution is not None
        and execution["executable"]
    )


    predicted_answer = (
        execution["result"]
        if executable
        else None
    )


    # -----------------------------------------------------------------
    # GOLD INFORMATION
    #
    # Accessed only after model generation.
    # -----------------------------------------------------------------

    gold_answer = (
        example["qa"]["exe_ans"]
    )

    gold_program = (
        example["qa"]["program"]
    )


    # -----------------------------------------------------------------
    # EXECUTION ACCURACY
    # -----------------------------------------------------------------

    execution_correct = bool(
        executable
        and predicted_answer
        == gold_answer
    )


    # -----------------------------------------------------------------
    # PROGRAM ACCURACY
    # -----------------------------------------------------------------

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


    # Official-style consistency:
    #
    # symbolic correctness alone is not enough;
    # the program must also execute successfully
    # to the correct gold result.

    program_correct = bool(
        program_symbolically_correct
        and executable
        and execution_correct
    )


    # -----------------------------------------------------------------
    # STRUCTURED ERRORS
    # -----------------------------------------------------------------

    validation_error = None

    if (
        validation is not None
        and not validation["valid"]
    ):

        validation_error = {
            "error_type":
                validation["error_type"],

            "error_message":
                validation["error_message"],
        }


    execution_error = None

    if (
        execution is not None
        and not execution["executable"]
    ):

        execution_error = {
            "error_type":
                execution["error_type"],

            "error_message":
                execution["error_message"],
        }


    program_evaluation_error = None

    if (
        program_evaluation[
            "error_type"
        ]
        is not None
    ):

        program_evaluation_error = {
            "error_type":
                program_evaluation[
                    "error_type"
                ],

            "error_message":
                program_evaluation[
                    "error_message"
                ],
        }


    # -----------------------------------------------------------------
    # RESULT RECORD
    # -----------------------------------------------------------------

    return {

        "example_id":
            example["id"],

        "dataset_index":
            dataset_index,

        "architecture":
            ARCHITECTURE,

        "configuration":
            CONFIGURATION,

        "prompt_mode":
            PROMPT_MODE,

        "run_id":
            run_id,

        "run_number":
            1,

        "dataset_split":
            dataset_split,

        "question":
            question,


        # -------------------------------------------------------------
        # MODEL
        # -------------------------------------------------------------

        "model":
            MODEL,

        "temperature":
            TEMPERATURE,


        # -------------------------------------------------------------
        # GENERATED PROGRAM
        # -------------------------------------------------------------

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


        # -------------------------------------------------------------
        # VALIDATION
        # -------------------------------------------------------------

        "program_valid":
            program_valid,

        "validation_error":
            validation_error,


        # -------------------------------------------------------------
        # EXECUTION
        # -------------------------------------------------------------

        "executable":
            executable,

        "execution_error":
            execution_error,

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


        # -------------------------------------------------------------
        # PROGRAM ACCURACY
        # -------------------------------------------------------------

        "program_symbolically_correct":
            program_symbolically_correct,

        "program_correct":
            program_correct,

        "program_evaluation_error":
            program_evaluation_error,

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


        # -------------------------------------------------------------
        # PARSING
        # -------------------------------------------------------------

        "parse_error":
            parse_error,


        # -------------------------------------------------------------
        # API / EFFICIENCY
        # -------------------------------------------------------------

        **model_result,


        # -------------------------------------------------------------
        # END-TO-END LATENCY
        # -------------------------------------------------------------

        "latency":
            time.perf_counter()
            - started,
    }


# ---------------------------------------------------------------------
# ARGUMENTS
# ---------------------------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Configuration A: "
            "single-agent FinQA baseline"
        )
    )


    parser.add_argument(
        "--split",
        choices=("dev", "test"),
        default="dev",
        help=(
            'Dataset split to run: "dev" or "test". '
            'Default: "dev".'
        ),
    )


    parser.add_argument(
        "--start",
        type=int,
        default=70,
        help=(
            "Starting index in dev.json. "
            "Default: 70"
        ),
    )


    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help=(
            "Number of examples to run. "
            "Default: 5"
        ),
    )


    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate configuration without "
            "making OpenRouter calls."
        ),
    )


    return parser.parse_args()


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    args = parse_args()


    dataset_path = get_dataset_path(
        args.split
    )


    if not dataset_path.exists():

        raise FileNotFoundError(
            f"Dataset file not found: {dataset_path}"
        )


    # -----------------------------------------------------------------
    # OFFLINE CHECK
    # -----------------------------------------------------------------

    if args.dry_run:

        print(
            "Configuration A dry run"
        )

        print(
            f"Architecture: "
            f"{ARCHITECTURE}"
        )

        print(
            f"Configuration: "
            f"{CONFIGURATION}"
        )

        print(
            f"Model: "
            f"{MODEL}"
        )

        print(
            f"Temperature: "
            f"{TEMPERATURE}"
        )

        print(
            f"Prompt mode: "
            f"{PROMPT_MODE}"
        )

        print(
            f"Requested dataset slice: "
            f"{args.split}[{args.start}:"
            f"{args.start + args.count}]"
        )


        print(
            f"Dataset file: "
            f"{dataset_path}"
        )

        demonstrations = (
            get_few_shot_examples()
        )

        if not demonstrations.strip():

            raise ValueError(
                "Few-shot examples are empty."
            )

        print(
            "Fixed training demonstrations: loaded"
        )

        print(
            "No API calls made."
        )

        return


    # -----------------------------------------------------------------
    # LOAD DATASET
    # -----------------------------------------------------------------

    with dataset_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        dataset = json.load(
            file
        )


    if args.start < 0:

        raise ValueError(
            "--start must be >= 0."
        )


    if args.count <= 0:

        raise ValueError(
            "--count must be > 0."
        )


    examples = dataset[
        args.start:
        args.start + args.count
    ]


    if len(examples) != args.count:

        raise ValueError(
            f"Requested {args.split} slice is outside "
            "the available dataset."
        )


    # -----------------------------------------------------------------
    # CLIENT
    # -----------------------------------------------------------------

    client = create_client()


    # -----------------------------------------------------------------
    # RUN ID
    # -----------------------------------------------------------------

    run_id = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%S_%fZ"
    )


    # -----------------------------------------------------------------
    # RESULT FILE
    # -----------------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    results_path = (
        RESULTS_DIR
        / (
            "configuration_A_"
            f"{args.split}_{args.start}_"
            f"{args.start + args.count}_"
            f"{run_id}.jsonl"
        )
    )


    # -----------------------------------------------------------------
    # AGGREGATES
    # -----------------------------------------------------------------

    valid_count = 0

    executable_count = 0

    execution_correct_count = 0

    symbolic_correct_count = 0

    program_correct_count = 0


    api_error_count = 0

    parse_error_count = 0

    validation_failure_count = 0

    execution_failure_count = 0


    total_input_tokens = 0

    total_output_tokens = 0

    total_latency = 0.0

    total_cost = 0.0

    cost_complete = True


    # -----------------------------------------------------------------
    # HEADER
    # -----------------------------------------------------------------

    print(
        "Configuration A — "
        "Single-Agent FinQA baseline"
    )

    print(
        f"Model: {MODEL}"
    )

    print(
        f"Temperature: "
        f"{TEMPERATURE}"
    )

    print(
        f"Prompt mode: "
        f"{PROMPT_MODE}"
    )

    print(
        f"Dataset slice: "
        f"{args.split}[{args.start}:"
        f"{args.start + args.count}]"
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


    # -----------------------------------------------------------------
    # RUN
    # -----------------------------------------------------------------

    with results_path.open(
        "x",
        encoding="utf-8",
    ) as results_file:


        for offset, example in enumerate(
            examples
        ):

            dataset_index = (
                args.start
                + offset
            )


            record = run_example(
                client,
                example,
                dataset_index,
                run_id,
                args.split,
            )


            # ---------------------------------------------------------
            # SAVE
            # ---------------------------------------------------------

            results_file.write(

                json.dumps(
                    record,
                    ensure_ascii=False,
                    allow_nan=False,
                )

                + "\n"
            )

            results_file.flush()


            # ---------------------------------------------------------
            # METRICS
            # ---------------------------------------------------------

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


            if record["api_error"] is not None:
                api_error_count += 1


            if record["parse_error"] is not None:
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


            total_latency += (
                record[
                    "latency"
                ]
            )


            if (
                record[
                    "effective_cost_usd"
                ]
                is not None
            ):

                total_cost += (
                    record[
                        "effective_cost_usd"
                    ]
                )

            else:

                cost_complete = False


            # ---------------------------------------------------------
            # TERMINAL OUTPUT
            # ---------------------------------------------------------

            print(
                f"[{offset + 1}/"
                f"{len(examples)}] "
                f"{args.split} index "
                f"{dataset_index} | "
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
                f"  Execution correct: "
                f"{record['execution_correct']}"
            )

            print(
                f"  Program correct: "
                f"{record['program_correct']}"
            )

            print(
                f"  Latency: "
                f"{record['latency']:.2f}s"
            )

            print(
                f"  Cost: "
                f"{record['effective_cost_usd']}"
            )


            if (
                record[
                    "api_error"
                ]
                is not None
            ):

                print(
                    f"  api_error: "
                    f"{record['api_error']}"
                )


            print()


            # ---------------------------------------------------------
            # STOP WHOLE RUN ON FATAL API FAILURE
            # ---------------------------------------------------------

            if record[
                "fatal_api_error"
            ]:

                print(
                    "Fatal API error detected. "
                    "Stopping experiment early."
                )

                break


    # -----------------------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------------------

    completed = (
        valid_count
        + (
            api_error_count
            if api_error_count
            else 0
        )
    )

    # Use number of records actually attempted.
    attempted = (
        offset + 1
        if examples
        else 0
    )


    print(
        "----------------------------------------"
    )

    print(
        "CONFIGURATION A SUMMARY"
    )

    print(
        "----------------------------------------"
    )


    print(
        f"Attempted examples: "
        f"{attempted}"
    )


    print(
        f"Valid programs: "
        f"{valid_count}/{attempted}"
    )


    print(
        f"Executable programs: "
        f"{executable_count}/{attempted}"
    )


    print(
        f"Execution correct: "
        f"{execution_correct_count}/{attempted}"
    )


    print(
        f"Execution accuracy: "
        f"{execution_correct_count / attempted * 100:.1f}%"
        if attempted
        else
        "Execution accuracy: N/A"
    )


    print(
        f"Symbolically correct: "
        f"{symbolic_correct_count}/{attempted}"
    )


    print(
        f"Program correct: "
        f"{program_correct_count}/{attempted}"
    )


    print(
        f"Program accuracy: "
        f"{program_correct_count / attempted * 100:.1f}%"
        if attempted
        else
        "Program accuracy: N/A"
    )


    print()


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
        f"Total output tokens: "
        f"{total_output_tokens}"
    )


    if attempted:

        print(
            f"Average latency: "
            f"{total_latency / attempted:.2f}s"
        )


    if cost_complete:

        print(
            f"Total API cost: "
            f"${total_cost:.6f}"
        )

        if attempted:

            print(
                f"Average cost/question: "
                f"${total_cost / attempted:.8f}"
            )

    else:

        print(
            "Total API cost: incomplete"
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