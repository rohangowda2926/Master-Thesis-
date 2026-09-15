"""Configuration B: Reasoning-Agent FinQA system.

Architecture:

    FinQA question + financial evidence
        -> one Reasoning Agent
        -> short explicit reasoning plan
        -> FinQA program
        -> validator
        -> deterministic executor
        -> execution/program evaluation

The plan is logged for analysis but is NOT used directly for scoring.

Shared experimental controls come from agents/common.py.

Run:

Offline check:
    python -m agents.configuration_b --dry-run

Paired pilot on same questions as Configuration A:
    python -m agents.configuration_b --start 70 --count 5
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

ARCHITECTURE = "reasoning_agent"

CONFIGURATION = "B"

PROMPT_MODE = "fixed_4_shot_plan_then_program"


# ---------------------------------------------------------------------
# PROMPT
# ---------------------------------------------------------------------

def build_prompt(
    question: str,
    context: str,
) -> str:
    """Build Configuration-B reasoning-agent prompt."""

    allowed_operations = ", ".join(
        OPERATION_ARITY
    )

    few_shot_examples = (
        get_few_shot_examples()
    )

    return f"""
You are the Reasoning Agent in a FinQA numerical reasoning system.

Your task has two stages:

1. Create a SHORT reasoning plan identifying:
   - the relevant numerical values,
   - the required arithmetic relationship,
   - whether percentage scaling is actually required.

2. Produce the valid sequential FinQA program that implements that plan.

Python will execute the program deterministically.

Return exactly one JSON object containing exactly these two keys:

{{
  "plan": "<short reasoning plan>",
  "program": "<FinQA program>"
}}

Return nothing else.

Do NOT output:
- Markdown
- code fences
- comments
- final numerical answer
- additional JSON keys
- text before or after the JSON object


============================================================
IMPORTANT PLAN RULE
============================================================

The plan must be concise.

It should explain which values and arithmetic relationship are needed.

Example:

"Use 60 and 243. Divide 60 by 243, then multiply by 100 because the question explicitly asks for a percentage."

Another example:

"Use 59.1 and 98.0. Divide 59.1 by 98.0. Do not multiply by 100 because the FinQA answer is expressed as a decimal ratio."


============================================================
ALLOWED FINQA OPERATIONS
============================================================

{allowed_operations}


============================================================
ARITHMETIC OPERANDS
============================================================

Prefer explicit numerical values directly from the supplied evidence.

Arithmetic operands must be:

1. numeric literals

Examples:

250
193.5
0.25


2. FinQA constants

Examples:

const_1
const_2
const_5
const_100
const_m1


3. previous operation references

Examples:

#0
#1
#2


Do NOT use labels such as:

Canada
revenue
sales
2020 revenue

as numerical operands.


============================================================
SEQUENTIAL PROGRAMS
============================================================

Correct:

divide(60, 243), multiply(#0, const_100)

Incorrect:

multiply(divide(60, 243), const_100)

Never nest operations.


============================================================
REFERENCES
============================================================

At operation 0:
no # reference exists.

At operation 1:
#0 may be used.

At operation 2:
#0 and #1 may be used.

Never reference future operations.


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

If the needed numerical values are directly visible in the table,
prefer those values directly.

Example:

Country | Sales
Canada  | 120
USA     | 200

Question:
What is the difference between USA and Canada sales?

Correct program:

subtract(200, 120)

Incorrect:

subtract(usa, canada)


============================================================
TABLE OPERATIONS
============================================================

Use:

table_sum
table_average
table_max
table_min

ONLY when genuine aggregation over an entire numeric row is required.

Rules:

- first argument = exact raw table row label
- second argument = none
- do not quote the row label
- do not use table operations simply to retrieve one cell
- do not invent row labels


============================================================
PERCENTAGE / RATIO DECISION
============================================================

Pay particular attention to scaling.

Do NOT automatically multiply by 100.

Do NOT automatically avoid multiplying by 100.

Before writing the program, determine whether the requested FinQA result
is a decimal ratio or a percentage-scaled value.

Example:

60 / 243 × 100

Program:

divide(60, 243), multiply(#0, const_100)

But:

59.1 / 98.0

Program:

divide(59.1, 98.0)


============================================================
FINAL SELF-CHECK
============================================================

Before returning the JSON:

1. Are the selected numbers actually relevant to the question?
2. Is subtraction direction correct?
3. Is the denominator correct?
4. Is multiplying by 100 really necessary?
5. Are all FinQA operations allowed?
6. Is the program sequential with valid # references?
7. Are there no nested operations?
8. Is the JSON exactly:
   plan + program?


============================================================
FINQA TRAINING DEMONSTRATIONS
============================================================

The following examples come only from the FinQA training split.

Use them to understand FinQA numerical reasoning and program format.

Do not copy their numerical values into the current problem.

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
    # MODEL
    # -----------------------------------------------------------------

    model_result = call_model(
        client,
        prompt,
    )


    reasoning_plan = None

    generated_program = None

    parse_error = None

    validation = None

    execution = None


    # -----------------------------------------------------------------
    # PARSE
    # -----------------------------------------------------------------

    if model_result["api_error"] is None:

        parsed, parse_error = (
            parse_json_response(
                model_result["raw_response"],
                required_keys={
                    "plan",
                    "program",
                },
            )
        )


        if parse_error is None:

            reasoning_plan = (
                parsed["plan"]
            )

            generated_program = (
                parsed["program"]
            )


            if not isinstance(
                reasoning_plan,
                str,
            ):

                parse_error = {
                    "error_type":
                        "invalid_plan_type",

                    "error_message":
                        '"plan" must be a string.',
                }

                reasoning_plan = None

                generated_program = None


            elif not isinstance(
                generated_program,
                str,
            ):

                parse_error = {
                    "error_type":
                        "invalid_program_type",

                    "error_message":
                        '"program" must be a string.',
                }

                generated_program = None


    # -----------------------------------------------------------------
    # VALIDATION
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
    # EXECUTION
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
    # GOLD
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


    program_correct = bool(
        program_symbolically_correct
        and executable
        and execution_correct
    )


    # -----------------------------------------------------------------
    # ERRORS
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
    # RECORD
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


        # Reasoning

        "reasoning_plan":
            reasoning_plan,


        # Model

        "model":
            MODEL,

        "temperature":
            TEMPERATURE,


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

        "validation_error":
            validation_error,


        # Execution

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


        # Program accuracy

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


        # Parsing

        "parse_error":
            parse_error,


        # API / efficiency

        **model_result,


        # End-to-end latency

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
            "Configuration B: "
            "reasoning-agent FinQA system"
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
    )


    parser.add_argument(
        "--count",
        type=int,
        default=5,
    )


    parser.add_argument(
        "--dry-run",
        action="store_true",
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
    # DRY RUN
    # -----------------------------------------------------------------

    if args.dry_run:

        print(
            "Configuration B dry run"
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
            "Output schema: plan + program"
        )

        print(
            "No API calls made."
        )

        return


    # -----------------------------------------------------------------
    # DATASET
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
            f"Requested {args.split} slice is outside dataset."
        )


    # -----------------------------------------------------------------
    # CLIENT / RUN
    # -----------------------------------------------------------------

    client = create_client()


    run_id = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%S_%fZ"
    )


    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    results_path = (
        RESULTS_DIR
        / (
            "configuration_B_"
            f"{args.split}_{args.start}_"
            f"{args.start + args.count}_"
            f"{run_id}.jsonl"
        )
    )


    # -----------------------------------------------------------------
    # COUNTERS
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
        "Configuration B — "
        "Reasoning Agent"
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

    attempted = 0


    with results_path.open(
        "x",
        encoding="utf-8",
    ) as results_file:


        for offset, example in enumerate(
            examples
        ):

            attempted += 1

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


            results_file.write(

                json.dumps(
                    record,
                    ensure_ascii=False,
                    allow_nan=False,
                )

                + "\n"
            )

            results_file.flush()


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
                record["api_error"]
                is not None
            ):

                api_error_count += 1


            if (
                record["parse_error"]
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


            print(
                f"[{attempted}/"
                f"{len(examples)}] "
                f"{args.split} index "
                f"{dataset_index} | "
                f"{record['example_id']}"
            )


            print(
                f"  Plan: "
                f"{record['reasoning_plan']}"
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

    print(
        "----------------------------------------"
    )

    print(
        "CONFIGURATION B SUMMARY"
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