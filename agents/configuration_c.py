"""Configuration C: Reasoning + Independent Verification.

Architecture:

    Question + FinQA evidence
              |
              v
       Reasoning Agent
       (same stage as B)
              |
       candidate program
              |
              v
      Verification Agent
       + original evidence
              |
     independently solve
              |
     compare with candidate
              |
        final program
              |
              v
    validator -> executor -> evaluator

Important independence rule:

The Verification Agent does NOT receive the Reasoning Agent's plan.
It receives:
    - the original question
    - the original financial evidence
    - the fixed train demonstrations
    - the candidate program to check

It must independently derive its own program before deciding whether
to accept or replace the candidate.

Run:

Offline:
    python -m agents.configuration_c --dry-run

Paired pilot:
    python -m agents.configuration_c --start 70 --count 5
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

from agents.common import (
    DEV_PATH,
    MODEL,
    RESULTS_DIR,
    TEMPERATURE,
    build_context,
    call_model,
    create_client,
    get_few_shot_examples,
    parse_json_response,
)

# Configuration C deliberately reuses the exact same Reasoning-Agent
# prompt used in Configuration B.
from agents.configuration_b import (
    build_prompt as build_reasoning_prompt,
)

from evaluation.finqa_executor import execute_program
from evaluation.finqa_program_evaluator import compare_programs
from evaluation.finqa_validator import (
    OPERATION_ARITY,
    validate_program,
)


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

ARCHITECTURE = "reasoning_plus_verification"

CONFIGURATION = "C"

PROMPT_MODE = (
    "reasoning_agent_plus_independent_verification"
)


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def _sum_if_complete(*values):
    """Sum values only when every value is available."""

    if any(
        value is None
        for value in values
    ):
        return None

    return sum(values)


def validate_and_execute(
    program,
    table,
):
    """Validate and deterministically execute one program."""

    validation = None

    execution = None


    if isinstance(
        program,
        str,
    ):

        validation = validate_program(
            program
        )


        if validation["valid"]:

            execution = execute_program(
                program,
                table=table,
            )


    program_valid = bool(
        validation is not None
        and validation["valid"]
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


    return {
        "validation":
            validation,

        "execution":
            execution,

        "program_valid":
            program_valid,

        "executable":
            executable,

        "predicted_answer":
            predicted_answer,

        "validation_error":
            validation_error,

        "execution_error":
            execution_error,
    }


# ---------------------------------------------------------------------
# VERIFICATION PROMPT
# ---------------------------------------------------------------------

def build_verification_prompt(
    question,
    context,
    candidate_program,
):
    """Build the independent Verification-Agent prompt."""

    allowed_operations = ", ".join(
        OPERATION_ARITY
    )

    few_shot_examples = (
        get_few_shot_examples()
    )


    candidate_text = (
        candidate_program
        if isinstance(
            candidate_program,
            str,
        )
        else "<candidate unavailable>"
    )


    return f"""
You are the independent Verification Agent in a FinQA numerical
reasoning system.

Your responsibility is to verify a candidate FinQA program using the
ORIGINAL financial evidence.

You must NOT assume that the candidate is correct.

Follow this order:

1. Independently solve the question from the original evidence.
2. Identify the relevant numerical values.
3. Determine the required arithmetic relationship.
4. Decide whether percentage scaling is actually required.
5. Produce your own independent FinQA program.
6. Only then compare your independent solution with the candidate.
7. Return the final verified FinQA program.

You do NOT receive the Reasoning Agent's reasoning plan.

Return exactly one JSON object with exactly these four keys:

{{
  "independent_plan": "<short independent reasoning>",
  "independent_program": "<your independently derived FinQA program>",
  "candidate_verdict": "agree",
  "final_program": "<verified final FinQA program>"
}}

candidate_verdict must be exactly:

"agree"

or

"disagree"

If the candidate is wrong:

- candidate_verdict must be "disagree"
- final_program must contain the corrected program

If the candidate agrees with your independently derived solution:

- candidate_verdict must be "agree"
- final_program may use the verified program

Return nothing outside the JSON object.


============================================================
INDEPENDENCE REQUIREMENT
============================================================

Do not copy the candidate merely because it looks plausible.

First determine the solution yourself from:

- the question
- the financial evidence

Your independent_program must represent the calculation you independently
believe is correct.


============================================================
ALLOWED FINQA OPERATIONS
============================================================

{allowed_operations}


============================================================
VALID OPERANDS
============================================================

Arithmetic operands must be:

1. numerical literals

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

Do not use descriptive labels as arithmetic operands.


============================================================
SEQUENTIAL PROGRAM RULE
============================================================

Correct:

divide(60, 243), multiply(#0, const_100)

Incorrect:

multiply(divide(60, 243), const_100)

Never nest operations.


============================================================
REFERENCE RULE
============================================================

At step 0:
no # reference exists.

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

When required numerical values are directly visible in a table,
prefer the numerical values directly.

Do not use row or column labels as arithmetic operands.


============================================================
TABLE AGGREGATION
============================================================

Use:

table_sum
table_average
table_max
table_min

ONLY when genuine aggregation over a numeric row is required.

Rules:

- first argument = exact raw table row label
- second argument = none
- do not quote row labels
- do not use aggregation simply to retrieve one table cell


============================================================
PERCENTAGE AND RATIO CHECK
============================================================

This is especially important.

Do NOT automatically multiply by 100.

Do NOT automatically avoid multiplying by 100.

Determine from the question and FinQA representation whether the desired
result is:

- a decimal ratio, or
- a percentage-scaled value.

Example requiring scaling:

divide(60, 243), multiply(#0, const_100)

Example not requiring scaling:

divide(59.1, 98.0)


============================================================
VERIFICATION CHECKLIST
============================================================

Before returning the final program verify:

1. Are the selected numerical values actually relevant?
2. Is subtraction direction correct?
3. Is the denominator correct?
4. Is percentage scaling justified?
5. Are all operations allowed?
6. Are all # references valid?
7. Is there no operation nesting?
8. Does your independent program answer the actual question?
9. Does candidate_verdict accurately reflect the comparison?
10. Is final_program the program you believe should be executed?


============================================================
FINQA TRAINING DEMONSTRATIONS
============================================================

These examples come only from the FinQA training split.

Use them to understand FinQA numerical reasoning and program syntax.

Do not copy their numerical values into the current problem.

{few_shot_examples}


============================================================
QUESTION
============================================================

{question}


============================================================
ORIGINAL FINANCIAL EVIDENCE
============================================================

{context}


============================================================
CANDIDATE PROGRAM TO VERIFY
============================================================

{candidate_text}
""".strip()


# ---------------------------------------------------------------------
# REASONING OUTPUT PARSER
# ---------------------------------------------------------------------

def parse_reasoning_output(
    model_result,
):

    plan = None

    program = None

    parse_error = None


    if model_result["api_error"] is not None:

        return (
            plan,
            program,
            parse_error,
        )


    parsed, parse_error = (
        parse_json_response(
            model_result[
                "raw_response"
            ],
            required_keys={
                "plan",
                "program",
            },
        )
    )


    if parse_error is not None:

        return (
            plan,
            program,
            parse_error,
        )


    if not isinstance(
        parsed["plan"],
        str,
    ):

        return (
            None,
            None,
            {
                "error_type":
                    "invalid_plan_type",

                "error_message":
                    '"plan" must be a string.',
            },
        )


    if not isinstance(
        parsed["program"],
        str,
    ):

        return (
            None,
            None,
            {
                "error_type":
                    "invalid_program_type",

                "error_message":
                    '"program" must be a string.',
            },
        )


    return (
        parsed["plan"],
        parsed["program"],
        None,
    )


# ---------------------------------------------------------------------
# VERIFICATION OUTPUT PARSER
# ---------------------------------------------------------------------

def parse_verification_output(
    model_result,
):

    independent_plan = None

    independent_program = None

    candidate_verdict = None

    final_program = None

    parse_error = None


    if model_result["api_error"] is not None:

        return (
            independent_plan,
            independent_program,
            candidate_verdict,
            final_program,
            parse_error,
        )


    parsed, parse_error = (
        parse_json_response(
            model_result[
                "raw_response"
            ],
            required_keys={
                "independent_plan",
                "independent_program",
                "candidate_verdict",
                "final_program",
            },
        )
    )


    if parse_error is not None:

        return (
            independent_plan,
            independent_program,
            candidate_verdict,
            final_program,
            parse_error,
        )


    for key in (
        "independent_plan",
        "independent_program",
        "candidate_verdict",
        "final_program",
    ):

        if not isinstance(
            parsed[key],
            str,
        ):

            return (
                None,
                None,
                None,
                None,
                {
                    "error_type":
                        "invalid_verification_field_type",

                    "error_message":
                        f'"{key}" must be a string.',
                },
            )


    candidate_verdict = (
        parsed[
            "candidate_verdict"
        ]
        .strip()
        .lower()
    )


    if candidate_verdict not in {
        "agree",
        "disagree",
    }:

        return (
            None,
            None,
            None,
            None,
            {
                "error_type":
                    "invalid_candidate_verdict",

                "error_message":
                    (
                        'candidate_verdict must be '
                        '"agree" or "disagree".'
                    ),
            },
        )


    return (
        parsed[
            "independent_plan"
        ],

        parsed[
            "independent_program"
        ],

        candidate_verdict,

        parsed[
            "final_program"
        ],

        None,
    )


# ---------------------------------------------------------------------
# RUN ONE EXAMPLE
# ---------------------------------------------------------------------

def run_example(
    client,
    example,
    dataset_index,
    run_id,
):

    started = time.perf_counter()

    question = (
        example["qa"]["question"]
    )

    context = build_context(
        example
    )

    table = example.get(
        "table"
    )


    # =============================================================
    # STAGE 1 — REASONING AGENT
    # =============================================================

    reasoning_prompt = (
        build_reasoning_prompt(
            question,
            context,
        )
    )


    reasoning_result = call_model(
        client,
        reasoning_prompt,
    )


    (
        reasoning_plan,
        reasoning_program,
        reasoning_parse_error,
    ) = parse_reasoning_output(
        reasoning_result
    )


    reasoning_pipeline = (
        validate_and_execute(
            reasoning_program,
            table,
        )
    )


    # =============================================================
    # STAGE 2 — VERIFICATION AGENT
    # =============================================================

    verification_result = {
        "raw_response":
            None,

        "finish_reason":
            None,

        "api_latency":
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

        "effective_cost_usd":
            None,

        "cost_source":
            None,

        "api_error":
            None,

        "fatal_api_error":
            False,

        "retry_count":
            0,

        "attempt_count":
            0,

        "attempts":
            [],
    }


    independent_plan = None

    independent_program = None

    candidate_verdict = None

    final_program = None

    verification_parse_error = None


    # A fatal reasoning API error normally indicates that another
    # immediate request will also fail, so do not waste another call.
    if not reasoning_result[
        "fatal_api_error"
    ]:

        verification_prompt = (
            build_verification_prompt(
                question,
                context,
                reasoning_program,
            )
        )


        verification_result = call_model(
            client,
            verification_prompt,
        )


        (
            independent_plan,
            independent_program,
            candidate_verdict,
            final_program,
            verification_parse_error,
        ) = parse_verification_output(
            verification_result
        )


    # -----------------------------------------------------------------
    # INDEPENDENT VERIFIER PROGRAM
    # -----------------------------------------------------------------

    independent_pipeline = (
        validate_and_execute(
            independent_program,
            table,
        )
    )


    # -----------------------------------------------------------------
    # FINAL VERIFIED PROGRAM
    # -----------------------------------------------------------------

    final_pipeline = (
        validate_and_execute(
            final_program,
            table,
        )
    )


    # =============================================================
    # GOLD — accessed only after model generation
    # =============================================================

    gold_answer = (
        example["qa"]["exe_ans"]
    )

    gold_program = (
        example["qa"]["program"]
    )


    # =============================================================
    # STAGE-LEVEL ACCURACY
    # =============================================================

    reasoning_execution_correct = bool(
        reasoning_pipeline[
            "executable"
        ]
        and
        reasoning_pipeline[
            "predicted_answer"
        ]
        == gold_answer
    )


    independent_execution_correct = bool(
        independent_pipeline[
            "executable"
        ]
        and
        independent_pipeline[
            "predicted_answer"
        ]
        == gold_answer
    )


    final_execution_correct = bool(
        final_pipeline[
            "executable"
        ]
        and
        final_pipeline[
            "predicted_answer"
        ]
        == gold_answer
    )


    # =============================================================
    # FINAL PROGRAM ACCURACY
    # =============================================================

    final_program_evaluation = (
        compare_programs(
            gold_program,
            final_program,
        )
    )


    final_symbolically_correct = bool(
        final_program_evaluation[
            "program_correct"
        ]
    )


    final_program_correct = bool(
        final_symbolically_correct
        and
        final_pipeline[
            "executable"
        ]
        and
        final_execution_correct
    )


    # =============================================================
    # CORRECTION BEHAVIOUR
    # =============================================================

    verifier_corrected_error = bool(
        not reasoning_execution_correct
        and final_execution_correct
    )


    verifier_broke_correct_answer = bool(
        reasoning_execution_correct
        and not final_execution_correct
    )


    # =============================================================
    # TOTAL EFFICIENCY
    # =============================================================

    combined_input_tokens = (
        _sum_if_complete(
            reasoning_result[
                "input_tokens"
            ],
            verification_result[
                "input_tokens"
            ],
        )
    )


    combined_output_tokens = (
        _sum_if_complete(
            reasoning_result[
                "output_tokens"
            ],
            verification_result[
                "output_tokens"
            ],
        )
    )


    combined_total_tokens = (
        _sum_if_complete(
            reasoning_result[
                "total_tokens"
            ],
            verification_result[
                "total_tokens"
            ],
        )
    )


    combined_cost = (
        _sum_if_complete(
            reasoning_result[
                "effective_cost_usd"
            ],
            verification_result[
                "effective_cost_usd"
            ],
        )
    )


    fatal_api_error = bool(
        reasoning_result[
            "fatal_api_error"
        ]
        or
        verification_result[
            "fatal_api_error"
        ]
    )


    # Overall API error, while preserving stage-specific errors.
    overall_api_error = (
        verification_result[
            "api_error"
        ]
        if verification_result[
            "api_error"
        ]
        is not None
        else reasoning_result[
            "api_error"
        ]
    )


    # =============================================================
    # RECORD
    # =============================================================

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
            "dev",

        "question":
            question,

        "model":
            MODEL,

        "temperature":
            TEMPERATURE,


        # ---------------------------------------------------------
        # REASONING AGENT
        # ---------------------------------------------------------

        "reasoning_plan":
            reasoning_plan,

        "reasoning_program":
            reasoning_program,

        "reasoning_parse_error":
            reasoning_parse_error,

        "reasoning_program_valid":
            reasoning_pipeline[
                "program_valid"
            ],

        "reasoning_validation_error":
            reasoning_pipeline[
                "validation_error"
            ],

        "reasoning_executable":
            reasoning_pipeline[
                "executable"
            ],

        "reasoning_execution_error":
            reasoning_pipeline[
                "execution_error"
            ],

        "reasoning_predicted_answer":
            reasoning_pipeline[
                "predicted_answer"
            ],

        "reasoning_execution_correct":
            reasoning_execution_correct,

        "reasoning_api":
            reasoning_result,


        # ---------------------------------------------------------
        # VERIFICATION AGENT — INDEPENDENT SOLUTION
        # ---------------------------------------------------------

        "verification_independent_plan":
            independent_plan,

        "verification_independent_program":
            independent_program,

        "candidate_verdict":
            candidate_verdict,

        "verification_parse_error":
            verification_parse_error,

        "verification_independent_program_valid":
            independent_pipeline[
                "program_valid"
            ],

        "verification_independent_validation_error":
            independent_pipeline[
                "validation_error"
            ],

        "verification_independent_executable":
            independent_pipeline[
                "executable"
            ],

        "verification_independent_execution_error":
            independent_pipeline[
                "execution_error"
            ],

        "verification_independent_predicted_answer":
            independent_pipeline[
                "predicted_answer"
            ],

        "verification_independent_execution_correct":
            independent_execution_correct,

        "verification_api":
            verification_result,


        # ---------------------------------------------------------
        # FINAL VERIFIED OUTPUT
        # ---------------------------------------------------------

        "generated_program":
            final_program,

        "normalized_program":
            (
                final_pipeline[
                    "validation"
                ][
                    "normalized_program"
                ]
                if final_pipeline[
                    "validation"
                ]
                is not None
                else None
            ),

        "gold_program":
            gold_program,

        "program_valid":
            final_pipeline[
                "program_valid"
            ],

        "validation_error":
            final_pipeline[
                "validation_error"
            ],

        "executable":
            final_pipeline[
                "executable"
            ],

        "execution_error":
            final_pipeline[
                "execution_error"
            ],

        "intermediate_results":
            (
                final_pipeline[
                    "execution"
                ][
                    "intermediate_results"
                ]
                if final_pipeline[
                    "execution"
                ]
                is not None
                else []
            ),

        "predicted_answer":
            final_pipeline[
                "predicted_answer"
            ],

        "gold_answer":
            gold_answer,

        "execution_correct":
            final_execution_correct,

        "program_symbolically_correct":
            final_symbolically_correct,

        "program_correct":
            final_program_correct,

        "program_evaluation_error":
            (
                {
                    "error_type":
                        final_program_evaluation[
                            "error_type"
                        ],

                    "error_message":
                        final_program_evaluation[
                            "error_message"
                        ],
                }
                if final_program_evaluation[
                    "error_type"
                ]
                is not None
                else None
            ),

        "gold_program_tokens":
            final_program_evaluation[
                "gold_tokens"
            ],

        "predicted_program_tokens":
            final_program_evaluation[
                "predicted_tokens"
            ],

        "gold_symbolic_program":
            final_program_evaluation[
                "gold_symbolic"
            ],

        "predicted_symbolic_program":
            final_program_evaluation[
                "predicted_symbolic"
            ],


        # ---------------------------------------------------------
        # VERIFICATION EFFECT
        # ---------------------------------------------------------

        "verifier_corrected_error":
            verifier_corrected_error,

        "verifier_broke_correct_answer":
            verifier_broke_correct_answer,


        # ---------------------------------------------------------
        # AGGREGATED EFFICIENCY
        # ---------------------------------------------------------

        "input_tokens":
            combined_input_tokens,

        "output_tokens":
            combined_output_tokens,

        "total_tokens":
            combined_total_tokens,

        "effective_cost_usd":
            combined_cost,

        "api_error":
            overall_api_error,

        "fatal_api_error":
            fatal_api_error,

        "model_call_count":
            (
                reasoning_result[
                    "attempt_count"
                ]
                +
                verification_result[
                    "attempt_count"
                ]
            ),

        "latency":
            (
                time.perf_counter()
                - started
            ),
    }


# ---------------------------------------------------------------------
# ARGUMENTS
# ---------------------------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Configuration C: "
            "Reasoning + Independent Verification"
        )
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


    # =============================================================
    # DRY RUN
    # =============================================================

    if args.dry_run:

        print(
            "Configuration C dry run"
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
            f"Requested dev slice: "
            f"dev[{args.start}:"
            f"{args.start + args.count}]"
        )

        print(
            "Reasoning stage: "
            "identical to Configuration B"
        )

        print(
            "Verification stage: "
            "independent solve + candidate check"
        )

        print(
            "Reasoning plan is NOT passed "
            "to Verification Agent"
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
            "Expected model calls/question: 2"
        )

        print(
            "No API calls made."
        )

        return


    # =============================================================
    # DATA
    # =============================================================

    with DEV_PATH.open(
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
            "Requested dev slice is outside dataset."
        )


    # =============================================================
    # CLIENT / RUN
    # =============================================================

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
            "configuration_C_"
            f"dev_{args.start}_"
            f"{args.start + args.count}_"
            f"{run_id}.jsonl"
        )
    )


    # =============================================================
    # COUNTERS
    # =============================================================

    attempted = 0

    valid_count = 0

    executable_count = 0

    execution_correct_count = 0

    symbolic_correct_count = 0

    program_correct_count = 0


    reasoning_correct_count = 0

    verifier_independent_correct_count = 0

    corrected_error_count = 0

    broke_correct_count = 0


    api_error_count = 0


    total_input_tokens = 0

    total_output_tokens = 0

    total_latency = 0.0

    total_cost = 0.0

    cost_complete = True


    # =============================================================
    # HEADER
    # =============================================================

    print(
        "Configuration C — "
        "Reasoning + Independent Verification"
    )

    print(
        f"Model: {MODEL}"
    )

    print(
        f"Temperature: "
        f"{TEMPERATURE}"
    )

    print(
        f"Dataset slice: "
        f"dev[{args.start}:"
        f"{args.start + args.count}]"
    )

    print(
        f"Examples: "
        f"{len(examples)}"
    )

    print(
        "Model calls/question: 2"
    )

    print(
        f"Results: "
        f"{results_path}"
    )

    print()


    # =============================================================
    # RUN
    # =============================================================

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


            # -----------------------------------------------------
            # METRICS
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


            reasoning_correct_count += int(
                record[
                    "reasoning_execution_correct"
                ]
            )


            verifier_independent_correct_count += int(
                record[
                    "verification_independent_execution_correct"
                ]
            )


            corrected_error_count += int(
                record[
                    "verifier_corrected_error"
                ]
            )


            broke_correct_count += int(
                record[
                    "verifier_broke_correct_answer"
                ]
            )


            if record[
                "api_error"
            ] is not None:

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


            # -----------------------------------------------------
            # TERMINAL
            # -----------------------------------------------------

            print(
                f"[{attempted}/"
                f"{len(examples)}] "
                f"dev index "
                f"{dataset_index} | "
                f"{record['example_id']}"
            )


            print(
                "  Reasoning candidate: "
                f"{record['reasoning_program']}"
            )


            print(
                "  Independent verifier: "
                f"{record['verification_independent_program']}"
            )


            print(
                "  Verdict: "
                f"{record['candidate_verdict']}"
            )


            print(
                "  Final program: "
                f"{record['generated_program']}"
            )


            print(
                "  Reasoning correct: "
                f"{record['reasoning_execution_correct']}"
            )


            print(
                "  Independent verifier correct: "
                f"{record['verification_independent_execution_correct']}"
            )


            print(
                "  Final execution correct: "
                f"{record['execution_correct']}"
            )


            print(
                "  Final program correct: "
                f"{record['program_correct']}"
            )


            print(
                "  Verifier corrected error: "
                f"{record['verifier_corrected_error']}"
            )


            print(
                "  Verifier broke correct answer: "
                f"{record['verifier_broke_correct_answer']}"
            )


            print(
                f"  Latency: "
                f"{record['latency']:.2f}s"
            )


            print(
                f"  Cost: "
                f"{record['effective_cost_usd']}"
            )


            if record[
                "api_error"
            ] is not None:

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


    # =============================================================
    # SUMMARY
    # =============================================================

    print(
        "----------------------------------------"
    )

    print(
        "CONFIGURATION C SUMMARY"
    )

    print(
        "----------------------------------------"
    )


    print(
        f"Attempted examples: "
        f"{attempted}"
    )


    print(
        f"Reasoning candidate correct: "
        f"{reasoning_correct_count}/{attempted}"
    )


    print(
        f"Independent verifier correct: "
        f"{verifier_independent_correct_count}/{attempted}"
    )


    print()


    print(
        f"Final valid programs: "
        f"{valid_count}/{attempted}"
    )


    print(
        f"Final executable programs: "
        f"{executable_count}/{attempted}"
    )


    print(
        f"Final execution correct: "
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
        f"Final symbolically correct: "
        f"{symbolic_correct_count}/{attempted}"
    )


    print(
        f"Final program correct: "
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
        f"Verifier corrected reasoning errors: "
        f"{corrected_error_count}"
    )


    print(
        f"Verifier broke correct reasoning answers: "
        f"{broke_correct_count}"
    )


    print(
        f"API errors: "
        f"{api_error_count}"
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