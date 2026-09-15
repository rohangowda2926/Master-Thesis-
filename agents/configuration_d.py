"""Configuration D: Full Multi-Agent FinQA architecture.

Architecture:

    Original question + evidence
              |
              v
       Reasoning Agent
              |
        candidate program
              |
              v
    Independent Verification Agent
       + original question/evidence
              |
       independent program
       + verification decision
              |
              v
          Coordinator
       + original evidence
       + candidate outputs
              |
              v
          final program
              |
              v
    validator -> executor -> evaluator

All agents use:

    qwen/qwen3-30b-a3b-instruct-2507
    temperature = 0

Run:

Offline:
    python -m agents.configuration_d --dry-run

Paired pilot:
    python -m agents.configuration_d --start 70 --count 5
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

from agents.configuration_b import (
    build_prompt as build_reasoning_prompt,
)

from agents.configuration_c import (
    build_verification_prompt,
    parse_reasoning_output,
    parse_verification_output,
    validate_and_execute,
)

from evaluation.finqa_program_evaluator import (
    compare_programs,
)

from evaluation.finqa_validator import (
    OPERATION_ARITY,
)


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

ARCHITECTURE = (
    "reasoning_verification_coordinator"
)

CONFIGURATION = "D"

PROMPT_MODE = (
    "reasoning_plus_verification_plus_coordinator"
)


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def sum_if_complete(*values):

    if any(
        value is None
        for value in values
    ):

        return None

    return sum(values)


def empty_api_result():

    return {
        "raw_response": None,
        "finish_reason": None,
        "api_latency": None,
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "actual_cost_usd": None,
        "estimated_cost_usd": None,
        "effective_cost_usd": None,
        "cost_source": None,
        "api_error": None,
        "fatal_api_error": False,
        "retry_count": 0,
        "attempt_count": 0,
        "attempts": [],
    }


# ---------------------------------------------------------------------
# COORDINATOR PROMPT
# ---------------------------------------------------------------------

def build_coordinator_prompt(
    question,
    context,
    reasoning_program,
    independent_program,
    verifier_verdict,
    verifier_final_program,
):

    allowed_operations = ", ".join(
        OPERATION_ARITY
    )

    demonstrations = (
        get_few_shot_examples()
    )


    reasoning_program = (
        reasoning_program
        if isinstance(
            reasoning_program,
            str,
        )
        else "<unavailable>"
    )


    independent_program = (
        independent_program
        if isinstance(
            independent_program,
            str,
        )
        else "<unavailable>"
    )


    verifier_final_program = (
        verifier_final_program
        if isinstance(
            verifier_final_program,
            str,
        )
        else "<unavailable>"
    )


    verifier_verdict = (
        verifier_verdict
        if isinstance(
            verifier_verdict,
            str,
        )
        else "<unavailable>"
    )


    return f"""
You are the Coordinator in a multi-agent FinQA system.

Two preceding agents have attempted the same financial numerical
reasoning problem.

Your job is NOT to automatically trust either agent.

You must use the ORIGINAL question and financial evidence to decide
which proposed calculation is correct.

You receive:

1. the Reasoning Agent's candidate program,
2. the Verification Agent's independently derived program,
3. the Verification Agent's verdict,
4. the Verification Agent's proposed verified program.

Review these against the original evidence.

You may:

- select the Reasoning Agent program,
- select the Verification Agent solution,
- or produce a corrected FinQA program if both are wrong.

Return exactly one JSON object with exactly these keys:

{{
  "decision": "<short explanation of which calculation is correct>",
  "final_program": "<final FinQA program>"
}}

Return nothing else.

Do NOT output:

- Markdown
- code fences
- final numerical answer
- additional JSON keys
- text before or after the JSON


============================================================
COORDINATOR RESPONSIBILITY
============================================================

Do not use majority voting blindly.

The same underlying model is used for all agents, so multiple agents
can make the same mistake.

Check the financial evidence yourself.

Pay particular attention to:

- selecting the correct values,
- subtraction direction,
- denominator choice,
- whether percentage scaling is required,
- unit conversion,
- whether an unnecessary multiply by 100 has been added.


============================================================
ALLOWED FINQA OPERATIONS
============================================================

{allowed_operations}


============================================================
VALID OPERANDS
============================================================

Arithmetic operands must be:

1. numerical literals,

2. FinQA constants such as:

const_1
const_2
const_5
const_100
const_m1

3. previous operation references:

#0
#1
#2


============================================================
SEQUENTIAL FORMAT
============================================================

Correct:

divide(60, 243), multiply(#0, const_100)

Incorrect:

multiply(divide(60, 243), const_100)

Never nest operations.


============================================================
REFERENCE RULE
============================================================

At operation 0:

no # reference exists.

At operation 1:

#0 may be used.

At operation 2:

#0 and #1 may be used.

Never use a future reference.


============================================================
PERCENTAGE CHECK
============================================================

Do NOT automatically multiply by 100.

Do NOT automatically avoid multiplying by 100.

Determine from the actual question whether the FinQA result should
be represented as:

- a decimal ratio,
- or a percentage-scaled value.


============================================================
FINAL CHECK
============================================================

Before returning your answer:

1. Verify that the numerical values exist in the evidence.
2. Verify the arithmetic relationship.
3. Check subtraction direction.
4. Check the denominator.
5. Check percentage scaling.
6. Check unit conversions.
7. Ensure only allowed FinQA operations are used.
8. Ensure the program is sequential.
9. Ensure final_program actually answers the question.


============================================================
FINQA TRAINING DEMONSTRATIONS
============================================================

These examples come only from the FinQA training split.

Use them only to understand FinQA program syntax and numerical reasoning.

{demonstrations}


============================================================
ORIGINAL QUESTION
============================================================

{question}


============================================================
ORIGINAL FINANCIAL EVIDENCE
============================================================

{context}


============================================================
REASONING AGENT CANDIDATE
============================================================

{reasoning_program}


============================================================
INDEPENDENT VERIFICATION PROGRAM
============================================================

{independent_program}


============================================================
VERIFIER VERDICT
============================================================

{verifier_verdict}


============================================================
VERIFIER PROPOSED FINAL PROGRAM
============================================================

{verifier_final_program}
""".strip()


# ---------------------------------------------------------------------
# COORDINATOR OUTPUT
# ---------------------------------------------------------------------

def parse_coordinator_output(
    model_result,
):

    decision = None

    final_program = None

    parse_error = None


    if model_result[
        "api_error"
    ] is not None:

        return (
            decision,
            final_program,
            parse_error,
        )


    parsed, parse_error = (
        parse_json_response(
            model_result[
                "raw_response"
            ],
            required_keys={
                "decision",
                "final_program",
            },
        )
    )


    if parse_error is not None:

        return (
            decision,
            final_program,
            parse_error,
        )


    if not isinstance(
        parsed["decision"],
        str,
    ):

        return (
            None,
            None,
            {
                "error_type":
                    "invalid_decision_type",

                "error_message":
                    '"decision" must be a string.',
            },
        )


    if not isinstance(
        parsed["final_program"],
        str,
    ):

        return (
            None,
            None,
            {
                "error_type":
                    "invalid_program_type",

                "error_message":
                    '"final_program" must be a string.',
            },
        )


    return (
        parsed["decision"],
        parsed["final_program"],
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
    dataset_split,
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

    verification_result = (
        empty_api_result()
    )


    independent_plan = None

    independent_program = None

    verifier_verdict = None

    verifier_final_program = None

    verification_parse_error = None


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


        verification_result = (
            call_model(
                client,
                verification_prompt,
            )
        )


        (
            independent_plan,
            independent_program,
            verifier_verdict,
            verifier_final_program,
            verification_parse_error,
        ) = parse_verification_output(
            verification_result
        )


    independent_pipeline = (
        validate_and_execute(
            independent_program,
            table,
        )
    )


    verifier_final_pipeline = (
        validate_and_execute(
            verifier_final_program,
            table,
        )
    )


    # =============================================================
    # STAGE 3 — COORDINATOR
    # =============================================================

    coordinator_result = (
        empty_api_result()
    )


    coordinator_decision = None

    coordinator_program = None

    coordinator_parse_error = None


    if (
        not reasoning_result[
            "fatal_api_error"
        ]
        and
        not verification_result[
            "fatal_api_error"
        ]
    ):

        coordinator_prompt = (
            build_coordinator_prompt(
                question=
                    question,

                context=
                    context,

                reasoning_program=
                    reasoning_program,

                independent_program=
                    independent_program,

                verifier_verdict=
                    verifier_verdict,

                verifier_final_program=
                    verifier_final_program,
            )
        )


        coordinator_result = (
            call_model(
                client,
                coordinator_prompt,
            )
        )


        (
            coordinator_decision,
            coordinator_program,
            coordinator_parse_error,
        ) = parse_coordinator_output(
            coordinator_result
        )


    final_pipeline = (
        validate_and_execute(
            coordinator_program,
            table,
        )
    )


    # =============================================================
    # GOLD
    #
    # Accessed only after all agent generation.
    # =============================================================

    gold_answer = (
        example["qa"]["exe_ans"]
    )


    gold_program = (
        example["qa"]["program"]
    )


    # =============================================================
    # STAGE ACCURACIES
    # =============================================================

    reasoning_correct = bool(

        reasoning_pipeline[
            "executable"
        ]

        and

        reasoning_pipeline[
            "predicted_answer"
        ]
        == gold_answer
    )


    independent_correct = bool(

        independent_pipeline[
            "executable"
        ]

        and

        independent_pipeline[
            "predicted_answer"
        ]
        == gold_answer
    )


    verifier_final_correct = bool(

        verifier_final_pipeline[
            "executable"
        ]

        and

        verifier_final_pipeline[
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

    program_evaluation = (
        compare_programs(
            gold_program,
            coordinator_program,
        )
    )


    final_symbolically_correct = bool(
        program_evaluation[
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
    # COORDINATOR CONTRIBUTION
    # =============================================================

    coordinator_corrected_error = bool(
        not verifier_final_correct
        and final_execution_correct
    )


    coordinator_broke_correct_answer = bool(
        verifier_final_correct
        and not final_execution_correct
    )


    # =============================================================
    # EFFICIENCY
    # =============================================================

    input_tokens = (
        sum_if_complete(

            reasoning_result[
                "input_tokens"
            ],

            verification_result[
                "input_tokens"
            ],

            coordinator_result[
                "input_tokens"
            ],
        )
    )


    output_tokens = (
        sum_if_complete(

            reasoning_result[
                "output_tokens"
            ],

            verification_result[
                "output_tokens"
            ],

            coordinator_result[
                "output_tokens"
            ],
        )
    )


    total_tokens = (
        sum_if_complete(

            reasoning_result[
                "total_tokens"
            ],

            verification_result[
                "total_tokens"
            ],

            coordinator_result[
                "total_tokens"
            ],
        )
    )


    effective_cost_usd = (
        sum_if_complete(

            reasoning_result[
                "effective_cost_usd"
            ],

            verification_result[
                "effective_cost_usd"
            ],

            coordinator_result[
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

        or

        coordinator_result[
            "fatal_api_error"
        ]
    )


    api_error = None


    for stage_result in (
        reasoning_result,
        verification_result,
        coordinator_result,
    ):

        if stage_result[
            "api_error"
        ] is not None:

            api_error = (
                stage_result[
                    "api_error"
                ]
            )

            break


    # =============================================================
    # FINAL PROGRAM ERRORS
    # =============================================================

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
            dataset_split,

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

        "reasoning_executable":
            reasoning_pipeline[
                "executable"
            ],

        "reasoning_predicted_answer":
            reasoning_pipeline[
                "predicted_answer"
            ],

        "reasoning_execution_correct":
            reasoning_correct,

        "reasoning_api":
            reasoning_result,


        # ---------------------------------------------------------
        # VERIFICATION AGENT
        # ---------------------------------------------------------

        "verification_independent_plan":
            independent_plan,

        "verification_independent_program":
            independent_program,

        "verification_independent_correct":
            independent_correct,

        "verification_verdict":
            verifier_verdict,

        "verification_final_program":
            verifier_final_program,

        "verification_final_correct":
            verifier_final_correct,

        "verification_parse_error":
            verification_parse_error,

        "verification_api":
            verification_result,


        # ---------------------------------------------------------
        # COORDINATOR
        # ---------------------------------------------------------

        "coordinator_decision":
            coordinator_decision,

        "coordinator_parse_error":
            coordinator_parse_error,

        "coordinator_api":
            coordinator_result,


        # ---------------------------------------------------------
        # FINAL OUTPUT
        # ---------------------------------------------------------

        "generated_program":
            coordinator_program,

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


        # ---------------------------------------------------------
        # COORDINATOR EFFECT
        # ---------------------------------------------------------

        "coordinator_corrected_verification_error":
            coordinator_corrected_error,

        "coordinator_broke_verified_answer":
            coordinator_broke_correct_answer,


        # ---------------------------------------------------------
        # EFFICIENCY
        # ---------------------------------------------------------

        "input_tokens":
            input_tokens,

        "output_tokens":
            output_tokens,

        "total_tokens":
            total_tokens,

        "effective_cost_usd":
            effective_cost_usd,

        "model_call_count":
            (
                reasoning_result[
                    "attempt_count"
                ]

                +

                verification_result[
                    "attempt_count"
                ]

                +

                coordinator_result[
                    "attempt_count"
                ]
            ),

        "api_error":
            api_error,

        "fatal_api_error":
            fatal_api_error,

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
            "Configuration D: "
            "Reasoning + Verification + Coordinator"
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


    # =============================================================
    # DRY RUN
    # =============================================================

    if args.dry_run:

        print(
            "Configuration D dry run"
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

        print(
            "Stage 1: Reasoning Agent"
        )

        print(
            "Stage 2: Independent Verification Agent"
        )

        print(
            "Stage 3: Coordinator"
        )

        print(
            "Expected model calls/question: 3"
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


    # =============================================================
    # DATASET
    # =============================================================

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
            "configuration_D_"
            f"{args.split}_{args.start}_"
            f"{args.start + args.count}_"
            f"{run_id}.jsonl"
        )
    )


    # =============================================================
    # COUNTERS
    # =============================================================

    attempted = 0

    reasoning_correct_count = 0

    independent_correct_count = 0

    verifier_final_correct_count = 0

    final_valid_count = 0

    final_executable_count = 0

    final_execution_correct_count = 0

    final_symbolically_correct_count = 0

    final_program_correct_count = 0


    coordinator_corrected_count = 0

    coordinator_broke_count = 0

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
        "Configuration D — "
        "Reasoning + Verification + Coordinator"
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
        f"{args.split}[{args.start}:"
        f"{args.start + args.count}]"
    )

    print(
        f"Examples: "
        f"{len(examples)}"
    )

    print(
        "Model calls/question: 3"
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


            # -----------------------------------------------------
            # COUNTERS
            # -----------------------------------------------------

            reasoning_correct_count += int(
                record[
                    "reasoning_execution_correct"
                ]
            )


            independent_correct_count += int(
                record[
                    "verification_independent_correct"
                ]
            )


            verifier_final_correct_count += int(
                record[
                    "verification_final_correct"
                ]
            )


            final_valid_count += int(
                record[
                    "program_valid"
                ]
            )


            final_executable_count += int(
                record[
                    "executable"
                ]
            )


            final_execution_correct_count += int(
                record[
                    "execution_correct"
                ]
            )


            final_symbolically_correct_count += int(
                record[
                    "program_symbolically_correct"
                ]
            )


            final_program_correct_count += int(
                record[
                    "program_correct"
                ]
            )


            coordinator_corrected_count += int(
                record[
                    "coordinator_corrected_verification_error"
                ]
            )


            coordinator_broke_count += int(
                record[
                    "coordinator_broke_verified_answer"
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
                f"{args.split} index "
                f"{dataset_index} | "
                f"{record['example_id']}"
            )


            print(
                "  Reasoning: "
                f"{record['reasoning_program']}"
            )


            print(
                "  Verifier independent: "
                f"{record['verification_independent_program']}"
            )


            print(
                "  Verifier final: "
                f"{record['verification_final_program']}"
            )


            print(
                "  Coordinator decision: "
                f"{record['coordinator_decision']}"
            )


            print(
                "  Coordinator final: "
                f"{record['generated_program']}"
            )


            print(
                "  Reasoning correct: "
                f"{record['reasoning_execution_correct']}"
            )


            print(
                "  Verifier final correct: "
                f"{record['verification_final_correct']}"
            )


            print(
                "  Coordinator execution correct: "
                f"{record['execution_correct']}"
            )


            print(
                "  Coordinator program correct: "
                f"{record['program_correct']}"
            )


            print(
                "  Coordinator corrected verifier error: "
                f"{record['coordinator_corrected_verification_error']}"
            )


            print(
                "  Coordinator broke verified answer: "
                f"{record['coordinator_broke_verified_answer']}"
            )


            print(
                f"  Latency: "
                f"{record['latency']:.2f}s"
            )


            print(
                f"  Cost: "
                f"{record['effective_cost_usd']}"
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
        "CONFIGURATION D SUMMARY"
    )

    print(
        "----------------------------------------"
    )


    print(
        f"Attempted examples: "
        f"{attempted}"
    )


    print(
        f"Reasoning correct: "
        f"{reasoning_correct_count}/{attempted}"
    )


    print(
        f"Independent verifier correct: "
        f"{independent_correct_count}/{attempted}"
    )


    print(
        f"Verifier final correct: "
        f"{verifier_final_correct_count}/{attempted}"
    )


    print()


    print(
        f"Final valid programs: "
        f"{final_valid_count}/{attempted}"
    )


    print(
        f"Final executable programs: "
        f"{final_executable_count}/{attempted}"
    )


    print(
        f"Final execution correct: "
        f"{final_execution_correct_count}/{attempted}"
    )


    print(
        f"Execution accuracy: "
        f"{final_execution_correct_count / attempted * 100:.1f}%"
        if attempted
        else
        "Execution accuracy: N/A"
    )


    print(
        f"Final symbolically correct: "
        f"{final_symbolically_correct_count}/{attempted}"
    )


    print(
        f"Final program correct: "
        f"{final_program_correct_count}/{attempted}"
    )


    print(
        f"Program accuracy: "
        f"{final_program_correct_count / attempted * 100:.1f}%"
        if attempted
        else
        "Program accuracy: N/A"
    )


    print()


    print(
        "Coordinator corrected verifier errors: "
        f"{coordinator_corrected_count}"
    )


    print(
        "Coordinator broke verified answers: "
        f"{coordinator_broke_count}"
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