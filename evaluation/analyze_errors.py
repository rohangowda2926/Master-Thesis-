"""Offline qualitative/error analysis for final FinQA A/B/C/D runs.

NO API calls are made.

This script analyzes the frozen full FinQA test results and identifies:

- questions all architectures solve correctly
- questions all architectures fail
- questions solved by exactly one architecture
- questions solved by exactly 2 or 3 architectures
- B -> C verification rescues/losses
- C -> D coordinator rescues/losses
- Configuration C internal verifier corrections/failures
- Configuration D internal coordinator corrections/failures
- invalid-program failures
- execution failures
- executable-but-wrong-answer failures
- execution-correct but symbolic-program-mismatch cases
- representative examples for manual dissertation error analysis

It exports:
1. JSON summary
2. question-level pattern CSV
3. representative-case CSV
4. architecture failure taxonomy CSV

Run:

    python -m evaluation.analyze_errors
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------
# REUSE FINAL-ANALYSIS RUN DISCOVERY
# ---------------------------------------------------------------------

from evaluation.analyze_final_test import (
    RESULTS_DIR,
    find_latest_complete_run,
    index_records,
    verify_alignment,
)


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

ARCHITECTURES = {
    "A": "Single-Agent",
    "B": "Reasoning Agent",
    "C": "Reasoning + Verification",
    "D": "Reasoning + Verification + Coordinator",
}


# ---------------------------------------------------------------------
# ARGUMENTS
# ---------------------------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Offline error analysis for final "
            "FinQA A/B/C/D experiments."
        )
    )

    parser.add_argument(
        "--split",
        choices=("dev", "test"),
        default="test",
    )

    parser.add_argument(
        "--start",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--count",
        type=int,
        default=1147,
    )

    parser.add_argument(
        "--examples-per-category",
        type=int,
        default=5,
        help=(
            "Maximum representative examples "
            "exported per error category."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# BASIC CLASSIFICATION
# ---------------------------------------------------------------------

def classify_final_outcome(record):
    """Classify one architecture's final observable outcome.

    Categories are objective pipeline outcomes rather than subjective
    semantic reasoning labels.
    """

    if bool(
        record.get(
            "program_correct"
        )
    ):

        return "program_correct"


    if bool(
        record.get(
            "execution_correct"
        )
    ):

        return (
            "execution_correct_program_mismatch"
        )


    if not bool(
        record.get(
            "program_valid"
        )
    ):

        return "invalid_program"


    if not bool(
        record.get(
            "executable"
        )
    ):

        return "execution_failure"


    return "executable_wrong_answer"


# ---------------------------------------------------------------------
# CROSS-ARCHITECTURE CORRECTNESS PATTERN
# ---------------------------------------------------------------------

def correctness_pattern(
    records_by_config,
    example_id,
):

    return "".join(

        configuration

        for configuration in (
            "A",
            "B",
            "C",
            "D",
        )

        if bool(
            records_by_config[
                configuration
            ][
                example_id
            ].get(
                "execution_correct"
            )
        )
    )


def readable_pattern(
    pattern,
):

    if pattern == "":
        return "none_correct"

    if pattern == "ABCD":
        return "all_correct"

    return (
        f"correct_{pattern}"
    )


# ---------------------------------------------------------------------
# SAFE FIELD ACCESS
# ---------------------------------------------------------------------

def field(
    record,
    key,
    default=None,
):

    value = record.get(
        key,
        default,
    )

    return value


def clean_text(
    value,
):

    if value is None:
        return ""

    return str(
        value
    ).replace(
        "\n",
        " ",
    ).strip()


# ---------------------------------------------------------------------
# QUESTION-LEVEL ROW
# ---------------------------------------------------------------------

def build_question_row(
    example_id,
    indexed,
):

    a = indexed["A"][example_id]
    b = indexed["B"][example_id]
    c = indexed["C"][example_id]
    d = indexed["D"][example_id]


    pattern = correctness_pattern(
        indexed,
        example_id,
    )


    row = {

        "dataset_index":
            a.get(
                "dataset_index"
            ),

        "example_id":
            example_id,

        "question":
            clean_text(
                a.get(
                    "question"
                )
            ),

        "gold_answer":
            a.get(
                "gold_answer"
            ),

        "gold_program":
            clean_text(
                a.get(
                    "gold_program"
                )
            ),

        "execution_correct_pattern":
            pattern,

        "pattern_label":
            readable_pattern(
                pattern
            ),

        "number_correct_architectures":
            sum(
                bool(
                    indexed[
                        configuration
                    ][
                        example_id
                    ].get(
                        "execution_correct"
                    )
                )

                for configuration
                in (
                    "A",
                    "B",
                    "C",
                    "D",
                )
            ),
    }


    # -----------------------------------------------------------------
    # FINAL ARCHITECTURE OUTPUTS
    # -----------------------------------------------------------------

    for configuration, record in (
        ("A", a),
        ("B", b),
        ("C", c),
        ("D", d),
    ):

        prefix = (
            configuration
            .lower()
        )


        row[
            f"{prefix}_execution_correct"
        ] = bool(
            record.get(
                "execution_correct"
            )
        )


        row[
            f"{prefix}_program_correct"
        ] = bool(
            record.get(
                "program_correct"
            )
        )


        row[
            f"{prefix}_program_valid"
        ] = bool(
            record.get(
                "program_valid"
            )
        )


        row[
            f"{prefix}_executable"
        ] = bool(
            record.get(
                "executable"
            )
        )


        row[
            f"{prefix}_failure_category"
        ] = classify_final_outcome(
            record
        )


        row[
            f"{prefix}_generated_program"
        ] = clean_text(
            record.get(
                "generated_program"
            )
        )


        row[
            f"{prefix}_predicted_answer"
        ] = record.get(
            "predicted_answer"
        )


    # -----------------------------------------------------------------
    # CONFIGURATION C INTERNAL BEHAVIOUR
    # -----------------------------------------------------------------

    row[
        "c_reasoning_program"
    ] = clean_text(
        c.get(
            "reasoning_program"
        )
    )


    row[
        "c_reasoning_correct"
    ] = bool(
        c.get(
            "reasoning_execution_correct"
        )
    )


    row[
        "c_independent_verifier_program"
    ] = clean_text(
        c.get(
            "verification_independent_program"
        )
    )


    row[
        "c_independent_verifier_correct"
    ] = bool(
        c.get(
            "verification_independent_execution_correct"
        )
    )


    row[
        "c_verifier_verdict"
    ] = clean_text(
        c.get(
            "candidate_verdict"
        )
    )


    row[
        "c_verifier_corrected_error"
    ] = bool(
        c.get(
            "verifier_corrected_error"
        )
    )


    row[
        "c_verifier_broke_correct"
    ] = bool(
        c.get(
            "verifier_broke_correct_answer"
        )
    )


    # -----------------------------------------------------------------
    # CONFIGURATION D INTERNAL BEHAVIOUR
    # -----------------------------------------------------------------

    row[
        "d_reasoning_program"
    ] = clean_text(
        d.get(
            "reasoning_program"
        )
    )


    row[
        "d_reasoning_correct"
    ] = bool(
        d.get(
            "reasoning_execution_correct"
        )
    )


    row[
        "d_independent_verifier_program"
    ] = clean_text(
        d.get(
            "verification_independent_program"
        )
    )


    row[
        "d_independent_verifier_correct"
    ] = bool(
        d.get(
            "verification_independent_correct"
        )
    )


    row[
        "d_verifier_final_program"
    ] = clean_text(
        d.get(
            "verification_final_program"
        )
    )


    row[
        "d_verifier_final_correct"
    ] = bool(
        d.get(
            "verification_final_correct"
        )
    )


    row[
        "d_coordinator_decision"
    ] = clean_text(
        d.get(
            "coordinator_decision"
        )
    )


    row[
        "d_coordinator_corrected_error"
    ] = bool(
        d.get(
            "coordinator_corrected_verification_error"
        )
    )


    row[
        "d_coordinator_broke_correct"
    ] = bool(
        d.get(
            "coordinator_broke_verified_answer"
        )
    )


    return row


# ---------------------------------------------------------------------
# REPRESENTATIVE CATEGORY RULES
# ---------------------------------------------------------------------

def build_category_memberships(
    row,
):

    categories = []


    # -----------------------------------------------------------------
    # GLOBAL
    # -----------------------------------------------------------------

    if (
        row[
            "number_correct_architectures"
        ]
        == 0
    ):

        categories.append(
            "all_architectures_wrong"
        )


    if (
        row[
            "number_correct_architectures"
        ]
        == 4
    ):

        categories.append(
            "all_architectures_correct"
        )


    if (
        row[
            "number_correct_architectures"
        ]
        == 1
    ):

        categories.append(
            "exactly_one_architecture_correct"
        )


    if (
        row[
            "number_correct_architectures"
        ]
        == 2
    ):

        categories.append(
            "exactly_two_architectures_correct"
        )


    if (
        row[
            "number_correct_architectures"
        ]
        == 3
    ):

        categories.append(
            "exactly_three_architectures_correct"
        )


    # -----------------------------------------------------------------
    # ARCHITECTURE-ONLY SUCCESSES
    # -----------------------------------------------------------------

    pattern = (
        row[
            "execution_correct_pattern"
        ]
    )


    if pattern in {
        "A",
        "B",
        "C",
        "D",
    }:

        categories.append(
            f"{pattern}_only_correct"
        )


    # -----------------------------------------------------------------
    # A -> B
    # -----------------------------------------------------------------

    if (
        row[
            "a_execution_correct"
        ]
        and not
        row[
            "b_execution_correct"
        ]
    ):

        categories.append(
            "B_lost_A_correct_answer"
        )


    if (
        not row[
            "a_execution_correct"
        ]
        and
        row[
            "b_execution_correct"
        ]
    ):

        categories.append(
            "B_rescued_A_error"
        )


    # -----------------------------------------------------------------
    # B -> C
    # -----------------------------------------------------------------

    if (
        not row[
            "b_execution_correct"
        ]
        and
        row[
            "c_execution_correct"
        ]
    ):

        categories.append(
            "C_rescued_B_error"
        )


    if (
        row[
            "b_execution_correct"
        ]
        and not
        row[
            "c_execution_correct"
        ]
    ):

        categories.append(
            "C_lost_B_correct_answer"
        )


    # -----------------------------------------------------------------
    # C -> D
    # -----------------------------------------------------------------

    if (
        not row[
            "c_execution_correct"
        ]
        and
        row[
            "d_execution_correct"
        ]
    ):

        categories.append(
            "D_rescued_C_error"
        )


    if (
        row[
            "c_execution_correct"
        ]
        and not
        row[
            "d_execution_correct"
        ]
    ):

        categories.append(
            "D_lost_C_correct_answer"
        )


    # -----------------------------------------------------------------
    # C INTERNAL VERIFIER
    # -----------------------------------------------------------------

    if row[
        "c_verifier_corrected_error"
    ]:

        categories.append(
            "C_verifier_corrected_reasoning_error"
        )


    if row[
        "c_verifier_broke_correct"
    ]:

        categories.append(
            "C_verifier_broke_correct_reasoning"
        )


    if (
        not row[
            "c_reasoning_correct"
        ]
        and
        not row[
            "c_independent_verifier_correct"
        ]
        and
        not row[
            "c_execution_correct"
        ]
    ):

        categories.append(
            "C_reasoner_and_verifier_both_wrong"
        )


    # -----------------------------------------------------------------
    # D INTERNAL COORDINATOR
    # -----------------------------------------------------------------

    if row[
        "d_coordinator_corrected_error"
    ]:

        categories.append(
            "D_coordinator_corrected_verifier_error"
        )


    if row[
        "d_coordinator_broke_correct"
    ]:

        categories.append(
            "D_coordinator_broke_verified_answer"
        )


    if (
        not row[
            "d_reasoning_correct"
        ]
        and
        not row[
            "d_independent_verifier_correct"
        ]
        and
        not row[
            "d_verifier_final_correct"
        ]
        and
        not row[
            "d_execution_correct"
        ]
    ):

        categories.append(
            "D_all_three_stages_wrong"
        )


    # -----------------------------------------------------------------
    # STRUCTURAL FAILURE TYPES
    # -----------------------------------------------------------------

    for configuration in (
        "a",
        "b",
        "c",
        "d",
    ):

        failure_category = (
            row[
                f"{configuration}_failure_category"
            ]
        )


        if failure_category in {
            "invalid_program",
            "execution_failure",
            "execution_correct_program_mismatch",
        }:

            categories.append(
                f"{configuration.upper()}_"
                f"{failure_category}"
            )


    return categories


# ---------------------------------------------------------------------
# FAILURE TAXONOMY SUMMARY
# ---------------------------------------------------------------------

def architecture_failure_taxonomy(
    records,
):

    counter = Counter()


    for record in records:

        counter[
            classify_final_outcome(
                record
            )
        ] += 1


    total = len(
        records
    )


    rows = []


    categories = (
        "program_correct",
        "execution_correct_program_mismatch",
        "executable_wrong_answer",
        "invalid_program",
        "execution_failure",
    )


    for category in categories:

        count = (
            counter[
                category
            ]
        )


        rows.append({

            "category":
                category,

            "count":
                count,

            "percentage":
                (
                    count
                    / total
                    * 100
                ),
        })


    return rows


# ---------------------------------------------------------------------
# SAVE GENERIC CSV
# ---------------------------------------------------------------------

def save_csv(
    path,
    rows,
):

    if not rows:

        return


    fieldnames = list(
        rows[0].keys()
    )


    with path.open(
        "x",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ---------------------------------------------------------------------
# TERMINAL SUMMARY
# ---------------------------------------------------------------------

def print_failure_taxonomy(
    taxonomy,
):

    print()
    print(
        "=" * 90
    )

    print(
        "FINAL-OUTPUT FAILURE TAXONOMY"
    )

    print(
        "=" * 90
    )


    for configuration in (
        "A",
        "B",
        "C",
        "D",
    ):

        print()

        print(
            f"Configuration {configuration} — "
            f"{ARCHITECTURES[configuration]}"
        )

        print(
            "-" * 60
        )


        for row in taxonomy[
            configuration
        ]:

            print(
                f"{row['category']:<38}"
                f"{row['count']:>5}  "
                f"{row['percentage']:>7.2f}%"
            )


def print_cross_architecture_summary(
    pattern_counter,
    category_counter,
):

    print()
    print(
        "=" * 90
    )

    print(
        "CROSS-ARCHITECTURE CORRECTNESS PATTERNS"
    )

    print(
        "=" * 90
    )


    for pattern, count in sorted(
        pattern_counter.items(),
        key=lambda item:
            (
                -item[1],
                item[0],
            ),
    ):

        print(
            f"{readable_pattern(pattern):<35}"
            f"{count:>5}"
        )


    print()
    print(
        "=" * 90
    )

    print(
        "KEY ERROR-ANALYSIS CATEGORIES"
    )

    print(
        "=" * 90
    )


    important_categories = (
        "all_architectures_wrong",
        "all_architectures_correct",
        "A_only_correct",
        "B_only_correct",
        "C_only_correct",
        "D_only_correct",
        "B_rescued_A_error",
        "B_lost_A_correct_answer",
        "C_rescued_B_error",
        "C_lost_B_correct_answer",
        "D_rescued_C_error",
        "D_lost_C_correct_answer",
        "C_verifier_corrected_reasoning_error",
        "C_verifier_broke_correct_reasoning",
        "C_reasoner_and_verifier_both_wrong",
        "D_coordinator_corrected_verifier_error",
        "D_coordinator_broke_verified_answer",
        "D_all_three_stages_wrong",
    )


    for category in important_categories:

        print(
            f"{category:<48}"
            f"{category_counter[category]:>5}"
        )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    args = parse_args()


    if args.start < 0:

        raise ValueError(
            "--start must be >= 0."
        )


    if args.count <= 0:

        raise ValueError(
            "--count must be > 0."
        )


    if (
        args.examples_per_category
        <= 0
    ):

        raise ValueError(
            "--examples-per-category "
            "must be > 0."
        )


    print(
        "Final FinQA error analysis"
    )

    print(
        f"Dataset: "
        f"{args.split}["
        f"{args.start}:"
        f"{args.start + args.count}]"
    )

    print(
        "No API calls made."
    )

    print()


    # =============================================================
    # LOAD FINAL RUNS
    # =============================================================

    source_files = {}

    records = {}


    for configuration in (
        "A",
        "B",
        "C",
        "D",
    ):

        path, run_records = (
            find_latest_complete_run(
                configuration,
                args.split,
                args.start,
                args.count,
            )
        )


        source_files[
            configuration
        ] = path


        records[
            configuration
        ] = run_records


        print(
            f"Configuration "
            f"{configuration}: "
            f"{path.name}"
        )


    # =============================================================
    # ALIGN
    # =============================================================

    ordered_ids = (
        verify_alignment(
            records
        )
    )


    print()

    print(
        f"Aligned questions: "
        f"{len(ordered_ids)}"
    )


    indexed = {

        configuration:
            index_records(
                records[
                    configuration
                ]
            )

        for configuration
        in (
            "A",
            "B",
            "C",
            "D",
        )
    }


    # =============================================================
    # QUESTION-LEVEL ANALYSIS
    # =============================================================

    question_rows = []

    pattern_counter = Counter()

    category_counter = Counter()

    category_examples = defaultdict(
        list
    )


    for example_id in ordered_ids:

        row = build_question_row(
            example_id,
            indexed,
        )


        question_rows.append(
            row
        )


        pattern_counter[
            row[
                "execution_correct_pattern"
            ]
        ] += 1


        categories = (
            build_category_memberships(
                row
            )
        )


        for category in categories:

            category_counter[
                category
            ] += 1


            if (
                len(
                    category_examples[
                        category
                    ]
                )
                <
                args.examples_per_category
            ):

                category_examples[
                    category
                ].append(
                    row
                )


    # =============================================================
    # FAILURE TAXONOMY
    # =============================================================

    taxonomy = {

        configuration:
            architecture_failure_taxonomy(
                records[
                    configuration
                ]
            )

        for configuration
        in (
            "A",
            "B",
            "C",
            "D",
        )
    }


    # =============================================================
    # DISPLAY
    # =============================================================

    print_failure_taxonomy(
        taxonomy
    )


    print_cross_architecture_summary(
        pattern_counter,
        category_counter,
    )


    # =============================================================
    # REPRESENTATIVE-CASE TABLE
    # =============================================================

    representative_rows = []


    for category in sorted(
        category_examples
    ):

        for rank, row in enumerate(
            category_examples[
                category
            ],
            start=1,
        ):

            representative_rows.append({

                "category":
                    category,

                "rank":
                    rank,

                "dataset_index":
                    row[
                        "dataset_index"
                    ],

                "example_id":
                    row[
                        "example_id"
                    ],

                "question":
                    row[
                        "question"
                    ],

                "gold_answer":
                    row[
                        "gold_answer"
                    ],

                "gold_program":
                    row[
                        "gold_program"
                    ],

                "A_program":
                    row[
                        "a_generated_program"
                    ],

                "A_correct":
                    row[
                        "a_execution_correct"
                    ],

                "B_program":
                    row[
                        "b_generated_program"
                    ],

                "B_correct":
                    row[
                        "b_execution_correct"
                    ],

                "C_reasoning_program":
                    row[
                        "c_reasoning_program"
                    ],

                "C_independent_program":
                    row[
                        "c_independent_verifier_program"
                    ],

                "C_final_program":
                    row[
                        "c_generated_program"
                    ],

                "C_correct":
                    row[
                        "c_execution_correct"
                    ],

                "D_reasoning_program":
                    row[
                        "d_reasoning_program"
                    ],

                "D_independent_program":
                    row[
                        "d_independent_verifier_program"
                    ],

                "D_verifier_final_program":
                    row[
                        "d_verifier_final_program"
                    ],

                "D_coordinator_decision":
                    row[
                        "d_coordinator_decision"
                    ],

                "D_final_program":
                    row[
                        "d_generated_program"
                    ],

                "D_correct":
                    row[
                        "d_execution_correct"
                    ],
            })


    # =============================================================
    # FAILURE TAXONOMY CSV ROWS
    # =============================================================

    taxonomy_rows = []


    for configuration in (
        "A",
        "B",
        "C",
        "D",
    ):

        for row in taxonomy[
            configuration
        ]:

            taxonomy_rows.append({

                "configuration":
                    configuration,

                "architecture":
                    ARCHITECTURES[
                        configuration
                    ],

                "category":
                    row[
                        "category"
                    ],

                "count":
                    row[
                        "count"
                    ],

                "percentage":
                    row[
                        "percentage"
                    ],
            })


    # =============================================================
    # SAVE
    # =============================================================

    timestamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%S_%fZ"
    )


    prefix = (
        f"final_error_analysis_"
        f"{args.split}_"
        f"{args.start}_"
        f"{args.start + args.count}_"
        f"{timestamp}"
    )


    json_path = (
        RESULTS_DIR
        / f"{prefix}.json"
    )


    question_csv_path = (
        RESULTS_DIR
        / (
            "final_error_question_patterns_"
            f"{timestamp}.csv"
        )
    )


    representative_csv_path = (
        RESULTS_DIR
        / (
            "final_error_representative_cases_"
            f"{timestamp}.csv"
        )
    )


    taxonomy_csv_path = (
        RESULTS_DIR
        / (
            "final_error_failure_taxonomy_"
            f"{timestamp}.csv"
        )
    )


    output = {

        "dataset_split":
            args.split,

        "start_index":
            args.start,

        "end_index":
            args.start
            + args.count,

        "questions":
            len(
                ordered_ids
            ),

        "source_files": {

            configuration:
                str(path)

            for configuration, path
            in source_files.items()
        },

        "execution_correctness_patterns":
            dict(
                pattern_counter
            ),

        "category_counts":
            dict(
                category_counter
            ),

        "failure_taxonomy":
            taxonomy,

        "representative_examples_per_category":
            args.examples_per_category,

        "methodological_note":
            (
                "Automatic categories represent observable "
                "pipeline outcomes and architecture behaviour. "
                "Semantic reasoning-error labels such as wrong "
                "denominator, wrong value selection, percentage "
                "scaling, or unit conversion require manual "
                "inspection and are not automatically inferred."
            ),
    }


    with json_path.open(
        "x",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )


    save_csv(
        question_csv_path,
        question_rows,
    )


    save_csv(
        representative_csv_path,
        representative_rows,
    )


    save_csv(
        taxonomy_csv_path,
        taxonomy_rows,
    )


    # =============================================================
    # FINISH
    # =============================================================

    print()
    print(
        "=" * 90
    )

    print(
        "ERROR ANALYSIS COMPLETE"
    )

    print(
        "=" * 90
    )


    print(
        f"JSON summary:\n"
        f"{json_path}"
    )


    print()

    print(
        f"Question-level patterns:\n"
        f"{question_csv_path}"
    )


    print()

    print(
        f"Representative cases:\n"
        f"{representative_csv_path}"
    )


    print()

    print(
        f"Failure taxonomy:\n"
        f"{taxonomy_csv_path}"
    )


    print()

    print(
        "No API calls were made."
    )


# ---------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()