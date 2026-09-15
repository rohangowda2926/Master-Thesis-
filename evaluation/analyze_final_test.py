"""Final FinQA test-set analysis for thesis A/B/C/D architectures.

This script is completely OFFLINE.
It makes NO API calls.

It analyzes the frozen full FinQA test runs:

A = Single-Agent direct program
B = Reasoning Agent
C = Reasoning + Independent Verification
D = Reasoning + Verification + Coordinator

Default dataset:
    test[0:1147]

Outputs:
    1. Full JSON analysis
    2. Architecture summary CSV
    3. Pairwise statistical comparison CSV

Run:
    python -m evaluation.analyze_final_test
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics

from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results"


# ---------------------------------------------------------------------
# ARCHITECTURES
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
            "Offline final FinQA test-set analysis "
            "for configurations A/B/C/D."
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

    return parser.parse_args()


# ---------------------------------------------------------------------
# JSONL
# ---------------------------------------------------------------------

def load_jsonl(path):

    records = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:

                record = json.loads(
                    line
                )

            except json.JSONDecodeError as exc:

                raise ValueError(
                    f"Invalid JSON in {path.name}, "
                    f"line {line_number}: {exc}"
                ) from exc

            records.append(
                record
            )

    return records


# ---------------------------------------------------------------------
# RUN DISCOVERY
# ---------------------------------------------------------------------

def run_is_complete(
    records,
    configuration,
    split,
    start,
    count,
):

    if len(records) != count:
        return False


    ids = [
        record.get(
            "example_id"
        )
        for record in records
    ]


    if (
        None in ids
        or len(ids)
        != len(set(ids))
    ):

        return False


    expected_indexes = set(
        range(
            start,
            start + count,
        )
    )


    actual_indexes = {
        record.get(
            "dataset_index"
        )
        for record in records
    }


    if actual_indexes != expected_indexes:

        return False


    for record in records:

        if (
            record.get(
                "configuration"
            )
            != configuration
        ):

            return False


        if (
            record.get(
                "dataset_split"
            )
            != split
        ):

            return False


    return True


def find_latest_complete_run(
    configuration,
    split,
    start,
    count,
):

    end = (
        start
        + count
    )


    pattern = (
        f"configuration_{configuration}_"
        f"{split}_{start}_{end}_*.jsonl"
    )


    candidates = sorted(
        RESULTS_DIR.glob(
            pattern
        ),
        key=lambda path:
            path.stat().st_mtime,
        reverse=True,
    )


    if not candidates:

        raise FileNotFoundError(
            f"No files found for Configuration "
            f"{configuration}.\n"
            f"Pattern: {pattern}"
        )


    rejected = []


    for path in candidates:

        records = load_jsonl(
            path
        )


        if run_is_complete(
            records,
            configuration,
            split,
            start,
            count,
        ):

            return path, records


        rejected.append(
            path.name
        )


    raise ValueError(
        f"No complete {count}-question run found "
        f"for Configuration {configuration}.\n"
        f"Rejected files: {rejected}"
    )


# ---------------------------------------------------------------------
# ALIGNMENT
# ---------------------------------------------------------------------

def index_records(
    records,
):

    indexed = {}


    for record in records:

        example_id = (
            record.get(
                "example_id"
            )
        )


        if example_id is None:

            raise ValueError(
                "Record missing example_id."
            )


        if example_id in indexed:

            raise ValueError(
                f"Duplicate example ID: "
                f"{example_id}"
            )


        indexed[
            example_id
        ] = record


    return indexed


def verify_alignment(
    all_records,
):

    reference_configuration = "A"

    reference = index_records(
        all_records[
            reference_configuration
        ]
    )


    reference_ids = set(
        reference
    )


    reference_index_map = {
        example_id:
            record[
                "dataset_index"
            ]
        for example_id, record
        in reference.items()
    }


    for configuration in (
        "B",
        "C",
        "D",
    ):

        indexed = index_records(
            all_records[
                configuration
            ]
        )


        ids = set(
            indexed
        )


        if ids != reference_ids:

            missing = (
                reference_ids
                - ids
            )

            extra = (
                ids
                - reference_ids
            )

            raise ValueError(
                f"Question alignment mismatch "
                f"for Configuration "
                f"{configuration}.\n"
                f"Missing IDs: {len(missing)}\n"
                f"Extra IDs: {len(extra)}"
            )


        index_map = {
            example_id:
                record[
                    "dataset_index"
                ]
            for example_id, record
            in indexed.items()
        }


        if (
            index_map
            != reference_index_map
        ):

            raise ValueError(
                f"Dataset-index alignment "
                f"mismatch for "
                f"Configuration "
                f"{configuration}."
            )


    ordered_ids = [

        example_id

        for example_id, _ in sorted(
            reference_index_map.items(),
            key=lambda item:
                item[1],
        )
    ]


    return ordered_ids


# ---------------------------------------------------------------------
# NUMERIC HELPERS
# ---------------------------------------------------------------------

def is_number(
    value,
):

    return (
        isinstance(
            value,
            (int, float),
        )
        and
        not isinstance(
            value,
            bool,
        )
        and
        math.isfinite(
            value
        )
    )


def numeric_values(
    values,
):

    return [
        float(value)
        for value in values
        if is_number(
            value
        )
    ]


def safe_mean(
    values,
):

    values = numeric_values(
        values
    )

    if not values:
        return None

    return statistics.mean(
        values
    )


def safe_median(
    values,
):

    values = numeric_values(
        values
    )

    if not values:
        return None

    return statistics.median(
        values
    )


def percentile(
    values,
    probability,
):

    values = sorted(
        numeric_values(
            values
        )
    )


    if not values:
        return None


    if len(values) == 1:
        return values[0]


    position = (
        probability
        * (
            len(values)
            - 1
        )
    )


    lower_index = (
        math.floor(
            position
        )
    )

    upper_index = (
        math.ceil(
            position
        )
    )


    if (
        lower_index
        == upper_index
    ):

        return values[
            lower_index
        ]


    weight = (
        position
        - lower_index
    )


    return (
        values[
            lower_index
        ]
        * (
            1
            - weight
        )

        +

        values[
            upper_index
        ]
        * weight
    )


def safe_sum_complete(
    values,
):

    values = list(
        values
    )


    if not values:

        return None


    if not all(
        is_number(value)
        for value in values
    ):

        return None


    return sum(
        float(value)
        for value in values
    )


def percentage(
    numerator,
    denominator,
):

    if denominator == 0:
        return None

    return (
        numerator
        / denominator
        * 100.0
    )


# ---------------------------------------------------------------------
# CONFIDENCE INTERVAL
# ---------------------------------------------------------------------

def wilson_interval(
    successes,
    total,
    z=1.959963984540054,
):

    if total <= 0:

        return (
            None,
            None,
        )


    p = (
        successes
        / total
    )


    denominator = (
        1
        + z**2
        / total
    )


    centre = (
        p
        + z**2
        / (
            2
            * total
        )
    )


    adjustment = (
        z
        * math.sqrt(
            (
                p
                * (
                    1
                    - p
                )
                / total
            )
            +
            (
                z**2
                / (
                    4
                    * total**2
                )
            )
        )
    )


    lower = (
        (
            centre
            - adjustment
        )
        / denominator
    )


    upper = (
        (
            centre
            + adjustment
        )
        / denominator
    )


    return (
        lower
        * 100,
        upper
        * 100,
    )


# ---------------------------------------------------------------------
# PARSE ERROR DETECTION
# ---------------------------------------------------------------------

def has_parse_error(
    record,
):

    possible_fields = (
        "parse_error",
        "reasoning_parse_error",
        "verification_parse_error",
        "coordinator_parse_error",
    )


    return any(
        record.get(
            field
        )
        is not None

        for field
        in possible_fields
    )


# ---------------------------------------------------------------------
# ARCHITECTURE SUMMARY
# ---------------------------------------------------------------------

def summarize_architecture(
    configuration,
    records,
):

    n = len(
        records
    )


    execution_correct = sum(
        bool(
            record.get(
                "execution_correct"
            )
        )
        for record
        in records
    )


    program_correct = sum(
        bool(
            record.get(
                "program_correct"
            )
        )
        for record
        in records
    )


    valid = sum(
        bool(
            record.get(
                "program_valid"
            )
        )
        for record
        in records
    )


    executable = sum(
        bool(
            record.get(
                "executable"
            )
        )
        for record
        in records
    )


    api_errors = sum(
        record.get(
            "api_error"
        )
        is not None
        for record
        in records
    )


    parse_errors = sum(
        has_parse_error(
            record
        )
        for record
        in records
    )


    validation_failures = (
        n
        - valid
    )


    execution_failures = sum(

        bool(
            record.get(
                "program_valid"
            )
        )
        and
        not bool(
            record.get(
                "executable"
            )
        )

        for record
        in records
    )


    latencies = [
        record.get(
            "latency"
        )
        for record
        in records
    ]


    input_tokens = [
        record.get(
            "input_tokens"
        )
        for record
        in records
    ]


    output_tokens = [
        record.get(
            "output_tokens"
        )
        for record
        in records
    ]


    costs = [
        record.get(
            "effective_cost_usd"
        )
        for record
        in records
    ]


    exec_ci = (
        wilson_interval(
            execution_correct,
            n,
        )
    )


    program_ci = (
        wilson_interval(
            program_correct,
            n,
        )
    )


    summary = {

        "configuration":
            configuration,

        "architecture":
            ARCHITECTURES[
                configuration
            ],

        "examples":
            n,


        # ---------------------------------------------------------
        # RQ1
        # ---------------------------------------------------------

        "execution_correct":
            execution_correct,

        "execution_accuracy":
            percentage(
                execution_correct,
                n,
            ),

        "execution_accuracy_ci95":
            {
                "lower":
                    exec_ci[0],

                "upper":
                    exec_ci[1],
            },


        "program_correct":
            program_correct,

        "program_accuracy":
            percentage(
                program_correct,
                n,
            ),

        "program_accuracy_ci95":
            {
                "lower":
                    program_ci[0],

                "upper":
                    program_ci[1],
            },


        # ---------------------------------------------------------
        # STRUCTURAL
        # ---------------------------------------------------------

        "valid_programs":
            valid,

        "valid_rate":
            percentage(
                valid,
                n,
            ),

        "executable_programs":
            executable,

        "executable_rate":
            percentage(
                executable,
                n,
            ),

        "api_errors":
            api_errors,

        "parse_errors":
            parse_errors,

        "validation_failures":
            validation_failures,

        "execution_failures":
            execution_failures,


        # ---------------------------------------------------------
        # RQ3
        # ---------------------------------------------------------

        "latency": {

            "mean_seconds":
                safe_mean(
                    latencies
                ),

            "median_seconds":
                safe_median(
                    latencies
                ),

            "q1_seconds":
                percentile(
                    latencies,
                    0.25,
                ),

            "q3_seconds":
                percentile(
                    latencies,
                    0.75,
                ),
        },


        "tokens": {

            "total_input":
                safe_sum_complete(
                    input_tokens
                ),

            "total_output":
                safe_sum_complete(
                    output_tokens
                ),

            "mean_input":
                safe_mean(
                    input_tokens
                ),

            "mean_output":
                safe_mean(
                    output_tokens
                ),

            "complete_input_records":
                len(
                    numeric_values(
                        input_tokens
                    )
                ),

            "complete_output_records":
                len(
                    numeric_values(
                        output_tokens
                    )
                ),
        },


        "cost": {

            "total_usd":
                safe_sum_complete(
                    costs
                ),

            "mean_usd":
                safe_mean(
                    costs
                ),

            "median_usd":
                safe_median(
                    costs
                ),

            "complete_records":
                len(
                    numeric_values(
                        costs
                    )
                ),
        },
    }


    # -------------------------------------------------------------
    # RQ4 — COMPONENT CONTRIBUTION
    # -------------------------------------------------------------

    if configuration == "C":

        reasoning_correct = sum(
            bool(
                record.get(
                    "reasoning_execution_correct"
                )
            )
            for record
            in records
        )


        independent_correct = sum(
            bool(
                record.get(
                    "verification_independent_execution_correct"
                )
            )
            for record
            in records
        )


        corrected = sum(
            bool(
                record.get(
                    "verifier_corrected_error"
                )
            )
            for record
            in records
        )


        broke = sum(
            bool(
                record.get(
                    "verifier_broke_correct_answer"
                )
            )
            for record
            in records
        )


        summary[
            "component_contribution"
        ] = {

            "reasoning_correct":
                reasoning_correct,

            "independent_verifier_correct":
                independent_correct,

            "final_correct":
                execution_correct,

            "verifier_corrected_errors":
                corrected,

            "verifier_broke_correct_answers":
                broke,

            "verifier_net_corrections":
                corrected
                - broke,
        }


    elif configuration == "D":

        reasoning_correct = sum(
            bool(
                record.get(
                    "reasoning_execution_correct"
                )
            )
            for record
            in records
        )


        independent_correct = sum(
            bool(
                record.get(
                    "verification_independent_correct"
                )
            )
            for record
            in records
        )


        verifier_final_correct = sum(
            bool(
                record.get(
                    "verification_final_correct"
                )
            )
            for record
            in records
        )


        corrected = sum(
            bool(
                record.get(
                    "coordinator_corrected_verification_error"
                )
            )
            for record
            in records
        )


        broke = sum(
            bool(
                record.get(
                    "coordinator_broke_verified_answer"
                )
            )
            for record
            in records
        )


        summary[
            "component_contribution"
        ] = {

            "reasoning_correct":
                reasoning_correct,

            "independent_verifier_correct":
                independent_correct,

            "verifier_final_correct":
                verifier_final_correct,

            "coordinator_final_correct":
                execution_correct,

            "coordinator_corrected_errors":
                corrected,

            "coordinator_broke_correct_answers":
                broke,

            "coordinator_net_corrections":
                corrected
                - broke,
        }


    else:

        summary[
            "component_contribution"
        ] = None


    return summary


# ---------------------------------------------------------------------
# EXACT MCNEMAR TEST
# ---------------------------------------------------------------------

def log_binomial_probability_half(
    n,
    k,
):

    return (
        math.lgamma(
            n + 1
        )
        -
        math.lgamma(
            k + 1
        )
        -
        math.lgamma(
            n - k + 1
        )
        -
        n
        * math.log(
            2
        )
    )


def exact_mcnemar_pvalue(
    first_only,
    second_only,
):

    discordant = (
        first_only
        + second_only
    )


    if discordant == 0:

        return 1.0


    tail_limit = min(
        first_only,
        second_only,
    )


    log_terms = [

        log_binomial_probability_half(
            discordant,
            k,
        )

        for k in range(
            tail_limit + 1
        )
    ]


    max_log = max(
        log_terms
    )


    lower_tail = (

        math.exp(
            max_log
        )

        * sum(
            math.exp(
                value
                - max_log
            )
            for value
            in log_terms
        )
    )


    p_value = min(
        1.0,
        2.0
        * lower_tail,
    )


    return p_value


# ---------------------------------------------------------------------
# PAIRED DIFFERENCE CI
# ---------------------------------------------------------------------

def paired_difference_ci(
    first_only,
    second_only,
    n,
    z=1.959963984540054,
):

    if n <= 1:

        return {
            "difference_pp":
                None,

            "lower_pp":
                None,

            "upper_pp":
                None,
        }


    # Difference is second minus first.
    difference = (
        second_only
        - first_only
    )


    mean_difference = (
        difference
        / n
    )


    sum_squares = (
        first_only
        + second_only
    )


    sample_variance = (

        sum_squares
        -
        n
        * mean_difference**2

    ) / (
        n - 1
    )


    sample_variance = max(
        0.0,
        sample_variance,
    )


    standard_error = math.sqrt(
        sample_variance
        / n
    )


    lower = (
        mean_difference
        - z
        * standard_error
    )


    upper = (
        mean_difference
        + z
        * standard_error
    )


    return {

        "difference_pp":
            mean_difference
            * 100,

        "lower_pp":
            lower
            * 100,

        "upper_pp":
            upper
            * 100,
    }


# ---------------------------------------------------------------------
# PAIRWISE ANALYSIS
# ---------------------------------------------------------------------

def paired_comparison(
    configuration_1,
    configuration_2,
    records_1,
    records_2,
    field,
):

    first = index_records(
        records_1
    )

    second = index_records(
        records_2
    )


    ids = sorted(
        first
    )


    both_correct = 0

    first_only = 0

    second_only = 0

    neither_correct = 0


    for example_id in ids:

        first_value = bool(
            first[
                example_id
            ].get(
                field
            )
        )


        second_value = bool(
            second[
                example_id
            ].get(
                field
            )
        )


        if (
            first_value
            and second_value
        ):

            both_correct += 1


        elif first_value:

            first_only += 1


        elif second_value:

            second_only += 1


        else:

            neither_correct += 1


    n = len(
        ids
    )


    p_value = (
        exact_mcnemar_pvalue(
            first_only,
            second_only,
        )
    )


    difference_ci = (
        paired_difference_ci(
            first_only,
            second_only,
            n,
        )
    )


    return {

        "comparison":
            (
                f"{configuration_1}"
                f"_vs_"
                f"{configuration_2}"
            ),

        "first_configuration":
            configuration_1,

        "second_configuration":
            configuration_2,

        "metric":
            field,

        "questions":
            n,

        "both_correct":
            both_correct,

        "first_only_correct":
            first_only,

        "second_only_correct":
            second_only,

        "neither_correct":
            neither_correct,

        "net_second_minus_first":
            (
                second_only
                - first_only
            ),

        "difference_percentage_points":
            difference_ci[
                "difference_pp"
            ],

        "difference_ci95_lower_pp":
            difference_ci[
                "lower_pp"
            ],

        "difference_ci95_upper_pp":
            difference_ci[
                "upper_pp"
            ],

        "mcnemar_exact_p":
            p_value,

        "holm_adjusted_p":
            None,
    }


# ---------------------------------------------------------------------
# HOLM MULTIPLE-COMPARISON CORRECTION
# ---------------------------------------------------------------------

def apply_holm_correction(
    comparisons,
):

    if not comparisons:
        return


    ordered_indexes = sorted(
        range(
            len(
                comparisons
            )
        ),
        key=lambda index:
            comparisons[
                index
            ][
                "mcnemar_exact_p"
            ],
    )


    number_of_tests = len(
        comparisons
    )


    previous_adjusted = 0.0


    for rank, index in enumerate(
        ordered_indexes
    ):

        raw_p = (
            comparisons[
                index
            ][
                "mcnemar_exact_p"
            ]
        )


        multiplier = (
            number_of_tests
            - rank
        )


        adjusted = min(
            1.0,
            raw_p
            * multiplier,
        )


        adjusted = max(
            adjusted,
            previous_adjusted,
        )


        comparisons[
            index
        ][
            "holm_adjusted_p"
        ] = adjusted


        previous_adjusted = (
            adjusted
        )


# ---------------------------------------------------------------------
# ALL PAIRWISE COMPARISONS
# ---------------------------------------------------------------------

def build_pairwise_results(
    all_records,
):

    configurations = (
        "A",
        "B",
        "C",
        "D",
    )


    execution_results = []

    program_results = []


    for first, second in combinations(
        configurations,
        2,
    ):

        execution_results.append(

            paired_comparison(
                first,
                second,
                all_records[
                    first
                ],
                all_records[
                    second
                ],
                "execution_correct",
            )
        )


        program_results.append(

            paired_comparison(
                first,
                second,
                all_records[
                    first
                ],
                all_records[
                    second
                ],
                "program_correct",
            )
        )


    apply_holm_correction(
        execution_results
    )


    apply_holm_correction(
        program_results
    )


    return {
        "execution_accuracy":
            execution_results,

        "program_accuracy":
            program_results,
    }


# ---------------------------------------------------------------------
# SEQUENTIAL ABLATION
# ---------------------------------------------------------------------

def build_ablation_results(
    summaries,
):

    steps = (
        ("A", "B"),
        ("B", "C"),
        ("C", "D"),
    )


    results = []


    for first, second in steps:

        first_summary = (
            summaries[
                first
            ]
        )


        second_summary = (
            summaries[
                second
            ]
        )


        first_cost = (
            first_summary[
                "cost"
            ][
                "mean_usd"
            ]
        )


        second_cost = (
            second_summary[
                "cost"
            ][
                "mean_usd"
            ]
        )


        results.append({

            "step":
                (
                    f"{first}"
                    f"_to_"
                    f"{second}"
                ),

            "from":
                first,

            "to":
                second,

            "execution_accuracy_delta_pp":
                (
                    second_summary[
                        "execution_accuracy"
                    ]
                    -
                    first_summary[
                        "execution_accuracy"
                    ]
                ),

            "program_accuracy_delta_pp":
                (
                    second_summary[
                        "program_accuracy"
                    ]
                    -
                    first_summary[
                        "program_accuracy"
                    ]
                ),

            "mean_latency_delta_seconds":
                (
                    second_summary[
                        "latency"
                    ][
                        "mean_seconds"
                    ]
                    -
                    first_summary[
                        "latency"
                    ][
                        "mean_seconds"
                    ]
                ),

            "mean_cost_delta_usd":
                (
                    (
                        second_cost
                        -
                        first_cost
                    )

                    if (
                        first_cost
                        is not None
                        and
                        second_cost
                        is not None
                    )

                    else None
                ),
        })


    return results


# ---------------------------------------------------------------------
# FORMATTING
# ---------------------------------------------------------------------

def fmt_percent(
    value,
):

    if value is None:
        return "N/A"

    return f"{value:.2f}%"


def fmt_seconds(
    value,
):

    if value is None:
        return "N/A"

    return f"{value:.2f}s"


def fmt_cost(
    value,
):

    if value is None:
        return "N/A"

    return f"${value:.6f}"


def fmt_p(
    value,
):

    if value is None:
        return "N/A"

    if value < 0.0001:
        return "<0.0001"

    return f"{value:.4f}"


# ---------------------------------------------------------------------
# TERMINAL REPORT
# ---------------------------------------------------------------------

def print_architecture_table(
    summaries,
):

    print()
    print(
        "=" * 122
    )

    print(
        "FINAL FINQA TEST-SET ARCHITECTURE RESULTS"
    )

    print(
        "=" * 122
    )


    print(
        f"{'Config':<7}"
        f"{'Architecture':<40}"
        f"{'Exec Acc':>11}"
        f"{'Prog Acc':>11}"
        f"{'Valid':>9}"
        f"{'Exec':>9}"
        f"{'Latency':>11}"
        f"{'Cost/Q':>13}"
    )


    print(
        "-" * 122
    )


    for configuration in (
        "A",
        "B",
        "C",
        "D",
    ):

        summary = summaries[
            configuration
        ]


        print(

            f"{configuration:<7}"

            f"{summary['architecture']:<40}"

            f"{fmt_percent(summary['execution_accuracy']):>11}"

            f"{fmt_percent(summary['program_accuracy']):>11}"

            f"{fmt_percent(summary['valid_rate']):>9}"

            f"{fmt_percent(summary['executable_rate']):>9}"

            f"{fmt_seconds(summary['latency']['mean_seconds']):>11}"

            f"{fmt_cost(summary['cost']['mean_usd']):>13}"
        )


    print(
        "=" * 122
    )


def print_pairwise_table(
    title,
    comparisons,
):

    print()
    print(
        title
    )

    print(
        "-" * 115
    )


    print(
        f"{'Pair':<10}"
        f"{'Both':>8}"
        f"{'1 only':>9}"
        f"{'2 only':>9}"
        f"{'Neither':>9}"
        f"{'Δ 2-1 pp':>12}"
        f"{'95% CI':>23}"
        f"{'McNemar p':>13}"
        f"{'Holm p':>11}"
    )


    for result in comparisons:

        ci = (
            f"["
            f"{result['difference_ci95_lower_pp']:.2f}, "
            f"{result['difference_ci95_upper_pp']:.2f}"
            f"]"
        )


        print(

            f"{result['comparison']:<10}"

            f"{result['both_correct']:>8}"

            f"{result['first_only_correct']:>9}"

            f"{result['second_only_correct']:>9}"

            f"{result['neither_correct']:>9}"

            f"{result['difference_percentage_points']:>12.2f}"

            f"{ci:>23}"

            f"{fmt_p(result['mcnemar_exact_p']):>13}"

            f"{fmt_p(result['holm_adjusted_p']):>11}"
        )


def print_ablation(
    results,
):

    print()
    print(
        "SEQUENTIAL ABLATION: A -> B -> C -> D"
    )

    print(
        "-" * 90
    )


    print(
        f"{'Step':<10}"
        f"{'Exec Δ pp':>13}"
        f"{'Program Δ pp':>15}"
        f"{'Latency Δ':>15}"
        f"{'Cost/Q Δ':>15}"
    )


    for result in results:

        print(

            f"{result['step']:<10}"

            f"{result['execution_accuracy_delta_pp']:>13.2f}"

            f"{result['program_accuracy_delta_pp']:>15.2f}"

            f"{result['mean_latency_delta_seconds']:>14.2f}s"

            f"{result['mean_cost_delta_usd']:>15.8f}"
        )


def print_component_results(
    summaries,
):

    print()
    print(
        "RQ4 COMPONENT CONTRIBUTION"
    )

    print(
        "-" * 70
    )


    c = summaries[
        "C"
    ][
        "component_contribution"
    ]


    print(
        "Configuration C — Verification Agent"
    )

    print(
        f"  Reasoning correct: "
        f"{c['reasoning_correct']}"
    )

    print(
        f"  Independent verifier correct: "
        f"{c['independent_verifier_correct']}"
    )

    print(
        f"  Final correct: "
        f"{c['final_correct']}"
    )

    print(
        f"  Corrected reasoning errors: "
        f"{c['verifier_corrected_errors']}"
    )

    print(
        f"  Broke correct reasoning answers: "
        f"{c['verifier_broke_correct_answers']}"
    )

    print(
        f"  Net correction effect: "
        f"{c['verifier_net_corrections']:+d}"
    )


    print()


    d = summaries[
        "D"
    ][
        "component_contribution"
    ]


    print(
        "Configuration D — Coordinator"
    )

    print(
        f"  Reasoning correct: "
        f"{d['reasoning_correct']}"
    )

    print(
        f"  Independent verifier correct: "
        f"{d['independent_verifier_correct']}"
    )

    print(
        f"  Verifier final correct: "
        f"{d['verifier_final_correct']}"
    )

    print(
        f"  Coordinator final correct: "
        f"{d['coordinator_final_correct']}"
    )

    print(
        f"  Corrected verifier errors: "
        f"{d['coordinator_corrected_errors']}"
    )

    print(
        f"  Broke verified answers: "
        f"{d['coordinator_broke_correct_answers']}"
    )

    print(
        f"  Net coordinator effect: "
        f"{d['coordinator_net_corrections']:+d}"
    )


# ---------------------------------------------------------------------
# CSV EXPORT
# ---------------------------------------------------------------------

def save_architecture_csv(
    path,
    summaries,
):

    fieldnames = [
        "configuration",
        "architecture",
        "examples",
        "execution_correct",
        "execution_accuracy",
        "execution_ci95_lower",
        "execution_ci95_upper",
        "program_correct",
        "program_accuracy",
        "program_ci95_lower",
        "program_ci95_upper",
        "valid_programs",
        "valid_rate",
        "executable_programs",
        "executable_rate",
        "api_errors",
        "parse_errors",
        "validation_failures",
        "execution_failures",
        "mean_latency_seconds",
        "median_latency_seconds",
        "latency_q1_seconds",
        "latency_q3_seconds",
        "total_input_tokens",
        "total_output_tokens",
        "mean_input_tokens",
        "mean_output_tokens",
        "total_cost_usd",
        "mean_cost_usd",
        "median_cost_usd",
    ]


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


        for configuration in (
            "A",
            "B",
            "C",
            "D",
        ):

            summary = summaries[
                configuration
            ]


            writer.writerow({

                "configuration":
                    configuration,

                "architecture":
                    summary[
                        "architecture"
                    ],

                "examples":
                    summary[
                        "examples"
                    ],

                "execution_correct":
                    summary[
                        "execution_correct"
                    ],

                "execution_accuracy":
                    summary[
                        "execution_accuracy"
                    ],

                "execution_ci95_lower":
                    summary[
                        "execution_accuracy_ci95"
                    ][
                        "lower"
                    ],

                "execution_ci95_upper":
                    summary[
                        "execution_accuracy_ci95"
                    ][
                        "upper"
                    ],

                "program_correct":
                    summary[
                        "program_correct"
                    ],

                "program_accuracy":
                    summary[
                        "program_accuracy"
                    ],

                "program_ci95_lower":
                    summary[
                        "program_accuracy_ci95"
                    ][
                        "lower"
                    ],

                "program_ci95_upper":
                    summary[
                        "program_accuracy_ci95"
                    ][
                        "upper"
                    ],

                "valid_programs":
                    summary[
                        "valid_programs"
                    ],

                "valid_rate":
                    summary[
                        "valid_rate"
                    ],

                "executable_programs":
                    summary[
                        "executable_programs"
                    ],

                "executable_rate":
                    summary[
                        "executable_rate"
                    ],

                "api_errors":
                    summary[
                        "api_errors"
                    ],

                "parse_errors":
                    summary[
                        "parse_errors"
                    ],

                "validation_failures":
                    summary[
                        "validation_failures"
                    ],

                "execution_failures":
                    summary[
                        "execution_failures"
                    ],

                "mean_latency_seconds":
                    summary[
                        "latency"
                    ][
                        "mean_seconds"
                    ],

                "median_latency_seconds":
                    summary[
                        "latency"
                    ][
                        "median_seconds"
                    ],

                "latency_q1_seconds":
                    summary[
                        "latency"
                    ][
                        "q1_seconds"
                    ],

                "latency_q3_seconds":
                    summary[
                        "latency"
                    ][
                        "q3_seconds"
                    ],

                "total_input_tokens":
                    summary[
                        "tokens"
                    ][
                        "total_input"
                    ],

                "total_output_tokens":
                    summary[
                        "tokens"
                    ][
                        "total_output"
                    ],

                "mean_input_tokens":
                    summary[
                        "tokens"
                    ][
                        "mean_input"
                    ],

                "mean_output_tokens":
                    summary[
                        "tokens"
                    ][
                        "mean_output"
                    ],

                "total_cost_usd":
                    summary[
                        "cost"
                    ][
                        "total_usd"
                    ],

                "mean_cost_usd":
                    summary[
                        "cost"
                    ][
                        "mean_usd"
                    ],

                "median_cost_usd":
                    summary[
                        "cost"
                    ][
                        "median_usd"
                    ],
            })


def save_pairwise_csv(
    path,
    pairwise,
):

    fieldnames = [
        "metric",
        "comparison",
        "first_configuration",
        "second_configuration",
        "questions",
        "both_correct",
        "first_only_correct",
        "second_only_correct",
        "neither_correct",
        "net_second_minus_first",
        "difference_percentage_points",
        "difference_ci95_lower_pp",
        "difference_ci95_upper_pp",
        "mcnemar_exact_p",
        "holm_adjusted_p",
    ]


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


        for metric_name, results in (
            (
                "execution_accuracy",
                pairwise[
                    "execution_accuracy"
                ],
            ),
            (
                "program_accuracy",
                pairwise[
                    "program_accuracy"
                ],
            ),
        ):

            for result in results:

                writer.writerow({

                    "metric":
                        metric_name,

                    "comparison":
                        result[
                            "comparison"
                        ],

                    "first_configuration":
                        result[
                            "first_configuration"
                        ],

                    "second_configuration":
                        result[
                            "second_configuration"
                        ],

                    "questions":
                        result[
                            "questions"
                        ],

                    "both_correct":
                        result[
                            "both_correct"
                        ],

                    "first_only_correct":
                        result[
                            "first_only_correct"
                        ],

                    "second_only_correct":
                        result[
                            "second_only_correct"
                        ],

                    "neither_correct":
                        result[
                            "neither_correct"
                        ],

                    "net_second_minus_first":
                        result[
                            "net_second_minus_first"
                        ],

                    "difference_percentage_points":
                        result[
                            "difference_percentage_points"
                        ],

                    "difference_ci95_lower_pp":
                        result[
                            "difference_ci95_lower_pp"
                        ],

                    "difference_ci95_upper_pp":
                        result[
                            "difference_ci95_upper_pp"
                        ],

                    "mcnemar_exact_p":
                        result[
                            "mcnemar_exact_p"
                        ],

                    "holm_adjusted_p":
                        result[
                            "holm_adjusted_p"
                        ],
                })


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


    print(
        "Final thesis experiment analysis"
    )

    print(
        f"Dataset: "
        f"{args.split}[{args.start}:"
        f"{args.start + args.count}]"
    )

    print(
        "No API calls made."
    )

    print()


    # -----------------------------------------------------------------
    # FIND FINAL RUNS
    # -----------------------------------------------------------------

    source_files = {}

    all_records = {}


    for configuration in (
        "A",
        "B",
        "C",
        "D",
    ):

        path, records = (
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


        all_records[
            configuration
        ] = records


        print(
            f"Configuration {configuration}: "
            f"{path.name}"
        )


    # -----------------------------------------------------------------
    # ALIGNMENT
    # -----------------------------------------------------------------

    aligned_ids = verify_alignment(
        all_records
    )


    print()

    print(
        f"Aligned questions: "
        f"{len(aligned_ids)}"
    )


    # -----------------------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------------------

    summaries = {

        configuration:
            summarize_architecture(
                configuration,
                all_records[
                    configuration
                ],
            )

        for configuration in (
            "A",
            "B",
            "C",
            "D",
        )
    }


    # -----------------------------------------------------------------
    # PAIRED STATISTICS
    # -----------------------------------------------------------------

    pairwise = (
        build_pairwise_results(
            all_records
        )
    )


    # -----------------------------------------------------------------
    # ABLATION
    # -----------------------------------------------------------------

    ablation = (
        build_ablation_results(
            summaries
        )
    )


    # -----------------------------------------------------------------
    # TERMINAL OUTPUT
    # -----------------------------------------------------------------

    print_architecture_table(
        summaries
    )


    print_pairwise_table(
        "PAIRED EXECUTION-ACCURACY COMPARISONS",
        pairwise[
            "execution_accuracy"
        ],
    )


    print_pairwise_table(
        "PAIRED PROGRAM-ACCURACY COMPARISONS",
        pairwise[
            "program_accuracy"
        ],
    )


    print_ablation(
        ablation
    )


    print_component_results(
        summaries
    )


    # -----------------------------------------------------------------
    # SAVE
    # -----------------------------------------------------------------

    timestamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%S_%fZ"
    )


    prefix = (
        f"final_test_analysis_"
        f"{args.split}_"
        f"{args.start}_"
        f"{args.start + args.count}_"
        f"{timestamp}"
    )


    json_path = (
        RESULTS_DIR
        / f"{prefix}.json"
    )


    architecture_csv_path = (
        RESULTS_DIR
        / (
            f"final_test_architecture_summary_"
            f"{timestamp}.csv"
        )
    )


    pairwise_csv_path = (
        RESULTS_DIR
        / (
            f"final_test_pairwise_statistics_"
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

        "question_count":
            len(
                aligned_ids
            ),

        "source_files": {

            configuration:
                str(path)

            for configuration, path
            in source_files.items()
        },

        "architectures":
            summaries,

        "pairwise_statistics":
            pairwise,

        "sequential_ablation":
            ablation,

        "statistical_method": {

            "paired_test":
                (
                    "Two-sided exact McNemar test "
                    "on question-level correctness"
                ),

            "multiple_comparison_correction":
                (
                    "Holm correction applied "
                    "separately to six execution "
                    "and six program comparisons"
                ),

            "accuracy_confidence_interval":
                "Wilson 95% confidence interval",

            "paired_difference_confidence_interval":
                (
                    "Approximate 95% confidence "
                    "interval for paired binary "
                    "difference"
                ),
        },
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


    save_architecture_csv(
        architecture_csv_path,
        summaries,
    )


    save_pairwise_csv(
        pairwise_csv_path,
        pairwise,
    )


    print()
    print(
        "=" * 70
    )

    print(
        "ANALYSIS COMPLETE"
    )

    print(
        "=" * 70
    )


    print(
        f"JSON: "
        f"{json_path}"
    )

    print(
        f"Architecture CSV: "
        f"{architecture_csv_path}"
    )

    print(
        f"Pairwise statistics CSV: "
        f"{pairwise_csv_path}"
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