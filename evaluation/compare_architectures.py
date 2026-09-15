"""Offline comparison of thesis architectures A/B/C/D.

Compares the latest saved JSONL results for the same FinQA dev slice.

Default experiment:
    dev[75:175]

Architectures:
    A = Single-Agent direct program
    B = Reasoning Agent
    C = Reasoning + Independent Verification
    D = Reasoning + Verification + Coordinator

Run:

    python -m evaluation.compare_architectures

Optional custom slice:

    python -m evaluation.compare_architectures --start 75 --end 175

This script makes NO OpenRouter/API calls.
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results"


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

CONFIGURATIONS = {
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
            "Offline comparison of "
            "FinQA architectures A/B/C/D."
        )
    )

    parser.add_argument(
        "--start",
        type=int,
        default=75,
        help="Starting dev index. Default: 75",
    )

    parser.add_argument(
        "--end",
        type=int,
        default=175,
        help="Exclusive ending dev index. Default: 175",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# FILE DISCOVERY
# ---------------------------------------------------------------------

def find_latest_result_file(
    configuration,
    start,
    end,
):
    """Find latest saved result file for one architecture."""

    pattern = (
        f"configuration_{configuration}_"
        f"dev_{start}_{end}_*.jsonl"
    )

    candidates = list(
        RESULTS_DIR.glob(pattern)
    )

    if not candidates:

        raise FileNotFoundError(
            f"No result file found for Configuration "
            f"{configuration} using pattern:\n"
            f"{pattern}"
        )

    return max(
        candidates,
        key=lambda path: path.stat().st_mtime,
    )


# ---------------------------------------------------------------------
# LOAD JSONL
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
# BASIC HELPERS
# ---------------------------------------------------------------------

def safe_mean(values):

    values = [
        value
        for value in values
        if isinstance(
            value,
            (int, float),
        )
        and not isinstance(
            value,
            bool,
        )
    ]

    if not values:
        return None

    return statistics.mean(
        values
    )


def safe_sum(values):

    values = [
        value
        for value in values
        if isinstance(
            value,
            (int, float),
        )
        and not isinstance(
            value,
            bool,
        )
    ]

    if not values:
        return None

    return sum(values)


def percentage(
    numerator,
    denominator,
):

    if not denominator:
        return None

    return (
        numerator
        / denominator
        * 100
    )


# ---------------------------------------------------------------------
# ARCHITECTURE SUMMARY
# ---------------------------------------------------------------------

def summarize_configuration(
    records,
):

    total = len(records)

    valid = sum(
        bool(record.get("program_valid"))
        for record in records
    )

    executable = sum(
        bool(record.get("executable"))
        for record in records
    )

    execution_correct = sum(
        bool(record.get("execution_correct"))
        for record in records
    )

    program_correct = sum(
        bool(record.get("program_correct"))
        for record in records
    )

    symbolic_correct = sum(
        bool(
            record.get(
                "program_symbolically_correct"
            )
        )
        for record in records
    )

    api_errors = sum(
        record.get("api_error")
        is not None
        for record in records
    )

    latencies = [
        record.get("latency")
        for record in records
    ]

    input_tokens = [
        record.get("input_tokens")
        for record in records
    ]

    output_tokens = [
        record.get("output_tokens")
        for record in records
    ]

    costs = [
        record.get(
            "effective_cost_usd"
        )
        for record in records
    ]

    return {

        "examples":
            total,

        "valid_count":
            valid,

        "valid_rate":
            percentage(
                valid,
                total,
            ),

        "executable_count":
            executable,

        "executable_rate":
            percentage(
                executable,
                total,
            ),

        "execution_correct":
            execution_correct,

        "execution_accuracy":
            percentage(
                execution_correct,
                total,
            ),

        "symbolically_correct":
            symbolic_correct,

        "program_correct":
            program_correct,

        "program_accuracy":
            percentage(
                program_correct,
                total,
            ),

        "api_errors":
            api_errors,

        "api_error_rate":
            percentage(
                api_errors,
                total,
            ),

        "total_input_tokens":
            safe_sum(
                input_tokens
            ),

        "average_input_tokens":
            safe_mean(
                input_tokens
            ),

        "total_output_tokens":
            safe_sum(
                output_tokens
            ),

        "average_output_tokens":
            safe_mean(
                output_tokens
            ),

        "average_latency_seconds":
            safe_mean(
                latencies
            ),

        "total_cost_usd":
            safe_sum(
                costs
            ),

        "average_cost_usd":
            safe_mean(
                costs
            ),
    }


# ---------------------------------------------------------------------
# ALIGN QUESTION-LEVEL RESULTS
# ---------------------------------------------------------------------

def index_records(records):

    indexed = {}

    for record in records:

        example_id = record.get(
            "example_id"
        )

        if example_id is None:

            raise ValueError(
                "A record is missing example_id."
            )

        if example_id in indexed:

            raise ValueError(
                f"Duplicate example_id: "
                f"{example_id}"
            )

        indexed[example_id] = record

    return indexed


def verify_alignment(
    all_records,
):

    id_sets = {
        configuration:
            set(
                index_records(records)
            )
        for configuration, records
        in all_records.items()
    }

    reference_ids = (
        id_sets["A"]
    )

    for configuration in (
        "B",
        "C",
        "D",
    ):

        if (
            id_sets[configuration]
            != reference_ids
        ):

            missing = (
                reference_ids
                - id_sets[configuration]
            )

            extra = (
                id_sets[configuration]
                - reference_ids
            )

            raise ValueError(
                f"Question alignment mismatch "
                f"for Configuration "
                f"{configuration}.\n"
                f"Missing: {len(missing)}\n"
                f"Extra: {len(extra)}"
            )

    return sorted(
        reference_ids
    )


# ---------------------------------------------------------------------
# PAIRED COMPARISON
# ---------------------------------------------------------------------

def paired_result(
    records_a,
    records_b,
    metric,
):

    a_index = index_records(
        records_a
    )

    b_index = index_records(
        records_b
    )

    ids = sorted(
        set(a_index)
        & set(b_index)
    )

    both_correct = 0

    a_only = 0

    b_only = 0

    neither = 0


    for example_id in ids:

        a_correct = bool(
            a_index[
                example_id
            ].get(metric)
        )

        b_correct = bool(
            b_index[
                example_id
            ].get(metric)
        )


        if (
            a_correct
            and b_correct
        ):

            both_correct += 1


        elif a_correct:

            a_only += 1


        elif b_correct:

            b_only += 1


        else:

            neither += 1


    return {

        "n":
            len(ids),

        "both_correct":
            both_correct,

        "first_only":
            a_only,

        "second_only":
            b_only,

        "neither_correct":
            neither,

        "net_second_advantage":
            b_only - a_only,
    }


# ---------------------------------------------------------------------
# ABLATION CONTRIBUTION
# ---------------------------------------------------------------------

def metric_delta(
    newer,
    older,
    metric,
):

    new_value = (
        newer.get(metric)
    )

    old_value = (
        older.get(metric)
    )

    if (
        new_value is None
        or old_value is None
    ):

        return None

    return (
        new_value
        - old_value
    )


def build_ablation_summary(
    summaries,
):

    comparisons = [
        (
            "B_minus_A",
            "B",
            "A",
        ),
        (
            "C_minus_B",
            "C",
            "B",
        ),
        (
            "D_minus_C",
            "D",
            "C",
        ),
    ]

    result = {}


    for (
        name,
        newer_key,
        older_key,
    ) in comparisons:

        newer = summaries[
            newer_key
        ]

        older = summaries[
            older_key
        ]

        result[name] = {

            "execution_accuracy_pp":
                metric_delta(
                    newer,
                    older,
                    "execution_accuracy",
                ),

            "program_accuracy_pp":
                metric_delta(
                    newer,
                    older,
                    "program_accuracy",
                ),

            "valid_rate_pp":
                metric_delta(
                    newer,
                    older,
                    "valid_rate",
                ),

            "executable_rate_pp":
                metric_delta(
                    newer,
                    older,
                    "executable_rate",
                ),

            "latency_seconds":
                metric_delta(
                    newer,
                    older,
                    "average_latency_seconds",
                ),

            "average_cost_usd":
                metric_delta(
                    newer,
                    older,
                    "average_cost_usd",
                ),
        }

    return result


# ---------------------------------------------------------------------
# COMPONENT CONTRIBUTION
# ---------------------------------------------------------------------

def component_contribution(
    all_records,
):

    c_records = (
        all_records["C"]
    )

    d_records = (
        all_records["D"]
    )


    verifier_corrected = sum(
        bool(
            record.get(
                "verifier_corrected_error"
            )
        )
        for record in c_records
    )


    verifier_broke = sum(
        bool(
            record.get(
                "verifier_broke_correct_answer"
            )
        )
        for record in c_records
    )


    coordinator_corrected = sum(
        bool(
            record.get(
                "coordinator_corrected_verification_error"
            )
        )
        for record in d_records
    )


    coordinator_broke = sum(
        bool(
            record.get(
                "coordinator_broke_verified_answer"
            )
        )
        for record in d_records
    )


    return {

        "configuration_C_verifier":
            {
                "corrected_errors":
                    verifier_corrected,

                "broke_correct_answers":
                    verifier_broke,
            },

        "configuration_D_coordinator":
            {
                "corrected_errors":
                    coordinator_corrected,

                "broke_correct_answers":
                    coordinator_broke,
            },
    }


# ---------------------------------------------------------------------
# FORMATTERS
# ---------------------------------------------------------------------

def fmt_percent(value):

    if value is None:
        return "N/A"

    return f"{value:.1f}%"


def fmt_seconds(value):

    if value is None:
        return "N/A"

    return f"{value:.2f}s"


def fmt_cost(value):

    if value is None:
        return "N/A"

    return f"${value:.6f}"


def print_table(
    summaries,
):

    print()
    print(
        "=" * 108
    )

    print(
        "ARCHITECTURE COMPARISON"
    )

    print(
        "=" * 108
    )


    header = (
        f"{'Config':<8}"
        f"{'Architecture':<38}"
        f"{'Exec Acc':>11}"
        f"{'Prog Acc':>11}"
        f"{'Valid':>9}"
        f"{'Exec':>9}"
        f"{'Latency':>10}"
        f"{'Avg Cost':>12}"
    )

    print(
        header
    )

    print(
        "-" * 108
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

        architecture = (
            CONFIGURATIONS[
                configuration
            ]
        )


        print(

            f"{configuration:<8}"

            f"{architecture:<38}"

            f"{fmt_percent(summary['execution_accuracy']):>11}"

            f"{fmt_percent(summary['program_accuracy']):>11}"

            f"{fmt_percent(summary['valid_rate']):>9}"

            f"{fmt_percent(summary['executable_rate']):>9}"

            f"{fmt_seconds(summary['average_latency_seconds']):>10}"

            f"{fmt_cost(summary['average_cost_usd']):>12}"
        )


    print(
        "=" * 108
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


    if args.end <= args.start:

        raise ValueError(
            "--end must be greater than --start."
        )


    # -----------------------------------------------------------------
    # FIND FILES
    # -----------------------------------------------------------------

    result_files = {}

    all_records = {}


    for configuration in (
        "A",
        "B",
        "C",
        "D",
    ):

        path = find_latest_result_file(
            configuration,
            args.start,
            args.end,
        )

        result_files[
            configuration
        ] = path


        all_records[
            configuration
        ] = load_jsonl(
            path
        )


    # -----------------------------------------------------------------
    # ALIGNMENT CHECK
    # -----------------------------------------------------------------

    aligned_ids = (
        verify_alignment(
            all_records
        )
    )


    print(
        "Offline architecture comparison"
    )

    print(
        f"Dataset slice: "
        f"dev[{args.start}:{args.end}]"
    )

    print(
        f"Aligned questions: "
        f"{len(aligned_ids)}"
    )

    print(
        "No API calls made."
    )

    print()


    for configuration in (
        "A",
        "B",
        "C",
        "D",
    ):

        print(
            f"{configuration}: "
            f"{result_files[configuration].name}"
        )


    # -----------------------------------------------------------------
    # SUMMARIES
    # -----------------------------------------------------------------

    summaries = {

        configuration:
            summarize_configuration(
                all_records[
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


    print_table(
        summaries
    )


    # -----------------------------------------------------------------
    # PAIRED EXECUTION COMPARISONS
    # -----------------------------------------------------------------

    pair_names = [
        ("A", "B"),
        ("A", "C"),
        ("A", "D"),
        ("B", "C"),
        ("B", "D"),
        ("C", "D"),
    ]


    execution_pairs = {}

    program_pairs = {}


    print()
    print(
        "PAIRED QUESTION-LEVEL EXECUTION RESULTS"
    )

    print(
        "-" * 70
    )


    for first, second in pair_names:

        comparison = paired_result(

            all_records[first],

            all_records[second],

            "execution_correct",
        )


        execution_pairs[
            f"{first}_vs_{second}"
        ] = comparison


        print(
            f"{first} vs {second}: "
            f"both={comparison['both_correct']}, "
            f"{first}-only={comparison['first_only']}, "
            f"{second}-only={comparison['second_only']}, "
            f"neither={comparison['neither_correct']}, "
            f"net advantage {second}="
            f"{comparison['net_second_advantage']:+d}"
        )


    print()
    print(
        "PAIRED QUESTION-LEVEL PROGRAM RESULTS"
    )

    print(
        "-" * 70
    )


    for first, second in pair_names:

        comparison = paired_result(

            all_records[first],

            all_records[second],

            "program_correct",
        )


        program_pairs[
            f"{first}_vs_{second}"
        ] = comparison


        print(
            f"{first} vs {second}: "
            f"both={comparison['both_correct']}, "
            f"{first}-only={comparison['first_only']}, "
            f"{second}-only={comparison['second_only']}, "
            f"neither={comparison['neither_correct']}, "
            f"net advantage {second}="
            f"{comparison['net_second_advantage']:+d}"
        )


    # -----------------------------------------------------------------
    # ABLATION
    # -----------------------------------------------------------------

    ablation = build_ablation_summary(
        summaries
    )


    print()
    print(
        "ABLATION CONTRIBUTION"
    )

    print(
        "-" * 70
    )


    for name, values in ablation.items():

        print(
            f"{name}: "
            f"execution "
            f"{values['execution_accuracy_pp']:+.1f} pp, "
            f"program "
            f"{values['program_accuracy_pp']:+.1f} pp, "
            f"latency "
            f"{values['latency_seconds']:+.2f}s, "
            f"avg cost "
            f"{values['average_cost_usd']:+.6f}"
        )


    # -----------------------------------------------------------------
    # COMPONENT CONTRIBUTION
    # -----------------------------------------------------------------

    components = (
        component_contribution(
            all_records
        )
    )


    print()
    print(
        "COMPONENT CONTRIBUTION"
    )

    print(
        "-" * 70
    )


    verifier = (
        components[
            "configuration_C_verifier"
        ]
    )


    coordinator = (
        components[
            "configuration_D_coordinator"
        ]
    )


    print(
        "Configuration C verifier:"
    )

    print(
        f"  corrected errors: "
        f"{verifier['corrected_errors']}"
    )

    print(
        f"  broke correct answers: "
        f"{verifier['broke_correct_answers']}"
    )


    print(
        "Configuration D coordinator:"
    )

    print(
        f"  corrected errors: "
        f"{coordinator['corrected_errors']}"
    )

    print(
        f"  broke correct answers: "
        f"{coordinator['broke_correct_answers']}"
    )


    # -----------------------------------------------------------------
    # SAVE COMPARISON
    # -----------------------------------------------------------------

    timestamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%S_%fZ"
    )


    output_path = (
        RESULTS_DIR
        / (
            "architecture_comparison_"
            f"dev_{args.start}_{args.end}_"
            f"{timestamp}.json"
        )
    )


    output = {

        "dataset_split":
            "dev",

        "start_index":
            args.start,

        "end_index":
            args.end,

        "question_count":
            len(aligned_ids),

        "source_files":
            {
                configuration:
                    str(path)

                for configuration, path
                in result_files.items()
            },

        "architecture_summaries":
            summaries,

        "paired_execution_results":
            execution_pairs,

        "paired_program_results":
            program_pairs,

        "ablation":
            ablation,

        "component_contribution":
            components,
    }


    with output_path.open(
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


    print()
    print(
        f"Comparison saved to:\n"
        f"{output_path}"
    )


# ---------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()