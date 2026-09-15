"""RQ2 reliability and repeated-run analysis.

Analyzes repeated A/B/C/D FinQA experiments offline.

Default:
    dev[75:175]
    3 runs per architecture

Metrics:
- mean execution accuracy + standard deviation
- mean program accuracy + standard deviation
- validity/executability
- question-level correct/incorrect consistency
- correct <-> incorrect flip rate
- exact generated-program agreement
- normalized-program agreement
- API / parsing / validation / execution failure rates
- latency
- input/output tokens
- API cost
- verifier/coordinator contribution

NO API calls are made.

Run:

    python -m evaluation.evaluate_reliability
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results"


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
            "Offline RQ2 reliability analysis "
            "for repeated A/B/C/D experiments."
        )
    )

    parser.add_argument(
        "--start",
        type=int,
        default=75,
    )

    parser.add_argument(
        "--end",
        type=int,
        default=175,
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of repeated runs per architecture.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# FILE DISCOVERY
# ---------------------------------------------------------------------

def find_run_files(
    configuration,
    start,
    end,
    required_runs,
):

    pattern = (
        f"configuration_{configuration}_"
        f"dev_{start}_{end}_*.jsonl"
    )

    files = list(
        RESULTS_DIR.glob(pattern)
    )

    if len(files) < required_runs:

        raise FileNotFoundError(
            f"Configuration {configuration}: "
            f"found only {len(files)} matching files; "
            f"{required_runs} required.\n"
            f"Pattern: {pattern}"
        )

    # Most recent N runs.
    files = sorted(
        files,
        key=lambda path: path.stat().st_mtime,
    )[-required_runs:]

    # Return them chronologically as Run 1, Run 2, Run 3.
    return sorted(
        files,
        key=lambda path: path.stat().st_mtime,
    )


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
# HELPERS
# ---------------------------------------------------------------------

def is_number(value):

    return (
        isinstance(
            value,
            (int, float),
        )
        and not isinstance(
            value,
            bool,
        )
    )


def numeric_values(values):

    return [
        value
        for value in values
        if is_number(value)
    ]


def safe_mean(values):

    values = numeric_values(
        values
    )

    if not values:
        return None

    return statistics.mean(
        values
    )


def safe_std(values):

    values = numeric_values(
        values
    )

    if len(values) < 2:
        return 0.0

    return statistics.stdev(
        values
    )


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


def index_records(records):

    indexed = {}

    for record in records:

        example_id = record.get(
            "example_id"
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


# ---------------------------------------------------------------------
# ALIGNMENT
# ---------------------------------------------------------------------

def verify_all_runs_aligned(
    all_runs,
):

    reference_ids = None

    for configuration, runs in all_runs.items():

        for run_index, records in enumerate(
            runs,
            start=1,
        ):

            ids = set(
                index_records(
                    records
                )
            )

            if reference_ids is None:

                reference_ids = ids

            elif ids != reference_ids:

                missing = (
                    reference_ids - ids
                )

                extra = (
                    ids - reference_ids
                )

                raise ValueError(
                    f"Alignment mismatch: "
                    f"Configuration {configuration}, "
                    f"Run {run_index}.\n"
                    f"Missing: {len(missing)}\n"
                    f"Extra: {len(extra)}"
                )

    return sorted(
        reference_ids
    )


# ---------------------------------------------------------------------
# ERROR DETECTION
# ---------------------------------------------------------------------

def has_parse_error(record):

    possible_fields = (
        "parse_error",
        "reasoning_parse_error",
        "verification_parse_error",
        "coordinator_parse_error",
    )

    return any(
        record.get(field)
        is not None

        for field in possible_fields
    )


def has_validation_failure(record):

    return (
        record.get(
            "validation_error"
        )
        is not None
        or
        not bool(
            record.get(
                "program_valid"
            )
        )
    )


def has_execution_failure(record):

    # Only count an execution failure if the program
    # was valid but deterministic execution failed.
    return bool(
        record.get(
            "program_valid"
        )
    ) and not bool(
        record.get(
            "executable"
        )
    )


# ---------------------------------------------------------------------
# SINGLE RUN SUMMARY
# ---------------------------------------------------------------------

def summarize_run(records):

    n = len(records)

    execution_correct = sum(
        bool(
            record.get(
                "execution_correct"
            )
        )
        for record in records
    )

    program_correct = sum(
        bool(
            record.get(
                "program_correct"
            )
        )
        for record in records
    )

    valid = sum(
        bool(
            record.get(
                "program_valid"
            )
        )
        for record in records
    )

    executable = sum(
        bool(
            record.get(
                "executable"
            )
        )
        for record in records
    )

    api_errors = sum(
        record.get(
            "api_error"
        )
        is not None
        for record in records
    )

    parse_errors = sum(
        has_parse_error(
            record
        )
        for record in records
    )

    validation_failures = sum(
        has_validation_failure(
            record
        )
        for record in records
    )

    execution_failures = sum(
        has_execution_failure(
            record
        )
        for record in records
    )

    latency_values = [
        record.get("latency")
        for record in records
    ]

    input_token_values = [
        record.get(
            "input_tokens"
        )
        for record in records
    ]

    output_token_values = [
        record.get(
            "output_tokens"
        )
        for record in records
    ]

    cost_values = [
        record.get(
            "effective_cost_usd"
        )
        for record in records
    ]

    return {

        "examples":
            n,

        "execution_correct":
            execution_correct,

        "execution_accuracy":
            percentage(
                execution_correct,
                n,
            ),

        "program_correct":
            program_correct,

        "program_accuracy":
            percentage(
                program_correct,
                n,
            ),

        "valid_count":
            valid,

        "valid_rate":
            percentage(
                valid,
                n,
            ),

        "executable_count":
            executable,

        "executable_rate":
            percentage(
                executable,
                n,
            ),

        "api_error_count":
            api_errors,

        "api_error_rate":
            percentage(
                api_errors,
                n,
            ),

        "parse_error_count":
            parse_errors,

        "parse_error_rate":
            percentage(
                parse_errors,
                n,
            ),

        "validation_failure_count":
            validation_failures,

        "validation_failure_rate":
            percentage(
                validation_failures,
                n,
            ),

        "execution_failure_count":
            execution_failures,

        "execution_failure_rate":
            percentage(
                execution_failures,
                n,
            ),

        "average_latency_seconds":
            safe_mean(
                latency_values
            ),

        "average_input_tokens":
            safe_mean(
                input_token_values
            ),

        "average_output_tokens":
            safe_mean(
                output_token_values
            ),

        "average_cost_usd":
            safe_mean(
                cost_values
            ),
    }


# ---------------------------------------------------------------------
# QUESTION-LEVEL BOOLEAN CONSISTENCY
# ---------------------------------------------------------------------

def boolean_consistency(
    runs,
    field,
):

    indexed_runs = [
        index_records(
            records
        )
        for records in runs
    ]

    ids = sorted(
        indexed_runs[0]
    )

    stable_true = 0

    stable_false = 0

    flip_count = 0

    patterns = {}


    for example_id in ids:

        values = tuple(
            bool(
                run[
                    example_id
                ].get(field)
            )
            for run in indexed_runs
        )

        pattern = "".join(
            "1" if value else "0"
            for value in values
        )

        patterns[
            pattern
        ] = (
            patterns.get(
                pattern,
                0,
            )
            + 1
        )


        if all(values):

            stable_true += 1


        elif not any(values):

            stable_false += 1


        else:

            flip_count += 1


    n = len(ids)

    stable_count = (
        stable_true
        + stable_false
    )

    return {

        "questions":
            n,

        "stable_true":
            stable_true,

        "stable_false":
            stable_false,

        "stable_total":
            stable_count,

        "consistency_rate":
            percentage(
                stable_count,
                n,
            ),

        "flip_count":
            flip_count,

        "flip_rate":
            percentage(
                flip_count,
                n,
            ),

        "patterns":
            patterns,
    }


# ---------------------------------------------------------------------
# PROGRAM AGREEMENT
# ---------------------------------------------------------------------

def program_agreement(
    runs,
    field,
):

    indexed_runs = [
        index_records(
            records
        )
        for records in runs
    ]

    ids = sorted(
        indexed_runs[0]
    )

    agreement_count = 0


    for example_id in ids:

        programs = [

            run[
                example_id
            ].get(field)

            for run in indexed_runs
        ]


        if len(
            set(programs)
        ) == 1:

            agreement_count += 1


    n = len(ids)

    return {

        "questions":
            n,

        "agreement_count":
            agreement_count,

        "agreement_rate":
            percentage(
                agreement_count,
                n,
            ),
    }


# ---------------------------------------------------------------------
# COMPONENT CONTRIBUTION
# ---------------------------------------------------------------------

def component_reliability(
    configuration,
    runs,
):

    if configuration == "C":

        corrected = []

        broke = []


        for records in runs:

            corrected.append(
                sum(
                    bool(
                        record.get(
                            "verifier_corrected_error"
                        )
                    )
                    for record in records
                )
            )

            broke.append(
                sum(
                    bool(
                        record.get(
                            "verifier_broke_correct_answer"
                        )
                    )
                    for record in records
                )
            )


        return {

            "component":
                "Verification Agent",

            "corrected_errors_per_run":
                corrected,

            "mean_corrected_errors":
                safe_mean(
                    corrected
                ),

            "broke_correct_answers_per_run":
                broke,

            "mean_broke_correct_answers":
                safe_mean(
                    broke
                ),
        }


    if configuration == "D":

        corrected = []

        broke = []


        for records in runs:

            corrected.append(
                sum(
                    bool(
                        record.get(
                            "coordinator_corrected_verification_error"
                        )
                    )
                    for record in records
                )
            )

            broke.append(
                sum(
                    bool(
                        record.get(
                            "coordinator_broke_verified_answer"
                        )
                    )
                    for record in records
                )
            )


        return {

            "component":
                "Coordinator",

            "corrected_errors_per_run":
                corrected,

            "mean_corrected_errors":
                safe_mean(
                    corrected
                ),

            "broke_correct_answers_per_run":
                broke,

            "mean_broke_correct_answers":
                safe_mean(
                    broke
                ),
        }


    return None


# ---------------------------------------------------------------------
# ARCHITECTURE RELIABILITY
# ---------------------------------------------------------------------

def analyze_architecture(
    configuration,
    runs,
):

    run_summaries = [
        summarize_run(
            records
        )
        for records in runs
    ]


    execution_accuracies = [
        summary[
            "execution_accuracy"
        ]
        for summary in run_summaries
    ]


    program_accuracies = [
        summary[
            "program_accuracy"
        ]
        for summary in run_summaries
    ]


    valid_rates = [
        summary[
            "valid_rate"
        ]
        for summary in run_summaries
    ]


    executable_rates = [
        summary[
            "executable_rate"
        ]
        for summary in run_summaries
    ]


    latencies = [
        summary[
            "average_latency_seconds"
        ]
        for summary in run_summaries
    ]


    input_tokens = [
        summary[
            "average_input_tokens"
        ]
        for summary in run_summaries
    ]


    output_tokens = [
        summary[
            "average_output_tokens"
        ]
        for summary in run_summaries
    ]


    costs = [
        summary[
            "average_cost_usd"
        ]
        for summary in run_summaries
    ]


    api_error_rates = [
        summary[
            "api_error_rate"
        ]
        for summary in run_summaries
    ]


    parse_error_rates = [
        summary[
            "parse_error_rate"
        ]
        for summary in run_summaries
    ]


    validation_failure_rates = [
        summary[
            "validation_failure_rate"
        ]
        for summary in run_summaries
    ]


    execution_failure_rates = [
        summary[
            "execution_failure_rate"
        ]
        for summary in run_summaries
    ]


    return {

        "architecture":
            CONFIGURATIONS[
                configuration
            ],

        "number_of_runs":
            len(runs),

        "per_run":
            run_summaries,


        # ---------------------------------------------------------
        # ACCURACY
        # ---------------------------------------------------------

        "execution_accuracy": {

            "per_run":
                execution_accuracies,

            "mean":
                safe_mean(
                    execution_accuracies
                ),

            "sample_std":
                safe_std(
                    execution_accuracies
                ),
        },


        "program_accuracy": {

            "per_run":
                program_accuracies,

            "mean":
                safe_mean(
                    program_accuracies
                ),

            "sample_std":
                safe_std(
                    program_accuracies
                ),
        },


        # ---------------------------------------------------------
        # STRUCTURAL RELIABILITY
        # ---------------------------------------------------------

        "valid_rate": {

            "per_run":
                valid_rates,

            "mean":
                safe_mean(
                    valid_rates
                ),

            "sample_std":
                safe_std(
                    valid_rates
                ),
        },


        "executable_rate": {

            "per_run":
                executable_rates,

            "mean":
                safe_mean(
                    executable_rates
                ),

            "sample_std":
                safe_std(
                    executable_rates
                ),
        },


        # ---------------------------------------------------------
        # QUESTION-LEVEL CONSISTENCY
        # ---------------------------------------------------------

        "execution_consistency":
            boolean_consistency(
                runs,
                "execution_correct",
            ),


        "program_correctness_consistency":
            boolean_consistency(
                runs,
                "program_correct",
            ),


        "validity_consistency":
            boolean_consistency(
                runs,
                "program_valid",
            ),


        "executability_consistency":
            boolean_consistency(
                runs,
                "executable",
            ),


        # ---------------------------------------------------------
        # PROGRAM OUTPUT CONSISTENCY
        # ---------------------------------------------------------

        "exact_program_agreement":
            program_agreement(
                runs,
                "generated_program",
            ),


        "normalized_program_agreement":
            program_agreement(
                runs,
                "normalized_program",
            ),


        # ---------------------------------------------------------
        # FAILURE RATES
        # ---------------------------------------------------------

        "failure_rates": {

            "mean_api_error_rate":
                safe_mean(
                    api_error_rates
                ),

            "mean_parse_error_rate":
                safe_mean(
                    parse_error_rates
                ),

            "mean_validation_failure_rate":
                safe_mean(
                    validation_failure_rates
                ),

            "mean_execution_failure_rate":
                safe_mean(
                    execution_failure_rates
                ),
        },


        # ---------------------------------------------------------
        # EFFICIENCY
        # ---------------------------------------------------------

        "efficiency": {

            "average_latency_seconds": {

                "per_run":
                    latencies,

                "mean":
                    safe_mean(
                        latencies
                    ),

                "sample_std":
                    safe_std(
                        latencies
                    ),
            },


            "average_input_tokens": {

                "per_run":
                    input_tokens,

                "mean":
                    safe_mean(
                        input_tokens
                    ),
            },


            "average_output_tokens": {

                "per_run":
                    output_tokens,

                "mean":
                    safe_mean(
                        output_tokens
                    ),
            },


            "average_cost_usd": {

                "per_run":
                    costs,

                "mean":
                    safe_mean(
                        costs
                    ),

                "sample_std":
                    safe_std(
                        costs
                    ),
            },
        },


        "component_reliability":
            component_reliability(
                configuration,
                runs,
            ),
    }


# ---------------------------------------------------------------------
# FORMATTING
# ---------------------------------------------------------------------

def fmt_percent(
    value,
):

    if value is None:
        return "N/A"

    return f"{value:.2f}%"


def fmt_mean_std(
    mean,
    std,
):

    if mean is None:
        return "N/A"

    return (
        f"{mean:.2f} ± "
        f"{std:.2f}"
    )


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

    return f"${value:.8f}"


# ---------------------------------------------------------------------
# TERMINAL REPORT
# ---------------------------------------------------------------------

def print_report(
    analyses,
):

    print()
    print(
        "=" * 120
    )

    print(
        "RQ2 — REPEATED-RUN RELIABILITY ANALYSIS"
    )

    print(
        "=" * 120
    )


    print(
        f"{'Config':<8}"
        f"{'Architecture':<39}"
        f"{'Exec Mean±SD':>16}"
        f"{'Prog Mean±SD':>16}"
        f"{'Exec Cons.':>13}"
        f"{'Prog Agree':>13}"
        f"{'Latency':>10}"
        f"{'Avg Cost':>14}"
    )

    print(
        "-" * 120
    )


    for configuration in (
        "A",
        "B",
        "C",
        "D",
    ):

        analysis = analyses[
            configuration
        ]


        exec_data = (
            analysis[
                "execution_accuracy"
            ]
        )


        program_data = (
            analysis[
                "program_accuracy"
            ]
        )


        exec_consistency = (
            analysis[
                "execution_consistency"
            ][
                "consistency_rate"
            ]
        )


        program_agreement_rate = (
            analysis[
                "exact_program_agreement"
            ][
                "agreement_rate"
            ]
        )


        latency = (
            analysis[
                "efficiency"
            ][
                "average_latency_seconds"
            ][
                "mean"
            ]
        )


        cost = (
            analysis[
                "efficiency"
            ][
                "average_cost_usd"
            ][
                "mean"
            ]
        )


        print(

            f"{configuration:<8}"

            f"{analysis['architecture']:<39}"

            f"{fmt_mean_std(exec_data['mean'], exec_data['sample_std']):>16}"

            f"{fmt_mean_std(program_data['mean'], program_data['sample_std']):>16}"

            f"{fmt_percent(exec_consistency):>13}"

            f"{fmt_percent(program_agreement_rate):>13}"

            f"{fmt_seconds(latency):>10}"

            f"{fmt_cost(cost):>14}"
        )


    print(
        "=" * 120
    )


    # -----------------------------------------------------------------
    # DETAILS
    # -----------------------------------------------------------------

    for configuration in (
        "A",
        "B",
        "C",
        "D",
    ):

        analysis = analyses[
            configuration
        ]


        print()
        print(
            f"CONFIGURATION {configuration} — "
            f"{analysis['architecture']}"
        )

        print(
            "-" * 70
        )


        print(
            "Execution accuracy runs: "
            f"{analysis['execution_accuracy']['per_run']}"
        )

        print(
            "Execution accuracy mean ± SD: "
            f"{analysis['execution_accuracy']['mean']:.2f}% "
            f"± "
            f"{analysis['execution_accuracy']['sample_std']:.2f}"
        )


        print(
            "Program accuracy runs: "
            f"{analysis['program_accuracy']['per_run']}"
        )

        print(
            "Program accuracy mean ± SD: "
            f"{analysis['program_accuracy']['mean']:.2f}% "
            f"± "
            f"{analysis['program_accuracy']['sample_std']:.2f}"
        )


        execution_consistency = (
            analysis[
                "execution_consistency"
            ]
        )


        print(
            "Execution consistency:"
        )

        print(
            f"  stable correct: "
            f"{execution_consistency['stable_true']}"
        )

        print(
            f"  stable incorrect: "
            f"{execution_consistency['stable_false']}"
        )

        print(
            f"  flipped across runs: "
            f"{execution_consistency['flip_count']}"
        )

        print(
            f"  consistency rate: "
            f"{execution_consistency['consistency_rate']:.2f}%"
        )

        print(
            f"  flip rate: "
            f"{execution_consistency['flip_rate']:.2f}%"
        )


        program_consistency = (
            analysis[
                "program_correctness_consistency"
            ]
        )


        print(
            "Program-correctness consistency:"
        )

        print(
            f"  stable correct: "
            f"{program_consistency['stable_true']}"
        )

        print(
            f"  stable incorrect: "
            f"{program_consistency['stable_false']}"
        )

        print(
            f"  flipped: "
            f"{program_consistency['flip_count']}"
        )

        print(
            f"  consistency rate: "
            f"{program_consistency['consistency_rate']:.2f}%"
        )


        exact_agreement = (
            analysis[
                "exact_program_agreement"
            ]
        )


        normalized_agreement = (
            analysis[
                "normalized_program_agreement"
            ]
        )


        print(
            "Program output agreement:"
        )

        print(
            f"  exact agreement: "
            f"{exact_agreement['agreement_count']}/"
            f"{exact_agreement['questions']} "
            f"({exact_agreement['agreement_rate']:.2f}%)"
        )

        print(
            f"  normalized agreement: "
            f"{normalized_agreement['agreement_count']}/"
            f"{normalized_agreement['questions']} "
            f"({normalized_agreement['agreement_rate']:.2f}%)"
        )


        failure_rates = (
            analysis[
                "failure_rates"
            ]
        )


        print(
            "Mean failure rates:"
        )

        print(
            f"  API: "
            f"{failure_rates['mean_api_error_rate']:.2f}%"
        )

        print(
            f"  parsing: "
            f"{failure_rates['mean_parse_error_rate']:.2f}%"
        )

        print(
            f"  validation: "
            f"{failure_rates['mean_validation_failure_rate']:.2f}%"
        )

        print(
            f"  execution: "
            f"{failure_rates['mean_execution_failure_rate']:.2f}%"
        )


        component = (
            analysis[
                "component_reliability"
            ]
        )


        if component is not None:

            print(
                f"{component['component']} contribution:"
            )

            print(
                "  corrected errors/run: "
                f"{component['corrected_errors_per_run']}"
            )

            print(
                "  mean corrected errors: "
                f"{component['mean_corrected_errors']:.2f}"
            )

            print(
                "  broke correct answers/run: "
                f"{component['broke_correct_answers_per_run']}"
            )

            print(
                "  mean broke correct answers: "
                f"{component['mean_broke_correct_answers']:.2f}"
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


    if args.runs < 2:

        raise ValueError(
            "Reliability analysis requires "
            "at least two runs."
        )


    print(
        "RQ2 repeated-run reliability analysis"
    )

    print(
        f"Dataset: dev[{args.start}:{args.end}]"
    )

    print(
        f"Runs per architecture: {args.runs}"
    )

    print(
        "No API calls made."
    )

    print()


    # -----------------------------------------------------------------
    # DISCOVER + LOAD
    # -----------------------------------------------------------------

    run_files = {}

    all_runs = {}


    for configuration in (
        "A",
        "B",
        "C",
        "D",
    ):

        files = find_run_files(
            configuration,
            args.start,
            args.end,
            args.runs,
        )


        run_files[
            configuration
        ] = files


        all_runs[
            configuration
        ] = [
            load_jsonl(
                path
            )
            for path in files
        ]


        print(
            f"Configuration {configuration}:"
        )


        for index, path in enumerate(
            files,
            start=1,
        ):

            print(
                f"  Run {index}: "
                f"{path.name}"
            )


    # -----------------------------------------------------------------
    # ALIGNMENT
    # -----------------------------------------------------------------

    aligned_ids = (
        verify_all_runs_aligned(
            all_runs
        )
    )


    print()

    print(
        f"Aligned questions across all runs: "
        f"{len(aligned_ids)}"
    )


    # -----------------------------------------------------------------
    # ANALYSIS
    # -----------------------------------------------------------------

    analyses = {

        configuration:
            analyze_architecture(
                configuration,
                all_runs[
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
    # DISPLAY
    # -----------------------------------------------------------------

    print_report(
        analyses
    )


    # -----------------------------------------------------------------
    # SAVE
    # -----------------------------------------------------------------

    timestamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%S_%fZ"
    )


    output_path = (
        RESULTS_DIR
        / (
            "reliability_analysis_"
            f"dev_{args.start}_{args.end}_"
            f"{args.runs}runs_"
            f"{timestamp}.json"
        )
    )


    output = {

        "research_question":
            "RQ2",

        "dataset_split":
            "dev",

        "start_index":
            args.start,

        "end_index":
            args.end,

        "question_count":
            len(aligned_ids),

        "runs_per_architecture":
            args.runs,

        "source_files": {

            configuration: [
                str(path)
                for path
                in paths
            ]

            for configuration, paths
            in run_files.items()
        },

        "architectures":
            analyses,
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
        "Reliability analysis saved to:"
    )

    print(
        output_path
    )


# ---------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()