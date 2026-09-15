"""Export dissertation-ready tables and figures from final thesis results.

NO API calls are made.

The script automatically finds the latest:
- final_test_architecture_summary_*.csv
- final_test_pairwise_statistics_*.csv
- reliability_analysis_dev_75_175_3runs_*.json
- final_error_analysis_test_0_1147_*.json

Outputs:
- thesis_execution_accuracy.png
- thesis_program_accuracy.png
- thesis_latency.png
- thesis_cost.png
- thesis_accuracy_cost_tradeoff.png
- thesis_final_summary.csv
- thesis_rq_summary.csv
- thesis_statistical_summary.csv

Run:
    python -m evaluation.export_thesis_results
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results"

EXPORT_DIR = RESULTS_DIR / "thesis_exports"


# ---------------------------------------------------------------------
# ARCHITECTURE LABELS
# ---------------------------------------------------------------------

CONFIG_ORDER = [
    "A",
    "B",
    "C",
    "D",
]


SHORT_LABELS = {
    "A": "A\nSingle-Agent",
    "B": "B\nReasoning",
    "C": "C\n+ Verification",
    "D": "D\n+ Coordinator",
}


FULL_LABELS = {
    "A": "Single-Agent",
    "B": "Reasoning Agent",
    "C": "Reasoning + Verification",
    "D": "Reasoning + Verification + Coordinator",
}


# ---------------------------------------------------------------------
# THESIS FIGURE STYLE
# ---------------------------------------------------------------------

COLORS = {
    "A": "#1f77b4",
    "B": "#ff7f0e",
    "C": "#2ca02c",
    "D": "#d62728",
}


plt.rcParams.update({

    "font.family":
        "serif",

    "font.serif":
        [
            "DejaVu Serif",
            "Times New Roman",
            "Times",
        ],

    "font.size":
        12,

    "axes.titlesize":
        17,

    "axes.labelsize":
        13,

    "xtick.labelsize":
        11,

    "ytick.labelsize":
        11,

    "axes.linewidth":
        1.0,

    "figure.dpi":
        120,

    "savefig.dpi":
        300,

    "savefig.facecolor":
        "white",
})


# ---------------------------------------------------------------------
# FILE DISCOVERY
# ---------------------------------------------------------------------

def latest_file(pattern: str) -> Path:
    """Return the most recently modified file matching a pattern."""

    files = list(
        RESULTS_DIR.glob(
            pattern
        )
    )

    if not files:

        raise FileNotFoundError(
            f"No result file matched:\n"
            f"{RESULTS_DIR / pattern}"
        )

    return max(
        files,
        key=lambda path:
            path.stat().st_mtime,
    )


# ---------------------------------------------------------------------
# CSV / JSON LOADERS
# ---------------------------------------------------------------------

def load_csv(path: Path):

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        return list(
            csv.DictReader(
                file
            )
        )


def load_json(path: Path):

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


# ---------------------------------------------------------------------
# SAFE CONVERSION
# ---------------------------------------------------------------------

def to_float(value):

    if value in (
        None,
        "",
    ):
        return None

    return float(
        value
    )


def to_int(value):

    if value in (
        None,
        "",
    ):
        return None

    return int(
        float(
            value
        )
    )


# ---------------------------------------------------------------------
# LOAD FINAL ARCHITECTURE RESULTS
# ---------------------------------------------------------------------

def load_architecture_results(
    path: Path,
):

    rows = load_csv(
        path
    )

    results = {}


    for row in rows:

        configuration = (
            row[
                "configuration"
            ]
        )


        results[
            configuration
        ] = {

            "configuration":
                configuration,

            "architecture":
                row[
                    "architecture"
                ],

            "examples":
                to_int(
                    row[
                        "examples"
                    ]
                ),

            "execution_correct":
                to_int(
                    row[
                        "execution_correct"
                    ]
                ),

            "execution_accuracy":
                to_float(
                    row[
                        "execution_accuracy"
                    ]
                ),

            "execution_ci95_lower":
                to_float(
                    row[
                        "execution_ci95_lower"
                    ]
                ),

            "execution_ci95_upper":
                to_float(
                    row[
                        "execution_ci95_upper"
                    ]
                ),

            "program_correct":
                to_int(
                    row[
                        "program_correct"
                    ]
                ),

            "program_accuracy":
                to_float(
                    row[
                        "program_accuracy"
                    ]
                ),

            "program_ci95_lower":
                to_float(
                    row[
                        "program_ci95_lower"
                    ]
                ),

            "program_ci95_upper":
                to_float(
                    row[
                        "program_ci95_upper"
                    ]
                ),

            "valid_rate":
                to_float(
                    row[
                        "valid_rate"
                    ]
                ),

            "executable_rate":
                to_float(
                    row[
                        "executable_rate"
                    ]
                ),

            "mean_latency_seconds":
                to_float(
                    row[
                        "mean_latency_seconds"
                    ]
                ),

            "median_latency_seconds":
                to_float(
                    row[
                        "median_latency_seconds"
                    ]
                ),

            "total_input_tokens":
                to_float(
                    row[
                        "total_input_tokens"
                    ]
                ),

            "total_output_tokens":
                to_float(
                    row[
                        "total_output_tokens"
                    ]
                ),

            "total_cost_usd":
                to_float(
                    row[
                        "total_cost_usd"
                    ]
                ),

            "mean_cost_usd":
                to_float(
                    row[
                        "mean_cost_usd"
                    ]
                ),

            "validation_failures":
                to_int(
                    row[
                        "validation_failures"
                    ]
                ),

            "execution_failures":
                to_int(
                    row[
                        "execution_failures"
                    ]
                ),

            "api_errors":
                to_int(
                    row[
                        "api_errors"
                    ]
                ),

            "parse_errors":
                to_int(
                    row[
                        "parse_errors"
                    ]
                ),
        }


    missing = [
        config
        for config in CONFIG_ORDER
        if config not in results
    ]


    if missing:

        raise ValueError(
            "Architecture summary is missing "
            f"configurations: {missing}"
        )


    return results


# ---------------------------------------------------------------------
# RELIABILITY RESULTS
# ---------------------------------------------------------------------

def load_reliability_results(
    path: Path,
):

    data = load_json(
        path
    )

    architectures = (
        data[
            "architectures"
        ]
    )


    results = {}


    for configuration in CONFIG_ORDER:

        architecture = (
            architectures[
                configuration
            ]
        )


        results[
            configuration
        ] = {

            "execution_mean":
                architecture[
                    "execution_accuracy"
                ][
                    "mean"
                ],

            "execution_std":
                architecture[
                    "execution_accuracy"
                ][
                    "sample_std"
                ],

            "program_mean":
                architecture[
                    "program_accuracy"
                ][
                    "mean"
                ],

            "program_std":
                architecture[
                    "program_accuracy"
                ][
                    "sample_std"
                ],

            "execution_consistency":
                architecture[
                    "execution_consistency"
                ][
                    "consistency_rate"
                ],

            "execution_flip_rate":
                architecture[
                    "execution_consistency"
                ][
                    "flip_rate"
                ],

            "exact_program_agreement":
                architecture[
                    "exact_program_agreement"
                ][
                    "agreement_rate"
                ],
        }


    return results


# ---------------------------------------------------------------------
# ERROR SUMMARY
# ---------------------------------------------------------------------

def load_error_results(
    path: Path,
):

    data = load_json(
        path
    )

    category_counts = (
        data.get(
            "category_counts",
            {},
        )
    )


    return {

        "all_architectures_wrong":
            category_counts.get(
                "all_architectures_wrong",
                0,
            ),

        "all_architectures_correct":
            category_counts.get(
                "all_architectures_correct",
                0,
            ),

        "B_rescued_A_error":
            category_counts.get(
                "B_rescued_A_error",
                0,
            ),

        "B_lost_A_correct_answer":
            category_counts.get(
                "B_lost_A_correct_answer",
                0,
            ),

        "C_rescued_B_error":
            category_counts.get(
                "C_rescued_B_error",
                0,
            ),

        "C_lost_B_correct_answer":
            category_counts.get(
                "C_lost_B_correct_answer",
                0,
            ),

        "D_rescued_C_error":
            category_counts.get(
                "D_rescued_C_error",
                0,
            ),

        "D_lost_C_correct_answer":
            category_counts.get(
                "D_lost_C_correct_answer",
                0,
            ),

        "C_verifier_corrected_reasoning_error":
            category_counts.get(
                "C_verifier_corrected_reasoning_error",
                0,
            ),

        "C_verifier_broke_correct_reasoning":
            category_counts.get(
                "C_verifier_broke_correct_reasoning",
                0,
            ),

        "D_coordinator_corrected_verifier_error":
            category_counts.get(
                "D_coordinator_corrected_verifier_error",
                0,
            ),

        "D_coordinator_broke_verified_answer":
            category_counts.get(
                "D_coordinator_broke_verified_answer",
                0,
            ),
    }


# ---------------------------------------------------------------------
# FIGURE HELPERS
# ---------------------------------------------------------------------

def save_bar_chart(
    values,
    ylabel,
    title,
    filename,
    value_format="{:.2f}",
    error_lower=None,
    error_upper=None,
    y_min=0,
    y_max=None,
    currency_axis=False,
):

    labels = [
        SHORT_LABELS[
            config
        ]
        for config
        in CONFIG_ORDER
    ]


    colors = [
        COLORS[
            config
        ]
        for config
        in CONFIG_ORDER
    ]


    fig, ax = plt.subplots(
        figsize=(
            9.5,
            6.0,
        )
    )


    # -----------------------------------------------------------------
    # CONFIDENCE INTERVAL ERROR BARS
    # -----------------------------------------------------------------

    yerr = None


    if (
        error_lower is not None
        and
        error_upper is not None
    ):

        lower_errors = [

            value - lower

            for value, lower
            in zip(
                values,
                error_lower,
            )
        ]


        upper_errors = [

            upper - value

            for value, upper
            in zip(
                values,
                error_upper,
            )
        ]


        yerr = [
            lower_errors,
            upper_errors,
        ]


    # -----------------------------------------------------------------
    # BAR CHART
    # -----------------------------------------------------------------

    bars = ax.bar(

        labels,
        values,

        color=
            colors,

        edgecolor=
            "black",

        linewidth=
            0.9,

        width=
            0.66,

        yerr=
            yerr,

        capsize=
            5
            if yerr is not None
            else 0,

        error_kw={
            "elinewidth":
                1.2,

            "capthick":
                1.2,

            "ecolor":
                "black",
        },
    )


    # -----------------------------------------------------------------
    # TITLE / AXIS LABEL
    # -----------------------------------------------------------------

    ax.set_ylabel(
        ylabel,
        fontweight="bold",
        labelpad=10,
    )


    ax.set_title(
        title,
        fontweight="bold",
        pad=16,
    )


    # -----------------------------------------------------------------
    # Y-AXIS LIMIT
    # -----------------------------------------------------------------

    if y_max is None:

        largest_value = max(

            error_upper
            if error_upper is not None
            else values
        )


        y_max = (
            largest_value
            * 1.18
        )


    ax.set_ylim(
        y_min,
        y_max,
    )


    # -----------------------------------------------------------------
    # GRID
    # -----------------------------------------------------------------

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.8,
        alpha=0.35,
    )


    ax.set_axisbelow(
        True
    )


    # -----------------------------------------------------------------
    # CLEAN SPINES
    # -----------------------------------------------------------------

    ax.spines[
        "top"
    ].set_visible(
        False
    )


    ax.spines[
        "right"
    ].set_visible(
        False
    )


    # -----------------------------------------------------------------
    # CURRENCY FORMAT
    # -----------------------------------------------------------------

    if currency_axis:

        ax.yaxis.set_major_formatter(

            FuncFormatter(
                lambda value, position:
                    f"${value:.4f}"
            )
        )


    # -----------------------------------------------------------------
    # VALUE LABELS
    # -----------------------------------------------------------------

    label_offset = (
        y_max
        * 0.018
    )


    for index, (
        bar,
        value,
    ) in enumerate(
        zip(
            bars,
            values,
        )
    ):

        if error_upper is not None:

            label_y = (
                error_upper[
                    index
                ]
                + label_offset
            )

        else:

            label_y = (
                bar.get_height()
                + label_offset
            )


        ax.text(

            bar.get_x()
            + bar.get_width()
            / 2,

            label_y,

            value_format.format(
                value
            ),

            ha="center",

            va="bottom",

            fontsize=11,

            fontweight="bold",
        )


    ax.tick_params(
        axis="x",
        length=0,
        pad=9,
    )


    fig.tight_layout()


    output_path = (
        EXPORT_DIR
        / filename
    )


    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )


    plt.close(
        fig
    )


    return output_path


# ---------------------------------------------------------------------
# EXECUTION ACCURACY FIGURE
# ---------------------------------------------------------------------

def export_execution_accuracy(
    results,
):

    values = [

        results[
            config
        ][
            "execution_accuracy"
        ]

        for config
        in CONFIG_ORDER
    ]


    lower = [

        results[
            config
        ][
            "execution_ci95_lower"
        ]

        for config
        in CONFIG_ORDER
    ]


    upper = [

        results[
            config
        ][
            "execution_ci95_upper"
        ]

        for config
        in CONFIG_ORDER
    ]


    return save_bar_chart(

        values=
            values,

        error_lower=
            lower,

        error_upper=
            upper,

        ylabel=
            "Execution Accuracy (%)",

        title=
            (
                "FinQA Test-Set Execution Accuracy "
                "by Architecture"
            ),

        filename=
            "thesis_execution_accuracy.png",

        y_min=
            0,

        y_max=
            42,

        value_format=
            "{:.2f}%",
    )


# ---------------------------------------------------------------------
# PROGRAM ACCURACY FIGURE
# ---------------------------------------------------------------------

def export_program_accuracy(
    results,
):

    values = [

        results[
            config
        ][
            "program_accuracy"
        ]

        for config
        in CONFIG_ORDER
    ]


    lower = [

        results[
            config
        ][
            "program_ci95_lower"
        ]

        for config
        in CONFIG_ORDER
    ]


    upper = [

        results[
            config
        ][
            "program_ci95_upper"
        ]

        for config
        in CONFIG_ORDER
    ]


    return save_bar_chart(

        values=
            values,

        error_lower=
            lower,

        error_upper=
            upper,

        ylabel=
            "Program Accuracy (%)",

        title=
            (
                "FinQA Test-Set Program Accuracy "
                "by Architecture"
            ),

        filename=
            "thesis_program_accuracy.png",

        y_min=
            0,

        y_max=
            38,

        value_format=
            "{:.2f}%",
    )


# ---------------------------------------------------------------------
# LATENCY FIGURE
# ---------------------------------------------------------------------

def export_latency(
    results,
):

    values = [

        results[
            config
        ][
            "mean_latency_seconds"
        ]

        for config
        in CONFIG_ORDER
    ]


    return save_bar_chart(

        values=
            values,

        ylabel=
            (
                "Average Latency "
                "(seconds/question)"
            ),

        title=
            (
                "Average End-to-End Latency "
                "by Architecture"
            ),

        filename=
            "thesis_latency.png",

        y_min=
            0,

        y_max=
            18,

        value_format=
            "{:.2f} s",
    )


# ---------------------------------------------------------------------
# COST FIGURE
# ---------------------------------------------------------------------

def export_cost(
    results,
):

    values = [

        results[
            config
        ][
            "mean_cost_usd"
        ]

        for config
        in CONFIG_ORDER
    ]


    return save_bar_chart(

        values=
            values,

        ylabel=
            (
                "Average API Cost "
                "(USD/question)"
            ),

        title=
            (
                "Average API Cost "
                "by Architecture"
            ),

        filename=
            "thesis_cost.png",

        y_min=
            0,

        y_max=
            0.00175,

        value_format=
            "${:.6f}",

        currency_axis=
            True,
    )


# ---------------------------------------------------------------------
# ACCURACY / COST TRADE-OFF
# ---------------------------------------------------------------------

def export_accuracy_cost_tradeoff(
    results,
):

    fig, ax = plt.subplots(
        figsize=(
            10,
            6.5,
        )
    )


    # -----------------------------------------------------------------
    # ANNOTATION OFFSETS
    # -----------------------------------------------------------------

    annotation_offsets = {

        "A":
            (18, 15),

        "B":
            (18, 15),

        "C":
            (15, 18),

        "D":
            (-240, 20),
    }


    # -----------------------------------------------------------------
    # PLOT EACH ARCHITECTURE
    # -----------------------------------------------------------------

    for configuration in CONFIG_ORDER:

        result = (
            results[
                configuration
            ]
        )


        x = (
            result[
                "mean_cost_usd"
            ]
        )


        y = (
            result[
                "execution_accuracy"
            ]
        )


        ax.scatter(

            x,
            y,

            s=
                160,

            color=
                COLORS[
                    configuration
                ],

            edgecolor=
                "black",

            linewidth=
                1.0,

            zorder=
                4,
        )


        label = (
            f"{configuration}: "
            f"{FULL_LABELS[configuration]}"
        )


        ax.annotate(

            label,

            xy=(
                x,
                y,
            ),

            xytext=
                annotation_offsets[
                    configuration
                ],

            textcoords=
                "offset points",

            fontsize=
                10.5,

            fontweight=
                "bold",

            bbox={
                "boxstyle":
                    "round,pad=0.35",

                "facecolor":
                    "white",

                "edgecolor":
                    COLORS[
                        configuration
                    ],

                "linewidth":
                    1.2,

                "alpha":
                    0.95,
            },

            arrowprops={
                "arrowstyle":
                    "-",

                "color":
                    COLORS[
                        configuration
                    ],

                "linewidth":
                    1.2,
            },
        )


    # -----------------------------------------------------------------
    # CONNECT C AND D
    #
    # Helps visually show the relatively small accuracy improvement
    # relative to the additional API cost.
    # -----------------------------------------------------------------

    c_cost = (
        results[
            "C"
        ][
            "mean_cost_usd"
        ]
    )


    c_accuracy = (
        results[
            "C"
        ][
            "execution_accuracy"
        ]
    )


    d_cost = (
        results[
            "D"
        ][
            "mean_cost_usd"
        ]
    )


    d_accuracy = (
        results[
            "D"
        ][
            "execution_accuracy"
        ]
    )


    ax.plot(

        [
            c_cost,
            d_cost,
        ],

        [
            c_accuracy,
            d_accuracy,
        ],

        linestyle=
            ":",

        linewidth=
            1.5,

        color=
            "gray",

        alpha=
            0.7,

        zorder=
            2,
    )


    # -----------------------------------------------------------------
    # AXIS LABELS
    # -----------------------------------------------------------------

    ax.set_xlabel(
        "Average API Cost (USD/question)",
        fontweight="bold",
        labelpad=10,
    )


    ax.set_ylabel(
        "Execution Accuracy (%)",
        fontweight="bold",
        labelpad=10,
    )


    ax.set_title(
        (
            "Accuracy–Cost Trade-off "
            "Across Architectures"
        ),
        fontweight="bold",
        pad=16,
    )


    # -----------------------------------------------------------------
    # AXIS LIMITS
    # -----------------------------------------------------------------

    ax.set_xlim(
        0.00038,
        0.00165,
    )


    ax.set_ylim(
        30.0,
        35.7,
    )


    # -----------------------------------------------------------------
    # COST AXIS FORMAT
    # -----------------------------------------------------------------

    ax.xaxis.set_major_formatter(

        FuncFormatter(
            lambda value, position:
                f"${value:.4f}"
        )
    )


    # -----------------------------------------------------------------
    # GRID
    # -----------------------------------------------------------------

    ax.grid(
        linestyle="--",
        linewidth=0.8,
        alpha=0.35,
    )


    ax.set_axisbelow(
        True
    )


    # -----------------------------------------------------------------
    # CLEAN SPINES
    # -----------------------------------------------------------------

    ax.spines[
        "top"
    ].set_visible(
        False
    )


    ax.spines[
        "right"
    ].set_visible(
        False
    )


    # -----------------------------------------------------------------
    # INTERPRETIVE NOTE
    # -----------------------------------------------------------------

    ax.text(

        0.99,
        0.025,

        (
            "Higher accuracy is preferable; "
            "lower cost is preferable."
        ),

        transform=
            ax.transAxes,

        ha=
            "right",

        va=
            "bottom",

        fontsize=
            9,

        style=
            "italic",

        color=
            "dimgray",
    )


    fig.tight_layout()


    output_path = (
        EXPORT_DIR
        / (
            "thesis_accuracy_cost_tradeoff.png"
        )
    )


    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )


    plt.close(
        fig
    )


    return output_path


# ---------------------------------------------------------------------
# FINAL SUMMARY CSV
# ---------------------------------------------------------------------

def export_final_summary(
    architecture_results,
    reliability_results,
):

    output_path = (
        EXPORT_DIR
        / "thesis_final_summary.csv"
    )


    fieldnames = [
        "configuration",
        "architecture",
        "test_examples",
        "execution_correct",
        "execution_accuracy_percent",
        "execution_ci95_lower_percent",
        "execution_ci95_upper_percent",
        "program_correct",
        "program_accuracy_percent",
        "program_ci95_lower_percent",
        "program_ci95_upper_percent",
        "valid_rate_percent",
        "executable_rate_percent",
        "mean_latency_seconds",
        "total_input_tokens",
        "total_output_tokens",
        "total_api_cost_usd",
        "mean_cost_per_question_usd",
        "reliability_execution_mean_percent",
        "reliability_execution_std",
        "reliability_program_mean_percent",
        "reliability_program_std",
        "execution_consistency_percent",
        "execution_flip_rate_percent",
        "exact_program_agreement_percent",
    ]


    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )


        writer.writeheader()


        for configuration in CONFIG_ORDER:

            final_result = (
                architecture_results[
                    configuration
                ]
            )


            reliability = (
                reliability_results[
                    configuration
                ]
            )


            writer.writerow({

                "configuration":
                    configuration,

                "architecture":
                    FULL_LABELS[
                        configuration
                    ],

                "test_examples":
                    final_result[
                        "examples"
                    ],

                "execution_correct":
                    final_result[
                        "execution_correct"
                    ],

                "execution_accuracy_percent":
                    final_result[
                        "execution_accuracy"
                    ],

                "execution_ci95_lower_percent":
                    final_result[
                        "execution_ci95_lower"
                    ],

                "execution_ci95_upper_percent":
                    final_result[
                        "execution_ci95_upper"
                    ],

                "program_correct":
                    final_result[
                        "program_correct"
                    ],

                "program_accuracy_percent":
                    final_result[
                        "program_accuracy"
                    ],

                "program_ci95_lower_percent":
                    final_result[
                        "program_ci95_lower"
                    ],

                "program_ci95_upper_percent":
                    final_result[
                        "program_ci95_upper"
                    ],

                "valid_rate_percent":
                    final_result[
                        "valid_rate"
                    ],

                "executable_rate_percent":
                    final_result[
                        "executable_rate"
                    ],

                "mean_latency_seconds":
                    final_result[
                        "mean_latency_seconds"
                    ],

                "total_input_tokens":
                    final_result[
                        "total_input_tokens"
                    ],

                "total_output_tokens":
                    final_result[
                        "total_output_tokens"
                    ],

                "total_api_cost_usd":
                    final_result[
                        "total_cost_usd"
                    ],

                "mean_cost_per_question_usd":
                    final_result[
                        "mean_cost_usd"
                    ],

                "reliability_execution_mean_percent":
                    reliability[
                        "execution_mean"
                    ],

                "reliability_execution_std":
                    reliability[
                        "execution_std"
                    ],

                "reliability_program_mean_percent":
                    reliability[
                        "program_mean"
                    ],

                "reliability_program_std":
                    reliability[
                        "program_std"
                    ],

                "execution_consistency_percent":
                    reliability[
                        "execution_consistency"
                    ],

                "execution_flip_rate_percent":
                    reliability[
                        "execution_flip_rate"
                    ],

                "exact_program_agreement_percent":
                    reliability[
                        "exact_program_agreement"
                    ],
            })


    return output_path


# ---------------------------------------------------------------------
# RQ SUMMARY
# ---------------------------------------------------------------------

def export_rq_summary(
    architecture_results,
    reliability_results,
    error_results,
):

    output_path = (
        EXPORT_DIR
        / "thesis_rq_summary.csv"
    )


    rows = [

        {
            "research_question":
                "RQ1",

            "focus":
                "Reasoning performance",

            "main_evidence":
                (
                    "Final FinQA test execution "
                    "and program accuracy"
                ),

            "result":
                (
                    "D highest execution accuracy "
                    f"({architecture_results['D']['execution_accuracy']:.2f}%) "
                    "and program accuracy "
                    f"({architecture_results['D']['program_accuracy']:.2f}%). "
                    "C closely follows at "
                    f"{architecture_results['C']['execution_accuracy']:.2f}% "
                    "execution accuracy."
                ),
        },

        {
            "research_question":
                "RQ2",

            "focus":
                "Reliability",

            "main_evidence":
                (
                    "Three repeated dev runs, "
                    "question-level consistency "
                    "and exact program agreement"
                ),

            "result":
                (
                    "Execution consistency ranged from "
                    f"{min(r['execution_consistency'] for r in reliability_results.values()):.2f}% "
                    "to "
                    f"{max(r['execution_consistency'] for r in reliability_results.values()):.2f}%. "
                    "Exact program agreement decreased from "
                    f"{reliability_results['A']['exact_program_agreement']:.2f}% "
                    "for A to "
                    f"{reliability_results['D']['exact_program_agreement']:.2f}% "
                    "for D."
                ),
        },

        {
            "research_question":
                "RQ3",

            "focus":
                "Efficiency",

            "main_evidence":
                (
                    "Latency, token usage "
                    "and API cost"
                ),

            "result":
                (
                    "Architectural complexity increased overhead: "
                    f"A averaged {architecture_results['A']['mean_latency_seconds']:.2f}s "
                    "and "
                    f"${architecture_results['A']['mean_cost_usd']:.6f}/question, "
                    "while D averaged "
                    f"{architecture_results['D']['mean_latency_seconds']:.2f}s "
                    "and "
                    f"${architecture_results['D']['mean_cost_usd']:.6f}/question."
                ),
        },

        {
            "research_question":
                "RQ4",

            "focus":
                "Component contribution",

            "main_evidence":
                (
                    "A→B→C→D staged ablation "
                    "and internal component corrections"
                ),

            "result":
                (
                    "Verification was the strongest added component. "
                    f"C corrected {error_results['C_verifier_corrected_reasoning_error']} "
                    "reasoning errors while breaking "
                    f"{error_results['C_verifier_broke_correct_reasoning']}. "
                    "The D coordinator corrected "
                    f"{error_results['D_coordinator_corrected_verifier_error']} "
                    "verifier errors while breaking "
                    f"{error_results['D_coordinator_broke_verified_answer']}."
                ),
        },
    ]


    fieldnames = [
        "research_question",
        "focus",
        "main_evidence",
        "result",
    ]


    with output_path.open(
        "w",
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


    return output_path


# ---------------------------------------------------------------------
# STATISTICAL SUMMARY
# ---------------------------------------------------------------------

def export_statistical_summary(
    pairwise_path: Path,
):

    rows = load_csv(
        pairwise_path
    )


    output_path = (
        EXPORT_DIR
        / "thesis_statistical_summary.csv"
    )


    selected_rows = []


    for row in rows:

        selected_rows.append({

            "metric":
                row[
                    "metric"
                ],

            "comparison":
                row[
                    "comparison"
                ],

            "difference_percentage_points":
                row[
                    "difference_percentage_points"
                ],

            "ci95_lower_pp":
                row[
                    "difference_ci95_lower_pp"
                ],

            "ci95_upper_pp":
                row[
                    "difference_ci95_upper_pp"
                ],

            "mcnemar_exact_p":
                row[
                    "mcnemar_exact_p"
                ],

            "holm_adjusted_p":
                row[
                    "holm_adjusted_p"
                ],

            "significant_after_holm_0_05":
                (
                    float(
                        row[
                            "holm_adjusted_p"
                        ]
                    )
                    < 0.05
                ),
        })


    fieldnames = list(
        selected_rows[
            0
        ].keys()
    )


    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )


        writer.writeheader()


        writer.writerows(
            selected_rows
        )


    return output_path


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    print(
        "Exporting dissertation-ready thesis results"
    )

    print(
        "No API calls made."
    )

    print()


    EXPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    # -----------------------------------------------------------------
    # DISCOVER SOURCES
    # -----------------------------------------------------------------

    architecture_path = latest_file(
        "final_test_architecture_summary_*.csv"
    )


    pairwise_path = latest_file(
        "final_test_pairwise_statistics_*.csv"
    )


    reliability_path = latest_file(
        "reliability_analysis_dev_75_175_3runs_*.json"
    )


    error_path = latest_file(
        "final_error_analysis_test_0_1147_*.json"
    )


    print(
        "Source files:"
    )

    print(
        f"  Final architecture results:\n"
        f"    {architecture_path.name}"
    )

    print(
        f"  Pairwise statistics:\n"
        f"    {pairwise_path.name}"
    )

    print(
        f"  Reliability analysis:\n"
        f"    {reliability_path.name}"
    )

    print(
        f"  Error analysis:\n"
        f"    {error_path.name}"
    )

    print()


    # -----------------------------------------------------------------
    # LOAD DATA
    # -----------------------------------------------------------------

    architecture_results = (
        load_architecture_results(
            architecture_path
        )
    )


    reliability_results = (
        load_reliability_results(
            reliability_path
        )
    )


    error_results = (
        load_error_results(
            error_path
        )
    )


    # -----------------------------------------------------------------
    # FIGURES
    # -----------------------------------------------------------------

    figure_paths = [

        export_execution_accuracy(
            architecture_results
        ),

        export_program_accuracy(
            architecture_results
        ),

        export_latency(
            architecture_results
        ),

        export_cost(
            architecture_results
        ),

        export_accuracy_cost_tradeoff(
            architecture_results
        ),
    ]


    # -----------------------------------------------------------------
    # TABLES
    # -----------------------------------------------------------------

    summary_csv = (
        export_final_summary(
            architecture_results,
            reliability_results,
        )
    )


    rq_csv = (
        export_rq_summary(
            architecture_results,
            reliability_results,
            error_results,
        )
    )


    statistics_csv = (
        export_statistical_summary(
            pairwise_path
        )
    )


    # -----------------------------------------------------------------
    # FINAL SANITY CHECK
    # -----------------------------------------------------------------

    assert (
        architecture_results[
            "A"
        ][
            "examples"
        ]
        == 1147
    )


    assert (
        architecture_results[
            "B"
        ][
            "examples"
        ]
        == 1147
    )


    assert (
        architecture_results[
            "C"
        ][
            "examples"
        ]
        == 1147
    )


    assert (
        architecture_results[
            "D"
        ][
            "examples"
        ]
        == 1147
    )


    assert (
        architecture_results[
            "A"
        ][
            "execution_correct"
        ]
        == 369
    )


    assert (
        architecture_results[
            "B"
        ][
            "execution_correct"
        ]
        == 351
    )


    assert (
        architecture_results[
            "C"
        ][
            "execution_correct"
        ]
        == 392
    )


    assert (
        architecture_results[
            "D"
        ][
            "execution_correct"
        ]
        == 398
    )


    # -----------------------------------------------------------------
    # DISPLAY FINAL RESULTS
    # -----------------------------------------------------------------

    print(
        "=" * 100
    )

    print(
        "FINAL THESIS RESULTS"
    )

    print(
        "=" * 100
    )


    print(

        f"{'Config':<8}"

        f"{'Architecture':<42}"

        f"{'Exec Acc':>12}"

        f"{'Prog Acc':>12}"

        f"{'Latency':>12}"

        f"{'Cost/Q':>15}"
    )


    print(
        "-" * 100
    )


    for configuration in CONFIG_ORDER:

        result = (
            architecture_results[
                configuration
            ]
        )


        print(

            f"{configuration:<8}"

            f"{FULL_LABELS[configuration]:<42}"

            f"{result['execution_accuracy']:>11.2f}%"

            f"{result['program_accuracy']:>11.2f}%"

            f"{result['mean_latency_seconds']:>10.2f} s"

            f"${result['mean_cost_usd']:>13.6f}"
        )


    print()

    print(
        "=" * 100
    )

    print(
        "FILES CREATED"
    )

    print(
        "=" * 100
    )


    for path in figure_paths:

        print(
            path
        )


    print(
        summary_csv
    )


    print(
        rq_csv
    )


    print(
        statistics_csv
    )


    print()

    print(
        "Final sanity checks passed."
    )

    print(
        "No API calls were made."
    )


if __name__ == "__main__":
    main()