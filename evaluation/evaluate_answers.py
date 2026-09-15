import json
import os
import math
from statistics import mean


# ============================================================
# CONFIGURATION
# ============================================================

SINGLE_AGENT_RESULTS = "results/single_agent_results.jsonl"
MULTI_AGENT_RESULTS = "results/multi_agent_results.jsonl"

OUTPUT_DIR = "results"

SINGLE_EVALUATION_OUTPUT = os.path.join(
    OUTPUT_DIR, "single_agent_evaluation.json"
)

MULTI_EVALUATION_OUTPUT = os.path.join(
    OUTPUT_DIR, "multi_agent_evaluation.json"
)

COMPARISON_OUTPUT = os.path.join(
    OUTPUT_DIR, "agent_comparison.json"
)


# ============================================================
# LOAD JSONL RESULTS
# ============================================================

def load_results(path):
    """
    Load experiment results from a JSONL file.
    Each line must contain one JSON object.
    """

    if not os.path.exists(path):
        print(f"WARNING: File not found: {path}")
        return []

    results = []

    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
                results.append(record)

            except json.JSONDecodeError as e:
                print(
                    f"WARNING: Could not parse line {line_number} "
                    f"in {path}: {e}"
                )

    return results


# ============================================================
# SAFE NUMBER CONVERSION
# ============================================================

def to_number(value):
    """
    Convert a value to float when it is a valid numerical answer.

    Returns None for:
    - None
    - empty strings
    - textual answers such as 'Not determinable'
    - non-numeric values
    """

    if value is None:
        return None

    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            return float(value)
        return None

    text = str(value).strip()

    if not text:
        return None

    # Explicitly reject non-numerical responses.
    lowered = text.lower()

    invalid_phrases = [
        "not determinable",
        "not determinable from the provided context",
        "cannot be determined",
        "cannot determine",
        "no valid answer",
        "no answer",
        "unknown",
        "n/a",
        "na",
        "none",
    ]

    for phrase in invalid_phrases:
        if phrase in lowered:
            return None

    # Remove common formatting.
    text = text.replace(",", "")
    text = text.replace("$", "")
    text = text.replace("€", "")
    text = text.replace("£", "")

    # Remove a trailing percentage symbol.
    # IMPORTANT:
    # We do NOT convert 93.5% to 0.935.
    # FinQA answer representation is preserved.
    if text.endswith("%"):
        text = text[:-1].strip()

    try:
        number = float(text)

        if math.isfinite(number):
            return number

    except ValueError:
        pass

    return None


# ============================================================
# EXTRACT PREDICTED ANSWER
# ============================================================

def get_predicted_answer(record):
    """
    Get the model's final predicted answer.

    We prefer normalized_prediction when it exists.
    Otherwise we use predicted_answer.
    """

    if "normalized_prediction" in record:
        value = record.get("normalized_prediction")

        if value is not None:
            number = to_number(value)

            if number is not None:
                return number

    if "predicted_answer" in record:
        value = record.get("predicted_answer")

        number = to_number(value)

        if number is not None:
            return number

    return None


# ============================================================
# GET GOLD ANSWER
# ============================================================

def get_gold_answer(record):
    """
    Get the benchmark gold numerical answer.
    """

    if "normalized_gold" in record:
        value = record.get("normalized_gold")

        number = to_number(value)

        if number is not None:
            return number

    if "gold_answer" in record:
        value = record.get("gold_answer")

        number = to_number(value)

        if number is not None:
            return number

    return None


# ============================================================
# STRICT NUMERICAL COMPARISON
# ============================================================

def strict_match(predicted, gold):
    """
    Strict numerical comparison.

    Example:
        127.4 == 127.4       -> True
        688.0 == 688          -> True
        24.69 == 24.69136    -> False
        93.5 == 0.935        -> False

    No percentage conversion is performed.
    No tolerance is applied.
    """

    if predicted is None or gold is None:
        return False

    return predicted == gold


# ============================================================
# CLASSIFY RESULT
# ============================================================

def classify_result(record):
    """
    Classify one experiment result.

    Categories:
        correct
        incorrect
        unparseable
        system_error
    """

    # Check for system-level failure.
    system_error = record.get("system_error")

    if system_error:
        return "system_error"

    predicted = get_predicted_answer(record)
    gold = get_gold_answer(record)

    if predicted is None:
        return "unparseable"

    if gold is None:
        return "unparseable"

    if strict_match(predicted, gold):
        return "correct"

    return "incorrect"


# ============================================================
# EVALUATE ONE AGENT
# ============================================================

def evaluate_agent(results, agent_name):
    """
    Evaluate all records for one architecture.
    """

    total = len(results)

    correct = 0
    incorrect = 0
    unparseable = 0
    system_errors = 0

    evaluated_latencies = []
    evaluated_input_tokens = []
    evaluated_output_tokens = []
    evaluated_costs = []

    detailed_results = []

    for record in results:

        classification = classify_result(record)

        if classification == "correct":
            correct += 1

        elif classification == "incorrect":
            incorrect += 1

        elif classification == "unparseable":
            unparseable += 1

        elif classification == "system_error":
            system_errors += 1

        # Metrics
        latency = record.get("latency_seconds")

        if isinstance(latency, (int, float)) and latency >= 0:
            evaluated_latencies.append(float(latency))

        input_tokens = record.get("input_tokens")

        if isinstance(input_tokens, (int, float)) and input_tokens >= 0:
            evaluated_input_tokens.append(int(input_tokens))

        output_tokens = record.get("output_tokens")

        if isinstance(output_tokens, (int, float)) and output_tokens >= 0:
            evaluated_output_tokens.append(int(output_tokens))

        cost = record.get("estimated_cost_usd")

        if isinstance(cost, (int, float)) and cost >= 0:
            evaluated_costs.append(float(cost))

        predicted = get_predicted_answer(record)
        gold = get_gold_answer(record)

        detailed_results.append({
            "id": record.get("id"),
            "question": record.get("question"),
            "gold_answer": record.get("gold_answer"),
            "predicted_answer": record.get("predicted_answer"),
            "normalized_gold": gold,
            "normalized_prediction": predicted,
            "classification": classification,
            "latency_seconds": latency,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost_usd": cost,
            "error": record.get("error"),
        })

    # Accuracy is calculated over all experiment examples.
    accuracy = correct / total if total > 0 else 0.0

    parseable = correct + incorrect

    parse_rate = parseable / total if total > 0 else 0.0

    results_summary = {
        "agent": agent_name,
        "total_examples": total,

        "correct": correct,
        "incorrect": incorrect,
        "unparseable": unparseable,
        "system_errors": system_errors,

        "strict_accuracy": accuracy,
        "strict_accuracy_percent": accuracy * 100,

        "parseable_examples": parseable,
        "parse_rate": parse_rate,
        "parse_rate_percent": parse_rate * 100,

        "average_latency_seconds": (
            mean(evaluated_latencies)
            if evaluated_latencies
            else None
        ),

        "average_input_tokens": (
            mean(evaluated_input_tokens)
            if evaluated_input_tokens
            else None
        ),

        "average_output_tokens": (
            mean(evaluated_output_tokens)
            if evaluated_output_tokens
            else None
        ),

        "average_cost_usd": (
            mean(evaluated_costs)
            if evaluated_costs
            else None
        ),

        "total_cost_usd": (
            sum(evaluated_costs)
            if evaluated_costs
            else 0.0
        ),

        "details": detailed_results,
    }

    return results_summary


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(summary):
    """
    Print a readable evaluation summary.
    """

    print()
    print("=" * 70)
    print(f"{summary['agent'].upper()} EVALUATION")
    print("=" * 70)

    print(f"Total examples:       {summary['total_examples']}")
    print(f"Correct:              {summary['correct']}")
    print(f"Incorrect:            {summary['incorrect']}")
    print(f"Unparseable:          {summary['unparseable']}")
    print(f"System errors:        {summary['system_errors']}")

    print("-" * 70)

    print(
        f"Strict accuracy:      "
        f"{summary['strict_accuracy_percent']:.2f}%"
    )

    print(
        f"Parse rate:           "
        f"{summary['parse_rate_percent']:.2f}%"
    )

    print("-" * 70)

    if summary["average_latency_seconds"] is not None:
        print(
            f"Average latency:      "
            f"{summary['average_latency_seconds']:.2f}s"
        )

    if summary["average_input_tokens"] is not None:
        print(
            f"Average input tokens: "
            f"{summary['average_input_tokens']:.0f}"
        )

    if summary["average_output_tokens"] is not None:
        print(
            f"Average output tokens:"
            f" {summary['average_output_tokens']:.0f}"
        )

    if summary["average_cost_usd"] is not None:
        print(
            f"Average cost/question:"
            f" ${summary['average_cost_usd']:.6f}"
        )

    print(
        f"Total cost:           "
        f"${summary['total_cost_usd']:.6f}"
    )

    print("=" * 70)


# ============================================================
# PRINT INDIVIDUAL ERRORS
# ============================================================

def print_errors(summary):
    """
    Print incorrect and unparseable examples.
    """

    print()
    print("-" * 70)
    print(f"{summary['agent'].upper()} ERROR ANALYSIS")
    print("-" * 70)

    errors_found = False

    for item in summary["details"]:

        if item["classification"] in [
            "incorrect",
            "unparseable",
            "system_error",
        ]:

            errors_found = True

            print()
            print(f"ID: {item['id']}")
            print(f"Classification: {item['classification']}")
            print(f"Gold: {item['gold_answer']}")
            print(f"Predicted: {item['predicted_answer']}")

    if not errors_found:
        print("No errors found.")


# ============================================================
# SAVE JSON
# ============================================================

def save_json(data, path):

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FINQA ANSWER EVALUATION")
    print("=" * 70)

    print()
    print("Evaluation rule:")
    print("STRICT numerical equality")
    print()
    print("Important:")
    print("- No percentage conversion")
    print("- No rounding tolerance")
    print("- No manual correction")
    print("- Gold answers remain in FinQA representation")
    print()

    # --------------------------------------------------------
    # LOAD RESULTS
    # --------------------------------------------------------

    single_results = load_results(
        SINGLE_AGENT_RESULTS
    )

    multi_results = load_results(
        MULTI_AGENT_RESULTS
    )

    print(
        f"Loaded single-agent results: "
        f"{len(single_results)}"
    )

    print(
        f"Loaded multi-agent results:  "
        f"{len(multi_results)}"
    )

    # --------------------------------------------------------
    # EVALUATE
    # --------------------------------------------------------

    single_summary = evaluate_agent(
        single_results,
        "single_agent"
    )

    multi_summary = evaluate_agent(
        multi_results,
        "multi_agent"
    )

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print_summary(single_summary)

    print_summary(multi_summary)

    print_errors(single_summary)

    print_errors(multi_summary)

    # --------------------------------------------------------
    # COMPARISON
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SINGLE-AGENT vs MULTI-AGENT")
    print("=" * 70)

    if single_summary["total_examples"] > 0:

        print(
            f"Single-agent accuracy: "
            f"{single_summary['strict_accuracy_percent']:.2f}%"
        )

    else:
        print("Single-agent accuracy: N/A")

    if multi_summary["total_examples"] > 0:

        print(
            f"Multi-agent accuracy:  "
            f"{multi_summary['strict_accuracy_percent']:.2f}%"
        )

    else:
        print("Multi-agent accuracy: N/A")

    print("-" * 70)

    if (
        single_summary["total_examples"] > 0
        and multi_summary["total_examples"] > 0
    ):

        accuracy_difference = (
            multi_summary["strict_accuracy_percent"]
            - single_summary["strict_accuracy_percent"]
        )

        print(
            f"Accuracy difference:    "
            f"{accuracy_difference:+.2f} percentage points"
        )

    if (
        single_summary["average_latency_seconds"] is not None
        and multi_summary["average_latency_seconds"] is not None
    ):

        latency_difference = (
            multi_summary["average_latency_seconds"]
            - single_summary["average_latency_seconds"]
        )

        print(
            f"Latency difference:     "
            f"{latency_difference:+.2f}s"
        )

    if (
        single_summary["average_cost_usd"] is not None
        and multi_summary["average_cost_usd"] is not None
    ):

        cost_difference = (
            multi_summary["average_cost_usd"]
            - single_summary["average_cost_usd"]
        )

        print(
            f"Cost difference:        "
            f"${cost_difference:+.6f}/question"
        )

    print("=" * 70)

    # --------------------------------------------------------
    # SAVE INDIVIDUAL EVALUATIONS
    # --------------------------------------------------------

    save_json(
        single_summary,
        SINGLE_EVALUATION_OUTPUT
    )

    save_json(
        multi_summary,
        MULTI_EVALUATION_OUTPUT
    )

    # --------------------------------------------------------
    # SAVE COMPARISON
    # --------------------------------------------------------

    comparison = {
        "evaluation_method": "strict_numerical_equality",

        "single_agent": {
            "total_examples":
                single_summary["total_examples"],

            "correct":
                single_summary["correct"],

            "incorrect":
                single_summary["incorrect"],

            "unparseable":
                single_summary["unparseable"],

            "system_errors":
                single_summary["system_errors"],

            "accuracy_percent":
                single_summary["strict_accuracy_percent"],

            "average_latency_seconds":
                single_summary["average_latency_seconds"],

            "average_input_tokens":
                single_summary["average_input_tokens"],

            "average_output_tokens":
                single_summary["average_output_tokens"],

            "average_cost_usd":
                single_summary["average_cost_usd"],

            "total_cost_usd":
                single_summary["total_cost_usd"],
        },

        "multi_agent": {
            "total_examples":
                multi_summary["total_examples"],

            "correct":
                multi_summary["correct"],

            "incorrect":
                multi_summary["incorrect"],

            "unparseable":
                multi_summary["unparseable"],

            "system_errors":
                multi_summary["system_errors"],

            "accuracy_percent":
                multi_summary["strict_accuracy_percent"],

            "average_latency_seconds":
                multi_summary["average_latency_seconds"],

            "average_input_tokens":
                multi_summary["average_input_tokens"],

            "average_output_tokens":
                multi_summary["average_output_tokens"],

            "average_cost_usd":
                multi_summary["average_cost_usd"],

            "total_cost_usd":
                multi_summary["total_cost_usd"],
        },
    }

    if (
        single_summary["total_examples"] > 0
        and multi_summary["total_examples"] > 0
    ):

        comparison["accuracy_difference_percentage_points"] = (
            multi_summary["strict_accuracy_percent"]
            - single_summary["strict_accuracy_percent"]
        )

    save_json(
        comparison,
        COMPARISON_OUTPUT
    )

    # --------------------------------------------------------
    # FINAL MESSAGE
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    print(
        f"Saved: {SINGLE_EVALUATION_OUTPUT}"
    )

    print(
        f"Saved: {MULTI_EVALUATION_OUTPUT}"
    )

    print(
        f"Saved: {COMPARISON_OUTPUT}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
    