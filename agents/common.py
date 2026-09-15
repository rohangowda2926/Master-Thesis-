"""Shared utilities for all thesis agent configurations.

This module contains functionality that MUST remain consistent across
Configurations A/B/C/D.

Shared model:
    qwen/qwen3-30b-a3b-instruct-2507

Shared controls:
    - temperature = 0
    - same FinQA evidence construction
    - same four training demonstrations
    - same API provider
    - same usage/cost logging
    - same retry/error policy

This module does NOT implement a particular agent architecture.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Iterable


# ---------------------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "finqa"
RESULTS_DIR = PROJECT_ROOT / "results"

TRAIN_PATH = DATA_DIR / "train.json"
DEV_PATH = DATA_DIR / "dev.json"
TEST_PATH = DATA_DIR / "test.json"



def get_dataset_path(split: str):
    """Return the frozen FinQA dataset path for a supported split."""

    if split == "dev":
        return DEV_PATH

    if split == "test":
        return TEST_PATH

    raise ValueError(
        f"Unsupported dataset split: {split!r}. "
        'Expected "dev" or "test".'
    )


# ---------------------------------------------------------------------
# FROZEN EXPERIMENTAL MODEL
# ---------------------------------------------------------------------

MODEL = "qwen/qwen3-30b-a3b-instruct-2507"

TEMPERATURE = 0

MAX_OUTPUT_TOKENS = 400

MAX_ATTEMPTS = 3

RETRY_DELAY_SECONDS = 2


# ---------------------------------------------------------------------
# FALLBACK PRICING
# ---------------------------------------------------------------------
#
# OpenRouter's returned usage.cost is preferred.
#
# These values are used only when actual request cost is unavailable.
# Final experiment analysis should use actual OpenRouter cost whenever
# it is returned.
# ---------------------------------------------------------------------

FALLBACK_INPUT_PRICE_PER_MILLION = 0.04815

FALLBACK_OUTPUT_PRICE_PER_MILLION = 0.1931


# ---------------------------------------------------------------------
# OPENROUTER CLIENT
# ---------------------------------------------------------------------

def create_client():
    """Create the shared OpenRouter client."""

    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(
        PROJECT_ROOT / ".env"
    )

    api_key = os.getenv(
        "OPENROUTER_API_KEY"
    )

    if not api_key:

        raise ValueError(
            "OPENROUTER_API_KEY was not found in .env/environment."
        )

    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,

        # All retries are explicitly logged by our own code.
        max_retries=0,
    )


# ---------------------------------------------------------------------
# FINQA CONTEXT
# ---------------------------------------------------------------------

def build_context(example):
    """Construct the shared FinQA evidence representation.

    The exact same representation should be supplied to all architectures.
    """

    pre_text = example.get(
        "pre_text",
        [],
    )

    post_text = example.get(
        "post_text",
        [],
    )

    table = example.get(
        "table",
        [],
    )

    context_parts = []


    if pre_text:

        context_parts.append(
            "Pre-text:\n"
            + "\n".join(pre_text)
        )


    if table:

        table_text = "\n".join(

            " | ".join(
                str(cell)
                for cell in row
            )

            for row in table
        )

        context_parts.append(
            "Table:\n"
            + table_text
        )


    if post_text:

        context_parts.append(
            "Post-text:\n"
            + "\n".join(post_text)
        )


    return "\n\n".join(
        context_parts
    )


# ---------------------------------------------------------------------
# FIXED FEW-SHOT DEMONSTRATIONS
# ---------------------------------------------------------------------

def get_few_shot_examples():
    """Return the fixed four FinQA training demonstrations.

    Demonstrations come only from train.json.

    Selection remains deterministic and is shared by every experimental
    configuration.
    """

    from single_agent.few_shot import (
        format_few_shot_examples,
    )

    return format_few_shot_examples()


# ---------------------------------------------------------------------
# STRICT JSON
# ---------------------------------------------------------------------

def _unique_json_object(pairs):
    """Reject duplicate JSON keys."""

    result = {}

    for key, value in pairs:

        if key in result:

            raise ValueError(
                f"Duplicate JSON key: {key!r}."
            )

        result[key] = value

    return result


def _reject_json_constant(value):
    """Reject non-standard JSON constants."""

    raise ValueError(
        f"Nonstandard JSON constant: {value}."
    )


def parse_json_response(
    raw_response,
    required_keys: Iterable[str],
):
    """Strictly parse an architecture's JSON response.

    No Markdown stripping.
    No substring extraction.
    No JSON repair.
    No silently added fields.

    Parameters
    ----------
    raw_response:
        Raw model content.

    required_keys:
        Exact keys that must appear in the JSON object.

    Returns
    -------
    tuple:
        (parsed_object, error)
    """

    required_keys = set(
        required_keys
    )


    if (
        not isinstance(
            raw_response,
            str,
        )
        or not raw_response.strip()
    ):

        return None, {
            "error_type":
                "empty_response",

            "error_message":
                "Expected a non-empty JSON response.",
        }


    try:

        parsed = json.loads(
            raw_response,
            object_pairs_hook=
                _unique_json_object,
            parse_constant=
                _reject_json_constant,
        )


    except ValueError as exc:

        return None, {
            "error_type":
                "invalid_json",

            "error_message":
                str(exc),
        }


    if not isinstance(
        parsed,
        dict,
    ):

        return None, {
            "error_type":
                "invalid_json_type",

            "error_message":
                "Expected a JSON object.",
        }


    if set(parsed) != required_keys:

        return None, {
            "error_type":
                "invalid_json_schema",

            "error_message":
                (
                    "Expected exactly these JSON keys: "
                    f"{sorted(required_keys)}; "
                    f"received {sorted(parsed.keys())}."
                ),
        }


    return parsed, None


# ---------------------------------------------------------------------
# TOKEN USAGE
# ---------------------------------------------------------------------

def _token_count(
    usage,
    field,
):

    value = getattr(
        usage,
        field,
        None,
    )

    if (
        type(value) is int
        and value >= 0
    ):
        return value

    return None


def _complete_usage_sum(
    attempts,
    field,
):

    values = [
        attempt[field]
        for attempt in attempts
    ]


    if all(
        value is not None
        for value in values
    ):

        return sum(values)


    return None


# ---------------------------------------------------------------------
# COST
# ---------------------------------------------------------------------

def calculate_fallback_cost(
    input_tokens,
    output_tokens,
):

    if (
        input_tokens is None
        or output_tokens is None
    ):

        return None


    return (
        input_tokens
        / 1_000_000
        * FALLBACK_INPUT_PRICE_PER_MILLION

        +

        output_tokens
        / 1_000_000
        * FALLBACK_OUTPUT_PRICE_PER_MILLION
    )


def extract_usage_cost(
    usage,
):
    """Extract OpenRouter's actual request cost when available."""

    if usage is None:
        return None


    value = getattr(
        usage,
        "cost",
        None,
    )


    if value is None:

        model_extra = getattr(
            usage,
            "model_extra",
            None,
        )

        if isinstance(
            model_extra,
            dict,
        ):

            value = model_extra.get(
                "cost"
            )


    try:

        if value is not None:

            return float(
                value
            )

    except (
        TypeError,
        ValueError,
    ):

        pass


    return None


# ---------------------------------------------------------------------
# API ERROR CLASSIFICATION
# ---------------------------------------------------------------------

def _status_code_from_exception(
    exc,
):

    value = getattr(
        exc,
        "status_code",
        None,
    )


    try:

        if value is not None:
            return int(value)

    except (
        TypeError,
        ValueError,
    ):

        pass


    return None


def is_fatal_api_error(
    status_code,
):
    """Identify API failures that should not be retried repeatedly.

    For example, retrying an insufficient-credit 402 error cannot succeed.
    """

    return status_code in {
        400,
        401,
        402,
        403,
        404,
    }


# ---------------------------------------------------------------------
# SHARED MODEL CALL
# ---------------------------------------------------------------------

def call_model(
    client,
    prompt,
    max_output_tokens=MAX_OUTPUT_TOKENS,
):
    """Run one controlled Qwen3-30B request.

    This function is architecture-independent.

    Agent-specific prompts are created outside this module.

    Returns raw output plus latency, token usage, cost, retries,
    and structured API error information.
    """

    attempts = []

    raw_response = None

    finish_reason = None

    api_error = None

    fatal_api_error = False


    started = time.perf_counter()


    for attempt_number in range(
        1,
        MAX_ATTEMPTS + 1,
    ):

        attempt_started = (
            time.perf_counter()
        )


        attempt = {

            "attempt_number":
                attempt_number,

            "latency":
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

            "api_error":
                None,

            "fatal_api_error":
                False,
        }


        try:

            response = (
                client.chat.completions.create(

                    model=
                        MODEL,

                    messages=[
                        {
                            "role":
                                "user",

                            "content":
                                prompt,
                        }
                    ],

                    temperature=
                        TEMPERATURE,

                    max_tokens=
                        max_output_tokens,

                    response_format={
                        "type":
                            "json_object"
                    },
                )
            )


        except Exception as exc:

            status_code = (
                _status_code_from_exception(
                    exc
                )
            )


            fatal_api_error = (
                is_fatal_api_error(
                    status_code
                )
            )


            api_error = {

                "error_type":
                    type(exc).__name__,

                "error_message":
                    str(exc),

                "status_code":
                    status_code,

                "fatal":
                    fatal_api_error,
            }


            attempt[
                "api_error"
            ] = api_error


            attempt[
                "fatal_api_error"
            ] = fatal_api_error


        else:

            usage = getattr(
                response,
                "usage",
                None,
            )


            attempt[
                "input_tokens"
            ] = _token_count(
                usage,
                "prompt_tokens",
            )


            attempt[
                "output_tokens"
            ] = _token_count(
                usage,
                "completion_tokens",
            )


            reported_total = (
                _token_count(
                    usage,
                    "total_tokens",
                )
            )


            if reported_total is not None:

                attempt[
                    "total_tokens"
                ] = reported_total


            elif (
                attempt[
                    "input_tokens"
                ]
                is not None

                and

                attempt[
                    "output_tokens"
                ]
                is not None
            ):

                attempt[
                    "total_tokens"
                ] = (

                    attempt[
                        "input_tokens"
                    ]

                    +

                    attempt[
                        "output_tokens"
                    ]
                )


            attempt[
                "actual_cost_usd"
            ] = extract_usage_cost(
                usage
            )


            attempt[
                "estimated_cost_usd"
            ] = calculate_fallback_cost(

                attempt[
                    "input_tokens"
                ],

                attempt[
                    "output_tokens"
                ],
            )


            choices = getattr(
                response,
                "choices",
                None,
            )


            if not choices:

                api_error = {

                    "error_type":
                        "invalid_api_response",

                    "error_message":
                        "API response contained no choices.",

                    "status_code":
                        None,

                    "fatal":
                        False,
                }


                attempt[
                    "api_error"
                ] = api_error


            else:

                message = getattr(
                    choices[0],
                    "message",
                    None,
                )


                raw_response = getattr(
                    message,
                    "content",
                    None,
                )


                finish_reason = getattr(
                    choices[0],
                    "finish_reason",
                    None,
                )


                api_error = None

                fatal_api_error = False


        attempt[
            "latency"
        ] = (
            time.perf_counter()
            - attempt_started
        )


        attempts.append(
            attempt
        )


        # Successful request
        if api_error is None:
            break


        # Permanent errors should not be retried.
        if fatal_api_error:
            break


        if (
            attempt_number
            < MAX_ATTEMPTS
        ):

            time.sleep(
                RETRY_DELAY_SECONDS
            )


    # -----------------------------------------------------------------
    # AGGREGATE USAGE
    # -----------------------------------------------------------------

    input_tokens = (
        _complete_usage_sum(
            attempts,
            "input_tokens",
        )
    )


    output_tokens = (
        _complete_usage_sum(
            attempts,
            "output_tokens",
        )
    )


    total_tokens = (
        _complete_usage_sum(
            attempts,
            "total_tokens",
        )
    )


    actual_cost_usd = (
        _complete_usage_sum(
            attempts,
            "actual_cost_usd",
        )
    )


    estimated_cost_usd = (
        calculate_fallback_cost(
            input_tokens,
            output_tokens,
        )
    )


    if actual_cost_usd is not None:

        effective_cost_usd = (
            actual_cost_usd
        )

        cost_source = (
            "openrouter_usage"
        )


    elif estimated_cost_usd is not None:

        effective_cost_usd = (
            estimated_cost_usd
        )

        cost_source = (
            "fallback_estimate"
        )


    else:

        effective_cost_usd = None

        cost_source = None


    return {

        "model":
            MODEL,

        "temperature":
            TEMPERATURE,

        "raw_response":
            raw_response,

        "finish_reason":
            finish_reason,

        "api_latency":
            (
                time.perf_counter()
                - started
            ),

        "input_tokens":
            input_tokens,

        "output_tokens":
            output_tokens,

        "total_tokens":
            total_tokens,

        "actual_cost_usd":
            actual_cost_usd,

        "estimated_cost_usd":
            estimated_cost_usd,

        "effective_cost_usd":
            effective_cost_usd,

        "cost_source":
            cost_source,

        "api_error":
            api_error,

        "fatal_api_error":
            fatal_api_error,

        "retry_count":
            len(attempts) - 1,

        "attempt_count":
            len(attempts),

        "attempts":
            attempts,
    }


# ---------------------------------------------------------------------
# OFFLINE SELF-CHECK
# ---------------------------------------------------------------------

def main():
    """Offline check only. No OpenRouter calls."""

    print(
        "Shared thesis agent configuration"
    )

    print(
        f"Model: {MODEL}"
    )

    print(
        f"Temperature: {TEMPERATURE}"
    )

    print(
        f"Maximum output tokens: "
        f"{MAX_OUTPUT_TOKENS}"
    )


    demonstrations = (
        get_few_shot_examples()
    )


    if not demonstrations.strip():

        raise ValueError(
            "Few-shot demonstrations are empty."
        )


    print(
        "Fixed FinQA training demonstrations: loaded"
    )

    print(
        "No API calls made."
    )


if __name__ == "__main__":
    main()