import os
import json
import time
import re
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

MODEL = "qwen/qwen3-8b"

DATASET_PATH = Path("data/finqa/dev.json")

RESULTS_PATH = Path("results/multi_agent_results.jsonl")
PREDICTIONS_PATH = Path("results/multi_agent_predictions.json")

NUM_EXAMPLES = 10

MAX_RETRIES = 3
MAX_OUTPUT_TOKENS = 300
TEMPERATURE = 0

INPUT_PRICE_PER_MILLION = 0.117
OUTPUT_PRICE_PER_MILLION = 0.455


# ============================================================
# OPENROUTER
# ============================================================

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError("OPENROUTER_API_KEY not found in .env")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)


# ============================================================
# HELPERS
# ============================================================

def clean_response(text):
    if not text:
        return ""

    text = text.strip()

    text = re.sub(
        r"```(?:text|json)?",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = text.replace("```", "")

    return text.strip()


def calculate_cost(input_tokens, output_tokens):
    input_cost = (
        input_tokens / 1_000_000
    ) * INPUT_PRICE_PER_MILLION

    output_cost = (
        output_tokens / 1_000_000
    ) * OUTPUT_PRICE_PER_MILLION

    return input_cost + output_cost


# ============================================================
# MODEL CALL
# ============================================================

def call_model(prompt):

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            start = time.time()

            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_OUTPUT_TOKENS,
            )

            latency = time.time() - start

            content = (
                response.choices[0].message.content
                or ""
            )

            usage = response.usage

            input_tokens = (
                getattr(
                    usage,
                    "prompt_tokens",
                    0,
                )
                if usage
                else 0
            )

            output_tokens = (
                getattr(
                    usage,
                    "completion_tokens",
                    0,
                )
                if usage
                else 0
            )

            return {
                "response": clean_response(content),
                "latency": latency,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "estimated_cost": calculate_cost(
                    input_tokens,
                    output_tokens,
                ),
                "error": None,
            }

        except Exception as e:

            if attempt == MAX_RETRIES:

                return {
                    "response": "",
                    "latency": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "estimated_cost": 0,
                    "error": str(e),
                }

            time.sleep(2)


# ============================================================
# PROGRAM EXTRACTION
# ============================================================

def extract_program(text):

    if not text:
        return None

    text = clean_response(text)

    match = re.search(
        r"(?:Final\s+program|Program|PROGRAM)\s*:\s*(.+)",
        text,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    program = match.group(1).strip()

    # Keep only the first line
    program = program.split("\n")[0].strip()

    # Remove markdown
    program = program.replace("`", "")

    # Remove accidental trailing period
    if program.endswith("."):
        program = program[:-1]

    return program if program else None


# ============================================================
# FINQA PROGRAM TOKENIZER
# ============================================================

def program_tokenization(original_program):
    if not original_program:
        return None

    original_program = original_program.strip().split(", ")

    program = []

    for tok in original_program:
        cur_tok = ""

        for c in tok:
            if c == ")":
                if cur_tok:
                    program.append(cur_tok)
                    cur_tok = ""

            cur_tok += c

            if c in ["(", ")"]:
                program.append(cur_tok)
                cur_tok = ""

        if cur_tok:
            program.append(cur_tok)

    program.append("EOF")

    return program

# ============================================================
# REASONING AGENT
# ============================================================

def run_reasoning_agent(question, context):

    prompt = f"""
You are the reasoning agent in a FinQA numerical reasoning system.

Solve the question using ONLY the supplied context.

Your task is to produce the correct FinQA reasoning program.

ALLOWED OPERATIONS ONLY:
add
subtract
multiply
divide
exp
greater
table_max
table_min
table_sum
table_average

Do NOT use any other operation.
In particular, NEVER use retrieve_value, lookup, search, or any invented operation.

Constants may include:
const_100
const_m1
and other FinQA constants when required.

Intermediate results are referenced as:
#0
#1
#2

IMPORTANT:
- Select the correct values from the supplied table/text.
- Determine the mathematical relationship required by the question.
- Do not invent values.
- Do not use outside knowledge.
- A question containing the word "percentage" does NOT automatically mean multiply by 100.
- Use multiply(#0, const_100) only when it is required by the mathematical interpretation of the question and the supplied data.
- Follow the numerical representation implied by the FinQA task.
- Return exactly ONE complete FinQA program.
- Keep the response concise.

Question:
{question}

Context:
{context}

Return ONLY these two lines:

Final program: <FinQA program>
Final answer: <numerical result>
"""

    return call_model(prompt)

# ============================================================
# VERIFICATION AGENT
# ============================================================

def run_verification_agent(
    question,
    context,
    reasoning_response,
):

    prompt = f"""
You are the verification agent in a FinQA numerical reasoning system.

Independently verify the reasoning agent's proposed program.

Check:

1. Are the selected values supported by the context?
2. Is the mathematical operation correct?
3. Is the operation order correct?
4. Are intermediate references such as #0 valid?
5. Is percentage/ratio handling mathematically appropriate?
6. Does the program answer the actual question?

ALLOWED OPERATIONS ONLY:
add
subtract
multiply
divide
exp
greater
table_max
table_min
table_sum
table_average

NEVER use:
retrieve_value
lookup
search
or any other operation.

Do NOT automatically agree with the reasoning agent.

If the reasoning is correct, keep the same program.

If it is incorrect, construct the corrected FinQA program yourself.

IMPORTANT:
A question containing "percentage" does not automatically require multiplication by 100.
Determine the required calculation from the question and supplied data.

Question:
{question}

Context:
{context}

Reasoning agent response:
{reasoning_response}

Return ONLY these two lines:

Final program: <correct FinQA program>
Final answer: <numerical result>
"""

    return call_model(prompt)


# ============================================================
# COORDINATOR
# ============================================================

def run_coordinator_agent(
    question,
    context,
    reasoning_response,
    verification_response,
):

    prompt = f"""
You are the final coordinator of a FinQA numerical reasoning system.

Use the reasoning and verification outputs to produce the final
FinQA program.

The verification agent has checked the candidate, but you must
ensure the final program is valid and answers the question.

ALLOWED OPERATIONS ONLY:
add
subtract
multiply
divide
exp
greater
table_max
table_min
table_sum
table_average

NEVER use:
retrieve_value
lookup
search
or any invented operation.

Rules:
- Return exactly ONE complete FinQA program.
- Use only values supported by the context.
- Preserve the correct operation order.
- Do not invent values.
- Do not automatically multiply percentages by 100.
- Use const_100 only when required by the actual calculation.
- Do not provide explanations.
- Do not provide multiple candidate programs.

Question:
{question}

Context:
{context}

Reasoning agent:
{reasoning_response}

Verification agent:
{verification_response}

Return ONLY these two lines:

Final program: <FinQA program>
Final answer: <numerical result>
"""

    return call_model(prompt)


# ============================================================
# DATASET
# ============================================================

def load_dataset():

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


# ============================================================
# CONTEXT
# ============================================================

def build_context(item):

    pre_text = " ".join(
        item.get("pre_text", [])
    )

    post_text = " ".join(
        item.get("post_text", [])
    )

    table = item.get("table", [])

    table_text = ""

    for row in table:

        table_text += (
            " | ".join(
                str(x)
                for x in row
            )
            + "\n"
        )

    return f"""
PRE-TEXT:
{pre_text}

TABLE:
{table_text}

POST-TEXT:
{post_text}
""".strip()


# ============================================================
# MAIN
# ============================================================

def main():

    dataset = load_dataset()

    examples = dataset[:NUM_EXAMPLES]

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Start fresh
    RESULTS_PATH.write_text(
        "",
        encoding="utf-8",
    )

    predictions = []

    total_latency = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0

    successful_runs = 0

    print("=" * 70)
    print("MULTI-AGENT FINQA PROGRAM PILOT")
    print("=" * 70)

    print(f"Model: {MODEL}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Examples: {len(examples)}")
    print()

    for index, item in enumerate(
        examples,
        start=1,
    ):

        qa = item["qa"]

        example_id = item["id"]

        question = qa["question"]

        gold_answer = qa["exe_ans"]

        gold_program = qa["program"]

        context = build_context(item)

        print("-" * 70)
        print(
            f"Example {index}/{len(examples)}"
        )

        print(f"ID: {example_id}")
        print(f"Question: {question}")
        print(f"Gold answer: {gold_answer}")
        print(f"Gold program: {gold_program}")

        # ----------------------------------------------------
        # REASONING
        # ----------------------------------------------------

        reasoning = run_reasoning_agent(
            question,
            context,
        )

        print("\n[Reasoning Agent]")
        print(reasoning["response"])

        # ----------------------------------------------------
        # VERIFICATION
        # ----------------------------------------------------

        verification = run_verification_agent(
            question,
            context,
            reasoning["response"],
        )

        print("\n[Verification Agent]")
        print(verification["response"])

        # ----------------------------------------------------
        # COORDINATOR
        # ----------------------------------------------------

        coordinator = run_coordinator_agent(
            question,
            context,
            reasoning["response"],
            verification["response"],
        )

        print("\n[Coordinator]")
        print(coordinator["response"])

        # ----------------------------------------------------
        # EXTRACT PROGRAM
        # ----------------------------------------------------

        predicted_program = extract_program(
            coordinator["response"]
        )

        predicted_tokens = program_tokenization(
            predicted_program
        )

        print("\n[Predicted program]")
        print(predicted_program)

        print("\n[Predicted tokens]")
        print(predicted_tokens)

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        latency = (
            reasoning["latency"]
            + verification["latency"]
            + coordinator["latency"]
        )

        input_tokens = (
            reasoning["input_tokens"]
            + verification["input_tokens"]
            + coordinator["input_tokens"]
        )

        output_tokens = (
            reasoning["output_tokens"]
            + verification["output_tokens"]
            + coordinator["output_tokens"]
        )

        cost = (
            reasoning["estimated_cost"]
            + verification["estimated_cost"]
            + coordinator["estimated_cost"]
        )

        errors = []

        for name, result in [
            ("reasoning", reasoning),
            ("verification", verification),
            ("coordinator", coordinator),
        ]:

            if result["error"]:
                errors.append(
                    f"{name}: {result['error']}"
                )

        error = (
            "; ".join(errors)
            if errors
            else None
        )

        if error is None:
            successful_runs += 1

        total_latency += latency
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        total_cost += cost

        # ----------------------------------------------------
        # SAVE RESULT
        # ----------------------------------------------------

        result = {
            "id": example_id,
            "question": question,

            "gold_answer": gold_answer,
            "gold_program": gold_program,

            "predicted_program": predicted_program,
            "predicted_tokens": predicted_tokens,

            "model": MODEL,
            "temperature": TEMPERATURE,

            "reasoning_response": reasoning["response"],
            "verification_response": verification["response"],
            "coordinator_response": coordinator["response"],

            "latency": latency,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost": cost,

            "reasoning_latency": reasoning["latency"],
            "verification_latency": verification["latency"],
            "coordinator_latency": coordinator["latency"],

            "error": error,
        }

        with open(
            RESULTS_PATH,
            "a",
            encoding="utf-8",
        ) as f:

            f.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

        # Official FinQA prediction format
        predictions.append(
            {
                "id": example_id,
                "predicted": predicted_tokens,
            }
        )

        print("\n[Metrics]")
        print(f"Latency: {latency:.2f}s")
        print(f"Input tokens: {input_tokens}")
        print(f"Output tokens: {output_tokens}")
        print(f"Estimated cost: ${cost:.6f}")

    # ========================================================
    # SAVE OFFICIAL PREDICTION FILE
    # ========================================================

    with open(
        PREDICTIONS_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            predictions,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("PILOT SUMMARY")
    print("=" * 70)

    print(
        f"Successful pipeline runs: "
        f"{successful_runs}/{len(examples)}"
    )

    print(
        f"Total latency: "
        f"{total_latency:.2f}s"
    )

    print(
        f"Average latency: "
        f"{total_latency / len(examples):.2f}s/question"
    )

    print(
        f"Total input tokens: "
        f"{total_input_tokens}"
    )

    print(
        f"Average input tokens: "
        f"{total_input_tokens / len(examples):.0f}"
    )

    print(
        f"Total output tokens: "
        f"{total_output_tokens}"
    )

    print(
        f"Average output tokens: "
        f"{total_output_tokens / len(examples):.0f}"
    )

    print(
        f"Total estimated cost: "
        f"${total_cost:.6f}"
    )

    print(
        f"Average estimated cost: "
        f"${total_cost / len(examples):.6f}/question"
    )

    print()
    print(
        f"Results saved to: {RESULTS_PATH}"
    )

    print(
        f"Official predictions saved to: "
        f"{PREDICTIONS_PATH}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()