import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = PROJECT_ROOT / "data" / "finqa" / "train.json"


def operation_sequence(program: str):
    if not isinstance(program, str):
        return []

    operations = []

    for step in program.split(", "):
        if "(" not in step:
            continue

        operations.append(
            step.split("(", 1)[0].strip()
        )

    return operations


def build_context(example):
    pre_text = example.get("pre_text", [])
    post_text = example.get("post_text", [])
    table = example.get("table", [])

    parts = []

    if pre_text:
        parts.append(
            "Pre-text:\n"
            + "\n".join(pre_text)
        )

    if table:
        table_text = "\n".join(
            " | ".join(str(cell) for cell in row)
            for row in table
        )

        parts.append(
            "Table:\n"
            + table_text
        )

    if post_text:
        parts.append(
            "Post-text:\n"
            + "\n".join(post_text)
        )

    return "\n\n".join(parts)


def select_few_shot_examples():
    with TRAIN_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        train = json.load(file)

    targets = [
        ("subtract", ["subtract"]),
        ("divide", ["divide"]),
        (
            "percentage",
            ["divide", "multiply"],
        ),
        (
            "multi_step",
            ["subtract", "divide"],
        ),
    ]

    selected = {}

    for example in train:

        qa = example.get("qa", {})

        program = qa.get("program")

        if not program:
            continue

        operations = operation_sequence(program)

        for name, required_sequence in targets:

            if name in selected:
                continue

            if operations != required_sequence:
                continue

            # Additional rule for percentage example:
            # require explicit const_100 so the demonstration
            # actually teaches percentage scaling.
            if (
                name == "percentage"
                and "const_100" not in program
            ):
                continue

            selected[name] = example

        if len(selected) == len(targets):
            break

    missing = [
        name
        for name, _ in targets
        if name not in selected
    ]

    if missing:
        raise ValueError(
            f"Could not find training examples for: {missing}"
        )

    return [
        selected[name]
        for name, _ in targets
    ]


def format_few_shot_examples():
    examples = select_few_shot_examples()

    formatted = []

    for index, example in enumerate(
        examples,
        start=1,
    ):

        question = example["qa"]["question"]
        program = example["qa"]["program"]
        context = build_context(example)

        formatted.append(
            f"""
DEMONSTRATION {index}

Question:
{question}

Financial evidence:
{context}

Correct FinQA output:
{{"program": "{program}"}}
""".strip()
        )

    return "\n\n".join(formatted)


if __name__ == "__main__":
    examples = select_few_shot_examples()

    print(
        f"Selected {len(examples)} training demonstrations."
    )

    for index, example in enumerate(
        examples,
        start=1,
    ):
        print()
        print(
            f"Demo {index}: {example['id']}"
        )
        print(
            f"Question: {example['qa']['question']}"
        )
        print(
            f"Program: {example['qa']['program']}"
        )