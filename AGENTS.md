# Master Thesis Coding Instructions

## Project

Thesis title:

**Comparative Evaluation of Multi-Agent and Single-Agent LLM Systems for Business Decision Support**

The project experimentally compares different LLM agent architectures for financial numerical reasoning using the FinQA dataset.

---

## Core Experimental Principle

The main independent variable is:

**Agent Architecture**

The underlying LLM must remain the same across all experimental configurations.

Current model:

`qwen/qwen3-8b`

Provider:

OpenRouter

Do not switch models between configurations unless explicitly required for a separate experiment.

---

## Planned Experimental Configurations

### A. Single-Agent Baseline

One Qwen3-8B agent generates the FinQA program.

### B. Reasoning Agent Only

One dedicated reasoning agent generates the FinQA program.

### C. Reasoning + Independent Verification

The Reasoning Agent and Verification Agent independently solve the same FinQA problem.

The Verification Agent must not simply approve or reject the Reasoning Agent's answer.

### D. Full Multi-Agent

Reasoning Agent + Independent Verification Agent + Coordinator.

The Coordinator receives the independently generated candidate programs and produces the final FinQA program.

---

## Experimental Controls

Keep these constant across A/B/C/D:

- Qwen3-8B
- same FinQA examples
- same financial context
- same generation settings
- same temperature
- same output representation
- same validator
- same deterministic executor
- same evaluation procedure
- same logging procedure

Architecture should be the main variable being changed.

---

## Research Questions

### RQ1 — Reasoning Performance

Measure mainly with:

- execution accuracy
- program accuracy

### RQ2 — Reliability

Measure mainly with:

- repeated-run consistency
- malformed output rate
- invalid program rate
- parsing failures
- execution failures
- API failures

### RQ3 — Efficiency

Measure with:

- latency
- input tokens
- output tokens
- total tokens
- estimated API cost

### RQ4 — Component Contribution

Use ablation/configuration comparisons to determine which architectural components contribute to performance.

---

## Important Interpretation Rule

Do not assume Multi-Agent will outperform Single-Agent.

The experiment must objectively determine:

- whether additional agents improve reasoning,
- whether they improve reliability,
- whether any gain justifies extra latency,
- token usage,
- API cost,
- and implementation complexity.

A result where Single-Agent performs better is still valid.

---

## Dataset

Primary dataset:

**FinQA**

Local files:

`data/finqa/dev.json`

`data/finqa/test.json`

Use `dev.json` for:

- development
- debugging
- prompt refinement
- smoke tests
- pilot experiments

Reserve `test.json` for final evaluation after methodology and prompts are frozen.

Do not tune prompts against the test set.

---

## FinQA Output Format

The LLM should generate a FinQA program.

Example:

`divide(60, 243), multiply(#0, const_100)`

The final numerical answer should be produced by deterministic Python execution.

Do not depend on the LLM's own numerical answer.

---

## Allowed FinQA Operations

Allowed operations include:

- add
- subtract
- multiply
- divide
- exp
- greater
- table_max
- table_min
- table_sum
- table_average

Reject unsupported invented operations such as:

- table_value
- retrieve_value
- lookup
- extract_value

---

## FinQA Syntax Rules

Use sequential FinQA syntax.

Correct:

`divide(60, 243), multiply(#0, const_100)`

Incorrect nested form:

`multiply(divide(60, 243), const_100)`

Do not use assignment syntax.

Incorrect:

`#0 = divide(60, 243)`

Use intermediate references:

- `#0`
- `#1`
- `#2`

References must point only to previously completed operations.

---

## Percentage Handling

Do not hardcode percentage conversion rules.

Do not always multiply by `const_100`.

Do not always avoid percentage conversion.

Infer the required calculation from the question and financial evidence.

Do not introduce dataset-specific hacks to improve pilot results.

---

## Deterministic Execution

Required pipeline:

LLM
→ FinQA program
→ validator
→ deterministic Python executor
→ predicted result
→ evaluator

Keep reasoning generation separate from numerical execution.

---

## Verification Agent Rule

The Verification Agent must be independent.

It should receive the same:

- question
- table
- pre_text
- post_text

as the Reasoning Agent.

It should independently generate its own candidate program.

It should not simply receive the Reasoning Agent result and say whether it is correct.

---

## Validation Requirements

Validate generated programs for:

1. empty output
2. unsupported operations
3. assignment syntax
4. nested operations
5. malformed syntax
6. wrong operation arity
7. invalid references
8. forward references
9. tokenization failure
10. execution failure

Always record the validation failure reason.

---

## Logging Requirements

Log at least:

- example ID
- architecture/configuration
- run number
- generated program
- program validity
- validation error
- predicted answer
- gold answer
- execution correctness
- program correctness
- latency
- input tokens
- output tokens
- total tokens
- estimated cost
- parse error
- API error
- retry count

For multi-agent systems also log:

- reasoning-agent program
- reasoning-agent validity
- verification-agent program
- verification-agent validity
- coordinator program
- final selected program

---

## Scaling Strategy

Do not immediately run the full dev set after changing code.

Use this sequence:

1. 3-5 example smoke test
2. inspect outputs manually
3. 10-example pilot
4. larger dev evaluation
5. freeze prompts/methodology
6. final test evaluation

Do not run all 883 dev examples until the pipeline is valid.

---

## Coding Approach

Prefer:

- simple Python
- modular code
- reproducibility
- shared evaluation code
- explicit configuration
- transparent logging

Avoid unnecessary framework complexity.

Do not migrate to LangGraph, CrewAI, AutoGen or another framework unless there is a clear experimental reason.

The current direct Python/OpenRouter orchestration is acceptable.

---

## Repository Design

Prefer reusable shared modules for:

- FinQA validation
- FinQA execution
- evaluation
- metrics
- experiment configuration
- logging

Avoid duplicating evaluator logic separately inside each architecture implementation.

All configurations should use the same evaluation components.

---

## Research Integrity Rules

Never:

- hardcode gold answers
- use gold answers during generation
- use gold programs during generation
- manually repair outputs during final evaluation
- modify outputs depending on whether they match gold
- tune prompts on the final test set
- silently introduce dataset-specific fixes

Generic deterministic normalization is acceptable only if:

- it does not use gold information,
- it applies equally to all configurations,
- it is documented in the thesis.

---

## Before Modifying Code

Always:

1. Read `THESIS_CODING_CONTEXT.md`
2. Inspect existing code
3. Reuse working components where possible
4. Propose the smallest necessary change
5. Do not rewrite the entire project from scratch unless explicitly asked
6. Do not change the experimental methodology without discussing it first