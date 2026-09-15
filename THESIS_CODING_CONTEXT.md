# Thesis Coding Context

## 1. Project Overview

Thesis title:

**Comparative Evaluation of Multi-Agent and Single-Agent LLM Systems for Business Decision Support**

The project experimentally compares different LLM agent architectures for financial numerical reasoning and decision-support tasks.

The main objective is not to prove that Multi-Agent systems are better.

The experiment should determine:

- whether additional agents improve reasoning performance,
- whether they improve reliability,
- how much additional latency/token usage/cost they introduce,
- and whether each architectural component provides measurable value.

The main independent variable is:

**Agent Architecture**

The underlying model must remain fixed across configurations.

---

## 2. Current Model

Model:

**Qwen3-8B**

OpenRouter model ID:

`qwen/qwen3-8b`

The same Qwen3-8B model must be used for all experiment configurations.

Reason for using Qwen3-8B:

- capable of numerical reasoning,
- supports agent-style workflows,
- relatively practical for repeated experiments,
- affordable enough for larger-scale FinQA experiments,
- keeping one model fixed allows architecture to remain the main independent variable.

Do not claim Qwen3-8B is the best model.

---

## 3. API

Current implementation uses:

**OpenRouter API**

Python orchestration is preferred.

The current implementation does NOT require LangGraph or CrewAI.

Do not migrate to LangGraph, CrewAI, AutoGen or another agent framework unless there is a clear experimental reason.

The purpose of the thesis is to compare architectures, not frameworks.

---

## 4. Dataset

Primary dataset:

**FinQA**

Paper:

Chen et al. (2021), "FinQA: A Dataset of Numerical Reasoning over Financial Data."

Official GitHub:

https://github.com/czyssrs/FinQA

Current local files:

`data/finqa/dev.json`

Contains approximately:

883 examples

`data/finqa/test.json`

Contains approximately:

1,147 examples

---

## 5. Dataset Usage Policy

Use:

`dev.json`

for:

- development,
- debugging,
- prompt refinement,
- implementation validation,
- smoke tests,
- pilot experiments.

Use:

`test.json`

only after:

- the architecture is finalized,
- prompts are frozen,
- evaluation code is finalized,
- methodology is frozen.

Do not tune prompts using the test set.

---

## 6. FinQA Data Structure

Important FinQA fields include:

- `pre_text`
- `post_text`
- `table`
- `id`
- `qa.question`
- `qa.program`
- `qa.exe_ans`
- `qa.gold_inds`
- `qa.program_re`

The model should receive:

- question,
- table,
- relevant financial context from pre_text/post_text.

The system should produce a FinQA program.

---

## 7. FinQA Program Representation

Example:

`divide(60, 243), multiply(#0, const_100)`

This represents sequential operations.

Operation 1:

`divide(60, 243)`

Operation 2 uses the result:

`multiply(#0, const_100)`

---

## 8. Allowed FinQA Operations

Supported operations include:

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

Do not allow unsupported invented operations such as:

- table_value
- retrieve_value
- lookup
- value
- extract

---

## 9. FinQA Syntax Rules

The generated program should use sequential FinQA syntax.

Correct:

`divide(60, 243), multiply(#0, const_100)`

Incorrect:

`multiply(divide(60, 243), const_100)`

Nested expressions should not be accepted.

Incorrect:

`#0 = divide(60, 243)`

Assignments should not be accepted.

Correct references:

- `#0`
- `#1`
- `#2`

A reference may only point to a previously executed operation.

---

## 10. Important Percentage Rule

Do NOT automatically convert decimal outputs to percentages.

Do NOT always multiply by `const_100`.

FinQA questions vary.

Examples include cases where the gold answer is a decimal and cases where percentage conversion is actually required.

The program must infer the correct transformation from the question and evidence.

Earlier experimentation showed that adding a blanket instruction such as:

"Do not convert decimals to percentages"

or

"Always multiply percentages by 100"

can artificially improve some examples while breaking others.

Do not introduce dataset-specific shortcuts.

---

## 11. Deterministic Execution

The LLM should generate the FinQA program.

The numerical result should then be calculated by Python.

Architecture:

LLM
→ generated FinQA program
→ validation
→ deterministic Python executor
→ numerical result

Do not rely on the LLM's own numerical answer as the primary final answer.

This separates:

**reasoning generation**

from

**program execution**

and makes evaluation reproducible.

---

## 12. Official FinQA Evaluation

The official evaluator uses the generated program and execution result.

Important concepts:

### Execution Accuracy

Whether the generated program executes to the correct numerical result.

Gold value:

`qa.exe_ans`

### Program Accuracy

Whether the generated reasoning program is symbolically equivalent to the gold FinQA program.

Gold program:

`qa.program`

The official FinQA evaluator allows symbolic equivalence and does not necessarily require identical surface text.

Official evaluator:

`code/evaluate/evaluate.py`

The final thesis evaluation should use or closely reproduce the official FinQA evaluation procedure.

---

## 13. Current Research Questions

### RQ1 — Reasoning Performance

How does agent architecture affect reasoning performance?

Primary measures:

- execution accuracy
- program accuracy

---

### RQ2 — Reliability

How does agent architecture affect reliability?

Primary measures:

- consistency across repeated runs
- malformed-output rate
- parsing failures
- invalid program rate
- execution failures
- API failures
- other error rates

---

### RQ3 — Efficiency

What efficiency trade-offs arise from additional agents?

Measure:

- latency
- input tokens
- output tokens
- total tokens
- estimated API cost

---

### RQ4 — Component Contribution

Which architectural components contribute to performance?

Use controlled ablation experiments.

---

## 14. Experimental Configurations

The current planned experiment has four configurations.

### Configuration A — Single-Agent Baseline

FinQA Input
→ Single Qwen3-8B Agent
→ FinQA Program
→ Validator
→ Deterministic Executor
→ Evaluation

---

### Configuration B — Reasoning Agent Only

FinQA Input
→ Reasoning Agent
→ FinQA Program
→ Validator
→ Deterministic Executor
→ Evaluation

This may be close to the baseline depending on final prompt design, but it should be implemented as a clearly defined experimental configuration.

---

### Configuration C — Reasoning + Verification

FinQA Input is supplied independently to:

- Reasoning Agent
- Verification Agent

Both independently generate candidate FinQA programs.

The Verification Agent must NOT simply see the Reasoning Agent's answer and approve or reject it.

It must independently solve the same FinQA problem from the same evidence.

A deterministic or clearly specified selection procedure should determine the final candidate.

---

### Configuration D — Full Multi-Agent

FinQA Input
→ Reasoning Agent

FinQA Input
→ Independent Verification Agent

Both candidate programs
→ Coordinator

Coordinator
→ Final FinQA Program

Final Program
→ Validator
→ Deterministic Executor
→ Evaluation

All agents use Qwen3-8B.

---

## 15. Independent Verification Requirement

This is important supervisor feedback.

The Verification Agent should receive the same:

- question,
- table,
- pre_text,
- post_text

as the Reasoning Agent.

However, it should independently solve the question.

It should not simply receive:

"Reasoning Agent produced X. Is X correct?"

That would create agreement bias.

Instead:

Reasoning Agent:

independent candidate A

Verification Agent:

independent candidate B

Coordinator:

compares candidate A and candidate B and selects or constructs the final program.

---

## 16. Experimental Controls

Across configurations A/B/C/D, keep the following constant:

- Qwen3-8B
- same FinQA questions
- same financial context
- same API/model settings
- same temperature
- same output format
- same program validator
- same deterministic executor
- same evaluation code
- same dataset split
- same logging logic

The purpose is to isolate:

**agent architecture**

as much as reasonably possible.

---

## 17. Previous Single-Agent Pilot

An earlier single-agent pilot used:

- Qwen3-8B
- OpenRouter
- temperature 0
- 10 dev examples
- custom numeric evaluator

Approximate results:

- 10 total examples
- 4 correct
- 6 incorrect
- 0 unparseable
- 0 API errors
- accuracy: approximately 40%
- average latency: approximately 27.85 seconds
- average input tokens: approximately 1,149
- average output tokens: approximately 1,334
- average cost/question: approximately $0.000742

Important:

This was a development pilot, not a final thesis result.

---

## 18. Previous Multi-Agent Pilot

Earlier multi-agent implementation:

Reasoning Agent
→ Verification Agent
→ Coordinator

Problems:

The Verification Agent was too dependent on the Reasoning Agent output.

The system often generated invalid FinQA syntax.

Approximate earlier metrics:

- 10 examples
- approximately 30% strict accuracy
- approximately 32.93 seconds average latency
- approximately 3,478 input tokens/question
- approximately 1,820 output tokens/question
- approximately $0.001235 per question

This result should not be interpreted as evidence that Multi-Agent is worse.

The implementation was not yet sufficiently valid.

---

## 19. Later Program-Based Multi-Agent Pilot

A later experiment switched from number-only output to FinQA program generation.

10/10 API pipelines completed successfully.

Approximate metrics:

- total latency: 447.62 seconds
- average latency: 44.76 seconds/question
- average input tokens: 3,512
- average output tokens: 2,910
- average estimated cost: $0.001735/question

However, many generated programs violated the FinQA DSL.

The most important conclusion:

**The immediate implementation problem was program-format compliance, not architecture performance.**

---

## 20. Specific Problems Found in Previous Pilot

### Example 1 — Valid Mathematical Answer

FinQA example:

`V/2008/page_17.pdf-1`

Model generated something similar to:

`divide(637, 5.0)`

This produced the correct numerical answer:

127.4

This is mathematically reasonable and may be symbolically equivalent to the FinQA constant representation.

---

### Example 2 — Unsupported Operation

For a C/2017 example, the model generated:

`table_value(...)`

This is unsupported.

The validator must reject it.

---

### Example 3 — Nested Program

For the DVN example, the model generated:

`multiply(divide(60, 243), const_100)`

Expected FinQA-style representation:

`divide(60, 243), multiply(#0, const_100)`

The validator must reject nested operations.

---

### Example 4 — Invalid Reference

For an ETR/2011 example, the system generated invalid references such as:

`subtract(#2, #0)`

when #2 had not been created.

The validator must verify reference order.

---

### Example 5 — Wrong Percentage Transformation

For ETR/2004, the model unnecessarily multiplied by 100.

Gold program:

`divide(59.1, 98.0)`

Gold answer:

0.60306

This confirms that automatic percentage conversion is incorrect.

---

### Example 6 — Assignment Syntax

For a JPM example, the model generated syntax similar to:

`#0 = divide(...)`

Assignments are not valid FinQA output.

The validator must reject assignment syntax.

---

### Example 7 — Wrong Financial Reasoning

For:

`PNC/2013/page_207.pdf-1`

Gold program:

`divide(36197, 1189)`

Gold answer:

30.44323

Earlier reasoning incorrectly changed the denominator.

This is a reasoning error, not a syntax error.

The system should distinguish:

- syntax/format errors
- reasoning errors

---

### Example 8 — Unsupported table_value Despite Correct Answer

Some examples generated:

`add(table_value(...), table_value(...))`

or similar.

The numerical answer could still be correct, but the program is not valid FinQA syntax.

Do not count such output as a fully valid program.

---

## 21. Important FinQA Examples

### V/2008

Question:

"What is the average payment volume per transaction for american express?"

Gold:

127.4

Gold program:

`divide(637, const_5)`

---

### DVN

Gold answer:

24.69136

Gold program:

`divide(60, 243), multiply(#0, const_100)`

---

### C/2017

Gold answer:

0.935

Gold program:

`subtract(193.5, const_100), divide(#0, const_100)`

---

### ETR/2004

Gold answer:

0.60306

Gold program:

`divide(59.1, 98.0)`

---

### PNC/2013 page_207

Gold answer:

30.44323

Gold program:

`divide(36197, 1189)`

---

### PNC/2013 page_62

Gold answer:

3576

Gold program:

`add(1356, 2220)`

---

## 22. Current Validation Requirements

Before accepting an LLM-generated program, validate:

1. program is not empty
2. contains supported operations only
3. no assignment syntax
4. no unsupported functions
5. no nested operations
6. correct operation syntax
7. valid number of arguments
8. intermediate references are valid
9. intermediate references point only backward
10. program can be tokenized
11. program can be executed deterministically

Record validation failure reasons.

---

## 23. Structured Output

Prefer structured JSON output from the LLM.

Example:

```json
{
  "program": "divide(60, 243), multiply(#0, const_100)"
}