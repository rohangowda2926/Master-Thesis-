# Comparative Evaluation of Multi-Agent and Single-Agent LLM Systems for Business Decision Support

## Master Thesis Project

This repository contains the implementation, experiments, evaluation pipeline, and reproducibility materials for the MSc dissertation:

**“Comparative Evaluation of Multi-Agent and Single-Agent LLM Systems for Business Decision Support.”**

The project evaluates whether increasing Large Language Model agent architecture complexity improves financial numerical reasoning performance, reliability, and decision-support efficiency.

The experiments are conducted using the **FinQA** benchmark while keeping the underlying language model fixed so that the primary experimental variable is the **agent architecture**.

---

## Research Questions

The study addresses four research questions:

### RQ1 — Reasoning Performance

How does agent architecture affect reasoning performance, measured through:

- FinQA execution accuracy
- FinQA program accuracy

### RQ2 — Reliability

How reliable are single-agent and multi-agent systems across repeated runs, measured through:

- repeated-run consistency
- correctness flips
- exact program agreement
- validation and execution failures

### RQ3 — Efficiency

What efficiency trade-offs arise when architecture complexity increases, measured through:

- latency
- input tokens
- output tokens
- API cost

### RQ4 — Component Contribution

What is the contribution of:

- explicit reasoning
- independent verification
- coordination

to the final system performance?

---

# Experimental Design

Four staged architectures are evaluated.

## Configuration A — Single-Agent

A single LLM directly generates a FinQA program from the question and financial evidence.

```text
Question + Evidence
        |
        v
   Single Agent
        |
        v
  FinQA Program
        |
        v
Validator -> Executor -> Evaluator