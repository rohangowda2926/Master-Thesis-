"""Offline symbolic-scoring regressions and fixtures for upstream comparison.

Run: python -m unittest evaluation.test_finqa_program_evaluator
No dataset, credentials, experiment logs, or model calls are used. Exported
matrices let a separate harness compare these fixtures with official FinQA code.
Malformed-input safeguards are tested separately from the equivalence matrix.

These expectations were differentially checked against the official functions at
https://github.com/czyssrs/FinQA/blob/main/code/evaluate/evaluate.py
on 2026-09-12 with SymPy 1.14.0. SHA256 of the retrieved reference text:
8fc4f5eaa2753eda4e4c779cf8310e60645008df4a673f99e3563ae4a2b359fa
The reference was loaded in memory; this test module never downloads/runs a
benchmark or calls a model. Pin the same SymPy version across experiment runs.
"""

import unittest

from evaluation.finqa_executor import execute_program
from evaluation.finqa_program_evaluator import compare_programs, program_tokenization


TOKENIZATION_CASES: list[str] = [
    "",
    "   ",
    "\n\t",
    "add(1, 2)",
    " add(1, 2) ",
    "\nadd(1, 2)\n",
    "add(1,2)",
    "add(1,  2)",
    "add(1, 2),multiply(#0, 3)",
    "add(1, 2), multiply(#0, 3)",
    "add(1, 2),\n multiply(#0, 3)",
    "table_sum(net revenue, none)",
    "table_sum(net revenue ,  none)",
    "table_average(revenue (net), none)",
    "add(1,234, 2)",
    "multiply(divide(1, 2), 3)",
    "#0 = add(1, 2)",
    "add(1, 2), ",
]


# Each tuple is (label, gold, prediction, expected program equivalence).
PROGRAM_CASES: list[tuple[str, str, str, bool]] = [
    (f"identity_{operation}", f"{operation}(2, 3)", f"{operation}(2, 3)", True)
    for operation in ("add", "subtract", "multiply", "divide", "exp", "greater")
] + [
    (f"identity_{operation}", f"{operation}(revenue, none)",
     f"{operation}(revenue, none)", True)
    for operation in ("table_max", "table_min", "table_sum", "table_average")
] + [
    ("constant_vs_integer", "divide(637, const_5)", "divide(637, 5)", False),
    ("constant_vs_decimal", "divide(637, const_5)", "divide(637, 5.0)", False),
    ("negative_constant_vs_literal", "multiply(2, const_m1)", "multiply(2, -1)", False),
    ("percentage_vs_decimal", "multiply(50%, 2)", "multiply(0.5, 2)", False),
    ("integer_vs_decimal", "divide(637, 5)", "divide(637, 5.0)", False),
    ("scientific_spelling", "add(5, 2)", "add(5e0, 2)", False),
    ("signed_spelling", "add(5, 2)", "add(+5, 2)", False),
    ("decimal_spelling", "add(0.5, 2)", "add(.5, 2)", False),
    ("constant_decimal_spelling", "add(const_5, 2)", "add(const_5.0, 2)", False),
    ("const_one_remains_symbolic", "multiply(5, const_1)",
     "divide(5, const_1)", False),
    ("literal_one_remains_symbolic", "multiply(5, 1)", "divide(5, 1)", False),
    ("const_zero_remains_symbolic", "add(5, const_0)", "subtract(5, const_0)", False),
    ("equal_answer_different_program", "add(2, 2)", "multiply(2, const_2)", False),
    ("commutative_add", "add(2, 3)", "add(3, 2)", True),
    ("commutative_multiply", "multiply(2, 3)", "multiply(3, 2)", True),
    ("subtract_order_matters", "subtract(2, 3)", "subtract(3, 2)", False),
    ("divide_order_matters", "divide(2, 3)", "divide(3, 2)", False),
    ("exponent_order_matters", "exp(2, 3)", "exp(3, 2)", False),
    ("comparison_order_matters", "greater(2, 3)", "greater(3, 2)", False),
    ("different_operation", "add(2, 3)", "subtract(2, 3)", False),
    ("outer_leading_whitespace", "add(1, 2)", " add(1, 2)", False),
    ("outer_trailing_whitespace", "add(1, 2)", "add(1, 2) ", False),
    ("missing_comma_space_is_not_repaired", "add(1, 2)", "add(1,2)", False),
    ("unused_step_known_operands", "add(1, 2)", "subtract(1, 2), add(1, 2)", True),
    ("unused_step_new_operand", "add(1, 2)", "add(99, 99), add(1, 2)", False),
    ("previous_reference_identity", "add(2, 3), multiply(#0, 4)",
     "add(2, 3), multiply(#0, 4)", True),
    ("changed_valid_reference", "add(2, 3), multiply(2, 3), subtract(#0, #1)",
     "add(2, 3), multiply(2, 3), subtract(#1, #0)", False),
    ("add_associativity", "add(1, 2), add(#0, 3)",
     "add(2, 3), add(1, #0)", True),
    ("multiply_associativity", "multiply(2, 3), multiply(#0, 4)",
     "multiply(3, 4), multiply(2, #0)", True),
    ("distribution", "add(2, 3), multiply(#0, 4)",
     "multiply(2, 4), multiply(3, 4), add(#0, #1)", True),
    ("equivalent_division", "divide(2, 3), multiply(#0, 4)",
     "multiply(2, 4), divide(#0, 3)", True),
    ("reordered_independent_arithmetic", "add(1, 2), multiply(3, 4), subtract(#0, #1)",
     "multiply(3, 4), add(1, 2), subtract(#1, #0)", True),
    ("table_name_matters", "table_sum(revenue, none)", "table_sum(cost, none)", False),
    ("table_operation_matters", "table_sum(revenue, none)",
     "table_average(revenue, none)", False),
    ("table_placeholder_matters", "table_sum(revenue, none)",
     "table_sum(revenue, zero)", False),
    ("table_label_space_matters", "table_sum(net revenue, none)",
     "table_sum(net  revenue, none)", False),
    ("table_argument_space_matters", "table_sum(revenue, none)",
     "table_sum(revenue , none)", False),
    ("table_leading_argument_space_matters", "table_sum(revenue, none)",
     "table_sum( revenue, none)", False),
    ("table_placeholder_space_matters", "table_sum(revenue, none)",
     "table_sum(revenue,  none)", False),
    ("table_reference_identity", "table_sum(revenue, none), divide(#0, const_2)",
     "table_sum(revenue, none), divide(#0, const_2)", True),
    # Official raw table identities retain the step-separator prefix. Moving a
    # table step between first and later positions therefore changes its symbol.
    ("table_moved_from_first_to_later", "table_sum(revenue, none), add(1, 2), add(#0, #1)",
     "add(1, 2), table_sum(revenue, none), add(#0, #1)", False),
    ("table_moved_from_later_to_first", "add(1, 2), table_sum(revenue, none), add(#0, #1)",
     "table_sum(revenue, none), add(1, 2), add(#0, #1)", False),
    ("reordered_tables_both_later", "add(1, 2), table_sum(revenue, none), table_max(cost, none), add(#1, #2)",
     "add(1, 2), table_max(cost, none), table_sum(revenue, none), add(#1, #2)", True),
    ("repeated_table_first_and_later_are_distinct", "table_sum(revenue, none), table_sum(revenue, none), subtract(#0, #1)",
     "table_sum(revenue, none), subtract(#0, #0)", False),
    ("repeated_table_identity", "table_sum(revenue, none), table_sum(revenue, none), subtract(#0, #1)",
     "table_sum(revenue, none), table_sum(revenue, none), subtract(#0, #1)", True),
    ("repeated_later_table_shares_symbol", "add(1, 2), table_sum(revenue, none), table_sum(revenue, none), subtract(#1, #2)",
     "add(1, 2), table_sum(revenue, none), subtract(#1, #1)", True),
]


class ProgramEvaluatorTests(unittest.TestCase):
    def assert_result_schema(self, result):
        self.assertEqual(set(result), {
            "program_correct", "gold_tokens", "predicted_tokens",
            "gold_symbolic", "predicted_symbolic", "error_type", "error_message",
        })
        self.assertIsInstance(result["program_correct"], bool)
        for field in ("gold_tokens", "predicted_tokens"):
            if result[field] is not None:
                self.assertIsInstance(result[field], list)
                self.assertTrue(all(isinstance(token, str) for token in result[field]))
        for field in ("gold_symbolic", "predicted_symbolic", "error_type", "error_message"):
            if result[field] is not None:
                self.assertIsInstance(result[field], str)

    def test_program_equivalence_matrix(self):
        for label, gold, prediction, expected in PROGRAM_CASES:
            with self.subTest(case=label):
                result = compare_programs(gold, prediction)
                self.assert_result_schema(result)
                self.assertEqual(result["program_correct"], expected, result)
                if expected:
                    self.assertIsNone(result["error_type"])
                    self.assertIsNone(result["error_message"])

    def test_tokenizer_preserves_upstream_raw_text_behavior(self):
        cases = [
            ("", ["EOF"]),
            ("   ", ["   ", "EOF"]),
            ("add(1, 2)", ["add(", "1", "2", ")", "EOF"]),
            (" add(1, 2) ", [" add(", "1", "2", ")", " ", "EOF"]),
            ("add(1,2)", ["add(", "1,2", ")", "EOF"]),
            ("add(1,  2)", ["add(", "1", " 2", ")", "EOF"]),
        ]
        for source, expected in cases:
            with self.subTest(source=source):
                self.assertEqual(program_tokenization(source), expected)
        for source in TOKENIZATION_CASES:
            with self.subTest(exported_source=source):
                self.assertEqual(program_tokenization(source)[-1], "EOF")

    def test_nonstring_tokenizer_input_is_rejected(self):
        for value in (None, 12, [], {}):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    program_tokenization(value)

    def test_malformed_predictions_return_structured_false(self):
        # These are safeguards, not claims about every upstream malformed-input
        # edge case. The baseline invokes comparison even after validation fails.
        predictions = [
            None, 12, [], {}, "", "   ",
            "add(1)", "add(1, 2, 3)", "add(1, 2", "add(1, 2))",
            "multiply(add(1, 2), 3)", "#0 = add(1, 2)",
            "lookup(1, 2)", "add(#0, 2)", "add(#1, 2)",
            "add(#x, 2)", "add(#-1, 2)",
            "add(1, 2), multiply(#2, 3)",
            "add(1, 2)add(1, 2)",
            "add(1, 2), table_sum(#0, none)",
            "add(1, 2), ", "add(1, 2) trailing prose",
            "```\nadd(1, 2)\n```",
        ]
        for prediction in predictions:
            with self.subTest(prediction=prediction):
                result = compare_programs("add(1, 2)", prediction)
                self.assert_result_schema(result)
                self.assertFalse(result["program_correct"], result)
                self.assertTrue(result["error_type"], result)
                self.assertTrue(result["error_message"], result)

    def test_malformed_gold_returns_structured_false(self):
        for gold in (None, 12, [], {}, "", "   ", "add(1)", "add(#0, 2)"):
            with self.subTest(gold=gold):
                result = compare_programs(gold, "add(1, 2)")
                self.assert_result_schema(result)
                self.assertFalse(result["program_correct"], result)
                self.assertTrue(result["error_type"], result)
                self.assertTrue(result["error_message"], result)

    def test_equal_execution_does_not_imply_program_equivalence(self):
        gold = "divide(637, const_5)"
        prediction = "divide(637, 5.0)"
        gold_execution = execute_program(gold)
        predicted_execution = execute_program(prediction)
        self.assertTrue(gold_execution["executable"])
        self.assertTrue(predicted_execution["executable"])
        self.assertEqual(gold_execution["result"], predicted_execution["result"])
        self.assertFalse(compare_programs(gold, prediction)["program_correct"])

    def test_symbolic_comparison_does_not_gate_on_runtime_success(self):
        # equal_program compares symbolic formulas. The official end-to-end
        # evaluator handles execution separately; this helper must not run code.
        gold = "add(1, 0), subtract(1, 1)"
        prediction = "divide(1, 0), subtract(1, 1)"
        self.assertTrue(compare_programs(gold, prediction)["program_correct"])
        self.assertTrue(execute_program(gold)["executable"])
        execution = execute_program(prediction)
        self.assertFalse(execution["executable"])
        self.assertEqual(execution["error_type"], "division_by_zero")


if __name__ == "__main__":
    unittest.main()
