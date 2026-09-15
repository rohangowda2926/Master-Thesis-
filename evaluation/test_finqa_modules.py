"""Offline checks for the shared FinQA validator and deterministic executor.

Run from the repository root with:
    python -m unittest evaluation.test_finqa_modules

All examples are synthetic or documented thesis-context fixtures. No dataset,
credentials, model calls, or saved experiment results are used.
"""

import unittest

from evaluation.finqa_executor import execute_program
from evaluation.finqa_validator import validate_program


class FinQAAssertions(unittest.TestCase):
    def assert_validation_schema(self, output):
        self.assertTrue({
            "valid", "normalized_program", "operations", "error_type",
            "error_message",
        }.issubset(output))
        self.assertIsInstance(output["valid"], bool)
        self.assertIsInstance(output["operations"], list)

    def assert_execution_schema(self, output):
        self.assertTrue({
            "executable", "result", "intermediate_results", "error_type",
            "error_message",
        }.issubset(output))
        self.assertIsInstance(output["executable"], bool)
        self.assertIsInstance(output["intermediate_results"], list)

    def assert_executes(self, program, expected, table=None):
        output = execute_program(program, table=table)
        self.assert_execution_schema(output)
        self.assertTrue(output["executable"], output)
        self.assertEqual(output["result"], expected)
        self.assertIsNone(output["error_type"])
        self.assertIsNone(output["error_message"])
        return output


class ValidatorTests(FinQAAssertions):
    def test_valid_program_exposes_operations_and_idempotent_normalization(self):
        output = validate_program("  divide(60, 243), multiply(#0, const_100)  ")
        self.assert_validation_schema(output)
        self.assertTrue(output["valid"], output)
        self.assertEqual(output["operations"], [
            {"operation": "divide", "arguments": ["60", "243"]},
            {"operation": "multiply", "arguments": ["#0", "const_100"]},
        ])
        self.assertIsInstance(output["normalized_program"], str)
        self.assertIsNone(output["error_type"])
        self.assertIsNone(output["error_message"])
        normalized = validate_program(output["normalized_program"])
        self.assertEqual(normalized["normalized_program"], output["normalized_program"])
        self.assertEqual(normalized["operations"], output["operations"])

    def test_invalid_programs_have_specific_reasons(self):
        cases = [
            ("", "empty_program"),
            ("   ", "empty_program"),
            (None, "invalid_input"),
            (12, "invalid_input"),
            ("#0 = divide(1, 2)", "assignment_syntax"),
            ("multiply(divide(60, 243), const_100)", "nested_operation"),
            ("table_value(revenue, none)", "unsupported_operation"),
            ("lookup(1, 2)", "unsupported_operation"),
            ("add(1, 2", "malformed_syntax"),
            ("add(1)", "invalid_arity"),
            ("add(, 2)", "invalid_arity"),
            ("add(1, 2, 3)", "invalid_arity"),
            ("add(1,234,567)", "invalid_arity"),
            ("add(#x, 1)", "invalid_reference"),
            ("add(#-1, 1)", "invalid_reference"),
            ("add(#00, 1)", "invalid_reference"),
            ("add(#0, 1)", "forward_reference"),
            ("add(1, 2), multiply(#2, 3)", "forward_reference"),
            ("add(one, 2)", "invalid_operand"),
            ("add(const_nope, 2)", "invalid_constant"),
            ("add(const_m2, 2)", "invalid_constant"),
            ("add(const_inf, 2)", "invalid_constant"),
            ("add(const_nan, 2)", "invalid_constant"),
            ("table_sum(revenue, 1)", "invalid_table_argument"),
            ("add(1, 2), table_sum(#0, none)", "invalid_table_argument"),
            ("The program is add(1, 2)", "malformed_syntax"),
            ("```\nadd(1, 2)\n```", "malformed_syntax"),
            ("add(1, 2) explanation", "malformed_syntax"),
            ("add(1, 2),", "malformed_syntax"),
            ("add(1, 2).", "malformed_syntax"),
        ]
        for program, error_type in cases:
            with self.subTest(program=program):
                output = validate_program(program)
                self.assert_validation_schema(output)
                self.assertFalse(output["valid"], output)
                self.assertEqual(output["error_type"], error_type)
                self.assertIsInstance(output["error_message"], str)
                self.assertTrue(output["error_message"])


class ExecutorTests(FinQAAssertions):
    def test_arithmetic_constants_percentages_and_comparisons(self):
        cases = [
            ("add(-2.5, +4e0)", 1.5),
            ("add(1,234, 2)", 1236.0),
            ("subtract(2, 5)", -3.0),
            ("multiply(3, const_m1)", -3.0),
            ("divide(7, 2)", 3.5),
            ("exp(2, 3)", 8.0),
            ("add(const_0.5, const_5)", 5.5),
            ("add(const_1, const_1000)", 1001.0),
            ("multiply(50%, 8)", 4.0),
            ("greater(3, 2)", "yes"),
            ("greater(2, 3)", "no"),
            ("greater(2, 2)", "no"),
        ]
        for program, expected in cases:
            with self.subTest(program=program):
                self.assert_executes(program, expected)

    def test_documented_thesis_program_fixtures(self):
        cases = [
            ("divide(637, const_5)", 127.4),
            ("divide(637, 5.0)", 127.4),
            ("divide(60, 243), multiply(#0, const_100)", 24.69136),
            ("subtract(193.5, const_100), divide(#0, const_100)", 0.935),
            ("divide(59.1, 98.0)", 0.60306),
            ("divide(36197, 1189)", 30.44323),
            ("add(1356, 2220)", 3576.0),
        ]
        for program, expected in cases:
            with self.subTest(program=program):
                self.assert_executes(program, expected)

    def test_rounds_only_final_result(self):
        output = self.assert_executes("divide(1, 3), multiply(#0, 3)", 1.0)
        self.assertEqual(output["intermediate_results"], [1 / 3, 1.0])
        self.assertNotEqual(output["intermediate_results"][0], round(1 / 3, 5))
        self.assert_executes("divide(1, 3)", 0.33333)

    def test_multiple_references_and_syntax_whitespace(self):
        program = " add ( 1,2 ),\n multiply(#0, 4), subtract(#1,#0), divide(#2,3) "
        output = self.assert_executes(program, 3.0)
        self.assertEqual(output["intermediate_results"], [3.0, 12.0, 9.0, 3.0])
        self.assertEqual(validate_program(program)["normalized_program"],
                         "add(1, 2), multiply(#0, 4), subtract(#1, #0), divide(#2, 3)")

    def test_all_table_operations_parse_financial_cells(self):
        table = [
            ["", "2022", "2023", "2024"],
            ["net revenue", "$1,000 (a)", "2,000", "3,000"],
        ]
        cases = {"table_max": 3000.0, "table_min": 1000.0,
                 "table_sum": 6000.0, "table_average": 2000.0}
        for operation, expected in cases.items():
            with self.subTest(operation=operation):
                self.assert_executes(f"{operation}(net revenue, none)", expected, table)
        self.assert_executes(
            "table_sum(net revenue, none), divide(#0, const_100)", 60.0, table
        )

    def test_table_percentage_and_duplicate_label_follow_upstream(self):
        table = [["rate", "1", "2"], ["rate", "25%", "75%"]]
        original = [row[:] for row in table]
        self.assert_executes("table_average(rate, none)", 0.5, table)
        self.assertEqual(table, original)
        overflow = execute_program("table_sum(rate, none)", [["rate", "1e308", "1e308"]])
        self.assertFalse(overflow["executable"])
        self.assertIsNone(overflow["result"])
        self.assertEqual(overflow["error_type"], "numeric_overflow")

    def test_table_failures_are_not_silently_skipped(self):
        cases = [
            (None, "missing_table"),
            ({"revenue": [1, 2]}, "invalid_table"),
            ([["revenue"]], "invalid_table"),
            ([["Revenue", "1", "2"]], "missing_table_row"),
            ([["revenue", "1", "not available"]], "invalid_table_cell"),
            ([["revenue", "1", "NaN"]], "invalid_table_cell"),
            ([["revenue", "(100)", "2"]], "invalid_table_cell"),
        ]
        for table, error_type in cases:
            with self.subTest(table=table):
                output = execute_program("table_sum(revenue, none)", table=table)
                self.assert_execution_schema(output)
                self.assertFalse(output["executable"], output)
                self.assertIsNone(output["result"])
                self.assertEqual(output["error_type"], error_type)
                self.assertTrue(output["error_message"])

    def test_validation_finishes_before_any_execution(self):
        output = execute_program("divide(1, 0), add(#2, 1)")
        self.assert_execution_schema(output)
        self.assertFalse(output["executable"])
        self.assertIsNone(output["result"])
        self.assertEqual(output["error_type"], "forward_reference")
        self.assertEqual(output["intermediate_results"], [])

    def test_runtime_failures_preserve_completed_intermediates(self):
        cases = [
            ("add(2, 3), divide(#0, 0)", "division_by_zero", [5.0]),
            ("add(2, 3), exp(1e308, 2)", "numeric_overflow", [5.0]),
            ("add(2, 3), exp(-1, 0.5)", "non_real_result", [5.0]),
            ("greater(2, 1), add(#0, 1)", "non_numeric_reference", ["yes"]),
        ]
        for program, error_type, intermediates in cases:
            with self.subTest(program=program):
                output = execute_program(program)
                self.assert_execution_schema(output)
                self.assertFalse(output["executable"], output)
                self.assertIsNone(output["result"])
                self.assertEqual(output["error_type"], error_type)
                self.assertEqual(output["intermediate_results"], intermediates)
                self.assertTrue(output["error_message"])


if __name__ == "__main__":
    unittest.main()
