import unittest

from memory_eval.evaluator import EvalResult


class MetricsTests(unittest.TestCase):
    def test_eval_result_shape(self) -> None:
        result = EvalResult(total_questions=10, correct_questions=7, overall_accuracy=0.7)
        self.assertEqual(result.total_questions, 10)
        self.assertEqual(result.correct_questions, 7)
        self.assertEqual(result.overall_accuracy, 0.7)


if __name__ == "__main__":
    unittest.main()

