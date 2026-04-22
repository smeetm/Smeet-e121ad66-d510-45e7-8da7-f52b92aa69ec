import unittest

from memory_eval.judge import parse_judge_verdict


class JudgeParserTests(unittest.TestCase):
    def test_parse_judge_verdict_valid_json(self) -> None:
        ok, reason = parse_judge_verdict('{"correct": true, "reason": "match"}')
        self.assertTrue(ok)
        self.assertEqual(reason, "match")

    def test_parse_judge_verdict_with_wrapper_text(self) -> None:
        ok, reason = parse_judge_verdict('Output:\n{"correct": false, "reason": "missing detail"}')
        self.assertFalse(ok)
        self.assertEqual(reason, "missing detail")


if __name__ == "__main__":
    unittest.main()

