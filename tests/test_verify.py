import sys
import unittest

from hbws import verify
from hbws.protocol import grade_with_feedback


class CodeGraderTests(unittest.TestCase):
    def test_known_correct_solution_passes(self):
        ok, feedback = verify.run_code_tests(
            "def add(a, b):\n    return a + b",
            "assert add(2, 3) == 5",
        )
        self.assertTrue(ok, feedback)
        self.assertEqual(feedback, "all tests passed")

    def test_wrong_solution_remains_a_model_failure(self):
        ok, feedback = verify.run_code_tests(
            "def add(a, b):\n    return a - b",
            "assert add(2, 3) == 5",
        )
        self.assertFalse(ok)
        self.assertIn("AssertionError", feedback)

    def test_darwin_prelude_does_not_set_address_space_limit(self):
        prelude = verify._sandbox_prelude()
        if sys.platform == "darwin":
            self.assertNotIn("setrlimit(resource.RLIMIT_AS", prelude)
        self.assertIn("setrlimit(resource.RLIMIT_CPU", prelude)

    def test_protocol_exposes_grader_feedback(self):
        task = {
            "family": "code",
            "grading_tests": "assert identity(7) == 7",
        }
        ok, feedback = grade_with_feedback(
            task, "def identity(x):\n    return x"
        )
        self.assertTrue(ok)
        self.assertEqual(feedback, "all tests passed")


if __name__ == "__main__":
    unittest.main()
