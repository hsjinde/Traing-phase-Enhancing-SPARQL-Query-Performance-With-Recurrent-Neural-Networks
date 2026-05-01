import unittest

import pandas as pd

from AutotagTarget import Convert_question_mark, GetSPO, GetTarget, get_lable, get_target


class AutoTagTargetTests(unittest.TestCase):
    def test_get_target_derives_label_column_without_mutating_input(self):
        source = pd.DataFrame({"query": ["SELECT DISTINCT ?x WHERE { <s> <p> ?x . }"]})

        result = GetTarget(source)

        self.assertNotIn("lable", source.columns)
        self.assertEqual(result["lable"].iloc[0], ["A"])

    def test_convert_question_mark_uses_selected_answer_variable(self):
        df = pd.DataFrame({"query": ["SELECT DISTINCT ?city WHERE { ?country <p> ?city . }"]})

        normalized = Convert_question_mark(["?country", "P", "?city"], df, 0)

        self.assertEqual(normalized, ["?x", "P", "?ans"])

    def test_unknown_patterns_are_ignored_instead_of_crashing(self):
        self.assertEqual(get_target(["?ans", "UNKNOWN", "O"]), [])

    def test_get_lable_expands_repeated_patterns_without_mutating_input(self):
        labels = ["A", "A", "B"]

        expanded = get_lable(labels)

        self.assertEqual(labels, ["A", "A", "B"])
        self.assertEqual(expanded, ["A", "E", "B"])

    def test_get_spo_removes_trailing_periods(self):
        self.assertEqual(GetSPO("<s> <p> ?ans."), ["S", "P", "?ans"])


if __name__ == "__main__":
    unittest.main()
