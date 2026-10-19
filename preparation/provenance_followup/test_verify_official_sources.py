import importlib.util
import pathlib
import tempfile
import unittest
from unittest.mock import patch

MODULE_PATH = pathlib.Path(__file__).with_name("verify_official_sources.py")
SPEC = importlib.util.spec_from_file_location("verify_official_sources", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FullSourceComparisonTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = pathlib.Path(self.directory.name)
        self.raw = self.root / "audit/data/raw"
        self.raw.mkdir(parents=True)

    def compare(self, role, filename, official_content, mirror_content):
        official = self.root / filename
        official.write_bytes(official_content)
        (self.raw / filename.replace(".txt", ".csv")).write_bytes(mirror_content)
        with patch.object(MODULE, "ROOT", self.root):
            return MODULE.compare((role, "Training_data/" + filename, len(official_content)), official)

    def test_crlf_is_only_representation_difference(self):
        result = self.compare("training_traits", "traits.csv", b"Env,Hybrid,Yield\r\nA,B,2.5\r\n", b"Env,Hybrid,Yield\nA,B,2.5\n")
        self.assertFalse(result["byte_match"])
        self.assertTrue(result["semantic_match"])
        self.assertEqual(result["different_cells_or_absent_rows"], 0)

    def test_genotype_tab_format_is_documented(self):
        result = self.compare("genotype", "geno.txt", b"<Numeric>\n<Marker>\tS1\tS2\nA\t1\tNA\n", b"Geno,S1,S2\nA,1,NA\n")
        self.assertTrue(result["semantic_match"])
        self.assertEqual(result["official_format_declaration"], "<Numeric>")

    def test_value_and_key_differences_are_not_silenced(self):
        result = self.compare("training_weather", "weather.csv", b"Env,Date,T\nA,2024-01-01,2\n", b"Env,Date,T\nB,2024-01-01,3\n")
        self.assertFalse(result["semantic_match"])
        self.assertFalse(result["ordered_keys_equal"])
        self.assertEqual(result["different_cells_or_absent_rows"], 2)

    def test_missing_final_row_fails(self):
        result = self.compare("training_traits", "traits.csv", b"Env,Hybrid\nA,B\nC,D\n", b"Env,Hybrid\nA,B\n")
        self.assertFalse(result["semantic_match"])
        self.assertEqual(result["official_rows"], 2)
        self.assertEqual(result["local_rows"], 1)


if __name__ == "__main__":
    unittest.main()
