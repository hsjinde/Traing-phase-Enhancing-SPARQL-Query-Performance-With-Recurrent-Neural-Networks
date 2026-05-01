import tempfile
import unittest
from pathlib import Path

import numpy as np

from preprocessing import PosEmbedding, Transform2WordEmbedding, WordEmbedding


class PreprocessingTests(unittest.TestCase):
    def test_word_embedding_loads_valid_vectors(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            embedding_dir = Path(temp_dir)
            (embedding_dir / "glove.6B.2d.txt").write_text("hello 0.1 0.2\nbad 0.1\n", encoding="utf8")

            words, word_idx, vectors = WordEmbedding(2, embedding_dir=embedding_dir)

        self.assertIn("hello", word_idx)
        self.assertNotIn("bad", word_idx)
        self.assertEqual(np.asarray(vectors).shape, (3, 2))

    def test_pos_embedding_loads_from_explicit_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            embedding_path = Path(temp_dir) / "pos.txt"
            embedding_path.write_text("NN " + " ".join(["0.1"] * 20) + "\n", encoding="utf8")

            tags, pos_idx, vectors = PosEmbedding(embedding_path=embedding_path)

        self.assertIn("NN", pos_idx)
        self.assertEqual(np.asarray(vectors).shape, (2, 20))

    def test_transform_word_embedding_rejects_missing_index(self):
        with self.assertRaises(ValueError):
            Transform2WordEmbedding("hello", ["hello"], None)


if __name__ == "__main__":
    unittest.main()
