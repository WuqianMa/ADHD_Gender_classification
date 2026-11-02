"""Regression checks for split workflows using synthetic, non-clinical data.

Run with: python -m unittest discover -s tests -v
"""

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from scripts.common import ADHD_FEATURES, ID_COLUMN
from scripts.data import attach_targets, find_table, load_table, read_table
from scripts.models import SignificantCorrelationSelector
from scripts.predict import predict
from scripts.train_ml import train


def make_data(root):
    rng = np.random.default_rng(8828)
    (root / "TRAIN_NEW").mkdir(parents=True)
    (root / "TEST").mkdir()
    labels = pd.DataFrame({ID_COLUMN: [f"sub-{i:03d}" for i in range(80)],
                           "ADHD_Outcome": np.tile([0, 0, 1, 1], 20),
                           "Sex_F": np.tile([0, 1, 0, 1], 20)})
    labels.to_csv(root / "TRAIN_NEW/TRAINING_SOLUTIONS.csv", index=False)
    for split, ids in [("TRAIN_NEW", labels[ID_COLUMN].tolist()), ("TEST", [f"test-{i}" for i in range(8)])]:
        prefix = "TRAIN" if split == "TRAIN_NEW" else "TEST"
        q = pd.DataFrame(rng.normal(size=(len(ids), len(ADHD_FEATURES))), columns=ADHD_FEATURES)
        if prefix == "TRAIN":
            q += labels["ADHD_Outcome"].to_numpy()[:, None] * 2
        q.loc[0, ADHD_FEATURES[0]] = np.nan
        q["MRI_Track_Age_at_Scan"] = rng.normal(10, 1, len(ids))
        q.insert(0, ID_COLUMN, ids)
        q.to_csv(root / split / f"{prefix}_QUANTITATIVE_METADATA.csv", index=False)
        conn = pd.DataFrame({f"{i}throw_{j}thcolumn": rng.normal(0, 0.01, len(ids)) +
                             (labels["Sex_F"].to_numpy() * 0.2 if prefix == "TRAIN" else 0)
                             for i in range(12) for j in range(i + 1, 12)})
        conn.insert(0, ID_COLUMN, ids)
        if prefix == "TEST":
            conn = conn.iloc[::-1]
        conn.to_csv(root / split / f"{prefix}_FUNCTIONAL_CONNECTOME_MATRICES.csv", index=False)
    return labels


class DataTests(unittest.TestCase):
    def test_join_uses_ids_and_preserves_feature_order(self):
        features = pd.DataFrame({ID_COLUMN: ["b", "a"], "feature": [1, 2]})
        targets = pd.DataFrame({ID_COLUMN: ["a", "b"], "Sex_F": [0, 1]})
        result = attach_targets(features, targets)
        self.assertEqual(result["Sex_F"].tolist(), [1, 0])
        with self.assertRaises(ValueError):
            attach_targets(features, targets.iloc[:1])

    def test_tsv_and_excel_and_duplicate_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frame = pd.DataFrame({ID_COLUMN: ["001", "002"], "value": [1, 2]})
            frame.to_csv(root / "table.tsv", sep="\t", index=False)
            frame.to_excel(root / "table.xlsx", index=False)
            self.assertEqual(read_table(root / "table.tsv")[ID_COLUMN].tolist(), ["001", "002"])
            self.assertEqual(read_table(root / "table.xlsx")[ID_COLUMN].tolist(), ["001", "002"])
            pd.concat([frame, frame]).to_csv(root / "duplicates.csv", index=False)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                read_table(root / "duplicates.csv")

    def test_revised_release_preferred_but_ambiguity_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for folder in ["TRAIN", "train_new_tsv"]:
                (root / folder).mkdir()
                (root / folder / "TRAIN_QUANTITATIVE.tsv").touch()
            self.assertEqual(find_table(root, "train", "QUANTITATIVE").parent.name, "train_new_tsv")
            (root / "train_new_tsv/TRAIN_QUANTITATIVE.csv").touch()
            with self.assertRaisesRegex(ValueError, "Ambiguous"):
                find_table(root, "train", "QUANTITATIVE")

    def test_selector_ignores_constants_and_uses_only_fit_labels(self):
        y = np.tile([0, 1], 20)
        X = pd.DataFrame({"signal": y, "constant": np.ones(40), "missing": np.nan})
        selector = SignificantCorrelationSelector(top_n=1).fit(X, y)
        self.assertEqual(selector.features_, ["signal"])
        self.assertEqual(selector.transform(X.iloc[:3]).shape, (3, 1))
        with self.assertRaisesRegex(ValueError, "No significant"):
            SignificantCorrelationSelector().fit(X[["constant", "missing"]], y)


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.data = cls.root / "data"
        make_data(cls.data)
        cls.models = cls.root / "models"
        train(argparse.Namespace(data_dir=cls.data, output_dir=cls.models,
                                 seed=8828, target="both", adhd_model="logistic", top_n=20))

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_saved_models_and_metrics_are_complete(self):
        for target in ["adhd", "sex"]:
            artifact = joblib.load(self.models / f"{target}_model.joblib")
            self.assertTrue(artifact["features"])
            self.assertIn("imputer", artifact["pipeline"].named_steps)
            self.assertNotIn("ADHD_Outcome", artifact["features"])
            self.assertNotIn("Sex_F", artifact["features"])
            metrics = json.loads((self.models / f"{target}_metrics.json").read_text())
            self.assertEqual(metrics["train_rows"] + metrics["validation_rows"], 80)

    def test_prediction_joins_shuffled_ids_and_never_fits(self):
        from sklearn.pipeline import Pipeline
        conn = load_table(self.data, "test", "CONNECTOME")
        sample = self.root / "sample_submission.csv"
        conn[[ID_COLUMN]].to_csv(sample, index=False)
        args = argparse.Namespace(data_dir=self.data, model_dir=self.models, sex_model="ml",
                                  device="cpu", batch_size=4, sample_submission=sample,
                                  output=self.root / "submission.csv")
        with patch.object(Pipeline, "fit", side_effect=AssertionError("Inference must not fit")):
            predict(args)
        result = pd.read_csv(args.output)
        self.assertEqual(result[ID_COLUMN].tolist(), conn[ID_COLUMN].tolist())
        sex = joblib.load(self.models / "sex_model.joblib")
        expected = sex["pipeline"].predict(conn[sex["features"]])
        np.testing.assert_array_equal(result["Sex_F"], expected)
        self.assertFalse(result.isna().any().any())

    def test_cpu_entry_points_do_not_import_gpu_packages(self):
        code = """
import importlib.abc
import sys
class BlockGPU(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torch_geometric'}:
            raise AssertionError('CPU workflow tried to import ' + fullname)
sys.meta_path.insert(0, BlockGPU())
import scripts.eda
import scripts.train_ml
import scripts.predict
import scripts.experiments
from scripts.models import adhd_pipeline, sex_pipeline
adhd_pipeline()
sex_pipeline()
"""
        subprocess.run([sys.executable, "-c", code], check=True, cwd=Path(__file__).resolve().parents[1])

    @unittest.skipUnless(importlib.util.find_spec("torch_geometric"), "Optional GNN dependencies are not installed")
    def test_graph_column_order_and_single_subject_batch(self):
        import torch
        from torch_geometric.loader import DataLoader
        from scripts.gnn import BrainGNN, ConnectomeDataset
        conn = load_table(self.data, "test", "CONNECTOME")
        original = ConnectomeDataset(conn, num_nodes=12)[0]
        shuffled = ConnectomeDataset(conn[conn.columns[::-1]], num_nodes=12)[0]
        torch.testing.assert_close(original.edge_index, shuffled.edge_index)
        torch.testing.assert_close(original.edge_attr, shuffled.edge_attr)
        model = BrainGNN(12, hidden_channels=4).eval()
        batch = next(iter(DataLoader([original], batch_size=1)))
        self.assertEqual(model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).shape, (1,))
        with self.assertRaisesRegex(ValueError, "Expected"):
            ConnectomeDataset(conn.iloc[:, :-1], num_nodes=12)


if __name__ == "__main__":
    unittest.main()
