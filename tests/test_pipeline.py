"""Offline end-to-end evaluation with a known PyTorch classifier and synthetic data."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from one_pixel_attack.evaluate import main


class KnownModel(torch.nn.Module):
    def forward(self, x):
        # Large response to one spatial location, using normalized input.
        value = x[:, 0, 0, 0]
        logits = torch.full((len(x), 10), -20.0, device=x.device)
        logits[:, 0] = -value
        logits[:, 1] = value
        return logits


class Dataset:
    data = np.zeros((5, 2, 2, 3), dtype=np.uint8)
    targets = [0, 0, 0, 0, 1]  # Last image is intentionally initially incorrect.
    classes = [str(i) for i in range(10)]

    def __len__(self):
        return len(self.data)


class PipelineTests(unittest.TestCase):
    def test_evaluate_writes_consistent_outputs_and_plots(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            config = temp / "config.json"
            config.write_text(
                json.dumps(
                    dict(
                        model="resnet18",
                        seed=42,
                        samples=5,
                        population_size=12,
                        generations=2,
                        mutation_factor=0.5,
                        stop_rule="budget",
                        method="de",
                        targeted=False,
                        batch_size=3,
                        save_examples=2,
                    )
                )
            )
            output = temp / "evaluation"
            with (
                patch("one_pixel_attack.evaluate.cifar10", return_value=Dataset()),
                patch(
                    "one_pixel_attack.evaluate.load_model", return_value=KnownModel()
                ),
            ):
                main(["--config", str(config), "--output", str(output)])
            rows = [
                json.loads(line)
                for line in (output / "records.jsonl").read_text().splitlines()
            ]
            summary = json.loads((output / "summary.json").read_text())
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(len(rows), 5)
            self.assertEqual(summary["attacks_attempted"], 4)
            self.assertEqual(summary["mean_attack_queries"], 36)
            self.assertEqual(summary["evaluation_queries"], 5 + 4 * 37)
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(len(list(output.glob("example-*.png"))), 2)
            for path in output.glob("example-*.npz"):
                with np.load(path) as arrays:
                    self.assertLessEqual(
                        np.any(arrays["original"] != arrays["adversarial"], axis=2).sum(), 1
                    )
            # Refuse to overwrite an existing experiment before loading data.
            with self.assertRaises(FileExistsError):
                main(["--config", str(config), "--output", str(output)])


if __name__ == "__main__":
    unittest.main()
