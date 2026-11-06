import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from one_pixel_attack import one_pixel_attack
from one_pixel_attack.models import MEAN, STD, build_model, load_model
from one_pixel_attack.predictor import CNNPredictor
from one_pixel_attack.train import run_epoch


class TorchTests(unittest.TestCase):
    def test_predictor_normalization_and_batching(self):
        class Inspect(torch.nn.Module):
            def forward(self, x):
                self_outer.assertFalse(torch.is_grad_enabled())
                self_outer.assertTrue(torch.allclose(x, torch.ones_like(x)))
                return torch.stack([x[:, 0, 0, 0], -x[:, 0, 0, 0]], dim=1)

        self_outer = self
        predict = CNNPredictor(Inspect(), [0.5] * 3, [0.5] * 3, batch_size=2)
        probs = predict(np.full((5, 2, 2, 3), 255, dtype=np.uint8))
        self.assertEqual(probs.shape, (5, 2))
        np.testing.assert_allclose(probs.sum(1), 1)

    def test_torch_attack_integration(self):
        class PixelModel(torch.nn.Module):
            def forward(self, x):
                value = x[:, 0, 0, 0]
                return torch.stack([1 - value, value], dim=1) * 8

        predictor = CNNPredictor(PixelModel(), [0] * 3, [1] * 3, batch_size=3)
        result = one_pixel_attack(
            np.zeros((1, 1, 3), dtype=np.uint8),
            0,
            predictor,
            population_size=12,
            generations=3,
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["changed_pixels"], 1)

    def test_training_step_and_checkpoint_roundtrip(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)
        model = build_model("small_cnn")
        before = model[0].weight.detach().clone()
        dataset = torch.utils.data.TensorDataset(
            torch.rand(8, 3, 32, 32), torch.arange(8) % 10
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)
        metrics = run_epoch(
            model, loader, "cpu", torch.optim.Adam(model.parameters(), lr=0.001)
        )
        self.assertEqual(metrics["samples"], 8)
        self.assertFalse(torch.equal(before, model[0].weight))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.pt"
            torch.save(
                {
                    "model": "small_cnn",
                    "mean": MEAN,
                    "std": STD,
                    "state_dict": model.state_dict(),
                },
                path,
            )
            loaded = load_model("small_cnn", path)
            model.eval()
            x = dataset.tensors[0]
            torch.testing.assert_close(model(x), loaded(x))

    def test_pretrained_architecture_shapes_without_downloads(self):
        torch.set_num_threads(1)
        for name in ["resnet18", "vgg16"]:
            model = build_model(name).eval()
            with torch.inference_mode():
                self.assertEqual(model(torch.zeros(1, 3, 32, 32)).shape, (1, 10))


if __name__ == "__main__":
    unittest.main()
