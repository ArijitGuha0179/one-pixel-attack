"""A deterministic PyTorch smoke demo, explicitly NOT a CIFAR-10 benchmark.

Run from the repository root: python examples/offline_demo.py --output runs/demo
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from one_pixel_attack import one_pixel_attack
from one_pixel_attack.predictor import CNNPredictor
from one_pixel_attack.utils import write_json
from one_pixel_attack.visualize import save_example


class PixelSensitiveModel(torch.nn.Module):
    def forward(self, images):
        value = images[:, 0, 3, 4]
        return torch.stack([1 - value, value], dim=1) * 8


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    image = np.full((8, 8, 3), 40, dtype=np.uint8)
    predict = CNNPredictor(PixelSensitiveModel(), [0, 0, 0], [1, 1, 1], batch_size=16)
    result = one_pixel_attack(
        image, 0, predict, population_size=64, generations=20, seed=42
    )
    save_example(
        image, result, output / "synthetic-example.png", ["class 0", "class 1"]
    )
    record = {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in result.items()
        if key != "adversarial"
    }
    record["evidence_type"] = "synthetic functionality demo; not a dataset benchmark"
    write_json(output / "synthetic-result.json", record)
    print(record["evidence_type"])
    print(f"success={result['success']}, queries={result['attack_queries']}")


if __name__ == "__main__":
    main()
