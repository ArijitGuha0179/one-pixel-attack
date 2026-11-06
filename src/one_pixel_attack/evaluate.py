"""Run paired, seeded evaluations and write auditable records incrementally."""

import argparse
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np

from .attack import one_pixel_attack
from .data import cifar10
from .models import MEAN, STD, URLS, load_model
from .predictor import CNNPredictor
from .utils import seed_everything, state_hash, summarize, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/resnet18.json")
    parser.add_argument("--data", default="data")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--checkpoint", help="Checkpoint produced by the train command")
    parser.add_argument(
        "--output", required=True, help="New output directory; existing paths refused"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--samples", type=int)
    parser.add_argument("--method", choices=["de", "random"])
    parser.add_argument(
        "--targeted", action="store_true", help="One random non-true target per image"
    )
    args = parser.parse_args(argv)
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.samples is not None:
        cfg["samples"] = args.samples
    if args.method is not None:
        cfg["method"] = args.method
    if args.targeted:
        cfg["targeted"] = True
    if not isinstance(cfg["samples"], int) or not 1 <= cfg["samples"] <= 10000:
        parser.error("samples must be between 1 and 10000")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    seed_everything(cfg["seed"])
    dataset = cifar10(args.data, download=args.download)
    model = load_model(cfg["model"], args.checkpoint)
    model_digest = state_hash(model)
    predict = CNNPredictor(model, MEAN, STD, args.device, cfg["batch_size"])
    indices = np.random.default_rng(cfg["seed"]).choice(
        len(dataset), cfg["samples"], replace=False
    )
    # Exactly the same subset/order for DE, random, and targeted runs at this seed.
    images = dataset.data[indices]
    labels = np.asarray(dataset.targets)[indices]
    clean = predict(images)
    manifest = {
        "config": cfg,
        "arguments": vars(args),
        "indices": indices.tolist(),
        "dataset": "torchvision CIFAR10 official test split; no resize",
        "model_state_sha256": model_digest,
        "checkpoint_source": args.checkpoint or URLS.get(cfg["model"]),
        "normalization": {"mean": MEAN, "std": STD},
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {
            p: version(p)
            for p in ["torch", "torchvision", "numpy", "timm", "matplotlib"]
        },
        "status": "running",
    }
    write_json(output / "manifest.json", manifest)
    records = []
    with (output / "records.jsonl").open("x", encoding="utf-8") as handle:
        for position, (index, image, label, probs) in enumerate(
            zip(indices, images, labels, clean)
        ):
            index, label, before = int(index), int(label), int(probs.argmax())
            target = None
            if cfg.get("targeted", False):
                target = int(
                    np.random.default_rng(cfg["seed"] + index).choice(
                        [c for c in range(10) if c != label]
                    )
                )
            row = {
                "index": index,
                "label": label,
                "target": target,
                "before": before,
                "after": before,
                "attacked": before == label,
                "success": False,
                "attack_queries": 0,
                "total_queries": 0,
                "clean_probabilities": probs.tolist(),
            }
            if row["attacked"]:
                result = one_pixel_attack(
                    image,
                    label,
                    predict,
                    target_label=target,
                    seed=cfg["seed"] + index,
                    population_size=cfg["population_size"],
                    generations=cfg["generations"],
                    mutation_factor=cfg["mutation_factor"],
                    stop_rule=cfg["stop_rule"],
                    method=cfg.get("method", "de"),
                )
                row.update(
                    {
                        k: v.tolist() if isinstance(v, np.ndarray) else v
                        for k, v in result.items()
                        if k != "adversarial"
                    }
                )
                row["after"] = result["adversarial_prediction"]
                diff = result["adversarial"].astype(float) - image.astype(float)
                row.update(
                    l2=float(np.linalg.norm(diff)),
                    linf=float(np.abs(diff).max()),
                    image_mae=float(np.abs(diff).mean()),
                )
                if sum(r["attacked"] for r in records) < cfg.get("save_examples", 6):
                    from .visualize import save_example

                    save_example(
                        image, result, output / f"example-{index}.png", dataset.classes
                    )
                    np.savez_compressed(
                        output / f"example-{index}.npz",
                        original=image,
                        adversarial=result["adversarial"],
                    )
            records.append(row)
            handle.write(json.dumps(row, allow_nan=False) + "\n")
            handle.flush()
            print(
                f"{position + 1}/{len(indices)} index={index} attacked={row['attacked']} success={row['success']}",
                flush=True,
            )
    summary = summarize(records)
    write_json(output / "summary.json", summary)
    manifest["status"] = "complete"
    write_json(output / "manifest.json", manifest)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
