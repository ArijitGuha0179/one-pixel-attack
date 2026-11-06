"""Train from scratch with a fixed training/validation split; test remains held out."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from .data import cifar10
from .models import MEAN, STD, build_model
from .utils import seed_everything, write_json


def run_epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    correct = total = 0
    loss_sum = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(optimizer is not None):
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits, y)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total += len(y)
        correct += int((logits.argmax(1) == y).sum())
        loss_sum += float(loss.detach()) * len(y)
    return {"loss": loss_sum / total, "accuracy": correct / total, "samples": total}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train.json")
    parser.add_argument("--data", default="data")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if cfg["epochs"] < 1 or cfg["batch_size"] < 1 or cfg["learning_rate"] <= 0:
        parser.error("epochs, batch_size and learning_rate must be positive")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    seed_everything(cfg["seed"])
    dataset = cifar10(args.data, train=True, download=args.download, tensor=True)
    generator = torch.Generator().manual_seed(cfg["seed"])
    train, validation = random_split(dataset, [45000, 5000], generator=generator)
    write_json(
        output / "split.json",
        {"train": train.indices, "validation": validation.indices},
    )
    train_loader = DataLoader(
        train,
        batch_size=cfg["batch_size"],
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    val_loader = DataLoader(validation, batch_size=cfg["batch_size"], num_workers=0)
    model = build_model(cfg["model"]).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    history, best = [], -1.0
    write_json(output / "config.json", cfg)
    for epoch in range(cfg["epochs"]):
        row = {
            "epoch": epoch + 1,
            "train": run_epoch(model, train_loader, args.device, optimizer),
            "validation": run_epoch(model, val_loader, args.device),
        }
        history.append(row)
        write_json(output / "history.json", history)
        print(json.dumps(row), flush=True)
        if row["validation"]["accuracy"] > best:
            best = row["validation"]["accuracy"]
            torch.save(
                {
                    "model": cfg["model"],
                    "mean": MEAN,
                    "std": STD,
                    "state_dict": model.state_dict(),
                    "config": cfg,
                    "epoch": epoch + 1,
                    "validation_accuracy": best,
                },
                output / "best.pt",
            )
    # Test once, only after selecting the checkpoint by validation accuracy.
    saved = torch.load(output / "best.pt", map_location=args.device, weights_only=True)
    model.load_state_dict(saved["state_dict"])
    test = cifar10(args.data, download=args.download, tensor=True)
    metrics = run_epoch(
        model,
        DataLoader(test, batch_size=cfg["batch_size"], num_workers=0),
        args.device,
    )
    write_json(output / "test.json", metrics)
    print(json.dumps({"test": metrics}))


if __name__ == "__main__":
    main()
