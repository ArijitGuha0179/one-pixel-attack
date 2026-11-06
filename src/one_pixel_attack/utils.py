import hashlib
import json
import random
from pathlib import Path

import numpy as np


def seed_everything(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def write_json(path, obj):
    Path(path).write_text(
        json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def state_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def summarize(records):
    if not records:
        raise ValueError("cannot summarize an empty evaluation")
    attacked = [r for r in records if r["attacked"]]
    success = sum(r["success"] for r in attacked)
    return {
        "images_evaluated": len(records),
        "attacks_attempted": len(attacked),
        "successful_attacks": success,
        "accuracy_before": sum(r["before"] == r["label"] for r in records)
        / len(records),
        "accuracy_after": sum(r["after"] == r["label"] for r in records) / len(records),
        "attack_success_rate": success / len(attacked) if attacked else None,
        "mean_attack_queries": float(np.mean([r["attack_queries"] for r in attacked]))
        if attacked
        else None,
        "mean_pixel_mae": float(np.mean([r["pixel_mae"] for r in attacked]))
        if attacked
        else None,
        "evaluation_queries": len(records) + sum(r["total_queries"] for r in attacked),
    }
