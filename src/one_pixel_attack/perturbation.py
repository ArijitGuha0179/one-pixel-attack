"""Decode (row, column, R, G, B) onto fresh uint8 image copies."""

import numpy as np


def apply_candidates(image, candidates):
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be a uint8 RGB array")
    candidates = np.asarray(candidates, dtype=float)
    if (
        candidates.ndim != 2
        or candidates.shape[1] != 5
        or not np.isfinite(candidates).all()
    ):
        raise ValueError("candidates must be a finite (N, 5) array")
    height, width, _ = image.shape
    if height == 0 or width == 0:
        raise ValueError("image dimensions must be positive")

    upper = np.array([height - 1, width - 1, 255, 255, 255])

    decoded = np.rint(np.clip(candidates, 0, upper)).astype(np.int64)

    images = np.repeat(image[None, ...], len(decoded), axis=0)

    indices = np.arange(len(decoded))

    images[indices, decoded[:, 0], decoded[:, 1], :] = decoded[:, 2:].astype(np.uint8)

    return images
