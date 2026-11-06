"""Save real attack examples and search traces; works without a display."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_example(image, result, path, names=None):
    names = names or [str(i) for i in range(len(result["clean_probabilities"]))]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
    axes[0].imshow(image)
    axes[0].set_title("Original: " + names[result["clean_prediction"]])
    axes[1].imshow(result["adversarial"])
    axes[1].set_title("Adversarial: " + names[result["adversarial_prediction"]])
    axes[2].imshow(
        np.any(image != result["adversarial"], axis=2), cmap="gray", vmin=0, vmax=1
    )
    axes[2].set_title("Changed pixel mask")
    for ax in axes[:3]:
        ax.axis("off")
    axes[3].plot(result["fitness_history"], color="#007d86")
    axes[3].set(
        xlabel="Generation (0 = initialization)",
        ylabel="Best objective",
        title="Search history",
    )
    fig.suptitle(
        f"Success: {result['success']} | Candidate queries: {result['attack_queries']:,}"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
