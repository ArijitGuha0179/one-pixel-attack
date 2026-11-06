"""Notebook attack with extracted DE operations and optional random baseline.

Each evaluated image counts as one query, including duplicate candidates.
The legacy 'paper' stop rule means notebook thresholds (0.90 / 0.05),
not an assertion of exact replication of every experiment in the paper.
"""

import numpy as np

from .differential_evolution import mutate, select
from .perturbation import apply_candidates


def one_pixel_attack(
    image,
    true_label,
    predict,
    target_label=None,
    population_size=400,
    generations=100,
    mutation_factor=0.5,
    stop_rule="success",
    seed=42,
    method="de",
):

    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be a uint8 RGB array.")

    if method not in {"de", "random"}:
        raise ValueError("method must be de or random")
    if not isinstance(population_size, (int, np.integer)) or not isinstance(
        generations, (int, np.integer)
    ):
        raise ValueError("population_size and generations must be integers")
    if not np.isfinite(mutation_factor):
        raise ValueError("mutation_factor must be finite")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("image dimensions must be positive")
    if population_size < 4:
        raise ValueError("population_size must be at least 4.")

    if generations < 0 or mutation_factor <= 0:
        raise ValueError("Invalid generation count or mutation factor.")

    if stop_rule not in {"paper", "success", "budget"}:
        raise ValueError("Unknown stop_rule.")

    rng = np.random.default_rng(seed)
    height, width, _ = image.shape

    upper = np.array(
        [height - 1, width - 1, 255, 255, 255],
        dtype=float,
    )

    # Check the original image before attempting an attack.
    clean_batch = np.asarray(predict(image[None]))
    if (
        clean_batch.ndim != 2
        or clean_batch.shape[0] != 1
        or clean_batch.shape[1] < 2
        or not np.isfinite(clean_batch).all()
        or np.any(clean_batch < 0)
        or not np.allclose(clean_batch.sum(axis=1), 1, atol=1e-5)
    ):
        raise ValueError("predict must return valid class probabilities")
    clean_probabilities = clean_batch[0]
    number_of_classes = len(clean_probabilities)

    if not 0 <= true_label < number_of_classes:
        raise ValueError("true_label is outside the model's class range.")

    if target_label is not None:
        if not 0 <= target_label < number_of_classes:
            raise ValueError("target_label is outside the class range.")
        if target_label == true_label:
            raise ValueError("Target must differ from the true label.")

    clean_prediction = int(clean_probabilities.argmax())

    if clean_prediction != true_label:
        raise ValueError(
            "The original image is already misclassified. "
            "Exclude it from attack-success evaluation."
        )

    # 1. INITIALIZATION
    population = np.empty((population_size, 5), dtype=float)

    population[:, 0] = rng.uniform(0, height - 1, population_size)
    population[:, 1] = rng.uniform(0, width - 1, population_size)
    # Initially we use a Guassian Dist. to pick the initial RGB values
    population[:, 2:] = rng.normal(loc=128, scale=127, size=(population_size, 3))
    # clip RGB values b/w 0-255
    population = np.clip(population, 0, upper)

    attack_queries = 0
    best_seen = None
    best_success = None
    history = []

    # 2. EVALUATION
    def evaluate(candidates):
        nonlocal attack_queries, best_seen, best_success

        modified = apply_candidates(image, candidates)
        # Query the CNN on modified img
        probabilities = np.asarray(predict(modified))

        if (
            probabilities.shape != (len(candidates), number_of_classes)
            or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0)
            or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)
        ):
            raise ValueError("predict must return valid class probabilities.")

        # One evaluated image = one query, regardless of batching.
        attack_queries += len(candidates)

        predictions = probabilities.argmax(axis=1)

        if target_label is None:
            # Untargeted: minimize the true-class probability
            fitness = probabilities[:, true_label]
            successful = predictions != true_label
            threshold_reached = probabilities[:, true_label] < 0.05
        else:
            # Targeted: minimize the negative target probability
            fitness = -probabilities[:, target_label]
            successful = predictions == target_label
            threshold_reached = probabilities[:, target_label] > 0.90

        # Preserve successful examples even if a different candidate
        # later has better fitness but does not satisfy the class criterion.
        for i in range(len(candidates)):
            record = {
                "fitness": float(fitness[i]),
                "candidate": candidates[i].copy(),
                "probabilities": probabilities[i].copy(),
            }

            if best_seen is None or record["fitness"] < best_seen["fitness"]:
                best_seen = record

            if successful[i]:
                if best_success is None or record["fitness"] < best_success["fitness"]:
                    best_success = record
        # stopping rule
        if stop_rule == "success":
            stop = bool(successful.any())
        elif stop_rule == "paper":
            stop = bool(threshold_reached.any())
        else:
            stop = False

        return fitness, stop

    fitness, should_stop = evaluate(population)
    history.append(best_seen["fitness"])
    completed_generations = 0

    for generation in range(generations):
        if should_stop:
            break

        # 3. DIFFERENTIAL MUTATION
        if method == "de":
            trials = mutate(population, upper, mutation_factor, rng)
        else:
            # Independent restarts with the notebook initialization distribution.
            trials = np.empty_like(population)
            trials[:, 0] = rng.uniform(0, height - 1, population_size)
            trials[:, 1] = rng.uniform(0, width - 1, population_size)
            trials[:, 2:] = rng.normal(128, 127, (population_size, 3))
            trials = np.clip(trials, 0, upper)

        # No crossover: evaluate the entire mutant vector.
        trial_fitness, should_stop = evaluate(trials)

        #  4. PARENT–CHILD SELECTION
        select(population, fitness, trials, trial_fitness)

        completed_generations = generation + 1
        history.append(best_seen["fitness"])

    chosen = best_success if best_success is not None else best_seen

    candidate = np.rint(np.clip(chosen["candidate"], 0, upper)).astype(int)
    # creates the final attacked image
    adversarial = apply_candidates(image, candidate[None])[0]

    changed_pixels = int(np.any(adversarial != image, axis=2).sum())
    assert changed_pixels <= 1

    row, column = candidate[:2]
    difference = np.abs(adversarial.astype(float) - image.astype(float))

    return {
        "success": best_success is not None,
        "adversarial": adversarial,
        "candidate": candidate,
        "clean_prediction": clean_prediction,
        "adversarial_prediction": int(chosen["probabilities"].argmax()),
        "clean_probabilities": clean_probabilities,
        "adversarial_probabilities": chosen["probabilities"],
        "attack_queries": attack_queries,
        "total_queries": attack_queries + 1,
        "generations": completed_generations,
        "changed_pixels": changed_pixels,
        "pixel_mae": float(difference[row, column].mean()),
        "fitness_history": history,
    }
