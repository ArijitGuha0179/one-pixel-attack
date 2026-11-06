"""Handwritten differential mutation and greedy parent-child selection.

No crossover, optimizer imports, gradients, or third-party attack code.
The caller supplies the seeded Generator so the notebook RNG order is retained.
"""

import numpy as np


def mutate(population, upper, factor, rng):
    """DE/rand/1: three distinct donors excluding each parent; clip bounds."""
    if len(population) < 4 or not np.isfinite(factor) or factor <= 0:
        raise ValueError("need at least four parents and a finite positive factor")
    trials = np.empty_like(population)
    for i in range(len(population)):
        donors = np.delete(np.arange(len(population)), i)
        a, b, c = rng.choice(donors, size=3, replace=False)
        trials[i] = population[a] + factor * (population[b] - population[c])
    return np.clip(trials, 0, upper)


def select(population, fitness, trials, trial_fitness):
    """Strict improvement wins; ties retain the parent (in-place)."""
    improved = trial_fitness < fitness
    population[improved] = trials[improved]
    fitness[improved] = trial_fitness[improved]
