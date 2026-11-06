# Method, paper relationship and notebook lineage

This project refactors the supplied notebook, not an external attack repository.
The complete original notebook is retained unchanged. The source hash is recorded
in `results/notebook/provenance.json`.

## Traceability (zero-based notebook cell indices)

| Notebook cell | Refactored module | Treatment |
|---|---|---|
| 8 | `predictor.py` | Batched uint8-to-tensor normalization; CPU-safe default and input checks |
| 11, 28 | `models.py` | Same timm architecture adaptations, checkpoint URLs and normalization |
| 13 | `perturbation.py` | Same rounding and clipping, with candidate validation |
| 15 | `attack.py`, `differential_evolution.py` | Same DE path and RNG order; extracted mutation and selection |
| 20, 25, 26, 29 | `evaluate.py`, `utils.py` | Seeded subset, eligible-image filtering, durable records and explicit denominators |
| 23 | `visualize.py` | Added search history and lossless example export |

New components are the random baseline, small CNN and training workflow, CLI,
run manifests, tests and CI. They are not attributed to the original notebook.
The notebook's code is exercised directly in a parity test using a synthetic
predictor; matching this test is not proof of identical cross-device CIFAR results.

## Paper relationship

The author manuscript is available at https://arxiv.org/html/1710.08864v6.
It describes a five-value pixel encoding, differential evolution, a population
of 400, scaling factor 0.5, no crossover and 100 iterations. This repository retains
those notebook defaults, interpreted as one initialization plus 100 generations.

The paper's displayed mutation formula contains a plus between the two donor
vectors. This implementation explicitly follows the notebook's conventional
differential mutation `a + F*(b-c)`, rather than silently claiming literal agreement
with that displayed equation.

The manuscript's Section III-C describes targeted stopping above 50% for its
Kaggle CIFAR-10 experiment and untargeted stopping below 5% for ImageNet. The
notebook's `paper` option instead uses 90% / 5%. That legacy option is preserved
for compatibility, while the repository defaults to actual prediction success.

These are distinct experimental settings, so this project is an implementation
and notebook refactor, **not an exact reproduction of the published tables**.
It uses the official CIFAR-10 test split and notebook pretrained checkpoints.
It does not recreate the paper's training or all-target evaluation protocol.

## Implementation details that affect results

- Spatial coordinates are zero-based `(row, column)`, not Cartesian `(x, y)`.
- Initialization uses continuous uniform coordinates and Gaussian RGB values;
  clipping and nearest-even rounding follow NumPy and the notebook. Edge coordinates
  consequently do not have the same initialization probability as interior ones.
- A no-op candidate is allowed; the assertion is at most one changed spatial pixel.
- Generations are synchronous: all mutants are constructed from the same parents.
- Ties preserve the parent. Donor indices are distinct and exclude the parent.
- Duplicate decoded candidates are counted again; there is no query cache.
- A whole population is queried before stopping, not one candidate at a time.
- The best successful candidate can differ from the best objective candidate in
  a multiclass model. Both are tracked; the final output prefers success.
- Target labels are sampled reproducibly from non-true classes, independently of
  the attack RNG, without consuming its draws.
- Normalization occurs inside the predictor after pixel substitution; no resizing
  or augmentation occurs between mutation and inference.
- The API rejects initially incorrect inputs, invalid targets and invalid probability outputs.
- Seeds are recorded and deterministic PyTorch operations requested. Hardware,
  package versions and mutable upstream checkpoint URLs still affect reproducibility.
  Each evaluation records the loaded model-state hash to identify actual weights.

## Scope

The attack requires class probabilities, not just a hard label. It does not access
gradients. It is a local model robustness experiment; no deployed service is queried.
It does not train an RL policy, claim imperceptibility, or establish certified robustness.
