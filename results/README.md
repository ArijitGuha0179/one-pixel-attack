# Results: keep evidence levels separate

## Historical notebook experiment

`notebook/console-output.txt` extracts saved text outputs with their source cells.
`notebook/provenance.json` records the unchanged notebook's SHA-256 and metric values.
The original notebook also contains the displayed attack visualization.

- ResNet-18: 100 sampled images, 96 correct before attack, 35 successful attacks.
  `35 / 96 = 36.4583%`; `(96 - 35) / 100 = 61%` accuracy after attack.
- VGG-16: 500 sampled images, 469 correct before attack, 222 successful attacks.
  `222 / 469 = 47.3348%`; `(469 - 222) / 500 = 49.4%` accuracy after attack.
- ResNet full-test clean accuracy printed by the notebook: 94.98%.
- Mean candidate queries printed: 26,425.0 (ResNet), 22,162.0 (VGG).

Those means cannot be independently recomputed from the supplied artifacts because
the complete query records were not attached. No new query distribution, confidence
interval, distortion average, trained-model accuracy, or random-baseline result is invented.
The notebook's original VGG output filename is misleading: the run covers 500 images.

## Validation and new experiments

See `docs/VALIDATION.md` for what actually ran in the build session. Automated tests
use synthetic data and classifiers to validate behavior; they are not a robustness
benchmark. New user-run experiments belong in `runs/` (ignored by Git).

For a publishable comparison, run DE and random search on the same image indices
and model hash across several seeds. Report sample sizes, budgets, stopping rules,
hardware, package versions, all attempted attacks, and uncertainty. Do not select
only images on which an attack succeeds.
