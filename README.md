# One Pixel Attack

## 1. What is One Pixel Attack?

One Pixel Attack is an adversarial attack that attempts to change an image
classifier's prediction by modifying just one pixel. The replacement can change
all three RGB values at that location while leaving the rest of the image unchanged.

The attack uses differential evolution to search for a useful pixel location and
replacement color. It is a score-based black-box attack: it uses the classifier's
output probabilities without accessing its gradients. An untargeted attack aims
to produce any incorrect prediction, while a targeted attack aims to produce a
specific chosen class. Changing one pixel does not guarantee success or make the
modification invisible.

## 2. How we implemented it

We implemented the classifier inference in **PyTorch** and wrote the core
**differential evolution algorithm manually using NumPy**, without using SciPy's
optimizer or an existing attack library. The original project notebook is retained,
and its attack code is organized into reusable Python modules.

Each candidate contains five values: `(row, column, R, G, B)`. The attack follows
these steps:

1. **Initialize:** Generate a population of candidate pixel locations and RGB values.
2. **Evaluate:** Apply each candidate to a fresh copy of the original image and query
   the classifier. For untargeted attacks, minimize the true-class probability;
   for targeted attacks, maximize the chosen target-class probability.
3. **Mutate:** Generate new candidates using `a + F * (b - c)`, with three distinct
   donors excluding the current parent. No crossover is used.
4. **Select:** Keep each new candidate only if it improves on its parent.
5. **Check success:** Use the predicted class to determine success and retain the
   best successful candidate found. Stop according to the configured stopping rule
   or generation limit.

The notebook experiments use a population of **400**, up to **100 generations**,
mutation factor **0.5**, and seed **42**. Coordinates and colors are clipped and
rounded to valid values, and every output is checked to ensure that at most one
spatial pixel changed. Only initially correctly classified images are attacked.

The repository also includes a random-search baseline, training and evaluation
commands, visualizations, and automated tests. These additions are separate from
the notebook experiments reported below.

## 3. Dataset and models tested

We used **CIFAR-10**, which contains 60,000 RGB images of size 32 × 32 pixels across
10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.
Its official split contains 50,000 training images and 10,000 test images. Its small
image size makes repeated candidate evaluations practical for this project.

The notebook tests the attack on two pretrained CIFAR-10 classifiers:

| Model | Test images sampled | Initially correct images attacked |
|---|---:|---:|
| ResNet-18 | 100 | 96 |
| VGG-16 with batch normalization | 500 | 469 |

Both models are loaded using `timm` with CIFAR-10 architecture adaptations and
pretrained weights from Hugging Face. Pixel changes are applied to the original
32 × 32 images before the model-specific normalization used in the notebook.
These experiments are **untargeted attacks**; targeted attacks are supported by
the code, but no targeted CIFAR-10 benchmark results are reported here.

A separate small CNN training workflow is included for further experiments. It
has been checked with synthetic training data, but no full CIFAR-10 training
results are claimed for it.

## Results and evidence

These values come from **saved notebook outputs**, not a rerun of the refactored
repository. Their denominators and arithmetic have been checked. Checkpoints,
dataset files, and the original per-image VGG JSONL were not supplied.

| Model | Sampled images | Initially correct / attacked | Successful | Attack success rate | Accuracy before → after | Mean candidate queries |
|---|---:|---:|---:|---:|---:|---:|
| ResNet-18 | 100 | 96 | 35 | 36.46% | 96.00% → 61.00% | 26,425.0 |
| VGG-16 BN | 500 | 469 | 222 | 47.33% | 93.80% → 49.40% | 22,162.0 |

The notebook separately reports **94.98% ResNet-18 accuracy on all 10,000 test
images**. The 96.00% above is accuracy on the sampled 100 images. VGG's output
filename contains `10k`, but its attack experiment covers **500 images**, not 10,000.

See [evidence and limitations](results/README.md), [validation](docs/VALIDATION.md),
and [notebook provenance](results/notebook/provenance.json).
No accuracy or attack-success claim is made for the newly added training workflow.

## Setup

Use Python 3.10 or newer (Python 3.12 is the local validation environment).
Run commands from the repository root.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux instead:
# source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

For CPU-only PyTorch, install `torch torchvision` using the official CPU index
before installing this package:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
```

For GPU use, select an appropriate PyTorch build from
[PyTorch's installation guide](https://pytorch.org/get-started/locally/).
CUDA deterministic execution may require setting `CUBLAS_WORKSPACE_CONFIG=:4096:8`
before launching Python. Results can vary across devices and library versions.
The included [validation environment](requirements-validated.txt) records the exact
packages used here; it is an environment record, not a universal GPU lockfile.

## Run an attack

For a quick check without downloading data or weights:

```bash
python examples/offline_demo.py --output runs/offline-demo
```

This uses an intentionally vulnerable synthetic PyTorch classifier. Its outputs
demonstrate functionality and are not CIFAR-10 performance measurements.

First run downloads CIFAR-10 and the model checkpoint. Later runs can omit `--download`.
CPU is the default; pass `--device cuda` to use a compatible GPU.
Every output directory must be new, preventing experiments from being overwritten.

```bash
# Quick pipeline check: 5 images, population 12, 2 generations
python -m one_pixel_attack evaluate --config configs/smoke.json --download --output runs/smoke

# Match the notebook's ResNet experiment settings
python -m one_pixel_attack evaluate --config configs/resnet18.json --download --output runs/resnet-de

# VGG experiment settings
python -m one_pixel_attack evaluate --config configs/vgg16.json --download --output runs/vgg-de

# Same subset and maximum query budget, independent random candidates
python -m one_pixel_attack evaluate --config configs/resnet18.json --method random --output runs/resnet-random

# One seeded random non-true target per sampled image
python -m one_pixel_attack evaluate --config configs/resnet18.json --targeted --output runs/resnet-targeted
```

Full experiments can take substantial time on CPU: the maximum is 40,400 candidate
evaluations per eligible image with the default settings. A five-image smoke run
is a functionality check, not a meaningful estimate of attack success.

Each run writes:

| File | Contents |
|---|---|
| `manifest.json` | Resolved config, ordered test indices, model-state SHA-256, versions, normalization, run status |
| `records.jsonl` | Incrementally flushed per-image results, probabilities, candidates, fitness histories and query counts |
| `summary.json` | Before/after accuracy, conditional success rate, distortion and query summaries |
| `example-*.png` | First eligible examples, including failures, with changed-pixel mask and fitness plot |
| `example-*.npz` | Lossless original and adversarial arrays for those examples |

An interrupted run has no completed summary; records already written remain useful.
Automatic resume is not implemented. Use a new output directory to rerun.

## Train your own classifier

This is an **added baseline**, not the source of the pretrained notebook results.
The default trains a small CNN for 20 epochs; change `model` in the training config
to `resnet18` or `vgg16` to train those architectures from scratch.

```bash
python -m one_pixel_attack train --config configs/train.json --download --output runs/training
python -m one_pixel_attack evaluate --config configs/small_cnn.json --checkpoint runs/training/best.pt --output runs/trained-cnn-attack
```

Training uses 45,000 CIFAR-10 training images and 5,000 validation images. Checkpoint
selection uses validation accuracy; the official test split is evaluated once after
selection. There is no augmentation or learning-rate schedule in this intentionally
simple baseline. No claim of state-of-the-art accuracy is made.

## Algorithm

Each candidate is `(row, column, R, G, B)`. Coordinates and RGB values are rounded
and clipped before being applied to a fresh copy of the original uint8 image.
The constraint is at most **one spatial pixel**, allowing all three RGB channels
at that location to change. This is not a one-channel constraint or an imperceptibility guarantee.

1. Initialize coordinates uniformly and colors from a clipped normal distribution.
2. Score candidates with the frozen classifier: minimize true-class probability
   for untargeted attacks, or negative target-class probability for targeted attacks.
3. For each parent, draw three distinct donors excluding that parent and form
   `a + F * (b - c)` with `F=0.5`. Evaluate the full mutant; no crossover is used.
4. Keep a child only when its objective improves on its parent's objective.
5. Retain the best successful candidate separately from the best objective value.

Success is determined by the predicted class, not a confidence threshold alone.
`success` stops after a population batch containing a successful candidate.
`budget` uses all generations. The legacy `paper` setting retains the notebook's
0.90 targeted / 0.05 untargeted thresholds; the name does **not** imply exact paper
reproduction. See [method and deviations](docs/METHOD.md).

## Metrics and fair comparisons

- **Attack success rate:** successes divided by initially correct images attacked.
  Already misclassified examples are excluded from this denominator.
- **Accuracy before/after:** both use the same sampled images, retaining initial errors.
- **Candidate queries:** every evaluated modified image counts once, even when batched
  or duplicated. Initialization is included. Maximum: `population_size × (generations + 1)`.
- **Total attack queries:** candidate queries plus one clean-image eligibility check.
  Evaluation-wide queries also include the initial clean pass over the sampled subset.
- **Pixel MAE:** mean absolute RGB change at the selected pixel, on the 0–255 scale.
  Per-image records also contain full-image MAE, L2 and L-infinity distortion.

DE and random search use the same subset, initialization distribution, population,
maximum budget, and stopping rule. Early stopping means actual queries can differ.
Use `stop_rule: "budget"` for equal candidate counts. No random-baseline performance
numbers are asserted until you run that comparison. A targeted run samples one
non-true target per image; it does not reproduce the paper's nine-target aggregation.

## Repository map

```text
configs/                   Reproducible experiment and training settings
src/one_pixel_attack/
  attack.py                Candidate scoring, success tracking, stopping, accounting
  differential_evolution.py  Manual mutation and selection
  perturbation.py          One-pixel decoding
  models.py, predictor.py  Model adapters and batched inference
  data.py, train.py        Dataset loading and training
  evaluate.py, visualize.py  Evaluation and saved plots
tests/                     Offline correctness and PyTorch integration tests
notebooks/                 Original notebook, byte-for-byte preserved
results/notebook/          Historical evidence, logs and source hash
docs/                      Method, validation and provenance notes
.github/workflows/         CPU test workflow
```

## Resume wording

> Implemented a score-based One Pixel Attack in PyTorch with handwritten differential
> evolution; built reproducible evaluation, query accounting, visualization, and
> automated tests for targeted and untargeted attacks.

If you cite the historical experiment, say “The project notebook recorded 47.33%
attack success on 469 initially correct images from a 500-image CIFAR-10 sample
using pretrained VGG-16.” Do not call this a full-test-set result or a newly
reproduced benchmark. Describe your own contributions accurately, including tool
assistance where your course requires it.

## References and license

Jiawei Su, Danilo Vasconcellos Vargas, and Kouichi Sakurai,
[*One Pixel Attack for Fooling Deep Neural Networks*](https://arxiv.org/abs/1710.08864),
IEEE Transactions on Evolutionary Computation (2019).

The referenced local PDF was not available in this session; the public author
manuscript was consulted. The repository does not redistribute the paper,
CIFAR-10, or pretrained weights. Their respective licenses apply separately.
Project code is provided under the [MIT License](LICENSE).
See [third-party sources](docs/THIRD_PARTY.md).
