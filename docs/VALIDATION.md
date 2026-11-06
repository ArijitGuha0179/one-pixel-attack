# Validation record

Validation performed on 6 November 2026, Windows, Python 3.12.14, CPU.
Exact relevant package versions are recorded in `requirements-validated.txt`.

## Executed successfully

- 18 offline unit/integration tests (`python -m unittest discover -s tests -v`).
- Exact notebook/refactor comparisons under the same synthetic predictor, seed,
  targeted/untargeted objectives and all three stopping rules.
- Donor selection, greedy selection, clipping, rounding, independent image copies,
  deterministic search, success retention and query accounting tests.
- PyTorch normalization, inference without gradients, and batched prediction tests.
- CIFAR-sized forward passes through both notebook-compatible architectures
  (random weights; no pretrained accuracy claim).
- A small-CNN optimizer step and checkpoint save/load round trip on synthetic data.
- End-to-end evaluation with synthetic images and a known PyTorch model, including
  JSONL, manifest, summary, PNG and NPZ output checks and overwrite prevention.
- The standalone synthetic demonstration produced an actual one-pixel attack in
  320 candidate queries. Its saved output is in `results/synthetic/`; it is not a
  CIFAR-10 result. Its plot was visually inspected.
- Python package wheel build, CLI help, syntax/static checks and ZIP integrity checks.
- Original notebook retained byte-for-byte, verified against its SHA-256.

## Not executed or independently established

- The full CIFAR-10 ResNet-18 and VGG-16 attack experiments were not rerun.
- Full classifier training and its final test accuracy were not measured.
- Pretrained weight compatibility was not verified by downloading the checkpoints;
  adapter code and URLs are preserved from the notebook, with architecture forward
  checks performed locally.
- GPU execution and the GitHub-hosted CI workflow were not run in this session.
- No measured DE-versus-random benchmark or targeted CIFAR-10 benchmark is included.
- The referenced `/mnt/data/one-pixel-attack.pdf` was unavailable. The public author
  manuscript was used; equivalence to that exact uploaded PDF cannot be verified.

Saved notebook percentages have been checked for arithmetic consistency. They
remain historical evidence, not independent replication. Complete query logs and
checkpoints would be needed to independently verify their reported means.
