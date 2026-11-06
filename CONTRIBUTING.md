# Contributing

Install the package with `python -m pip install -r requirements-dev.txt` and run
`python -m unittest discover -s tests -v` before proposing a change. Tests must run
offline: no datasets or pretrained checkpoints should be downloaded by tests.

For algorithm changes, extend coverage of query accounting, one-pixel support,
targeted/untargeted success, and reproducibility. Preserve the original notebook
as an immutable source artifact. Explain deliberate departures from it in
`docs/METHOD.md`; do not silently weaken the notebook-parity test.

For benchmark claims, include configuration, subset indices, model hash, library
versions, per-image records and denominator definitions. Keep synthetic tests,
saved notebook outputs, and newly reproduced experiments clearly distinguished.
Do not commit large model checkpoints or datasets.
