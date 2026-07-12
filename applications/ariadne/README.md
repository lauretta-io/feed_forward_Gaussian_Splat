# ARIADNE

ARIADNE is an isolated, CPU-testable application package for vision-first distributed UAV
autonomy. The current implementation covers Module 00 (bootstrap) and Module 01 (common time,
frame, pose, calibration, and transform types) from the
[build specification](../../documentation/ARIADNE_CODEX_BUILD_SPEC.md).

## Development setup

```bash
cd applications/ariadne
python3.12 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest
```

No command imports model runtimes, initializes CUDA, or downloads artifacts.

## Commands

```bash
ariadne validate-config --config configs/wingman/default.yaml
ariadne wingman run --config configs/wingman/default.yaml
ariadne intelligence run --config configs/intelligence/default.yaml
ariadne simulate --scenario configs/simulation/two_node.yaml
ariadne benchmark --suite smoke
```

Without installing the package, set `PYTHONPATH=src` and invoke `python -m ariadne`.
Runtime artifacts are placed beneath the configured `output_dir`, never in package source.

## Module documentation

- [Architecture](docs/architecture.md)
- [Coordinate frames](docs/coordinate_frames.md)
- [Dataset evaluation](docs/datasets.md)
- [Common types](src/ariadne/common/README.md)

## Dataset evaluation

```bash
python scripts/download_datasets.py
python scripts/run_dataset_sequence.py --wandb-mode offline
```

Use `--wandb-mode online --wandb-project <project>` to publish scalar metrics and JSON report
artifacts. Raw datasets are never uploaded to W&B.
