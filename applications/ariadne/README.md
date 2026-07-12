# ARIADNE

ARIADNE is an isolated, CPU-testable application package for vision-first distributed UAV
autonomy. It includes the bootstrap and common types plus deterministic reference implementations
for synchronized replay, model interfaces, VIO evaluation, visual features, temporal static
filtering, cross-agent association, and robust translation-graph optimization from the
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
scripts/replicate_ignored_assets.sh --skip-pull --wandb-mode offline
```

Use `--wandb-mode online --wandb-project <project>` to publish scalar metrics and JSON report
artifacts. Raw datasets are never uploaded to W&B.

## Phase 1 model benchmark

Steps 1-5 of the model work order are available as an integrated deterministic benchmark:

```bash
ariadne benchmark --suite phase1 \
  --output outputs/ariadne/phase1/benchmark.json \
  --wandb-mode offline
```

The suite exercises synchronized camera/IMU replay, interchangeable VIO backends, separate
geometric and semantic features, temporal static filtering, cross-agent association, and robust
incremental pose optimization. The included NumPy implementations are reference backends for
interfaces, metrics, and regression testing; production ORB-SLAM3, DPVO, DINO, and GTSAM adapters
must be evaluated through the same contracts.

See [Phase 1 models](docs/phase1_models.md) for metrics and adapter boundaries.
