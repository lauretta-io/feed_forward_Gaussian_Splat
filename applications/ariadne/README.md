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
- [Real VIO backends](docs/real_vio.md)
- [Module implementation plans](docs/implementation_plans.md)
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

The next reference gate carries a confirmed static observation from image preprocessing through
saliency, region formation, the local object store, compressed mesh uplink, and Intelligence-node
ingest. Runtime saliency defaults to the full official DUTS-trained U²-Net checkpoint; download and
verify it once from the repository root with:

```bash
.venv/bin/python applications/ariadne/scripts/download_u2net.py
```

The checkpoint is stored in the ignored `.cache/ariadne/models/u2net/u2net.pth` path. The
deterministic gradient/contrast backend remains available as the explicit `gradient_contrast`
fallback for offline contract tests.

```bash
ariadne benchmark --suite exchange \
  --output outputs/ariadne/exchange/benchmark.json \
  --wandb-mode offline
```

The global reference gate fuses two registered observations into a versioned Gaussian scene,
generates and bounds a correction delta, and exposes the result through unified context:

```bash
ariadne benchmark --suite global-scene \
  --output outputs/ariadne/global-scene/benchmark.json \
  --wandb-mode offline
```

Planning, health, 60-second network partition recovery, registry, trust, and deployment gates run
independently or as part of the composed reference system:

```bash
ariadne benchmark --suite operations \
  --output outputs/ariadne/operations/benchmark.json
ariadne benchmark --suite end-to-end \
  --output outputs/ariadne/end-to-end/benchmark.json
```

The operations gate now executes the ARIADNE-to-SKYLA boundary rather than presenting it only as
an architecture proposal. It creates and signs a versioned global-context hand-off, applies
mission priority and no-fly constraints, excludes unhealthy, low-link, or low-battery Wingmen, and
emits idempotent route requests that still require local flight-safety validation.

## Capability status website

From the repository root, regenerate the self-contained local status website from the current
JSON evaluation reports:

```bash
.venv/bin/python scripts/build_ariadne_status_site.py
```

The generated page is `reports/ariadne_status.html`. It summarizes validated capabilities,
reference components, dataset readiness, real VIO results, source paths, and known gaps. The page
also embeds one self-contained 20-frame TUM VI `corridor1` segment selected at ORB-SLAM3
initialization. Those frames are run through the real preprocessing and feature implementations,
the full pretrained U²-Net saliency model, region clustering, and production ORB-SLAM3 poses matched within
60 ms. The 20 Hz source segment spans 0.95 seconds and is replayed at 5 fps for inspection; stages
without meaningful camera output show saved benchmark metrics instead of decorative video.
