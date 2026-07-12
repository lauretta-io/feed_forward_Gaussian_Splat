# Dataset Evaluation

ARIADNE uses a representative corpus because the complete upstream releases require more storage
than the current workspace can safely provide. MILUV, D2SLAM, QDrone, and S3E together exceed 635
GB before extraction; the workspace had 480 GB available at selection time.

## Local corpus

| Dataset | Selected data | Purpose |
|---|---|---|
| MILUV | `default_3_random_0` | Primary three-UAV vision, IMU, UWB, and mocap replay |
| D2SLAM | aligned TUM corridor set | Five-agent stereo/IMU ROS1 bag replay |
| QDrone | complete Hugging Face release | Single-UAV IMU/UWB regression |
| S3E | v1 Playground 2 and v2 Playground 3 | Three-agent visual-inertial and network stress |

The registry at `configs/datasets/registry.yaml` records upstream URLs, licenses, expected sizes,
and publisher checksums. Dataset payloads remain under the ignored `datasets/ariadne` tree.

## Metrics

Each adapter emits a typed `DatasetEvaluation` with agent and modality inventories, message or
sample counts, duration, ground-truth coverage, warnings, and dataset-specific metrics. Visual-
inertial datasets calculate nearest camera-to-IMU timestamp errors in milliseconds. The CLI writes
the complete result to JSON and logs numeric metrics plus that JSON report as a W&B artifact.

```bash
ariadne evaluate \
  --dataset miluv \
  --path datasets/ariadne/miluv/archives/default_3_random_0.zip \
  --output outputs/ariadne/miluv.json \
  --wandb-mode online \
  --wandb-project gaussiansplat_test
```

Run the complete representative sequence with:

```bash
python applications/ariadne/scripts/run_dataset_sequence.py \
  --wandb-mode online \
  --wandb-project gaussiansplat_test
```

W&B receives metrics and reports only. It does not receive raw images, ROS bags, archives, or
credentials.

## Replicating another clone

The shell entry point below updates the current branch from `origin`, installs evaluation
dependencies, downloads and verifies every file in the representative corpus, extracts the
D2SLAM archive, and regenerates ignored evaluation outputs:

```bash
applications/ariadne/scripts/replicate_ignored_assets.sh \
  --wandb-mode online \
  --wandb-project gaussiansplat_test
```

Use `--skip-pull` when testing local uncommitted changes. The script is resumable and safe to run
again: complete files are revalidated by expected size and publisher checksum. Clone-local `.env`
credentials are loaded when present but are never copied or generated.
