#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
BRANCH="${ARIADNE_BRANCH:-$(git -C "$ROOT" branch --show-current)}"
PYTHON="${ARIADNE_PYTHON:-$ROOT/.venv/bin/python}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_PROJECT="${WANDB_PROJECT:-gaussiansplat_test}"
WANDB_GROUP="${WANDB_GROUP:-ariadne-replication}"
PULL=1
DOWNLOAD=1
INSTALL=1
RUN=1

usage() {
  cat <<'EOF'
Usage: replicate_ignored_assets.sh [options]

Recreate ignored ARIADNE datasets and evaluation outputs in another clone.

Options:
  --branch NAME          Branch to pull from origin (default: current branch)
  --python PATH          Python 3.12 interpreter or virtualenv Python
  --wandb-mode MODE      disabled, offline, or online (default: offline)
  --wandb-project NAME   W&B project (default: gaussiansplat_test)
  --wandb-group NAME     W&B run group (default: ariadne-replication)
  --skip-pull            Do not update the clone from origin
  --skip-download        Do not download or verify dataset files
  --skip-install         Do not install ARIADNE evaluation dependencies
  --skip-run             Do not regenerate outputs and W&B runs
  -h, --help             Show this help

For online W&B logging, provide WANDB_API_KEY through the environment or an
ignored .env file. This script never copies credentials between clones.
EOF
}

while (($#)); do
  case "$1" in
    --branch) BRANCH="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    --wandb-mode) WANDB_MODE="$2"; shift 2 ;;
    --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
    --wandb-group) WANDB_GROUP="$2"; shift 2 ;;
    --skip-pull) PULL=0; shift ;;
    --skip-download) DOWNLOAD=0; shift ;;
    --skip-install) INSTALL=0; shift ;;
    --skip-run) RUN=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$WANDB_MODE" in
  disabled|offline|online) ;;
  *) echo "invalid --wandb-mode: $WANDB_MODE" >&2; exit 2 ;;
esac

command -v git >/dev/null || { echo "git is required" >&2; exit 2; }
command -v 7z >/dev/null || { echo "7z is required (p7zip-full)" >&2; exit 2; }
git -C "$ROOT" rev-parse --show-toplevel >/dev/null

if ((PULL)); then
  if [[ -n "$(git -C "$ROOT" status --porcelain --untracked-files=no)" ]]; then
    echo "tracked changes are present; commit or stash them before pulling" >&2
    exit 2
  fi
  git -C "$ROOT" pull --ff-only origin "$BRANCH"
fi

if [[ ! -x "$PYTHON" ]]; then
  if command -v python3.12 >/dev/null; then
    python3.12 -m venv "$ROOT/.venv"
    PYTHON="$ROOT/.venv/bin/python"
  else
    echo "Python 3.12 is required; set ARIADNE_PYTHON or --python" >&2
    exit 2
  fi
fi

if ((INSTALL)); then
  "$PYTHON" -m pip install -e "$ROOT/applications/ariadne[evaluation]"
fi

if ((DOWNLOAD)); then
  "$PYTHON" "$ROOT/applications/ariadne/scripts/download_datasets.py"
fi

D2_ARCHIVE="$ROOT/datasets/ariadne/d2slam/archives/tum_corr.7z"
D2_OUTPUT="$ROOT/datasets/ariadne/d2slam/extracted"
if [[ ! -f "$D2_ARCHIVE" ]]; then
  echo "missing D2SLAM archive: $D2_ARCHIVE" >&2
  exit 2
fi
mkdir -p "$D2_OUTPUT"
EXISTING_D2_BAGS=0
if [[ -d "$D2_OUTPUT/tum_corr" ]]; then
  EXISTING_D2_BAGS="$(find "$D2_OUTPUT/tum_corr" -maxdepth 1 -name '*.bag' -type f | wc -l)"
fi
if [[ "$EXISTING_D2_BAGS" -ne 5 ]]; then
  7z x -y "-o$D2_OUTPUT" "$D2_ARCHIVE"
fi

if [[ "$(find "$D2_OUTPUT/tum_corr" -maxdepth 1 -name '*.bag' -type f | wc -l)" -ne 5 ]]; then
  echo "D2SLAM transformation did not produce five ROS bags" >&2
  exit 2
fi

if ((RUN)); then
  if [[ -f "$ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$ROOT/.env"
    set +a
  fi
  if [[ "$WANDB_MODE" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
    echo "WANDB_API_KEY is required for --wandb-mode online" >&2
    exit 2
  fi
  export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$ROOT/.wandb-cache}"
  "$PYTHON" "$ROOT/applications/ariadne/scripts/run_dataset_sequence.py" \
    --wandb-mode "$WANDB_MODE" \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-group "$WANDB_GROUP"
fi

echo "ARIADNE ignored assets replicated under: $ROOT/datasets/ariadne"
echo "Evaluation outputs: $ROOT/outputs/ariadne/dataset_sequence"
