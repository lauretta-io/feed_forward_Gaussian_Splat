#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
BACKEND_ROOT="${ARIADNE_BACKEND_ROOT:-$ROOT/.cache/ariadne/backends}"
EXECUTABLE="$BACKEND_ROOT/ORB_SLAM3/Examples/Stereo-Inertial/stereo_inertial_euroc"

if [[ ! -x "$EXECUTABLE" ]]; then
  echo "ORB-SLAM3 backend is not built: $EXECUTABLE" >&2
  exit 2
fi
command -v docker >/dev/null || { echo "docker is required to run ORB-SLAM3" >&2; exit 2; }
exec docker run --rm \
  --mount "type=bind,source=$ROOT,target=$ROOT" \
  --workdir "$PWD" \
  --env "LD_LIBRARY_PATH=$BACKEND_ROOT/ORB_SLAM3/lib:$BACKEND_ROOT/ORB_SLAM3/Thirdparty/DBoW2/lib:$BACKEND_ROOT/ORB_SLAM3/Thirdparty/g2o/lib:/usr/local/lib" \
  ariadne-orbslam3-build:22.04 "$EXECUTABLE" "$@"
