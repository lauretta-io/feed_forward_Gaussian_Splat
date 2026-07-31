#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
BACKEND_ROOT="${ARIADNE_BACKEND_ROOT:-$ROOT/.cache/ariadne/backends}"
MODE="${ARIADNE_ORBSLAM3_MODE:-stereo-inertial}"
DETERMINISTIC_RUNTIME=0
SYNC_LOCAL_MAPPING=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --deterministic-runtime)
      DETERMINISTIC_RUNTIME=1
      shift
      ;;
    --sync-local-mapping)
      SYNC_LOCAL_MAPPING=1
      shift
      ;;
    *)
      break
      ;;
  esac
done
case "$MODE" in
  stereo)
    EXECUTABLE="$BACKEND_ROOT/ORB_SLAM3/Examples/Stereo/stereo_euroc"
    ;;
  stereo-inertial)
    EXECUTABLE="$BACKEND_ROOT/ORB_SLAM3/Examples/Stereo-Inertial/stereo_inertial_euroc"
    ;;
  *)
    echo "unsupported ORB-SLAM3 mode: $MODE" >&2
    exit 2
    ;;
esac

if [[ ! -x "$EXECUTABLE" ]]; then
  echo "ORB-SLAM3 $MODE backend is not built: $EXECUTABLE" >&2
  exit 2
fi
command -v docker >/dev/null || { echo "docker is required to run ORB-SLAM3" >&2; exit 2; }
DOCKER_ARGS=(run --rm)
if [[ "$DETERMINISTIC_RUNTIME" == "1" ]]; then
  ALLOWED_CPU_LIST="$(awk '/^Cpus_allowed_list:/ {print $2}' /proc/self/status)"
  FIRST_CPU="${ALLOWED_CPU_LIST%%,*}"
  FIRST_CPU="${FIRST_CPU%%-*}"
  [[ "$FIRST_CPU" =~ ^[0-9]+$ ]] || {
    echo "cannot resolve a deterministic CPU from: $ALLOWED_CPU_LIST" >&2
    exit 2
  }
  DOCKER_ARGS+=(
    --cpuset-cpus "$FIRST_CPU"
    --env OMP_NUM_THREADS=1
    --env OMP_DYNAMIC=FALSE
    --env OPENBLAS_NUM_THREADS=1
    --env MKL_NUM_THREADS=1
    --env NUMEXPR_NUM_THREADS=1
    --env OPENCV_FOR_THREADS_NUM=1
  )
fi
if [[ "$SYNC_LOCAL_MAPPING" == "1" ]]; then
  DOCKER_ARGS+=(--env ARIADNE_ORBSLAM3_SYNC_LOCAL_MAPPING=1)
fi
exec docker "${DOCKER_ARGS[@]}" \
  --mount "type=bind,source=$ROOT,target=$ROOT" \
  --workdir "$PWD" \
  --env "LD_LIBRARY_PATH=$BACKEND_ROOT/ORB_SLAM3/lib:$BACKEND_ROOT/ORB_SLAM3/Thirdparty/DBoW2/lib:$BACKEND_ROOT/ORB_SLAM3/Thirdparty/g2o/lib:/usr/local/lib" \
  ariadne-orbslam3-build:22.04 "$EXECUTABLE" "$@"
