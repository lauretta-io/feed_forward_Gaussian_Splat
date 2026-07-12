#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
BACKEND_ROOT="${ARIADNE_BACKEND_ROOT:-$ROOT/.cache/ariadne/backends}"
OPENVINS_COMMIT="69488123ed9362dd44b6f28e7f4680abbff1442b"
ORBSLAM3_COMMIT="4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4"
BUILD="none"

usage() {
  cat <<'EOF'
Usage: setup_vio_backends.sh [--build openvins|orbslam3|all]

Clone pinned GPLv3 OpenVINS and ORB-SLAM3 sources into the ignored backend cache.
Builds use Docker so ROS and C++ dependencies do not enter the ARIADNE Python environment.
EOF
}

while (($#)); do
  case "$1" in
    --build) BUILD="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done
case "$BUILD" in none|openvins|orbslam3|all) ;; *) echo "invalid build target" >&2; exit 2 ;; esac

clone_pinned() {
  local url="$1" destination="$2" commit="$3"
  if [[ ! -d "$destination/.git" ]]; then
    mkdir -p "$(dirname "$destination")"
    git clone "$url" "$destination"
  fi
  git -C "$destination" fetch --depth 1 origin "$commit"
  git -C "$destination" checkout --detach "$commit"
}

OPENVINS_WS="$BACKEND_ROOT/openvins_ws"
OPENVINS_SOURCE="$OPENVINS_WS/src/open_vins"
ORBSLAM3_SOURCE="$BACKEND_ROOT/ORB_SLAM3"
clone_pinned https://github.com/rpng/open_vins.git "$OPENVINS_SOURCE" "$OPENVINS_COMMIT"
clone_pinned https://github.com/UZ-SLAMLab/ORB_SLAM3.git "$ORBSLAM3_SOURCE" "$ORBSLAM3_COMMIT"

if [[ "$BUILD" != "none" ]]; then
  command -v docker >/dev/null || { echo "docker is required for backend builds" >&2; exit 2; }
  docker info >/dev/null
fi

if [[ "$BUILD" == "openvins" || "$BUILD" == "all" ]]; then
  docker build -t ariadne-openvins:noetic \
    -f "$OPENVINS_SOURCE/Dockerfile_ros1_20_04" "$OPENVINS_SOURCE"
  docker run --rm --mount "type=bind,source=$OPENVINS_WS,target=/catkin_ws" \
    ariadne-openvins:noetic bash -lc \
    'source /opt/ros/noetic/setup.bash && cd /catkin_ws && catkin build'
fi

if [[ "$BUILD" == "orbslam3" || "$BUILD" == "all" ]]; then
  docker build -t ariadne-orbslam3-build:22.04 \
    -f "$ROOT/applications/ariadne/docker/Dockerfile.orbslam3-build" "$ROOT"
  docker run --rm --mount "type=bind,source=$ORBSLAM3_SOURCE,target=/opt/orbslam3" \
    ariadne-orbslam3-build:22.04 bash -lc \
    "sed -e 's/make -j$/make -j2/' -e 's/make -j4$/make -j2/' build.sh > /tmp/build-limited.sh && bash /tmp/build-limited.sh"
fi

echo "OpenVINS source: $OPENVINS_SOURCE"
echo "ORB-SLAM3 source: $ORBSLAM3_SOURCE"
