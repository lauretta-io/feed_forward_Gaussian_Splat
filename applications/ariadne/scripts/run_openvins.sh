#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
BACKEND_ROOT="${ARIADNE_BACKEND_ROOT:-$ROOT/.cache/ariadne/backends}"
OPENVINS_WS="$BACKEND_ROOT/openvins_ws"
SETUP="$OPENVINS_WS/devel/setup.bash"

if [[ ! -d "$OPENVINS_WS/devel" ]]; then
  echo "OpenVINS workspace is not built: $SETUP" >&2
  exit 2
fi
command -v docker >/dev/null || { echo "docker is required to run OpenVINS" >&2; exit 2; }
exec docker run --rm --network host \
  --mount "type=bind,source=$ROOT,target=$ROOT" \
  --mount "type=bind,source=$OPENVINS_WS,target=/catkin_ws" \
  ariadne-openvins:noetic bash -lc \
  "source /opt/ros/noetic/setup.bash && source /catkin_ws/devel/setup.bash && exec roslaunch \"\$@\"" \
  bash "$@"
