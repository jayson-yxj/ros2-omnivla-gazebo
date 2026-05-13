#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${VLN_PROJECT_ROOT:-}" ]]; then
  PROJECT_ROOT="${VLN_PROJECT_ROOT}"
elif [[ -f "${SCRIPT_DIR}/../../../install/setup.bash" ]]; then
  PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
elif [[ -f "${SCRIPT_DIR}/../../../../install/setup.bash" ]]; then
  PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
else
  PROJECT_ROOT="/root/Desktop/vln_project"
fi

set +u
source /opt/ros/humble/setup.bash
source "${PROJECT_ROOT}/install/setup.bash"
set -u

export VLN_PROJECT_ROOT="${PROJECT_ROOT}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"

ros2 launch vlnav_gazebo target_servo_gazebo.launch.py \
  gui:="${GUI:-true}" \
  target_text:="${TARGET_TEXT:-trash can,trash bin,garbage bin,waste bin,bin}" \
  detector_hz:="${DETECTOR_HZ:-4.0}" \
  detector_conf:="${DETECTOR_CONF:-0.005}" \
  target_min_score:="${TARGET_MIN_SCORE:-0.55}" \
  target_min_area:="${TARGET_MIN_AREA:-0.005}" \
  target_only:="${TARGET_ONLY:-true}" \
  max_linear:="${MAX_LINEAR:-0.05}" \
  max_angular:="${MAX_ANGULAR:-0.14}" \
  scan_angular:="${SCAN_ANGULAR:-0.08}" \
  target_area_stop:="${TARGET_AREA_STOP:-0.11}" \
  autostart:="${AUTOSTART:-false}" \
  rviz:="${RVIZ:-true}" \
  model_path:="${YOLO_WORLD_MODEL:-${PROJECT_ROOT}/models/yolov8s-worldv2.pt}" \
  "$@"
