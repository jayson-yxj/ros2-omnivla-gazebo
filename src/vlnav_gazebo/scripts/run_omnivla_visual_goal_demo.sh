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

ros2 launch vlnav_gazebo omnivla_visual_goal_gazebo.launch.py \
  gui:="${GUI:-true}" \
  rviz:="${RVIZ:-true}" \
  instruction:="${INSTRUCTION:-Move to the trash can and then stop}" \
  target_text:="${TARGET_TEXT:-trash can,trash bin,garbage bin,waste bin,bin}" \
  detector_hz:="${DETECTOR_HZ:-4.0}" \
  detector_conf:="${DETECTOR_CONF:-0.005}" \
  target_min_score:="${TARGET_MIN_SCORE:-0.55}" \
  target_min_area:="${TARGET_MIN_AREA:-0.005}" \
  target_only:="${TARGET_ONLY:-true}" \
  infer_hz:="${INFER_HZ:-1.0}" \
  max_linear:="${MAX_LINEAR:-0.08}" \
  max_angular:="${MAX_ANGULAR:-0.12}" \
  visual_goal_timeout_sec:="${VISUAL_GOAL_TIMEOUT_SEC:-3.0}" \
  visual_goal_stop_bottom_y:="${VISUAL_GOAL_STOP_BOTTOM_Y:-0.78}" \
  autostart_task:="${AUTOSTART_TASK:-false}" \
  yolo_model_path:="${YOLO_WORLD_MODEL:-${PROJECT_ROOT}/models/yolov8s-worldv2.pt}" \
  omnivla_model_dir:="${OMNIVLA_MODEL_DIR:-${PROJECT_ROOT}/models/omnivla/omnivla-original}" \
  "$@"
