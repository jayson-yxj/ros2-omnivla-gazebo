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

ros2 launch vlnav_gazebo omnivla_gazebo.launch.py \
  gui:="${GUI:-true}" \
  instruction:="${INSTRUCTION:-move toward the goal}" \
  goal_x:="${GOAL_X:-2.79818558693}" \
  goal_y:="${GOAL_Y:--3.52509260178}" \
  goal_yaw:="${GOAL_YAW:--1.59079641132}" \
  spawn_x:="${SPAWN_X:-5.18359947205}" \
  spawn_y:="${SPAWN_Y:-1.49744713306}" \
  spawn_yaw:="${SPAWN_YAW:-1.32731997641}" \
  max_linear:="${MAX_LINEAR:-0.08}" \
  max_angular:="${MAX_ANGULAR:-0.12}" \
  infer_hz:="${INFER_HZ:-1.0}" \
  autostart_task:="${AUTOSTART_TASK:-false}" \
  rviz:="${RVIZ:-false}" \
  model_dir:="${OMNIVLA_MODEL_DIR:-${PROJECT_ROOT}/models/omnivla/omnivla-original}" \
  "$@"
