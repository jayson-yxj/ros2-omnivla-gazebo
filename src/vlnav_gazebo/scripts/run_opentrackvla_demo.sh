#!/usr/bin/env bash
set -euo pipefail

set +u
source /opt/ros/humble/setup.bash
source /root/Desktop/vln_project/install/setup.bash
set -u

export TURTLEBOT3_MODEL=burger_cam
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"

HF_ROOT="${HF_ROOT:-/root/Desktop/vln_project/models/hf}"

ros2 launch vlnav_gazebo opentrackvla_gazebo.launch.py \
  gui:="${GUI:-true}" \
  instruction:="${INSTRUCTION:-follow the target person}" \
  hf_model_dir:="${HF_MODEL_DIR:-${HF_ROOT}/omlab__opentrackvla-qwen06b}" \
  qwen_model_name:="${QWEN_MODEL_NAME:-${HF_ROOT}/Qwen__Qwen3-0.6B}" \
  dino_model_name:="${DINO_MODEL_NAME:-${HF_ROOT}/facebook__dinov3-vits16-pretrain-lvd1689m}" \
  siglip_model_name:="${SIGLIP_MODEL_NAME:-${HF_ROOT}/google__siglip-so400m-patch14-384}" \
  device:="${DEVICE:-cuda:0}" \
  "$@"
