#!/usr/bin/env bash
set -euo pipefail

target_text="${1:-chair}"
gui="${GUI:-true}"

cd /root/Desktop/vln_project
set +u
source /opt/ros/humble/setup.bash
set -u
export TURTLEBOT3_MODEL=burger
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"

if [ ! -f install/setup.bash ]; then
  colcon build --merge-install --symlink-install
fi

set +u
source install/setup.bash
set -u

ros2 launch vlnav_gazebo open_vln_gazebo.launch.py gui:="${gui}" target_text:="${target_text}"
