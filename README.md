# OmniVLA Gazebo Validation

This workspace now runs a real OmniVLA policy in Gazebo with a low-center-of-mass differential-drive camera robot.

- Policy: official OmniVLA `NHirose/omnivla-original`
- Simulator: Gazebo Classic + AWS RoboMaker Small House + `stable_diff_cam`
- Inputs: `/camera/image_raw`, `/odom`, language instruction, and a goal pose
- Output: OmniVLA action chunk -> velocity controller -> `/cmd_vel`
- Useful topics: `/vl_nav/omnivla/status`, `/vl_nav/omnivla/waypoints`, `/vl_nav/omnivla/annotated_image`

The ROS adapter is not a semantic-goal mock. It loads the official OmniVLA checkpoint and uses the official `inference/run_omnivla.py` model path.

## Build

```bash
cd /root/Desktop/vln_project
scripts/prepare_third_party.sh
source /opt/ros/humble/setup.bash
colcon build --merge-install --symlink-install
source install/setup.bash
```

Large generated or downloaded folders are intentionally not tracked by git:

```text
build/
install/
log/
models/
third_party/
```

`scripts/prepare_third_party.sh` clones the required upstream repositories and applies the small OmniVLA inference-import patch used by this workspace.

## Download OmniVLA

The full checkpoint is stored at:

```text
models/omnivla/omnivla-original
```

If it needs to be re-downloaded:

```bash
export VLN_PROJECT_ROOT=/root/Desktop/vln_project
python3 src/vlnav_gazebo/scripts/download_omnivla_assets.py \
  --asset original \
  --proxy http://127.0.0.1:7897 \
  --max-workers 1
```

The script routes HuggingFace API traffic through Clash and lets the large `cas-bridge.xethub.hf.co` file downloads go direct, which was required in this container.

## Run

Headless validation:

```bash
GUI=false \
INSTRUCTION="move toward the goal" \
GOAL_X=2.79818558693 \
GOAL_Y=-3.52509260178 \
GOAL_YAW=-1.59079641132 \
SPAWN_X=5.18359947205 \
SPAWN_Y=1.49744713306 \
SPAWN_YAW=1.32731997641 \
MAX_LINEAR=0.08 \
MAX_ANGULAR=0.12 \
INFER_HZ=1.0 \
src/vlnav_gazebo/scripts/run_omnivla_demo.sh
```

With Gazebo GUI:

```bash
src/vlnav_gazebo/scripts/run_omnivla_demo.sh
```

With RViz trajectory visualization:

```bash
GUI=true RVIZ=true src/vlnav_gazebo/scripts/run_omnivla_demo.sh
```

The robot now waits for an explicit task. In another terminal:

```bash
source /opt/ros/humble/setup.bash
source /root/Desktop/vln_project/install/setup.bash
python3 src/vlnav_gazebo/scripts/send_omnivla_task.py --task "去到厨房然后停下来"
```

Or send an explicit pose goal:

```bash
python3 src/vlnav_gazebo/scripts/send_omnivla_task.py \
  --instruction "move toward the goal" \
  --goal-x 2.79818558693 \
  --goal-y -3.52509260178 \
  --goal-yaw -1.59079641132
```

Natural-language place names are grounded by a Small House semantic landmark table in `send_omnivla_task.py`; this keeps the experiment reproducible while still sending language to OmniVLA. Known places include `kitchen`, `living_room`, `dining_area`, and `bedroom`.

Stop the active task:

```bash
python3 src/vlnav_gazebo/scripts/stop_omnivla_task.py
```

Direct launch:

```bash
ros2 launch vlnav_gazebo omnivla_gazebo.launch.py \
  gui:=true \
  instruction:="move toward the goal" \
  spawn_x:=5.18359947205 \
  spawn_y:=1.49744713306 \
  spawn_yaw:=1.32731997641 \
  goal_x:=2.79818558693 \
  goal_y:=-3.52509260178 \
  goal_yaw:=-1.59079641132 \
  max_linear:=0.08 \
  max_angular:=0.12 \
  infer_hz:=1.0 \
  publish_cmd_vel:=true \
  autostart_task:=false
```

Monitor:

```bash
ros2 topic echo /vl_nav/omnivla/status
ros2 topic echo /vl_nav/omnivla/waypoints
ros2 topic echo /vl_nav/omnivla/executed_path
ros2 topic echo /vl_nav/omnivla/predicted_path
ros2 topic echo /vl_nav/omnivla/predicted_path_marker
ros2 topic echo /cmd_vel
ros2 run rqt_image_view rqt_image_view /vl_nav/omnivla/annotated_image
```

## Verified

- Official OmniVLA repo cloned under `third_party/OmniVLA`
- Official `NHirose/omnivla-original` checkpoint downloaded
- `define_model()` loads successfully
- Sample OmniVLA forward pass outputs an 8-step action chunk
- Gazebo headless launches the low-center-of-mass `stable_diff_cam`
- Default Gazebo world is AWS RoboMaker Small House, with spawn and goal taken from its route file
- The robot stays idle until `/vl_nav/omnivla/task` receives a task
- RViz can show executed trajectory and OmniVLA predicted short-horizon trajectory
- OmniVLA ROS node consumes real Gazebo camera and odom
- `publish_cmd_vel:=true` publishes velocity commands to the Gazebo diff-drive plugin

## Notes

The OmniVLA repository was patched locally only to remove eager package-level imports that pulled training/RLDS dependencies into inference. The actual model loading and forward path still use the official OmniVLA inference modules.

OpenTrackVLA and YOLO-World files remain in the workspace as auxiliary experiments and smoke tests, but the main route is now OmniVLA.
