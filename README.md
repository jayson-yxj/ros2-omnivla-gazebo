# ros2-omnivla-gazebo

ROS 2 Humble workspace for validating OmniVLA in Gazebo Classic with a differential-drive indoor robot.
<img width="480" height="240" alt="2026-05-11T11_56_31 464Z-243109" src="https://github.com/user-attachments/assets/b396d6bf-5b67-4d2b-bf31-03271c35be8c" />

This repository keeps the experiment boundary explicit:

- **OmniVLA** is the action and short-horizon trajectory policy.
- **YOLO-World** is only used for open-vocabulary visual grounding and stop evidence in visible object-goal tests.
- The auxiliary `target_servo` route is only a detector/control debug path. It is not the main VLN result.

## What is in this workspace

- Simulator: Gazebo Classic
- Indoor world: AWS RoboMaker Small House
- Robot: custom `stable_diff_cam`
- Policy: official `NHirose/omnivla-original`
- Object grounding: `yolov8s-worldv2.pt`

Core ROS inputs and outputs:

- Input: `/camera/image_raw`
- Input: `/odom`
- Input: `/vl_nav/omnivla/task`
- Output: `/cmd_vel`
- Status: `/vl_nav/omnivla/status`

## Build

```bash
cd /root/Desktop/vln_project
scripts/prepare_third_party.sh
source /opt/ros/humble/setup.bash
colcon build --merge-install --symlink-install
source install/setup.bash
```

Large generated or downloaded directories are intentionally untracked:

```text
build/
install/
log/
models/
third_party/
```

## Models

OmniVLA checkpoint:

```text
models/omnivla/omnivla-original
```

YOLO-World checkpoint:

```text
models/yolov8s-worldv2.pt
```

If OmniVLA needs to be re-downloaded:

```bash
export VLN_PROJECT_ROOT=/root/Desktop/vln_project
python3 src/vlnav_gazebo/scripts/download_omnivla_assets.py \
  --asset original \
  --proxy http://127.0.0.1:7897 \
  --max-workers 1
```

## Main Modes

### 1. OmniVLA with pose goal

This is the most stable baseline in the workspace.

Run:

```bash
cd /root/Desktop/vln_project
source /opt/ros/humble/setup.bash
source install/setup.bash
GUI=true RVIZ=true src/vlnav_gazebo/scripts/run_omnivla_demo.sh
```

Send a named-place task:

```bash
python3 src/vlnav_gazebo/scripts/send_omnivla_task.py \
  --task "去到厨房然后停下来"
```

Or send an explicit pose goal:

```bash
python3 src/vlnav_gazebo/scripts/send_omnivla_task.py \
  --instruction "move toward the goal" \
  --goal-x 2.79818558693 \
  --goal-y -3.52509260178 \
  --goal-yaw -1.59079641132
```

Known named places in `send_omnivla_task.py`:

- `kitchen`
- `living_room`
- `dining_area`
- `bedroom`

In this route the OmniVLA input is:

```text
current image + odom-derived relative pose goal + optional language instruction
```

### 2. OmniVLA with visual object grounding

This route is for visible object-goal experiments such as:

```text
Move to the trash can and then stop
Find the fridge and then stop
```

Run:

```bash
cd /root/Desktop/vln_project
source /opt/ros/humble/setup.bash
source install/setup.bash
GUI=true RVIZ=true src/vlnav_gazebo/scripts/run_omnivla_visual_goal_demo.sh
```

Send a task:

```bash
python3 src/vlnav_gazebo/scripts/send_omnivla_task.py \
  --language-only \
  --task "Move to the trash can and then stop"
```

Example for fridge:

```bash
python3 src/vlnav_gazebo/scripts/send_omnivla_task.py \
  --language-only \
  --task "Find the fridge and then stop"
```

Pipeline:

```text
Gazebo camera + instruction
-> YOLO-World open-vocabulary detection
-> grounded local visual goal pose
-> OmniVLA image + language + grounded pose
-> OmniVLA action chunk / waypoint
-> /cmd_vel
```

Important behavior:

- The task publisher now sends an explicit `target_text` when it can infer one from the instruction.
- YOLO-World only uses detections with `score >= 0.55` and sufficient area for control and stop logic.
- Lower-score detections are still shown in RViz for debugging, but they do not control the robot.
- When the target family changes, for example `trash can -> fridge`, the detector now resets and reloads the YOLO-World model. This avoids the GPU class-switch crash seen with in-place vocabulary changes.

This route is still **not full semantic exploration**. If the target is not visible, the system does not build a semantic map or perform frontier exploration.

## Useful Commands

Stop OmniVLA:

```bash
python3 src/vlnav_gazebo/scripts/stop_omnivla_task.py
```

Monitor status:

```bash
ros2 topic echo /vl_nav/omnivla/status
ros2 topic echo /vl_nav/omnivla/waypoints
ros2 topic echo /vl_nav/target_detection
ros2 topic echo /vl_nav/detections
```

Image debugging:

```bash
ros2 run rqt_image_view rqt_image_view /vl_nav/annotated_image
ros2 run rqt_image_view rqt_image_view /vl_nav/omnivla/annotated_image
```

## RViz

The default RViz configuration shows:

- executed path
- predicted short-horizon OmniVLA path
- YOLO annotated image
- OmniVLA annotated image

## Auxiliary Target Servo

This mode exists only for debugging detection and base control:

```bash
GUI=true src/vlnav_gazebo/scripts/run_target_servo_demo.sh
```

Send a task:

```bash
python3 src/vlnav_gazebo/scripts/send_target_servo_task.py \
  --task "Move to the trash can and then stop"
```

This is not the main OmniVLA/VLN path.

## Verified

- Official OmniVLA repository cloned under `third_party/OmniVLA`
- Official `NHirose/omnivla-original` checkpoint downloaded
- OmniVLA ROS node loads and publishes `/cmd_vel`
- Gazebo launches the custom `stable_diff_cam`
- RViz shows OmniVLA trajectory overlays and detector overlays
- Visual object-goal task publishing now includes explicit `target_text`
- YOLO-World target-family switching now recovers by model reload instead of crashing on GPU

## Known limitations

- OmniVLA language-only mode is not a complete indoor exploration system.
- Visual object-goal behavior still depends on whether YOLO-World can recognize the Gazebo asset with the chosen target vocabulary.
- Arrival for object-goal tasks is based on image geometry, not metric depth sensing.
- Do not run multiple nodes that publish `/cmd_vel` at the same time unless you are intentionally testing arbitration.
