# Project Memory

This file is the working memory for the `ros2-omnivla-gazebo` project. Read it before making future changes when the conversation context is long.

## Goal

Build a rigorous Gazebo validation workspace for vision-language navigation with a differential-drive indoor robot.

The main experiment must stay honest:

- OmniVLA should remain the action/trajectory policy when the experiment is presented as OmniVLA/VLN.
- YOLO-World may be used for open-vocabulary grounding, target detection, and stop conditions.
- YOLO-World-only servo is an auxiliary debug mode, not a VLN result.
- A semantic lookup table is acceptable for reproducible known-place tests, but it is not the same as autonomous semantic exploration.

## Environment

- Workspace: `/root/Desktop/vln_project`
- ROS: Humble
- Simulator: Gazebo Classic
- Main world: AWS RoboMaker Small House
- Robot: custom `stable_diff_cam`
- OmniVLA repo: `third_party/OmniVLA`
- OmniVLA checkpoint: `models/omnivla/omnivla-original`
- YOLO-World checkpoint: `models/yolov8s-worldv2.pt`

The repo uses a merged ROS install layout. Build with:

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select vlnav_gazebo --symlink-install --merge-install
```

## Main OmniVLA Flow

Recommended command:

```bash
cd /root/Desktop/vln_project
source /opt/ros/humble/setup.bash
source install/setup.bash
GUI=true RVIZ=true src/vlnav_gazebo/scripts/run_omnivla_demo.sh
```

Task examples:

```bash
python3 src/vlnav_gazebo/scripts/send_omnivla_task.py --task "去到厨房然后停下来"
```

```bash
python3 src/vlnav_gazebo/scripts/send_omnivla_task.py \
  --instruction "move toward the goal" \
  --goal-x 2.79818558693 \
  --goal-y -3.52509260178 \
  --goal-yaw -1.59079641132
```

Current OmniVLA inputs in this flow:

- `/camera/image_raw`
- `/odom`
- language instruction
- pose goal, either explicitly supplied or produced by semantic place lookup

Important topics:

- `/vl_nav/omnivla/task`
- `/vl_nav/omnivla/status`
- `/vl_nav/omnivla/waypoints`
- `/vl_nav/omnivla/executed_path`
- `/vl_nav/omnivla/predicted_path`
- `/vl_nav/omnivla/annotated_image`
- `/cmd_vel`

Stop:

```bash
python3 src/vlnav_gazebo/scripts/stop_omnivla_task.py
```

## OmniVLA Official Input Understanding

Official OmniVLA supports multiple goal modalities via modality IDs:

- `0`: satellite only
- `1`: pose + satellite
- `2`: satellite + goal image
- `3`: pose + satellite + goal image
- `4`: pose only
- `5`: pose + goal image
- `6`: goal image only
- `7`: language only
- `8`: language + pose

OmniVLA-edge additionally supports `9`: language + goal image.

The official inference path uses:

- current first-person image
- optional goal image
- optional GPS/current pose and goal pose converted to local relative pose
- optional language prompt
- action chunk output

In Gazebo, odom/map coordinates replace GPS/UTM.

## Visual Object Goal Flow

This is the preferred route for visible object goals like:

```text
Move to the trash can and then stop
```

Recommended command:

```bash
cd /root/Desktop/vln_project
source /opt/ros/humble/setup.bash
source install/setup.bash
GUI=true RVIZ=true src/vlnav_gazebo/scripts/run_omnivla_visual_goal_demo.sh
```

Task:

```bash
python3 src/vlnav_gazebo/scripts/send_omnivla_task.py \
  --language-only \
  --task "Move to the trash can and then stop"
```

The task publisher now sends an explicit `target_text` field when it can infer one from the instruction, for example:

- `"Move to the trash can and then stop"` -> `target_text="trash can"`
- `"Find the fridge and then stop"` -> `target_text="fridge"`
- `"靠近冰箱"` -> `target_text="fridge"`

Architecture:

```text
Gazebo camera + instruction
YOLO-World target detection -> online visual goal pose + stop condition
OmniVLA current image + instruction + grounded relative pose -> action chunk
OmniVLA waypoints -> /cmd_vel
```

Critical distinction:

- YOLO-World does not command `/cmd_vel` in this flow.
- OmniVLA still produces waypoints/actions.
- YOLO-World only provides target grounding and arrival/stop evidence.

Useful topics:

- `/vl_nav/target_detection`
- `/vl_nav/detections`
- `/vl_nav/annotated_image`
- `/vl_nav/omnivla/status`
- `/vl_nav/omnivla/waypoints`
- `/vl_nav/omnivla/predicted_path`
- `/vl_nav/omnivla/executed_path`

Detection visualization behavior:

- Only the current task target family is visualized.
- Boxes below the control threshold are still shown for debugging.
- Only detections with confidence `>= 0.55` and sufficient area are allowed to affect control or stopping.
- YOLO-World class switching on GPU is fragile if done in-place after inference. The detector now resets and reloads the model when the task target family changes, for example from `trash can` to `fridge`.

## Auxiliary Target Servo

The following mode exists only for debugging YOLO detection and base control:

```bash
GUI=true src/vlnav_gazebo/scripts/run_target_servo_demo.sh
```

Task:

```bash
python3 src/vlnav_gazebo/scripts/send_target_servo_task.py \
  --task "Move to the trash can and then stop"
```

This mode is not VLN and should not be presented as OmniVLA. It uses bounding-box geometry and a simple controller.

## Robot Model

Model path:

```text
src/vlnav_gazebo/models/stable_diff_cam/model.sdf
```

Current design intent:

- low center of mass
- small differential-drive robot
- camera height about `0.50 m`
- horizontal FOV `120 deg`
- camera tilted downward about `8 deg`

Past issue: wheel joint axis was wrong and caused bad movement. Be careful when editing wheel links/joints.

## Semantic Place Table

Path:

```text
src/vlnav_gazebo/scripts/send_omnivla_task.py
```

Known Small House places:

- `kitchen`
- `living_room`
- `dining_area`
- `bedroom`

This is a reproducibility mechanism, not autonomous discovery. If the user asks for unknown-object or unknown-room exploration, the correct answer is to add mapping/exploration plus open-vocabulary semantic memory, not to pretend the semantic table is VLN.

## Known Limitations

- Language-only OmniVLA alone is unreliable for long-horizon indoor object-goal navigation.
- Without pose, goal image, semantic lookup, or visual grounding, the robot has no robust way to know where a non-visible object is.
- Current visual object goal flow can only ground objects detected in or found by the camera stream; it does not build a persistent semantic map.
- YOLO detection quality depends on Gazebo object appearance, lighting, camera pose, target labels, and confidence threshold.
- Arrival for visual object goals is based on bounding-box area and center error, not metric depth.
- Do not run multiple nodes that publish `/cmd_vel` at the same time unless explicitly testing arbitration.

## Git Notes

- Current remote: `https://github.com:443/jayson-yxj/ros2-omnivla-gazebo.git`
- Push previously failed due to missing GitHub credentials.
- Do not commit/download large third-party artifacts into git:
  - `models/`
  - `third_party/`
  - `build/`
  - `install/`
  - `log/`

## Current Main Files

- `src/vlnav_gazebo/vlnav_gazebo/omnivla_policy.py`
- `src/vlnav_gazebo/vlnav_gazebo/yolo_world_detector.py`
- `src/vlnav_gazebo/vlnav_gazebo/target_servo.py`
- `src/vlnav_gazebo/launch/omnivla_gazebo.launch.py`
- `src/vlnav_gazebo/launch/omnivla_visual_goal_gazebo.launch.py`
- `src/vlnav_gazebo/launch/target_servo_gazebo.launch.py`
- `src/vlnav_gazebo/scripts/run_omnivla_demo.sh`
- `src/vlnav_gazebo/scripts/run_omnivla_visual_goal_demo.sh`
- `src/vlnav_gazebo/scripts/run_target_servo_demo.sh`
- `src/vlnav_gazebo/scripts/send_omnivla_task.py`
- `src/vlnav_gazebo/scripts/send_target_servo_task.py`
- `src/vlnav_gazebo/config/omnivla_trajectory.rviz`
- `src/vlnav_gazebo/config/target_servo.rviz`
