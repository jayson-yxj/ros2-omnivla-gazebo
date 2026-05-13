import json
import math
import os
import sys
import time
from typing import Optional

import cv2
import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, String
from visualization_msgs.msg import Marker


def yaw_from_quat(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class OmniVLAPolicy(Node):
    """ROS2 adapter around the official OmniVLA inference path."""

    def __init__(self):
        super().__init__('omnivla_policy')

        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('omnivla_root', '/root/Desktop/vln_project/third_party/OmniVLA')
        self.declare_parameter('model_dir', '/root/Desktop/vln_project/models/omnivla/omnivla-original')
        self.declare_parameter('resume_step', 120000)
        self.declare_parameter('instruction', 'move toward the goal')
        self.declare_parameter('goal_image_path', '/root/Desktop/vln_project/third_party/OmniVLA/inference/goal_img.jpg')
        self.declare_parameter('goal_x', 0.0)
        self.declare_parameter('goal_y', 0.0)
        self.declare_parameter('goal_yaw', 0.0)
        self.declare_parameter('use_pose_goal', True)
        self.declare_parameter('use_language_goal', True)
        self.declare_parameter('use_image_goal', False)
        self.declare_parameter('use_visual_goal_grounding', False)
        self.declare_parameter('visual_goal_detection_topic', '/vl_nav/target_detection')
        self.declare_parameter('visual_goal_hfov_rad', 2.094395102)
        self.declare_parameter('visual_goal_range_scale', 0.45)
        self.declare_parameter('visual_goal_min_range', 0.6)
        self.declare_parameter('visual_goal_max_range', 4.0)
        self.declare_parameter('visual_goal_timeout_sec', 1.0)
        self.declare_parameter('visual_goal_stop_area', 0.11)
        self.declare_parameter('visual_goal_stop_center_error', 0.16)
        self.declare_parameter('visual_goal_stop_bottom_y', 0.78)
        self.declare_parameter('use_visual_final_approach', True)
        self.declare_parameter('visual_final_approach_min_area', 0.05)
        self.declare_parameter('visual_final_approach_max_linear', 0.06)
        self.declare_parameter('visual_final_approach_max_angular', 0.18)
        self.declare_parameter('visual_final_approach_center_kp', 0.55)
        self.declare_parameter('visual_final_approach_center_deadband', 0.05)
        self.declare_parameter('visual_final_approach_slow_area', 0.08)
        self.declare_parameter('infer_hz', 0.33)
        self.declare_parameter('device', 'cuda:0')
        self.declare_parameter('waypoint_index', 4)
        self.declare_parameter('metric_waypoint_spacing', 0.1)
        self.declare_parameter('max_linear', 0.3)
        self.declare_parameter('max_angular', 0.3)
        self.declare_parameter('publish_cmd_vel', True)
        self.declare_parameter('stop_on_error', True)
        self.declare_parameter('autostart_task', False)
        self.declare_parameter('stop_at_goal', True)
        self.declare_parameter('goal_tolerance', 0.35)

        self.bridge = CvBridge()
        self.latest_rgb: Optional[np.ndarray] = None
        self.pose_xy_yaw: Optional[tuple[float, float, float]] = None
        self.active_instruction = str(self.get_parameter('instruction').value)
        self.active_goal_x = self._float_param('goal_x')
        self.active_goal_y = self._float_param('goal_y')
        self.active_goal_yaw = self._float_param('goal_yaw')
        self.active_use_pose_goal = self._bool_param('use_pose_goal')
        self.active_use_language_goal = self._bool_param('use_language_goal')
        self.active_use_image_goal = self._bool_param('use_image_goal')
        self.active_stop_at_goal = self._bool_param('stop_at_goal')
        self.active_goal_tolerance = self._float_param('goal_tolerance')
        self.task_active = self._bool_param('autostart_task')
        self.idle_status_published = False
        self.visual_detection: Optional[dict] = None
        self.visual_detection_time = 0.0
        self.visual_goal_available = False
        self.executed_path = Path()
        self.executed_path.header.frame_id = 'odom'
        self.model_loaded = False
        self.load_error: Optional[str] = None

        self.omni = None
        self.inference_helper = None
        self.vla = None
        self.action_head = None
        self.pose_projector = None
        self.device_id = None
        self.num_patches = None
        self.action_tokenizer = None
        self.processor = None
        self.goal_image_pil = None
        self.count_id = 0

        self.image_sub = self.create_subscription(
            Image, self.get_parameter('image_topic').value, self._on_image, 5
        )
        self.odom_sub = self.create_subscription(
            Odometry, self.get_parameter('odom_topic').value, self._on_odom, 20
        )
        self.task_sub = self.create_subscription(String, '/vl_nav/omnivla/task', self._on_task, 10)
        self.visual_detection_sub = self.create_subscription(
            String,
            self.get_parameter('visual_goal_detection_topic').value,
            self._on_visual_detection,
            10,
        )
        self.cmd_pub = self.create_publisher(Twist, self.get_parameter('cmd_vel_topic').value, 10)
        self.status_pub = self.create_publisher(String, '/vl_nav/omnivla/status', 10)
        self.waypoints_pub = self.create_publisher(Float32MultiArray, '/vl_nav/omnivla/waypoints', 10)
        self.annotated_pub = self.create_publisher(Image, '/vl_nav/omnivla/annotated_image', 2)
        self.executed_path_pub = self.create_publisher(Path, '/vl_nav/omnivla/executed_path', 10)
        self.predicted_path_pub = self.create_publisher(Path, '/vl_nav/omnivla/predicted_path', 10)
        self.predicted_marker_pub = self.create_publisher(Marker, '/vl_nav/omnivla/predicted_path_marker', 10)

        hz = max(0.02, self._float_param('infer_hz'))
        self.timer = self.create_timer(1.0 / hz, self._tick)
        self.get_logger().info(
            'OmniVLA adapter waiting for image and odom; '
            f'autostart_task={self.task_active}; task topic=/vl_nav/omnivla/task'
        )

    def _on_image(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Exception as exc:
            self._publish_status('image_error', {'error': str(exc)})

    def _on_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        yaw = yaw_from_quat(msg.pose.pose.orientation)
        self.pose_xy_yaw = (float(p.x), float(p.y), float(yaw))
        pose = PoseStamped()
        pose.header = msg.header
        pose.header.frame_id = msg.header.frame_id or 'odom'
        pose.pose = msg.pose.pose
        self.executed_path.header.stamp = msg.header.stamp
        self.executed_path.poses.append(pose)
        if len(self.executed_path.poses) > 5000:
            self.executed_path.poses = self.executed_path.poses[-5000:]
        try:
            self.executed_path_pub.publish(self.executed_path)
        except Exception:
            return

    def _on_task(self, msg: String):
        data = msg.data.strip()
        if not data:
            self._deactivate_task('empty_task')
            return
        if data.lower() in ('stop', 'pause', 'cancel'):
            self._deactivate_task(data.lower())
            return

        try:
            payload = json.loads(data)
            self.active_instruction = str(payload.get('instruction', self.active_instruction))
            self.active_goal_x = float(payload.get('goal_x', self.active_goal_x))
            self.active_goal_y = float(payload.get('goal_y', self.active_goal_y))
            self.active_goal_yaw = float(payload.get('goal_yaw', self.active_goal_yaw))
            self.active_use_pose_goal = self._json_bool(payload.get('use_pose_goal', self.active_use_pose_goal))
            self.active_use_language_goal = self._json_bool(
                payload.get('use_language_goal', self.active_use_language_goal)
            )
            self.active_use_image_goal = self._json_bool(payload.get('use_image_goal', self.active_use_image_goal))
            self.active_stop_at_goal = self._json_bool(payload.get('stop_at_goal', self.active_stop_at_goal))
            self.active_goal_tolerance = float(payload.get('goal_tolerance', self.active_goal_tolerance))
        except json.JSONDecodeError:
            self.active_instruction = data

        self.task_active = True
        self.idle_status_published = False
        if not self._bool_param('use_visual_goal_grounding'):
            self.visual_detection = None
            self.visual_detection_time = 0.0
            self.visual_goal_available = False
        self.executed_path.poses.clear()
        self.get_logger().info(
            'Accepted OmniVLA task: '
            f'instruction={self.active_instruction!r}, '
            f'goal=({self.active_goal_x:.3f}, {self.active_goal_y:.3f}, {self.active_goal_yaw:.3f}), '
            f'use_pose_goal={self.active_use_pose_goal}, '
            f'use_language_goal={self.active_use_language_goal}, '
            f'use_image_goal={self.active_use_image_goal}'
        )
        self._publish_status(
            'task_active',
            {
                'instruction': self.active_instruction,
                'goal_x': self.active_goal_x,
                'goal_y': self.active_goal_y,
                'goal_yaw': self.active_goal_yaw,
                'use_pose_goal': self.active_use_pose_goal,
                'use_language_goal': self.active_use_language_goal,
                'use_image_goal': self.active_use_image_goal,
                'stop_at_goal': self.active_stop_at_goal,
                'goal_tolerance': self.active_goal_tolerance,
            },
        )

    def _on_visual_detection(self, msg: String):
        if not self._bool_param('use_visual_goal_grounding'):
            return
        try:
            detection = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if not isinstance(detection, dict):
            return
        self.visual_detection = detection
        self.visual_detection_time = time.time()
        self._update_visual_goal_from_detection(detection)

    def _tick(self):
        if not self.task_active:
            if not self.idle_status_published:
                self.cmd_pub.publish(Twist())
                self._publish_status('idle_waiting_for_task', {'task_topic': '/vl_nav/omnivla/task'})
                self.idle_status_published = True
            return
        if self.latest_rgb is None or self.pose_xy_yaw is None:
            return
        if self._visual_goal_reached():
            self._deactivate_task('visual_goal_reached')
            return
        if self._should_use_visual_final_approach():
            twist = self._visual_final_approach_twist()
            if self._bool_param('publish_cmd_vel'):
                self.cmd_pub.publish(twist)
            self._publish_status(
                'visual_final_approach',
                {
                    'active_instruction': self.active_instruction,
                    'linear_x': twist.linear.x,
                    'angular_z': twist.angular.z,
                    'visual_goal_available': self.visual_goal_available,
                    'goal_x': self.active_goal_x,
                    'goal_y': self.active_goal_y,
                    'goal_yaw': self.active_goal_yaw,
                },
            )
            return
        if (
            self.active_use_pose_goal
            and not self._bool_param('use_visual_goal_grounding')
            and self.active_stop_at_goal
            and self._distance_to_goal() <= self.active_goal_tolerance
        ):
            self._deactivate_task('goal_reached')
            return
        if not self.model_loaded:
            if not self._load_model_once():
                self._stop_if_configured()
                return

        try:
            waypoints = self._predict_waypoints()
            twist = self._waypoints_to_twist(waypoints)
            self.waypoints_pub.publish(Float32MultiArray(data=waypoints.astype(np.float32).reshape(-1).tolist()))
            self._publish_predicted_path(waypoints)
            if self._bool_param('publish_cmd_vel'):
                self.cmd_pub.publish(twist)
            self._publish_overlay(self.latest_rgb, waypoints)
            self.get_logger().info(
                f'OmniVLA cmd_vel linear={twist.linear.x:.3f} angular={twist.angular.z:.3f}'
            )
            self._publish_status(
                'ok',
                {
                    'active_instruction': self.active_instruction,
                    'linear_x': twist.linear.x,
                    'angular_z': twist.angular.z,
                    'waypoint_index': self._int_param('waypoint_index'),
                    'use_pose_goal': self.active_use_pose_goal,
                    'effective_use_pose_goal': self._effective_use_pose_goal(),
                    'use_language_goal': self.active_use_language_goal,
                    'use_image_goal': self.active_use_image_goal,
                    'visual_goal_available': self.visual_goal_available,
                    'goal_x': self.active_goal_x,
                    'goal_y': self.active_goal_y,
                    'goal_yaw': self.active_goal_yaw,
                },
            )
        except Exception as exc:
            self.get_logger().error(f'OmniVLA inference failed: {exc}')
            self._publish_status('inference_error', {'error': str(exc)})
            self._stop_if_configured()

    def _load_model_once(self) -> bool:
        if self.load_error is not None:
            return False

        root = str(self.get_parameter('omnivla_root').value)
        model_dir = str(self.get_parameter('model_dir').value)
        try:
            if not self._looks_like_omnivla_model_dir(model_dir):
                raise FileNotFoundError(
                    f'OmniVLA checkpoint not found at {model_dir}. '
                    'Download NHirose/omnivla-original into this directory first.'
                )
            if root not in sys.path:
                sys.path.insert(0, root)

            from PIL import Image as PILImage
            import inference.run_omnivla as omni

            self.omni = omni
            cfg = omni.InferenceConfig()
            cfg.vla_path = model_dir
            cfg.resume_step = self._int_param('resume_step')

            requested_device = str(self.get_parameter('device').value)
            if requested_device.startswith('cuda') and not torch.cuda.is_available():
                self.get_logger().warning('CUDA requested but not available; using CPU')

            self.get_logger().info(f'Loading OmniVLA model from {model_dir}')
            (
                self.vla,
                self.action_head,
                self.pose_projector,
                self.device_id,
                self.num_patches,
                self.action_tokenizer,
                self.processor,
            ) = omni.define_model(cfg)

            goal_image_path = str(self.get_parameter('goal_image_path').value)
            self.goal_image_pil = PILImage.open(goal_image_path).convert('RGB')
            self.inference_helper = omni.Inference(
                save_dir='/tmp',
                lan_inst_prompt=self.active_instruction,
                goal_utm=(0.0, 0.0),
                goal_compass=0.0,
                goal_image_PIL=self.goal_image_pil,
                action_tokenizer=self.action_tokenizer,
                processor=self.processor,
            )
            self.model_loaded = True
            self._publish_status('model_loaded', {'model_dir': model_dir, 'device': str(self.device_id)})
            return True
        except Exception as exc:
            self.load_error = str(exc)
            self.get_logger().error(f'Failed to load OmniVLA: {self.load_error}')
            self._publish_status('model_load_error', {'error': self.load_error})
            return False

    def _predict_waypoints(self) -> np.ndarray:
        from PIL import Image as PILImage

        self._set_omnivla_modality_globals()
        current_image_pil = PILImage.fromarray(self.latest_rgb.astype(np.uint8)).convert('RGB')
        goal_pose = self._goal_pose_vector()
        instruction = self.active_instruction
        lan_inst = instruction if self.active_use_language_goal else 'xxxx'

        batch = self.inference_helper.data_transformer_omnivla(
            current_image_pil,
            lan_inst,
            self.goal_image_pil,
            goal_pose,
            prompt_builder=self.omni.PurePromptBuilder,
            action_tokenizer=self.action_tokenizer,
            processor=self.processor,
        )
        actions, modality_id = self.inference_helper.run_forward_pass(
            vla=self.vla.eval(),
            action_head=self.action_head.eval(),
            noisy_action_projector=None,
            pose_projector=self.pose_projector.eval(),
            batch=batch,
            action_tokenizer=self.action_tokenizer,
            device_id=self.device_id,
            use_l1_regression=True,
            use_diffusion=False,
            use_film=False,
            num_patches=self.num_patches,
            compute_diffusion_l1=False,
            num_diffusion_steps_train=None,
            mode='eval',
            idrun=self.count_id,
        )
        self.count_id += 1
        waypoints = actions.detach().float().cpu().numpy()[0]
        self._publish_status('raw_prediction', {'modality_id': int(modality_id.detach().cpu().numpy()[0])})
        return waypoints

    def _set_omnivla_modality_globals(self):
        self.omni.satellite = False
        self.omni.pose_goal = self._effective_use_pose_goal()
        self.omni.image_goal = self.active_use_image_goal
        self.omni.lan_prompt = self.active_use_language_goal

    def _goal_pose_vector(self) -> np.ndarray:
        if not self._effective_use_pose_goal():
            return np.zeros(4, dtype=np.float32)

        cur_x, cur_y, cur_yaw = self.pose_xy_yaw
        goal_x = self.active_goal_x
        goal_y = self.active_goal_y
        goal_yaw = self.active_goal_yaw
        spacing = max(1e-3, self._float_param('metric_waypoint_spacing'))
        thres_dist = 30.0

        dx = goal_x - cur_x
        dy = goal_y - cur_y
        rel_x = dx * math.cos(cur_yaw) + dy * math.sin(cur_yaw)
        rel_y = -dx * math.sin(cur_yaw) + dy * math.cos(cur_yaw)
        radius = math.sqrt(rel_x * rel_x + rel_y * rel_y)
        if radius > thres_dist:
            rel_x *= thres_dist / radius
            rel_y *= thres_dist / radius

        return np.array([
            rel_y / spacing,
            -rel_x / spacing,
            math.cos(goal_yaw - cur_yaw),
            math.sin(goal_yaw - cur_yaw),
        ], dtype=np.float32)

    def _effective_use_pose_goal(self) -> bool:
        return self.active_use_pose_goal or (
            self._bool_param('use_visual_goal_grounding') and self._visual_goal_is_fresh()
        )

    def _visual_goal_is_fresh(self) -> bool:
        if not self.visual_goal_available:
            return False
        return time.time() - self.visual_detection_time <= self._float_param('visual_goal_timeout_sec')

    def _update_visual_goal_from_detection(self, detection: dict):
        if self.pose_xy_yaw is None:
            return
        cur_x, cur_y, cur_yaw = self.pose_xy_yaw
        center_x = float(detection.get('center_x_norm', 0.5))
        area = max(1e-4, float(detection.get('area_norm', 0.0)))
        hfov = self._float_param('visual_goal_hfov_rad')
        bearing = (0.5 - center_x) * hfov

        target_range = self._float_param('visual_goal_range_scale') / math.sqrt(area)
        target_range = float(np.clip(
            target_range,
            self._float_param('visual_goal_min_range'),
            self._float_param('visual_goal_max_range'),
        ))

        rel_forward = max(0.25, target_range * math.cos(bearing))
        rel_left = target_range * math.sin(bearing)
        self.active_goal_x = cur_x + rel_forward * math.cos(cur_yaw) - rel_left * math.sin(cur_yaw)
        self.active_goal_y = cur_y + rel_forward * math.sin(cur_yaw) + rel_left * math.cos(cur_yaw)
        self.active_goal_yaw = math.atan2(self.active_goal_y - cur_y, self.active_goal_x - cur_x)
        self.visual_goal_available = True

    def _visual_goal_reached(self) -> bool:
        if not self._bool_param('use_visual_goal_grounding') or self.visual_detection is None:
            return False
        if not self._visual_goal_is_fresh():
            return False
        area = float(self.visual_detection.get('area_norm', 0.0))
        center_error = abs(float(self.visual_detection.get('center_x_norm', 0.5)) - 0.5)
        bottom_y = float(self.visual_detection.get('bottom_y_norm', 0.0))
        return (
            area >= self._float_param('visual_goal_stop_area')
            and center_error <= self._float_param('visual_goal_stop_center_error')
            and bottom_y >= self._float_param('visual_goal_stop_bottom_y')
        )

    def _should_use_visual_final_approach(self) -> bool:
        if not self._bool_param('use_visual_final_approach'):
            return False
        if self.visual_detection is None or not self._visual_goal_is_fresh():
            return False
        return float(self.visual_detection.get('area_norm', 0.0)) >= self._float_param('visual_final_approach_min_area')

    def _visual_final_approach_twist(self) -> Twist:
        twist = Twist()
        if self.visual_detection is None:
            return twist

        center_x = float(self.visual_detection.get('center_x_norm', 0.5))
        area = float(self.visual_detection.get('area_norm', 0.0))
        error = center_x - 0.5
        deadband = self._float_param('visual_final_approach_center_deadband')
        kp = self._float_param('visual_final_approach_center_kp')
        max_angular = self._float_param('visual_final_approach_max_angular')
        max_linear = self._float_param('visual_final_approach_max_linear')
        slow_area = self._float_param('visual_final_approach_slow_area')
        stop_area = self._float_param('visual_goal_stop_area')

        if abs(error) > deadband:
            twist.angular.z = float(np.clip(-kp * error, -max_angular, max_angular))

        if abs(error) <= 0.22:
            if area < slow_area:
                twist.linear.x = max_linear
            else:
                scale = max(0.2, (stop_area - area) / max(1e-3, stop_area - slow_area))
                twist.linear.x = max_linear * scale

        return twist

    def _waypoints_to_twist(self, waypoints: np.ndarray) -> Twist:
        twist = Twist()
        if waypoints.ndim != 2 or waypoints.shape[0] == 0 or waypoints.shape[1] < 4:
            return twist

        idx = min(max(0, self._int_param('waypoint_index')), waypoints.shape[0] - 1)
        chosen = waypoints[idx].copy()
        chosen[:2] *= max(1e-3, self._float_param('metric_waypoint_spacing'))
        dx, dy, hx, hy = [float(v) for v in chosen[:4]]

        eps = 1e-8
        dt = 1.0 / 3.0
        if abs(dx) < eps and abs(dy) < eps:
            linear = 0.0
            angular = self._clip_angle(math.atan2(hy, hx)) / dt
        elif abs(dx) < eps:
            linear = 0.0
            angular = math.copysign(math.pi / (2.0 * dt), dy)
        else:
            linear = dx / dt
            angular = math.atan(dy / dx) / dt

        linear = float(np.clip(linear, 0.0, self._float_param('max_linear')))
        angular = float(np.clip(angular, -self._float_param('max_angular'), self._float_param('max_angular')))
        twist.linear.x = linear
        twist.angular.z = angular
        return twist

    def _publish_overlay(self, rgb: np.ndarray, waypoints: np.ndarray):
        try:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            h, w = bgr.shape[:2]
            origin = (w // 2, int(h * 0.86))
            scale = 120.0
            pts = []
            spacing = self._float_param('metric_waypoint_spacing')
            for wp in waypoints[:16]:
                x = float(wp[0]) * spacing
                y = float(wp[1]) * spacing
                pts.append((origin[0] - int(y * scale), origin[1] - int(x * scale)))
            for p0, p1 in zip(pts[:-1], pts[1:]):
                cv2.line(bgr, p0, p1, (0, 0, 0), 6)
                cv2.line(bgr, p0, p1, (255, 180, 0), 3)
            if pts:
                cv2.circle(bgr, pts[0], 4, (0, 255, 0), -1)
            self.annotated_pub.publish(self.bridge.cv2_to_imgmsg(bgr, encoding='bgr8'))
        except Exception:
            return

    def _publish_predicted_path(self, waypoints: np.ndarray):
        if self.pose_xy_yaw is None:
            return
        cur_x, cur_y, cur_yaw = self.pose_xy_yaw
        spacing = max(1e-3, self._float_param('metric_waypoint_spacing'))

        path = Path()
        path.header.frame_id = 'odom'
        path.header.stamp = self.get_clock().now().to_msg()

        marker = Marker()
        marker.header = path.header
        marker.ns = 'omnivla'
        marker.id = 1
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.035
        marker.color.r = 0.0
        marker.color.g = 0.7
        marker.color.b = 1.0
        marker.color.a = 1.0
        marker.pose.orientation.w = 1.0

        start = PoseStamped()
        start.header = path.header
        start.pose.position.x = cur_x
        start.pose.position.y = cur_y
        start.pose.orientation.w = 1.0
        path.poses.append(start)
        marker.points.append(Point(x=cur_x, y=cur_y, z=0.05))

        for wp in waypoints[:16]:
            dx = float(wp[0]) * spacing
            dy = float(wp[1]) * spacing
            x = cur_x + dx * math.cos(cur_yaw) - dy * math.sin(cur_yaw)
            y = cur_y + dx * math.sin(cur_yaw) + dy * math.cos(cur_yaw)
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
            marker.points.append(Point(x=x, y=y, z=0.05))

        self.predicted_path_pub.publish(path)
        self.predicted_marker_pub.publish(marker)

    def _deactivate_task(self, reason: str):
        self.task_active = False
        self.idle_status_published = False
        self.cmd_pub.publish(Twist())
        self._publish_status('task_inactive', {'reason': reason})

    def _distance_to_goal(self) -> float:
        if self.pose_xy_yaw is None:
            return float('inf')
        cur_x, cur_y, _ = self.pose_xy_yaw
        return math.hypot(self.active_goal_x - cur_x, self.active_goal_y - cur_y)

    @staticmethod
    def _json_bool(value) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in ('1', 'true', 'yes', 'on')
        return bool(value)

    @staticmethod
    def _looks_like_omnivla_model_dir(path: str) -> bool:
        return (
            os.path.isdir(path)
            and os.path.isfile(os.path.join(path, 'config.json'))
            and os.path.isfile(os.path.join(path, 'action_head--120000_checkpoint.pt'))
            and (
                os.path.isfile(os.path.join(path, 'pose_projector--120000_checkpoint.pt'))
                or os.path.isfile(os.path.join(path, 'proprio_projector--120000_checkpoint.pt'))
            )
        )

    @staticmethod
    def _clip_angle(angle: float) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def _publish_status(self, state: str, extra: Optional[dict] = None):
        payload = {'state': state, 'stamp_wall_time': time.time()}
        if extra:
            payload.update(extra)
        self.status_pub.publish(String(data=json.dumps(payload, ensure_ascii=True)))

    def _stop_if_configured(self):
        if self._bool_param('stop_on_error') and self._bool_param('publish_cmd_vel'):
            self.cmd_pub.publish(Twist())

    def _float_param(self, name: str) -> float:
        return float(self.get_parameter(name).value)

    def _int_param(self, name: str) -> int:
        return int(self.get_parameter(name).value)

    def _bool_param(self, name: str) -> bool:
        value = self.get_parameter(name).value
        if isinstance(value, str):
            return value.strip().lower() in ('1', 'true', 'yes', 'on')
        return bool(value)


def main(args=None):
    rclpy.init(args=args)
    node = OmniVLAPolicy()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.cmd_pub.publish(Twist())
            node.destroy_node()
        except BaseException:
            pass
        try:
            rclpy.shutdown()
        except BaseException:
            pass


if __name__ == '__main__':
    main()
