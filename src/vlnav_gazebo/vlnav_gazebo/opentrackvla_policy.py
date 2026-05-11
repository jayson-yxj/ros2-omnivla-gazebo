import json
import math
import os
import sys
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, String


class OpenTrackVLAPolicy(Node):
    """ROS2 adapter for the upstream OpenTrackVLA waypoint model."""

    def __init__(self):
        super().__init__('opentrackvla_policy')

        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('instruction', 'follow the target person')
        self.declare_parameter('open_trackvla_root', '/root/Desktop/vln_project/third_party/OpenTrackVLA')
        self.declare_parameter('hf_model_dir', '/root/Desktop/vln_project/models/hf/omlab__opentrackvla-qwen06b')
        self.declare_parameter('hf_model_id', 'omlab/opentrackvla-qwen06b')
        self.declare_parameter('auto_download', False)
        self.declare_parameter('qwen_model_name', 'Qwen/Qwen3-0.6B')
        self.declare_parameter('dino_model_name', 'facebook/dinov3-vits16-pretrain-lvd1689m')
        self.declare_parameter('siglip_model_name', 'google/siglip-so400m-patch14-384')
        self.declare_parameter('device', 'cuda:0')
        self.declare_parameter('infer_hz', 0.5)
        self.declare_parameter('history', 31)
        self.declare_parameter('waypoint_index', 1)
        self.declare_parameter('control_dt', 0.2)
        self.declare_parameter('max_linear', 0.18)
        self.declare_parameter('max_angular', 0.55)
        self.declare_parameter('heading_gain', 1.2)
        self.declare_parameter('theta_gain', 0.3)
        self.declare_parameter('allow_reverse', False)
        self.declare_parameter('publish_cmd_vel', True)
        self.declare_parameter('stop_on_error', True)

        self.bridge = CvBridge()
        self.latest_rgb: Optional[np.ndarray] = None
        self.latest_stamp = None
        self.last_infer_wall_time = 0.0
        self.model = None
        self.vision_cache = None
        self.grid_pool_tokens = None
        self.OpenTrackVLAForWaypoint = None
        self.OpenTrackVLAConfig = None
        self.frame_tokens = deque(maxlen=max(1, self._int_param('history')))
        self.load_error: Optional[str] = None

        image_topic = self.get_parameter('image_topic').value
        cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.image_sub = self.create_subscription(Image, image_topic, self._on_image, 5)
        self.cmd_pub = self.create_publisher(Twist, cmd_vel_topic, 10)
        self.status_pub = self.create_publisher(String, '/vl_nav/opentrackvla/status', 10)
        self.waypoints_pub = self.create_publisher(Float32MultiArray, '/vl_nav/opentrackvla/waypoints', 10)
        self.annotated_pub = self.create_publisher(Image, '/vl_nav/opentrackvla/annotated_image', 2)

        infer_hz = max(0.05, self._float_param('infer_hz'))
        self.timer = self.create_timer(1.0 / infer_hz, self._tick)
        self.get_logger().info(
            'OpenTrackVLA ROS adapter waiting for camera frames; '
            f'image_topic={image_topic}, instruction={self.get_parameter("instruction").value!r}'
        )

    def _on_image(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            self.latest_stamp = msg.header.stamp
        except Exception as exc:
            self._publish_status('image_error', {'error': str(exc)})

    def _tick(self):
        if self.latest_rgb is None:
            return
        if self.model is None or self.vision_cache is None:
            if not self._load_model_once():
                self._stop_if_configured()
                return

        try:
            result = self._predict(self.latest_rgb)
            if result is None:
                self._stop_if_configured()
                return
            waypoints, twist = result
            self.waypoints_pub.publish(Float32MultiArray(data=waypoints.astype(np.float32).reshape(-1).tolist()))
            if self._bool_param('publish_cmd_vel'):
                self.cmd_pub.publish(twist)
            self._publish_status(
                'ok',
                {
                    'instruction': self.get_parameter('instruction').value,
                    'linear_x': twist.linear.x,
                    'angular_z': twist.angular.z,
                    'waypoint_index': self._int_param('waypoint_index'),
                    'waypoints': waypoints.tolist(),
                },
            )
            self._publish_overlay(self.latest_rgb, waypoints)
        except Exception as exc:
            self.get_logger().error(f'OpenTrackVLA inference failed: {exc}')
            self._publish_status('inference_error', {'error': str(exc)})
            self._stop_if_configured()

    def _load_model_once(self) -> bool:
        if self.load_error is not None:
            return False

        root = str(self.get_parameter('open_trackvla_root').value)
        model_dir = str(self.get_parameter('hf_model_dir').value)
        auto_download = self._bool_param('auto_download')
        try:
            if root not in sys.path:
                sys.path.insert(0, root)
            from cache_gridpool import VisionCacheConfig, VisionFeatureCacher, grid_pool_tokens
            from open_trackvla_hf import OpenTrackVLAConfig, OpenTrackVLAForWaypoint

            self.OpenTrackVLAConfig = OpenTrackVLAConfig
            self.OpenTrackVLAForWaypoint = OpenTrackVLAForWaypoint
            self.grid_pool_tokens = grid_pool_tokens

            if not self._looks_like_hf_model_dir(model_dir):
                if not auto_download:
                    raise FileNotFoundError(
                        f'OpenTrackVLA checkpoint not found at {model_dir}. '
                        'Run scripts/download_opentrackvla_assets.py first, or set auto_download:=true.'
                    )
                model_dir = self._download_hf_model()

            device = self._device()
            self.get_logger().info(f'Loading OpenTrackVLA checkpoint from {model_dir} on {device}')
            cfg = OpenTrackVLAConfig.from_pretrained(model_dir)
            qwen_name = str(self.get_parameter('qwen_model_name').value).strip()
            if qwen_name:
                cfg.llm_name = qwen_name
            model = OpenTrackVLAForWaypoint.from_pretrained(model_dir, config=cfg)
            self.model = model.to(device).eval()

            vision_cfg = VisionCacheConfig(
                dino_model_name=str(self.get_parameter('dino_model_name').value),
                siglip_model_name=str(self.get_parameter('siglip_model_name').value),
                image_size=384,
                batch_size=1,
                device=str(device),
            )
            self.get_logger().info(
                'Loading OpenTrackVLA vision encoders: '
                f'DINO={vision_cfg.dino_model_name}, SigLIP={vision_cfg.siglip_model_name}'
            )
            self.vision_cache = VisionFeatureCacher(vision_cfg).eval()
            self._publish_status('model_loaded', {'model_dir': model_dir, 'device': str(device)})
            return True
        except Exception as exc:
            self.load_error = str(exc)
            self.get_logger().error(f'Failed to load OpenTrackVLA: {self.load_error}')
            self._publish_status('model_load_error', {'error': self.load_error})
            return False

    def _predict(self, rgb: np.ndarray):
        coarse, fine = self._encode_frame_tokens(rgb)
        if coarse is None or fine is None:
            self._publish_status('encode_error', {'error': 'vision encoder returned no tokens'})
            return None

        device = self._device()
        history = max(1, self._int_param('history'))
        self.frame_tokens.append(coarse.cpu())
        hist = list(self.frame_tokens)[-history:]
        if len(hist) < history:
            hist = [hist[0]] * (history - len(hist)) + hist

        coarse_chunks = []
        coarse_tidx = []
        for t, tok4 in enumerate(hist):
            tok4 = tok4.to(device)
            coarse_chunks.append(tok4)
            coarse_tidx.append(torch.full((tok4.size(0),), t, dtype=torch.long, device=device))

        coarse_tokens = torch.cat(coarse_chunks, dim=0).unsqueeze(0)
        coarse_tidx = torch.cat(coarse_tidx, dim=0).unsqueeze(0)
        fine_tokens = fine.to(device).unsqueeze(0)
        fine_tidx = torch.full((1, fine_tokens.size(1)), history, dtype=torch.long, device=device)
        instruction = [str(self.get_parameter('instruction').value)]

        with torch.inference_mode():
            tau = self.model(coarse_tokens, coarse_tidx, fine_tokens, fine_tidx, instruction)

        waypoints = tau.detach().float().cpu().numpy()[0]
        twist = self._waypoints_to_twist(waypoints)
        return waypoints, twist

    def _encode_frame_tokens(self, rgb_np: np.ndarray):
        try:
            from PIL import Image as PILImage

            pil = PILImage.fromarray(rgb_np.astype(np.uint8))
            tok_dino, hp, wp = self.vision_cache._encode_dino([pil])
            tok_siglip = self.vision_cache._encode_siglip([pil], out_hw=(hp, wp))
            tokens = torch.cat([tok_dino, tok_siglip], dim=-1)
            fine = self.grid_pool_tokens(tokens, hp, wp, out_tokens=64)[0].float()
            coarse = self.grid_pool_tokens(tokens, hp, wp, out_tokens=4)[0].float()
            return coarse, fine
        except Exception as exc:
            self.get_logger().error(f'OpenTrackVLA vision encoding failed: {exc}')
            return None, None

    def _waypoints_to_twist(self, waypoints: np.ndarray) -> Twist:
        twist = Twist()
        if waypoints.ndim != 2 or waypoints.shape[0] == 0 or waypoints.shape[1] < 2:
            return twist

        idx = min(max(0, self._int_param('waypoint_index')), waypoints.shape[0] - 1)
        x = float(waypoints[idx, 0])
        y = float(waypoints[idx, 1])
        theta = float(waypoints[idx, 2]) if waypoints.shape[1] >= 3 else 0.0
        dt = max(0.05, self._float_param('control_dt'))
        max_linear = abs(self._float_param('max_linear'))
        max_angular = abs(self._float_param('max_angular'))

        if not self._bool_param('allow_reverse'):
            x = max(0.0, x)

        desired_heading = math.atan2(y, max(abs(x), 1e-3))
        linear = max(-max_linear, min(max_linear, x / dt))
        angular = (
            self._float_param('heading_gain') * desired_heading
            + self._float_param('theta_gain') * (theta / dt)
        )
        angular = max(-max_angular, min(max_angular, angular))

        twist.linear.x = float(linear)
        twist.angular.z = float(angular)
        return twist

    def _publish_overlay(self, rgb: np.ndarray, waypoints: np.ndarray):
        try:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            h, w = bgr.shape[:2]
            origin = (w // 2, int(h * 0.86))
            scale = 120.0
            points = []
            for wp in waypoints[:16]:
                x = float(wp[0])
                y = float(wp[1])
                points.append((origin[0] - int(y * scale), origin[1] - int(x * scale)))
            for p0, p1 in zip(points[:-1], points[1:]):
                cv2.line(bgr, p0, p1, (0, 0, 0), 6)
                cv2.line(bgr, p0, p1, (0, 255, 180), 3)
            if points:
                cv2.circle(bgr, points[0], 4, (0, 255, 0), -1)
            msg = self.bridge.cv2_to_imgmsg(bgr, encoding='bgr8')
            self.annotated_pub.publish(msg)
        except Exception:
            return

    def _download_hf_model(self) -> str:
        from huggingface_hub import snapshot_download

        repo_id = str(self.get_parameter('hf_model_id').value)
        model_dir = str(self.get_parameter('hf_model_dir').value)
        os.makedirs(model_dir, exist_ok=True)
        self.get_logger().info(f'Downloading {repo_id} to {model_dir}')
        snapshot_download(repo_id, repo_type='model', local_dir=model_dir, local_dir_use_symlinks=False)
        if not self._looks_like_hf_model_dir(model_dir):
            raise FileNotFoundError(f'Download finished but {model_dir} is not a valid HF model dir')
        return model_dir

    @staticmethod
    def _looks_like_hf_model_dir(path: str) -> bool:
        if not os.path.isdir(path):
            return False
        if not os.path.isfile(os.path.join(path, 'config.json')):
            return False
        for name in os.listdir(path):
            if name in ('pytorch_model.bin', 'model.safetensors'):
                return True
            if name.endswith(('.bin', '.safetensors')):
                return True
            if name.endswith(('.bin.index.json', '.safetensors.index.json')):
                return True
        return False

    def _device(self) -> torch.device:
        requested = str(self.get_parameter('device').value)
        if requested.startswith('cuda') and not torch.cuda.is_available():
            return torch.device('cpu')
        return torch.device(requested)

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
    node = OpenTrackVLAPolicy()
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
