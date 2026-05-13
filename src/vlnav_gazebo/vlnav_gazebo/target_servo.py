import json
import time
from typing import Optional

from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String


class TargetServo(Node):
    """Open-vocabulary target servo controller.

    The node consumes YOLO-World target detections and produces conservative
    differential-drive commands: scan until the target is visible, center the
    target in the image, approach it, then stop when the target occupies enough
    image area.
    """

    def __init__(self):
        super().__init__('target_servo')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('detection_topic', '/vl_nav/target_detection')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('max_linear', 0.06)
        self.declare_parameter('max_angular', 0.16)
        self.declare_parameter('scan_angular', 0.10)
        self.declare_parameter('center_kp', 0.45)
        self.declare_parameter('center_deadband', 0.08)
        self.declare_parameter('target_area_stop', 0.11)
        self.declare_parameter('target_area_slow', 0.06)
        self.declare_parameter('detection_timeout_sec', 1.2)
        self.declare_parameter('target_text', 'trash can')
        self.declare_parameter('autostart', False)

        self.target_text = str(self.get_parameter('target_text').value)
        self.active = self._bool_param('autostart')
        self.arrived = False
        self.last_detection: Optional[dict] = None
        self.last_detection_time = 0.0
        self.executed_path = Path()
        self.executed_path.header.frame_id = 'odom'

        self.cmd_pub = self.create_publisher(Twist, self.get_parameter('cmd_vel_topic').value, 10)
        self.status_pub = self.create_publisher(String, '/vl_nav/target_servo/status', 10)
        self.path_pub = self.create_publisher(Path, '/vl_nav/target_servo/executed_path', 10)
        self.det_sub = self.create_subscription(
            String,
            self.get_parameter('detection_topic').value,
            self.on_detection,
            10,
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            self.get_parameter('odom_topic').value,
            self.on_odom,
            20,
        )
        self.task_sub = self.create_subscription(String, '/vl_nav/target_servo/task', self.on_task, 10)
        self.timer = self.create_timer(0.1, self.tick)
        self.get_logger().info(
            f'Target servo ready; active={self.active}, target={self.target_text!r}, '
            'task topic=/vl_nav/target_servo/task'
        )

    def on_task(self, msg):
        data = msg.data.strip()
        if not data:
            self.deactivate('empty_task')
            return
        if data.lower() in ('stop', 'pause', 'cancel'):
            self.deactivate(data.lower())
            return

        try:
            payload = json.loads(data)
            if isinstance(payload, dict):
                self.target_text = str(payload.get('target_text') or payload.get('target') or self.target_text)
            else:
                self.target_text = str(payload)
        except json.JSONDecodeError:
            self.target_text = data

        self.active = True
        self.arrived = False
        self.last_detection = None
        self.last_detection_time = 0.0
        self.get_logger().info(f'Accepted target-servo task: target={self.target_text!r}')
        self.publish_status('task_active', {'target_text': self.target_text})

    def on_detection(self, msg):
        try:
            detection = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if not isinstance(detection, dict):
            return
        self.last_detection = detection
        self.last_detection_time = time.time()

    def on_odom(self, msg):
        pose = PoseStamped()
        pose.header = msg.header
        pose.header.frame_id = msg.header.frame_id or 'odom'
        pose.pose = msg.pose.pose
        self.executed_path.header.stamp = msg.header.stamp
        self.executed_path.poses.append(pose)
        if len(self.executed_path.poses) > 5000:
            self.executed_path.poses = self.executed_path.poses[-5000:]
        self.path_pub.publish(self.executed_path)

    def tick(self):
        if not self.active or self.arrived:
            return

        now = time.time()
        timeout = float(self.get_parameter('detection_timeout_sec').value)
        if self.last_detection is None or now - self.last_detection_time > timeout:
            cmd = Twist()
            cmd.angular.z = float(self.get_parameter('scan_angular').value)
            self.cmd_pub.publish(cmd)
            self.publish_status('scanning', {'target_text': self.target_text})
            return

        cx = float(self.last_detection.get('center_x_norm', 0.5))
        area = float(self.last_detection.get('area_norm', 0.0))
        score = float(self.last_detection.get('score', 0.0))
        label = str(self.last_detection.get('label', 'target'))

        error = cx - 0.5
        stop_area = float(self.get_parameter('target_area_stop').value)
        deadband = float(self.get_parameter('center_deadband').value)
        if area >= stop_area and abs(error) <= max(deadband * 2.0, 0.14):
            self.arrived = True
            self.active = False
            self.stop()
            self.get_logger().info(f'Target reached: {label} score={score:.2f} area={area:.3f}')
            self.publish_status(
                'target_reached',
                {'label': label, 'score': score, 'area_norm': area, 'center_x_norm': cx},
            )
            return

        cmd = Twist()
        max_w = float(self.get_parameter('max_angular').value)
        kp = float(self.get_parameter('center_kp').value)
        if abs(error) > deadband:
            cmd.angular.z = self.clip(-kp * error, -max_w, max_w)

        slow_area = float(self.get_parameter('target_area_slow').value)
        max_v = float(self.get_parameter('max_linear').value)
        if abs(error) <= 0.22:
            if area < slow_area:
                cmd.linear.x = max_v
            else:
                scale = max(0.25, (stop_area - area) / max(1e-3, stop_area - slow_area))
                cmd.linear.x = max_v * scale

        self.cmd_pub.publish(cmd)
        self.publish_status(
            'tracking',
            {
                'label': label,
                'score': score,
                'area_norm': area,
                'center_x_norm': cx,
                'linear_x': cmd.linear.x,
                'angular_z': cmd.angular.z,
            },
        )

    def deactivate(self, reason: str):
        self.active = False
        self.arrived = False
        self.stop()
        self.publish_status('inactive', {'reason': reason})

    def stop(self):
        self.cmd_pub.publish(Twist())

    def publish_status(self, state: str, extra: Optional[dict] = None):
        payload = {'state': state, 'target_text': self.target_text, 'stamp_wall_time': time.time()}
        if extra:
            payload.update(extra)
        self.status_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))

    def _bool_param(self, name: str) -> bool:
        value = self.get_parameter(name).value
        if isinstance(value, str):
            return value.strip().lower() in ('1', 'true', 'yes', 'on')
        return bool(value)

    @staticmethod
    def clip(value: float, lower: float, upper: float) -> float:
        return min(max(value, lower), upper)


def main(args=None):
    rclpy.init(args=args)
    node = TargetServo()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node.stop()
        try:
            node.destroy_node()
        except BaseException:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
