import json
import time

from geometry_msgs.msg import Twist
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String


class ScanForTarget(Node):
    def __init__(self):
        super().__init__('scan_for_target')
        self.declare_parameter('angular_speed', 0.35)
        self.declare_parameter('timeout_sec', 90.0)
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.hit_sub = self.create_subscription(String, '/vl_nav/target_detection', self.on_hit, 10)
        self.image_sub = self.create_subscription(
            Image,
            self.get_parameter('image_topic').value,
            self.on_image,
            10,
        )
        self.start = None
        self.done = False
        self.timer = self.create_timer(0.1, self.tick)
        self.get_logger().info('Waiting for camera before scanning; stopping on target hit.')

    def on_image(self, _msg):
        if self.start is None:
            self.start = time.time()
            self.get_logger().info('Camera ready; starting scan.')

    def on_hit(self, msg):
        if self.done:
            return
        self.done = True
        self.stop()
        try:
            hit = json.loads(msg.data)
            self.get_logger().info(f'Target acquired: {hit.get("label")} score={hit.get("score"):.2f}')
        except Exception:
            self.get_logger().info(f'Target acquired: {msg.data}')

    def tick(self):
        if self.done:
            return
        if self.start is None:
            return
        if time.time() - self.start > float(self.get_parameter('timeout_sec').value):
            self.done = True
            self.stop()
            self.get_logger().warn('Scan timed out without target detection.')
            return
        cmd = Twist()
        cmd.angular.z = float(self.get_parameter('angular_speed').value)
        self.cmd_pub.publish(cmd)

    def stop(self):
        self.cmd_pub.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = ScanForTarget()
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
