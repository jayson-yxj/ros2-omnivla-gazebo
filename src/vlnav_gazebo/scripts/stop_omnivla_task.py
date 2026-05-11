#!/usr/bin/env python3
import time

from ros_python_guard import ensure_ros_python


ensure_ros_python()

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


def main():
    rclpy.init()
    node = Node('stop_omnivla_task')
    pub = node.create_publisher(String, '/vl_nav/omnivla/task', 10)
    msg = String(data='stop')
    deadline = time.time() + 1.0
    while time.time() < deadline and pub.get_subscription_count() == 0:
        rclpy.spin_once(node, timeout_sec=0.05)
    for _ in range(5):
        pub.publish(msg)
        rclpy.spin_once(node, timeout_sec=0.05)
    node.get_logger().info('Published OmniVLA stop task.')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
