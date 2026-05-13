#!/usr/bin/env python3
import argparse
import json
import time

from ros_python_guard import ensure_ros_python


ensure_ros_python()

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


TARGET_ALIASES = {
    'trash_can': ('trash can', 'trash bin', 'garbage bin', 'waste bin', 'bin', '垃圾桶'),
    'chair': ('chair', '椅子'),
    'table': ('table', 'desk', '桌子'),
    'sofa': ('sofa', 'couch', '沙发', '沙發'),
    'bed': ('bed', '床'),
    'plant': ('plant', '盆栽', '植物'),
}


def infer_target(text: str) -> str:
    lowered = text.lower()
    for aliases in TARGET_ALIASES.values():
        for alias in aliases:
            if alias.lower() in lowered:
                return ','.join(item for item in aliases if not any('\u4e00' <= ch <= '\u9fff' for ch in item))
    return text


def main():
    parser = argparse.ArgumentParser(description='Publish a YOLO-World target-servo task.')
    parser.add_argument('--task', required=True, help='Natural-language task, e.g. "Move to the trash can and stop".')
    parser.add_argument('--target-text', help='Open-vocabulary target labels, comma-separated.')
    args = parser.parse_args()

    target_text = args.target_text or infer_target(args.task)
    payload = {
        'instruction': args.task,
        'target_text': target_text,
    }

    rclpy.init()
    node = Node('send_target_servo_task')
    pub = node.create_publisher(String, '/vl_nav/target_servo/task', 10)
    msg = String(data=json.dumps(payload, ensure_ascii=False))
    deadline = time.time() + 1.0
    while time.time() < deadline and pub.get_subscription_count() == 0:
        rclpy.spin_once(node, timeout_sec=0.05)
    for _ in range(5):
        pub.publish(msg)
        rclpy.spin_once(node, timeout_sec=0.05)
    node.get_logger().info(f'Published target-servo task: {msg.data}')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
