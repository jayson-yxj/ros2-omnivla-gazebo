#!/usr/bin/env python3
import argparse
import json
import math
import re
import time

from ros_python_guard import ensure_ros_python


ensure_ros_python()

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


PLACES = {
    'kitchen': {
        'aliases': ('kitchen', '厨房', '廚房'),
        'goal_x': 8.30155944824,
        'goal_y': 1.45175802708,
        'goal_yaw': 1.50536940356,
    },
    'living_room': {
        'aliases': ('living room', 'living_room', '客厅', '客廳'),
        'goal_x': 2.79818558693,
        'goal_y': -3.52509260178,
        'goal_yaw': -1.59079641132,
    },
    'dining_area': {
        'aliases': ('dining', 'dining area', '餐厅', '餐廳'),
        'goal_x': 5.18359947205,
        'goal_y': 1.49744713306,
        'goal_yaw': 1.32731997641,
    },
    'bedroom': {
        'aliases': ('bedroom', '卧室', '臥室'),
        'goal_x': -5.64541149139,
        'goal_y': -2.90587878227,
        'goal_yaw': 1.62412965699,
    },
}

TARGET_GROUPS = (
    (('trash can', 'trash bin', 'garbage bin', 'waste bin', 'bin', 'dustbin', 'wastebasket', '垃圾桶'), 'trash can'),
    (('chair', 'stool', 'seat', '椅子'), 'chair'),
    (('table', 'desk', 'coffee table', 'dining table', '桌子'), 'table'),
    (('fridge', 'refrigerator', 'icebox', 'freezer', '冰箱'), 'fridge'),
    (('cabinet', 'cupboard', 'locker', 'sideboard', '柜子', '櫃子'), 'cabinet'),
    (('sofa', 'couch', '沙发', '沙發'), 'sofa'),
    (('bed', '床'), 'bed'),
    (('plant', 'potted plant', '盆栽', '植物'), 'plant'),
    (('door', '门', '門'), 'door'),
    (('television', 'tv', 'monitor', '电视', '電視'), 'television'),
    (('sink', 'washbasin', 'basin', '水槽'), 'sink'),
    (('microwave', 'oven', 'stove', 'cooktop', '炉灶', '爐灶'), 'microwave'),
)


def infer_place(text: str):
    lowered = text.lower()
    for name, spec in PLACES.items():
        for alias in spec['aliases']:
            if alias.lower() in lowered:
                return name, spec
    return None, None


def infer_target_text(text: str) -> str:
    lowered = text.lower()
    for aliases, canonical in TARGET_GROUPS:
        if any(alias.lower() in lowered for alias in aliases):
            return canonical
    patterns = (
        r'(?:go to|move to|walk to|approach|find|locate|reach|head to|go near|move near)\s+(?:the\s+)?([a-z][a-z0-9 _-]{1,40})',
        r'(?:去到|到|前往|靠近|找到)([^，。,.\n]{1,12})',
    )
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if not match:
            continue
        phrase = match.group(1).strip()
        phrase = re.split(r'\b(and|then|stop)\b|停下|然后|再', phrase)[0].strip()
        if phrase:
            return phrase
    return ''


def main():
    parser = argparse.ArgumentParser(description='Publish an OmniVLA task.')
    parser.add_argument('--task', help='Natural-language task, e.g. "去到厨房然后停下来".')
    parser.add_argument('--instruction', default='move toward the goal')
    parser.add_argument('--target-text', help='Explicit target object or place text for YOLO-World grounding.')
    parser.add_argument('--place', choices=sorted(PLACES.keys()), help='Named place in the Small House world.')
    parser.add_argument('--goal-x', type=float)
    parser.add_argument('--goal-y', type=float)
    parser.add_argument('--goal-yaw', type=float)
    parser.add_argument('--goal-tolerance', type=float, default=0.35)
    parser.add_argument(
        '--language-only',
        action='store_true',
        help='Use OmniVLA language-only modality: current image + instruction, without pose goal.',
    )
    parser.add_argument(
        '--image-goal',
        action='store_true',
        help='Also enable OmniVLA goal-image modality using the launch-configured goal image.',
    )
    parser.add_argument('--no-stop-at-goal', action='store_true')
    args = parser.parse_args()

    instruction = args.task or args.instruction
    target_text = (args.target_text or infer_target_text(instruction)).strip()
    place_name = args.place
    place_spec = PLACES[place_name] if place_name else None
    if place_spec is None:
        place_name, place_spec = infer_place(instruction)

    if args.language_only:
        goal_x = 0.0
        goal_y = 0.0
        goal_yaw = 0.0
    elif args.goal_x is not None and args.goal_y is not None:
        goal_x = args.goal_x
        goal_y = args.goal_y
        goal_yaw = args.goal_yaw if args.goal_yaw is not None else 0.0
    elif place_spec is not None:
        goal_x = place_spec['goal_x']
        goal_y = place_spec['goal_y']
        goal_yaw = place_spec['goal_yaw']
    else:
        raise SystemExit(
            'No goal was specified. Use --language-only, use --place kitchen, include a known place '
            'in --task, or pass --goal-x/--goal-y.'
        )

    if not math.isfinite(goal_x) or not math.isfinite(goal_y) or not math.isfinite(goal_yaw):
        raise SystemExit('Goal contains non-finite values.')

    rclpy.init()
    node = Node('send_omnivla_task')
    pub = node.create_publisher(String, '/vl_nav/omnivla/task', 10)
    payload = {
        'instruction': instruction,
        'goal_x': goal_x,
        'goal_y': goal_y,
        'goal_yaw': goal_yaw,
        'goal_tolerance': args.goal_tolerance,
        'stop_at_goal': (not args.no_stop_at_goal) and (not args.language_only),
        'use_pose_goal': not args.language_only,
        'use_language_goal': True,
        'use_image_goal': args.image_goal,
    }
    if place_name:
        payload['place'] = place_name
    if target_text:
        payload['target_text'] = target_text
    msg = String(data=json.dumps(payload))
    deadline = time.time() + 1.0
    while time.time() < deadline and pub.get_subscription_count() == 0:
        rclpy.spin_once(node, timeout_sec=0.05)
    for _ in range(5):
        pub.publish(msg)
        rclpy.spin_once(node, timeout_sec=0.05)
    node.get_logger().info(f'Published OmniVLA task: {msg.data}')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
