import json
import re
import time
import traceback

import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String


class YoloWorldDetector(Node):
    TARGET_GROUPS = (
        (
            ('trash can', 'trash bin', 'garbage bin', 'waste bin', 'bin', 'dustbin', 'wastebasket', '垃圾桶'),
            [
                'trash can',
                'trash bin',
                'garbage bin',
                'waste bin',
                'bin',
                'dustbin',
                'wastebasket',
                'blue trash can',
                'blue trash bin',
                'dark trash can',
                'cylindrical trash can',
                'trash container',
                'waste container',
            ],
        ),
        (
            ('chair', 'stool', 'seat', '椅子'),
            ['chair', 'office chair', 'dining chair', 'stool', 'seat'],
        ),
        (
            ('table', 'desk', 'coffee table', 'dining table', '桌子'),
            ['table', 'desk', 'coffee table', 'dining table'],
        ),
        (
            ('fridge', 'refrigerator', 'icebox', 'freezer', '冰箱'),
            ['fridge', 'refrigerator', 'freezer', 'silver refrigerator', 'white refrigerator'],
        ),
        (
            ('cabinet', 'cupboard', 'locker', 'sideboard', '柜子', '櫃子'),
            ['cabinet', 'cupboard', 'locker', 'sideboard'],
        ),
        (
            ('sofa', 'couch', '沙发', '沙發'),
            ['sofa', 'couch', 'loveseat'],
        ),
        (
            ('bed', '床'),
            ['bed'],
        ),
        (
            ('plant', 'potted plant', '盆栽', '植物'),
            ['plant', 'potted plant'],
        ),
        (
            ('door', '门', '門'),
            ['door'],
        ),
        (
            ('television', 'tv', 'monitor', '电视', '電視'),
            ['television', 'tv', 'monitor'],
        ),
        (
            ('sink', 'washbasin', 'basin', '水槽'),
            ['sink', 'washbasin', 'basin'],
        ),
        (
            ('microwave', 'oven', 'stove', 'cooktop', '炉灶', '爐灶'),
            ['microwave', 'oven', 'stove', 'cooktop'],
        ),
    )

    def __init__(self):
        super().__init__('yolo_world_detector')
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('model_path', '/root/Desktop/vln_project/models/yolov8s-worldv2.pt')
        self.declare_parameter(
            'classes',
            'trash can,trash bin,garbage bin,waste bin,bin,chair,couch,sofa,table,bed,plant,door,television',
        )
        self.declare_parameter('target_text', 'trash can')
        self.declare_parameter('conf', 0.005)
        self.declare_parameter('target_min_score', 0.55)
        self.declare_parameter('target_min_area', 0.005)
        self.declare_parameter('target_only', True)
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('max_hz', 1.0)
        self.declare_parameter('device', 'cuda:0')

        self.bridge = CvBridge()
        self.last_infer = 0.0
        self.last_log = 0.0
        self.model = None
        self.class_names = self.parse_classes()
        self.target_terms = self.parse_target_terms()
        self.current_model_classes = []
        self.current_target_signature = tuple(term.lower() for term in self.target_terms)

        image_topic = self.get_parameter('image_topic').value
        self.detection_pub = self.create_publisher(String, '/vl_nav/detections', 10)
        self.target_pub = self.create_publisher(String, '/vl_nav/target_detection', 10)
        self.annotated_pub = self.create_publisher(Image, '/vl_nav/annotated_image', 10)
        self.sub = self.create_subscription(Image, image_topic, self.on_image, 10)
        self.task_sub = self.create_subscription(String, '/vl_nav/target_servo/task', self.on_task, 10)
        self.omnivla_task_sub = self.create_subscription(String, '/vl_nav/omnivla/task', self.on_task, 10)
        self.get_logger().info(f'YOLO-World detector subscribed to {image_topic}')

    def parse_classes(self):
        raw = self.get_parameter('classes').value
        return [item.strip() for item in raw.split(',') if item.strip()]

    def parse_target_terms(self):
        raw = self.get_parameter('target_text').value.lower()
        terms = self.normalize_target_terms(raw)
        return terms or ['chair']

    def normalize_target_terms(self, text):
        lowered = text.lower()
        for aliases, expanded_terms in self.TARGET_GROUPS:
            if any(term in lowered for term in aliases):
                return expanded_terms
        inferred = self.infer_target_phrase(lowered)
        if inferred:
            return [inferred]
        terms = [term.strip() for term in lowered.replace('，', ',').split(',') if term.strip()]
        return terms

    def infer_target_phrase(self, text):
        patterns = (
            r'(?:go to|move to|walk to|approach|find|locate|reach)\s+(?:the\s+)?([a-z][a-z0-9 _-]{1,40})',
            r'(?:go near|move near|head to)\s+(?:the\s+)?([a-z][a-z0-9 _-]{1,40})',
            r'(?:去到|到|前往|靠近|找到)([^，。,.\n]{1,12})',
        )
        for pattern in patterns:
            match = re.search(pattern, text)
            if not match:
                continue
            phrase = match.group(1).strip()
            phrase = re.split(r'\b(and|then|stop|停下|然后|再)\b', phrase)[0].strip()
            if phrase:
                return phrase
        return ''

    def ensure_model(self):
        if self.model is not None:
            return
        from ultralytics import YOLO

        model_path = self.get_parameter('model_path').value
        self.get_logger().info(f'Loading YOLO-World model: {model_path}')
        self.model = YOLO(model_path)
        self.set_model_classes(self.target_terms if self._bool_param('target_only') else self.class_names)

    def reset_model(self, reason):
        self.model = None
        self.current_model_classes = []
        self.get_logger().info(f'Resetting YOLO-World model: {reason}')

    def set_model_classes(self, classes):
        unique = []
        for item in classes:
            item = str(item).strip()
            if item and item.lower() not in [name.lower() for name in unique]:
                unique.append(item)
        if unique == self.current_model_classes:
            return
        self.model.set_classes(unique)
        self.current_model_classes = unique
        self.get_logger().info(f'Open-vocabulary classes: {self.current_model_classes}')

    def on_task(self, msg):
        target_text = self.extract_target_text(msg.data)
        if not target_text:
            return
        new_terms = self.normalize_target_terms(target_text)
        new_signature = tuple(term.lower() for term in new_terms)
        target_changed = new_signature != self.current_target_signature
        self.target_terms = new_terms
        self.current_target_signature = new_signature
        for term in reversed(self.target_terms):
            if term and term not in [name.lower() for name in self.class_names]:
                self.class_names.insert(0, term)
        if target_changed and self.model is not None and self._bool_param('target_only'):
            self.reset_model(f'target terms changed to {self.target_terms}')
        elif self.model is not None and self._bool_param('target_only'):
            self.set_model_classes(self.target_terms)
        self.get_logger().info(f'Updated target terms: {self.target_terms}')

    def extract_target_text(self, raw):
        raw = raw.strip()
        if not raw:
            return ''
        try:
            payload = json.loads(raw)
            if isinstance(payload, dict):
                return str(
                    payload.get('target_text')
                    or payload.get('target')
                    or payload.get('instruction')
                    or ''
                ).strip()
        except json.JSONDecodeError:
            pass
        return raw

    def on_image(self, msg):
        now = time.time()
        max_hz = float(self.get_parameter('max_hz').value)
        if now - self.last_infer < 1.0 / max_hz:
            return
        self.last_infer = now

        try:
            self.ensure_model()
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            if self._bool_param('target_only'):
                self.set_model_classes(self.target_terms)
            else:
                self.set_model_classes(self.class_names)
            result = self.model.predict(
                frame,
                conf=float(self.get_parameter('conf').value),
                imgsz=int(self.get_parameter('imgsz').value),
                device=self.get_parameter('device').value,
                verbose=False,
            )[0]
        except Exception as exc:
            self.get_logger().error(f'YOLO-World inference failed: {exc}')
            self.get_logger().error(traceback.format_exc())
            self.reset_model('recover after inference/class-switch failure')
            return

        height, width = frame.shape[:2]
        detections, candidate_hits = self.parse_result(
            result,
            width,
            height,
            force_target=self._bool_param('target_only'),
        )
        min_area = float(self.get_parameter('target_min_area').value)
        min_score = float(self.get_parameter('target_min_score').value)
        target_hits = [
            item
            for item in candidate_hits
            if item.get('area_norm', 0.0) >= min_area and item.get('score', 0.0) >= min_score
        ]
        target_hits = sorted(
            target_hits,
            key=lambda item: (float(item.get('score', 0.0)), float(item.get('area_norm', 0.0))),
            reverse=True,
        )

        payload = {
            'stamp': {'sec': msg.header.stamp.sec, 'nanosec': msg.header.stamp.nanosec},
            'target_text': self.get_parameter('target_text').value,
            'active_target_terms': self.target_terms,
            'target_min_score': min_score,
            'target_min_area': min_area,
            'detections': detections,
            'candidate_target_hits': candidate_hits,
            'target_hits': target_hits,
        }
        self.detection_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))
        if target_hits:
            self.target_pub.publish(String(data=json.dumps(target_hits[0], ensure_ascii=False)))
            self.get_logger().info(f'FOUND {target_hits[0]["label"]} score={target_hits[0]["score"]:.2f}')
        elif time.time() - self.last_log > 5.0:
            labels = [f'{item["label"]}:{item["score"]:.2f}' for item in detections[:5]]
            candidates = [f'{item["label"]}:{item["score"]:.2f}' for item in candidate_hits[:5]]
            self.get_logger().info(
                f'YOLO frame processed; target_terms={self.target_terms}; detections={labels}; candidate_hits={candidates}'
            )
            self.last_log = time.time()

        annotated = self.render_annotated(frame, candidate_hits, target_hits, min_score)
        out = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        out.header = msg.header
        self.annotated_pub.publish(out)

    def parse_result(self, result, width, height, force_target=False):
        detections = []
        target_hits = []
        names = result.names
        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls.item())
                label = str(names.get(cls_id, cls_id))
                score = float(box.conf.item())
                xyxy = [float(v) for v in box.xyxy[0].tolist()]
                x1, y1, x2, y2 = xyxy
                box_w = max(0.0, x2 - x1)
                box_h = max(0.0, y2 - y1)
                item = {
                    'label': label,
                    'score': score,
                    'xyxy': xyxy,
                    'image_width': width,
                    'image_height': height,
                    'top_y_norm': y1 / max(1, height),
                    'bottom_y_norm': y2 / max(1, height),
                    'center_x_norm': ((x1 + x2) * 0.5) / max(1, width),
                    'center_y_norm': ((y1 + y2) * 0.5) / max(1, height),
                    'area_norm': (box_w * box_h) / max(1, width * height),
                }
                detections.append(item)
                if force_target or self.is_target(label):
                    target_hits.append(item)
        return detections, target_hits

    def render_annotated(self, frame, candidate_hits, target_hits, min_score):
        annotated = frame.copy()
        for item in candidate_hits:
            x1, y1, x2, y2 = [int(round(v)) for v in item.get('xyxy', [0, 0, 0, 0])]
            label = str(item.get('label', 'target'))
            score = float(item.get('score', 0.0))
            confident = score >= float(min_score)
            color = (0, 220, 0) if confident else (0, 180, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            text = f'{label} {score:.2f}'
            cv2.putText(
                annotated,
                text,
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )
        if not target_hits:
            cv2.putText(
                annotated,
                f'No target hit >= {float(min_score):.2f}',
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 180, 255),
                2,
                cv2.LINE_AA,
            )
        return annotated

    def is_target(self, label):
        label = label.lower()
        return any(term in label or label in term for term in self.target_terms)

    def _bool_param(self, name):
        value = self.get_parameter(name).value
        if isinstance(value, str):
            return value.strip().lower() in ('1', 'true', 'yes', 'on')
        return bool(value)


def main(args=None):
    rclpy.init(args=args)
    node = YoloWorldDetector()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.destroy_node()
        except BaseException:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
