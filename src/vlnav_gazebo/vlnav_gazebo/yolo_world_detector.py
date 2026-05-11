import json
import time

import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String


class YoloWorldDetector(Node):
    def __init__(self):
        super().__init__('yolo_world_detector')
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('model_path', '/root/Desktop/vln_project/models/yolov8s-worldv2.pt')
        self.declare_parameter('classes', 'chair,couch,sofa,table,bed,plant,door,television')
        self.declare_parameter('target_text', 'chair')
        self.declare_parameter('conf', 0.08)
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('max_hz', 1.0)
        self.declare_parameter('device', 'cuda:0')

        self.bridge = CvBridge()
        self.last_infer = 0.0
        self.last_log = 0.0
        self.model = None
        self.class_names = self.parse_classes()
        self.target_terms = self.parse_target_terms()

        image_topic = self.get_parameter('image_topic').value
        self.detection_pub = self.create_publisher(String, '/vl_nav/detections', 10)
        self.target_pub = self.create_publisher(String, '/vl_nav/target_detection', 10)
        self.annotated_pub = self.create_publisher(Image, '/vl_nav/annotated_image', 10)
        self.sub = self.create_subscription(Image, image_topic, self.on_image, 10)
        self.get_logger().info(f'YOLO-World detector subscribed to {image_topic}')

    def parse_classes(self):
        raw = self.get_parameter('classes').value
        return [item.strip() for item in raw.split(',') if item.strip()]

    def parse_target_terms(self):
        raw = self.get_parameter('target_text').value.lower()
        terms = [term.strip() for term in raw.replace('，', ',').split(',') if term.strip()]
        return terms or ['chair']

    def ensure_model(self):
        if self.model is not None:
            return
        from ultralytics import YOLO

        model_path = self.get_parameter('model_path').value
        self.get_logger().info(f'Loading YOLO-World model: {model_path}')
        self.model = YOLO(model_path)
        self.model.set_classes(self.class_names)
        self.get_logger().info(f'Open-vocabulary classes: {self.class_names}')

    def on_image(self, msg):
        now = time.time()
        max_hz = float(self.get_parameter('max_hz').value)
        if now - self.last_infer < 1.0 / max_hz:
            return
        self.last_infer = now

        self.ensure_model()
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        result = self.model.predict(
            frame,
            conf=float(self.get_parameter('conf').value),
            imgsz=int(self.get_parameter('imgsz').value),
            device=self.get_parameter('device').value,
            verbose=False,
        )[0]

        detections = []
        target_hits = []
        names = result.names
        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls.item())
                label = str(names.get(cls_id, cls_id))
                score = float(box.conf.item())
                xyxy = [float(v) for v in box.xyxy[0].tolist()]
                item = {'label': label, 'score': score, 'xyxy': xyxy}
                detections.append(item)
                if self.is_target(label):
                    target_hits.append(item)

        payload = {
            'stamp': {'sec': msg.header.stamp.sec, 'nanosec': msg.header.stamp.nanosec},
            'target_text': self.get_parameter('target_text').value,
            'detections': detections,
            'target_hits': target_hits,
        }
        self.detection_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))
        if target_hits:
            self.target_pub.publish(String(data=json.dumps(target_hits[0], ensure_ascii=False)))
            self.get_logger().info(f'FOUND {target_hits[0]["label"]} score={target_hits[0]["score"]:.2f}')
        elif time.time() - self.last_log > 5.0:
            labels = [f'{item["label"]}:{item["score"]:.2f}' for item in detections[:5]]
            self.get_logger().info(f'YOLO frame processed; detections={labels}')
            self.last_log = time.time()

        annotated = result.plot()
        out = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        out.header = msg.header
        self.annotated_pub.publish(out)

    def is_target(self, label):
        label = label.lower()
        return any(term in label or label in term for term in self.target_terms)


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
