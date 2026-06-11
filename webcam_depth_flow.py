import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import String

from cv_bridge import CvBridge

import cv2
import numpy as np

from PIL import Image as PILImage

from transformers import pipeline


class DepthObstacleNode(Node):

    def __init__(self):

        super().__init__('depth_obstacle_node')

        print("Loading Depth Anything V2...")

        self.depth_estimator = pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf"
        )

        print("Depth Anything Loaded")

        self.bridge = CvBridge()

        # ===================================
        # CHANGE THIS IF YOUR CAMERA TOPIC
        # IS DIFFERENT
        # ===================================

        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        # ===================================
        # OUTPUT TOPIC
        # ===================================

        self.warning_pub = self.create_publisher(
            String,
            '/obstacle_warning',
            10
        )

        # ===================================
        # PARAMETERS
        # ===================================

        self.DANGER_DEPTH_THRESHOLD = 180

        self.WARNING_THRESHOLD = 0.03

        self.STOP_THRESHOLD = 0.10

        self.CONFIRM_FRAMES = 3

        self.danger_counter = 0

    def image_callback(self, msg):

        frame = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding='bgr8'
        )

        frame = cv2.resize(
            frame,
            (320, 240)
        )

        # ===================================
        # DEPTH ANYTHING
        # ===================================

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        pil_image = PILImage.fromarray(
            rgb_frame
        )

        result = self.depth_estimator(
            pil_image
        )

        depth = np.array(
            result["depth"]
        )

        depth = cv2.normalize(
            depth,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        )

        depth = depth.astype(
            np.uint8
        )

        h, w = depth.shape

        # ===================================
        # SAFETY CORRIDOR
        # ===================================

        left_x = int(0.35 * w)
        right_x = int(0.65 * w)

        top_y = int(0.25 * h)
        bottom_y = int(0.75 * h)

        corridor_depth = depth[
            top_y:bottom_y,
            left_x:right_x
        ]

        # ===================================
        # DANGER PIXELS
        # ===================================

        danger_pixels = np.sum(
            corridor_depth >
            self.DANGER_DEPTH_THRESHOLD
        )

        total_pixels = corridor_depth.size

        danger_ratio = (
            danger_pixels /
            total_pixels
        )

        # ===================================
        # STATUS LOGIC
        # ===================================

        status = "CLEAR"

        if danger_ratio > self.STOP_THRESHOLD:

            self.danger_counter += 1

        else:

            self.danger_counter = 0

        if self.danger_counter >= self.CONFIRM_FRAMES:

            status = "STOP"

        elif danger_ratio > self.WARNING_THRESHOLD:

            status = "WARNING"

        else:

            status = "CLEAR"

        # ===================================
        # PUBLISH STATUS
        # ===================================

        status_msg = String()

        status_msg.data = status

        self.warning_pub.publish(
            status_msg
        )

        # ===================================
        # DISPLAY
        # ===================================

        corridor_color = (
            0,
            255,
            0
        )

        if status == "WARNING":

            corridor_color = (
                0,
                255,
                255
            )

        if status == "STOP":

            corridor_color = (
                0,
                0,
                255
            )

        cv2.rectangle(
            frame,
            (left_x, top_y),
            (right_x, bottom_y),
            corridor_color,
            2
        )

        cv2.putText(
            frame,
            f"Danger Ratio: {danger_ratio:.3f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 0),
            2
        )

        cv2.putText(
            frame,
            f"Danger Pixels: {danger_pixels}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 0),
            2
        )

        cv2.putText(
            frame,
            f"Counter: {self.danger_counter}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 0),
            2
        )

        cv2.putText(
            frame,
            status,
            (100, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            corridor_color,
            3
        )

        cv2.imshow(
            "AI Deck View",
            frame
        )

        cv2.imshow(
            "Depth Map",
            depth
        )

        cv2.waitKey(1)


def main():

    rclpy.init()

    node = DepthObstacleNode()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
