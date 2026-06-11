import cv2
import numpy as np
from PIL import Image
from transformers import pipeline

# ==========================================
# LOAD DEPTH ANYTHING V2
# ==========================================

print("Loading Depth Anything V2...")

depth_estimator = pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf"
)

print("Model Loaded Successfully")

# ==========================================
# WEBCAM
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot access webcam")
    exit()

# ==========================================
# PARAMETERS
# ==========================================

DANGER_DEPTH_THRESHOLD = 180

WARNING_THRESHOLD = 0.03
STOP_THRESHOLD = 0.10

CONFIRM_FRAMES = 3

danger_counter = 0

# ==========================================
# MAIN LOOP
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (320, 240))

    # ======================================
    # DEPTH ANYTHING
    # ======================================

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    pil_image = Image.fromarray(
        rgb_frame
    )

    result = depth_estimator(
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

    # ======================================
    # SAFETY CORRIDOR
    # ======================================

    left_x = int(0.35 * w)
    right_x = int(0.65 * w)

    top_y = int(0.25 * h)
    bottom_y = int(0.75 * h)

    corridor_depth = depth[
        top_y:bottom_y,
        left_x:right_x
    ]

    # ======================================
    # DANGER PIXELS
    # ======================================

    danger_pixels = np.sum(
        corridor_depth >
        DANGER_DEPTH_THRESHOLD
    )

    total_pixels = (
        corridor_depth.size
    )

    danger_ratio = (
        danger_pixels /
        total_pixels
    )

    # ======================================
    # STATUS LOGIC
    # ======================================

    status = "CLEAR"

    if danger_ratio > STOP_THRESHOLD:

        danger_counter += 1

    else:

        danger_counter = 0

    if danger_counter >= CONFIRM_FRAMES:

        status = "STOP"

    elif danger_ratio > WARNING_THRESHOLD:

        status = "WARNING"

    else:

        status = "CLEAR"

    # ======================================
    # CORRIDOR COLOR
    # ======================================

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

    # ======================================
    # DISPLAY DATA
    # ======================================

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
        f"Counter: {danger_counter}",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 0, 0),
        2
    )

    # ======================================
    # STATUS DISPLAY
    # ======================================

    if status == "STOP":

        print("STOP")

        cv2.putText(
            frame,
            "STOP",
            (95, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 0, 255),
            3
        )

        cv2.putText(
            frame,
            "CHANGE ROUTE",
            (40, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )

    elif status == "WARNING":

        cv2.putText(
            frame,
            "WARNING",
            (70, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 255),
            3
        )

    else:

        cv2.putText(
            frame,
            "PATH CLEAR",
            (70, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    # ======================================
    # WINDOWS
    # ======================================

    cv2.imshow(
        "Drone View",
        frame
    )

    cv2.imshow(
        "Depth Map",
        depth
    )

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

# ==========================================
# CLEANUP
# ==========================================

cap.release()
cv2.destroyAllWindows()