from ultralytics import YOLO
import cv2
import numpy as np
from dt_apriltags import Detector
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
import time

# ----------------------------
# Helper Functions
# ----------------------------
def to_homogeneous(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    return T

def create_offset_transform(tx, ty, tz):
    T = np.eye(4)
    T[0, 3] = tx
    T[1, 3] = ty
    T[2, 3] = tz
    return T

def compute_position(T_in_cam):
    return np.linalg.inv(T_base_in_cam) @ T_in_cam

# ----------------------------
# Load YOLO Model
# ----------------------------
yolo_model = YOLO("runs/detect/sorting_yolo_model2/weights/best.pt")

FOOD_CLASSES = ['butter', 'biscuit', 'proteinbar', 'mentos', 'tictac']
MEDICINE_CLASSES = ['advil', 'moov', 'tylenol', 'ibuprofen', 'eyedrop']

# ----------------------------
# Load Calibration and Tag Detector
# ----------------------------
calib = np.load("/home/nishit/sorting_using_yolo/camera_calib.npz")
camera_matrix = calib["camera_matrix"]
dist_coeffs = calib["dist_coeffs"]
detector = Detector(families="tag36h11")

# ----------------------------
# Fixed Transforms
# ----------------------------
T_offset = create_offset_transform(-0.19, 0.0, 0.0)
T_tag_to_cube = create_offset_transform(0.0225, 0.0, 0.0)

T_base_in_cam = None
T_drop_green_cam = None
T_drop_brown_cam = None

# ----------------------------
# Capture Setup
# ----------------------------
cap = cv2.VideoCapture(4)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
tag_size = 0.03  # meters

object_positions = []
used_tags = set()
MAX_OBJECTS = 4
expected_tag_ids = {3, 7, 5, 6}  # Update this set based on the AprilTag IDs you're using

# ----------------------------
# Detection Loop
# ----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(gray)

    tag_poses = {}
    for det in detections:
        object_points = np.array([
            [-tag_size/2, -tag_size/2, 0],
            [ tag_size/2, -tag_size/2, 0],
            [ tag_size/2,  tag_size/2, 0],
            [-tag_size/2,  tag_size/2, 0]
        ], dtype=np.float64)
        image_points = np.array(det.corners, dtype=np.float64)
        success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
        if not success:
            continue
        R, _ = cv2.Rodrigues(rvec)
        T_tag_in_cam = to_homogeneous(R, tvec)
        tag_poses[det.tag_id] = T_tag_in_cam

        if det.tag_id == 0:
            T_base_in_cam = T_tag_in_cam @ T_offset
        elif det.tag_id == 2:
            T_drop_green_cam = T_tag_in_cam
        elif det.tag_id == 1:
            T_drop_brown_cam = T_tag_in_cam

    # YOLO Inference
    results = yolo_model.predict(source=frame, conf=0.5, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy()
    class_names = results[0].names

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.astype(int)
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        cls_name = class_names[int(class_ids[i])]

        min_dist = float('inf')
        closest_tag_id = None
        for tag_id, pose in tag_poses.items():
            if tag_id in [0, 1, 2]:  # Skip drop and base tags
                continue
            tag_px = camera_matrix @ pose[:3, 3]
            tag_px = tag_px[:2] / tag_px[2]
            dist = np.linalg.norm(np.array([cx, cy]) - tag_px)
            if dist < min_dist:
                min_dist = dist
                closest_tag_id = tag_id

        if closest_tag_id is not None and min_dist < 100 and closest_tag_id not in used_tags:
            used_tags.add(closest_tag_id)
            T_obj_cam = tag_poses[closest_tag_id]
            T_obj_base = compute_position(T_obj_cam)
            T_cube_base = T_obj_base @ T_tag_to_cube
            pos = {
                "x": T_cube_base[0, 3],
                "y": T_cube_base[1, 3],
                "z": T_cube_base[2, 3] + 0.05,
                "class": cls_name,
                "tag_id": closest_tag_id
            }
            object_positions.append(pos)
            print(f"✅ Added {cls_name} from Tag {closest_tag_id} at pixel distance {min_dist:.2f}")
        elif closest_tag_id in used_tags:
            print(f"⚠️ Skipping duplicate Tag {closest_tag_id} for {cls_name}")

    cv2.imshow("YOLOv8 Detection", results[0].plot())
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    if (T_base_in_cam is not None and 
        T_drop_green_cam is not None and 
        T_drop_brown_cam is not None and 
        used_tags.issuperset(expected_tag_ids)):
        print("✅ All required tags detected. Starting robot...")

        # Store the frame to stop YOLO updates during robot motion
        final_frame = results[0].plot().copy()
        break


cap.release()
cv2.destroyAllWindows()

# ----------------------------
# Convert Drop Locations
# ----------------------------
def transform_drop(tag_transform):
    drop_base = compute_position(tag_transform)
    return {
        "x": drop_base[0, 3],
        "y": drop_base[1, 3],
        "z": drop_base[2, 3] + 0.10
    }

green_drop_position = transform_drop(T_drop_green_cam)
brown_drop_position = transform_drop(T_drop_brown_cam)

# ----------------------------
# Robot Control
# ----------------------------
def pick_place_object(obj_pos, drop_pos, obj_name):
    bot = robot
    print(f"\n--- Pick-and-Place for {obj_name} ---")
    print("Object Position:", obj_pos)

    pick_safe_z = obj_pos["z"] + 0.10
    drop_safe_z = drop_pos["z"] + 0.25
    bot.gripper.release()

    pick_position = {
        "x": obj_pos["x"] + 0.03,
        "y": obj_pos["y"],
        "z": obj_pos["z"]
    }

    bot.arm.go_to_home_pose()
    bot.arm.set_ee_pose_components(x=pick_position["x"], y=pick_position["y"], z=pick_safe_z)
    time.sleep(0.5)

    bot.arm.set_ee_pose_components(x=pick_position["x"], y=pick_position["y"], z=pick_position["z"])
    time.sleep(0.5)

    bot.gripper.grasp()
    time.sleep(0.5)

    bot.arm.go_to_home_pose()
    time.sleep(0.5)

    bot.arm.set_ee_pose_components(x=drop_pos["x"], y=drop_pos["y"], z=drop_safe_z)
    time.sleep(0.8)

    bot.arm.set_ee_pose_components(x=drop_pos["x"], y=drop_pos["y"], z=drop_pos["z"] + 0.10)
    time.sleep(0.5)

    bot.gripper.release()
    bot.arm.set_ee_pose_components(
        x=drop_pos["x"],
        y=drop_pos["y"],
        z=drop_pos["z"] + 0.12
    )
    time.sleep(0.3)

    bot.arm.go_to_home_pose()

# ----------------------------
# Execute Pick and Place
# ----------------------------
robot = InterbotixManipulatorXS("rx200", "arm", "gripper")

for obj in object_positions:
    cls = obj["class"]
    if cls in FOOD_CLASSES:
        drop_pos = green_drop_position
    elif cls in MEDICINE_CLASSES:
        drop_pos = brown_drop_position
    else: 
        print(f"Skipping unclassified object: {cls}")
        continue
    pick_place_object(obj, drop_pos, cls)

    # Show YOLO frame while robot executes pick-place
cv2.namedWindow("Robot Working", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Robot Working", 800, 600)

while True:
    cv2.imshow("Robot Working", final_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

