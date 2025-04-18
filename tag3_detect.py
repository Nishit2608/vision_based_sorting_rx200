import cv2
import numpy as np
from dt_apriltags import Detector

# Load calibration
calib = np.load("/home/nishit/sorting_using_yolo/camera_calib.npz")
camera_matrix = calib["camera_matrix"]
dist_coeffs = calib["dist_coeffs"]

# AprilTag detector
detector = Detector(families="tag36h11")

def to_homogeneous(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    return T

cap = cv2.VideoCapture(4)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(gray)

    for det in detections:
        if det.tag_id != 3:  # Change to tag ID 2 (object)
            continue  # assuming tag 2 is on the object

        tag_size = 0.03  # Tag size (0.06 meters)
        object_points = np.array([
            [-tag_size / 2, -tag_size / 2, 0],
            [ tag_size / 2, -tag_size / 2, 0],
            [ tag_size / 2,  tag_size / 2, 0],
            [-tag_size / 2,  tag_size / 2, 0]
        ])
        image_points = np.array(det.corners, dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
        if success:
            # Rotation matrix from Rodrigues vector
            R, _ = cv2.Rodrigues(rvec)

            # Transform the tag pose to homogeneous matrix
            T_tag_in_cam = to_homogeneous(R, tvec)

            # Output the transformation matrices
            print(f"[Object (Tag 2) in Camera Frame] Transformation Matrix:")
            print(T_tag_in_cam)

    cv2.imshow("AprilTag Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()