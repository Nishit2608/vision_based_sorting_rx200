import cv2
import numpy as np

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((6*8, 3), np.float32)
objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)
objp *= 0.040  # 40mm square size in meters

objpoints = []
imgpoints = []

cap = cv2.VideoCapture(4)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
saved = 0

print("Show the chessboard to the camera. Press SPACE to capture, ESC to finish.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, (8, 6), None)

    if found:
        cv2.drawChessboardCorners(frame, (8, 6), corners, found)

    cv2.imshow("Calibration", frame)
    key = cv2.waitKey(1)

    if found and key == 32:  # Spacebar
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners2)
        saved += 1
        print(f"Captured image {saved}")
    elif key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()

# Calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("\n=== Calibration Result ===")
print("Camera Matrix:\n", mtx)
print("\nDistortion Coefficients:\n", dist)

# Save results
np.savez("camera_calib.npz", camera_matrix=mtx, dist_coeffs=dist)
print("\nSaved as camera_calib.npz")