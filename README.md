# 🦾 Vision-Based Sorting using YOLOv8 and Interbotix RX200

This project demonstrates a scalable, real-time industrial solution for **object sorting** using computer vision and robotics. It combines a **Logitech C270 webcam**, **YOLOv8 object detection**, and the **Interbotix RX200 robotic arm**, orchestrated via **ROS 2 Humble** on **Ubuntu 22.04**.

---

## 📸 Demo & Setup


https://github.com/user-attachments/assets/6160d7f2-7286-41a8-9952-e35261c66b2d



👉 Object detection using YOLOv8  
👉 Robot picks and sorts detected objects based on class  
👉 All sorting happens in real-time using webcam + AprilTags

![setup](https://github.com/user-attachments/assets/b9e7d6d9-d076-4a36-8dab-54abdaccdd48)


---

## 🔍 Object Detection with YOLOv8

- Trained on **11 real-world items** using **Roboflow**
- Grouped into 2 classes:
  - 🍪 **Food**: butter, biscuit, proteinbar, mentos, tictac
  - 💊 **Medicine**: advil, moov, tylenol, ibuprofen, volini, eyedrop
- Model used: YOLOv8m (`best.pt`)
- Labelled + exported in YOLOv8 PyTorch format

📷 
![Screenshot from 2025-04-18 02-39-23](https://github.com/user-attachments/assets/fa59e19b-d88c-4087-8967-4e400144eae4)
![results](https://github.com/user-attachments/assets/03c97dc8-b48b-4095-b3fa-73fb5f5f7b4e)


---

## 📐 AprilTag & Calibration Logic

### ✅ AprilTag Setup
- **Tag 0** – Base frame  
- **Tag 1** – Medicine bin 
- **Tag 2** – Food bin   
- **Tags 3–7** – Objects

### 🎯 Camera Calibration
- We used OpenCV to calibrate the Logitech C270 webcam using an 8×6 chessboard pattern with 40mm square size.
- Captured multiple images while showing the chessboard to the camera.
- Extracted object and image points for subpixel corner refinement.
- Computed the camera matrix and distortion coefficients.
- Saved calibration results to camera_calib.npz.
- ![Screenshot from 2025-04-18 04-15-10](https://github.com/user-attachments/assets/6a72eadb-0f71-4379-bc8b-eb9e4f0503f3)

### 📐 Transformations
We use AprilTag detection and pose estimation to localize objects in the camera frame and convert them into robot base frame coordinates.
![Figure_111](https://github.com/user-attachments/assets/91a16504-21d4-4839-86dd-45d754454c71)

Step-by-Step Workflow:

- Estimate Tag Pose
- AprilTag detections provide the pose of each tag in the camera frame using cv2.solvePnP.

      T_tag_in_cam = to_homogeneous(R, t)

- Set the Base Frame using Tag 0
- AprilTag with ID 0 is placed on the table to define the robot's base coordinate frame.
- A fixed offset is applied to align it with the robot's actual base link:

      T_base_in_cam = T_tag0_in_cam @ T_offset

- Convert Object Tag Pose to Robot Base Frame
- For each detected object (attached to its own tag), we transform the pose into the robot base frame:

      T_obj_in_base = np.linalg.inv(T_base_in_cam) @ T_tag_in_cam

- Compute Grasp Point from Tag Center
- Since the object is slightly offset from the center of the tag (e.g., the tag is on one side), we apply another fixed transform to shift to the graspable part of the object:

      T_grasp_in_base = T_obj_in_base @ T_tag_to_cube_offset

🦾 Robot Execution Logic

Once the 3D grasp poses are computed for all objects, the robot performs the following steps for each:

    Move to safe height above the object

    Descend and grasp the object

    Lift up and move to the assigned bin (based on food or medicine class)

    Place the object and return to home pose

The sorting process is autonomous and repeatable in real-time.




https://github.com/user-attachments/assets/2977b49c-d836-4402-aa4b-b92b4d079f59


📊 Project Structure
```text
sorting_using_yolo/
            ├── train_yolov8.py             # YOLOv8 training script
            ├── camera_calibration.py       # Camera calibration script using chessboard
            ├── pick_and_place_using_yolo.py # Main sorting script (YOLO + AprilTag + Interbotix)
            ├── camera_calib.npz            # Saved calibration (camera matrix + distortion)
            ├── yolo_inference.py           # YOLOv8 detection only script
            ├── tf_plot.py                  # 3D transform visualization
            ├── dataset_images/             # Captured training data
            ├── runs/                       # YOLO training outputs
            └── README.md                   # This file

```

📎 Dependencies

This project uses:

    🐍 Python 3.10+

    🤖 ROS 2 Humble on Ubuntu 22.04

    📦 OpenCV, NumPy, Matplotlib

    📦 ultralytics for YOLOv8

    🤖 Interbotix SDK (interbotix_xs_modules)

    🧠 dt_apriltags for AprilTag detection

Install YOLOv8:

      pip install ultralytics

🚀 To Run the Project

# Activate ROS 2 workspace if needed
      source ~/interbotix_ws/install/setup.bash

# Run the full pick-and-place script
      python3 pick_and_place_using_yolo.py

# Want to just test YOLO?

      python3 yolo_inference.py

📌 Future Improvements

    Add live YOLO overlay as a ROS 2 node

    Replace AprilTags with 3D perception

    Integrate multi-camera stereo depth

    Expand to handle more object types

🏭 Scalable for Industry

This setup can be adapted to real warehouse sorting, medical sample organization, or retail product handling. The current version handles YOLO-based classification + AprilTag-based localization + real-time robotic control — a full pipeline ready for scale.
            

