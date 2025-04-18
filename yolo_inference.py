# webcam_test_yolo.py
from ultralytics import YOLO
import cv2

# Load your trained model
model = YOLO("/home/nishit/sorting_using_yolo/runs/detect/sorting_yolo_model2/weights/best.pt")

# Open the webcam (use 0 or 1 or 2 depending on your system)
cap = cv2.VideoCapture(4)  # Change the index if needed

# Check if webcam opened successfully
if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

# Inference loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Run YOLO inference on the frame
    results = model.predict(source=frame, conf=0.5, save=False, verbose=False)

    # Draw results on the frame
    annotated_frame = results[0].plot()

    # Show the output
    cv2.imshow("YOLOv8 Detection", annotated_frame)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
