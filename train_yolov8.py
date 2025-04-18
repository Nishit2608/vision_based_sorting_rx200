from ultralytics import YOLO

# Load YOLOv8 model (Nano version; swap with yolov8s.pt or yolov8m.pt for better accuracy)
model = YOLO("yolov8n.pt")

# Train the model
model.train(
    data="/home/nishit/sorting_using_yolo/data.yaml",  # ✅ Your full dataset path
    epochs=50,
    imgsz=640,
    batch=8,  # Adjust based on GPU memory
    name="sorting_yolo_model",  # Saves to runs/detect/sorting_yolo_model
    project="runs/detect"
)

# Evaluate the trained model
metrics = model.val()
