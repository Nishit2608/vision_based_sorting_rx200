from ultralytics import YOLO

# Load YOLOv8 model (Nano version; swap with yolov8s.pt or yolov8m.pt for better accuracy)
model = YOLO("yolov8n.pt")

# Train the model
model.train(
    data="/home/nishit/sorting_using_yolo/data.yaml",  #full dataset path
    epochs=50,
    imgsz=640,
    batch=8,  #adjustments
    name="sorting_yolo_model",  #saves to runs/detect/sorting_yolo_model
    project="runs/detect"
)

# Evaluate the trained model
metrics = model.val()
