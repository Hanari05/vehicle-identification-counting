from ultralytics import YOLO

model = YOLO("runs/detect/vehicle_yolov8n_first_train-final/weights/best.pt")

results = model.predict(
    source="videos/Traffic_mixed.mp4",
    conf=0.45,
    save=True,
    stream=True,
    imgsz=640,
    project="outputs/predict",
    name="test_traffic"
)

for r in results:
    pass

print("Test xong.")