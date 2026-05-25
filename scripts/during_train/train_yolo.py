from ultralytics import YOLO
import os

DATA_YAML = "datasets/.main_datasets/data.yaml"

if not os.path.exists(DATA_YAML):
    raise FileNotFoundError(f"Không tìm thấy file data.yaml: {DATA_YAML}")

model = YOLO("yolov8n.pt")

results = model.train(
    data=DATA_YAML,
    epochs=20,
    imgsz=640,
    batch=4,
    name="vehicle_yolov8n_first_train"
)

print("Train xong.")