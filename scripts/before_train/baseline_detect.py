from ultralytics import YOLO
import cv2
import os

# Load model YOLO gốc
model = YOLO("yolov8n.pt")

video_path = "videos/Traffic_mixed.mp4"
output_path = "outputs/baseline_result.mp4"

os.makedirs("outputs", exist_ok=True)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Không mở được video.")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

# COCO class IDs:
# 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck
vehicle_classes = [1, 2, 3, 5, 7]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(
        frame,
        classes=vehicle_classes,
        conf=0.35,
        verbose=False
    )

    annotated_frame = results[0].plot()

    out.write(annotated_frame)

    display_width = 960
    scale = display_width / width
    display_height = int(height * scale)
    
    display_frame = cv2.resize(annotated_frame, (display_width, display_height))
    cv2.imshow("Baseline YOLO Detection", display_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Đã lưu video kết quả tại: {output_path}")