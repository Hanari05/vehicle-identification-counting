import cv2
import os

video_path = "videos/Traffic_mixed.mp4"
output_dir = "datasets"

os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Không mở được video.")
    exit()

frame_count = 0
saved_count = 0

# Lưu mỗi 30 frame.
# Nếu video 30 FPS thì tức là lấy khoảng 1 ảnh mỗi giây.
frame_interval = 30

while True:
    ret, frame = cap.read()

    if not ret:
        break

    if frame_count % frame_interval == 0:
        filename = os.path.join(output_dir, f"frame_{saved_count:05d}.jpg")
        cv2.imwrite(filename, frame)
        saved_count += 1

    frame_count += 1

cap.release()

print(f"Đã lưu {saved_count} ảnh vào thư mục {output_dir}")