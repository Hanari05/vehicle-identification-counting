import cv2
import threading #thư viện để tạo luồng xử lý video
from ultralytics import YOLO

class VideoStream:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.ret, self.frame = self.cap.read()
        self.stopped = False

        self.thread = threading.Thread(target = self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while True:
            if self.stopped:
                self.cap.release()
                return
            self.ret, self.frame = self.cap.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

model = YOLO('best.pt') # có thể bằng model đã huấn luyện
video = VideoStream("Traffic_Street.mp4")

line_pos_y = 550
offset = 5
ds_da_dem = []
dem_xe_pl = {
    'bus': 0,
    'car': 0,
    'motorbike': 0,
    'truck': 0
}

while True:
    frame = video.read()
    if frame is None:
        continue
    frame = cv2.resize(frame, (1080, 720))
    roi = frame[360:720, 0:1080]
    results = model.track(roi, persist = True, verbose = False)
    cv2.line(frame, (0, line_pos_y), (1080, line_pos_y), (0, 0, 255), 2)

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy()

        for box, track_id, class_id  in zip(boxes, track_ids, class_ids):
            x1, y1, x2, y2 = map(int, box)
            id = int(track_id)
            ten_loai_xe = model.names[int(class_id)]

            y1_goc = y1 + 360
            y2_goc = y2 + 360
            y_tam = int((y1_goc + y2_goc) / 2)
            if (line_pos_y - offset) < y_tam < (line_pos_y + offset) and (id not in ds_da_dem):
                dem_xe_pl[ten_loai_xe] += 1
                ds_da_dem.append(id)

                cv2.line(frame, (0, line_pos_y), (1080, line_pos_y), (0, 255, 0), 2)

            cv2.rectangle(frame, (x1, y1_goc), (x2, y2_goc), (0, 255, 0), 2)
            label = f"{ten_loai_xe.upper()} ID: {id}"
            cv2.putText(frame, label, (x1, y1_goc-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    y_offset = 50
    for loai_xe, so_luong in dem_xe_pl.items():
        text = f"{loai_xe.capitalize()}: {so_luong}"
        cv2.putText(frame, text, (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        y_offset += 30

    cv2.imshow("He thong dem xe", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        video.stop()
        break
cv2.destroyAllWindows()