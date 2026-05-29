import cv2
from ultralytics import YOLO

class VehicleCounter:
    def __init__(self, model_path = 'best.pt'):
        self.model = YOLO(model_path)
        self.line_pos_y = 450
        self.offset = 5
        self.ds_da_dem = []
        self.dem_xe_pl = {
            'bus': 0,
            'car': 0,
            'motorbike': 0,
            'truck': 0
        }

    def ProcessFrame(self, frame):
        frame = cv2.resize(frame, (1080, 720))
        roi = frame[360:720, 0:1080]
        results = self.model.track(roi, persist = True, verbose = False)

        cv2.line(frame, (0, self.line_pos_y), (1080, self.line_pos_y), (0, 0, 255), 2)

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy()

            for box, track_id, class_id  in zip(boxes, track_ids, class_ids):
                x1, y1, x2, y2 = map(int, box)
                id = int(track_id)
                ten_loai_xe = self.model.names[int(class_id)]

                y1_goc = y1 + 360
                y2_goc = y2 + 360
                y_tam = int((y1_goc + y2_goc) / 2)
                if (self.line_pos_y - self.offset) < y_tam < (self.line_pos_y + self.offset) and (id not in self.ds_da_dem):
                    self.dem_xe_pl[ten_loai_xe] += 1
                    self.ds_da_dem.append(id)

                    cv2.line(frame, (0, self.line_pos_y), (1080, self.line_pos_y), (0, 255, 0), 2)

                cv2.rectangle(frame, (x1, y1_goc), (x2, y2_goc), (0, 255, 0), 2)
                label = f"{ten_loai_xe.upper()} ID: {id}"
                cv2.putText(frame, label, (x1, y1_goc-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return frame, self.dem_xe_pl