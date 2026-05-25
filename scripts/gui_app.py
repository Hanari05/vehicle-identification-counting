import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
from collections import defaultdict
import threading


class VehicleDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Vehicle Counting System")
        self.root.geometry("1100x750")

        self.video_path = None
        self.model_path = None
        self.model = None
        self.cap = None
        self.running = False

        self.vehicle_counts = defaultdict(int)

        self.create_widgets()

    def create_widgets(self):
        title = tk.Label(
            self.root,
            text="Vehicle Detection & Counting System",
            font=("Arial", 18, "bold")
        )
        title.pack(pady=10)

        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)

        self.btn_select_video = tk.Button(
            button_frame,
            text="Select Video",
            width=15,
            command=self.select_video
        )
        self.btn_select_video.grid(row=0, column=0, padx=5)

        self.btn_select_model = tk.Button(
            button_frame,
            text="Select Model",
            width=15,
            command=self.select_model
        )
        self.btn_select_model.grid(row=0, column=1, padx=5)

        self.btn_start = tk.Button(
            button_frame,
            text="Start",
            width=15,
            command=self.start_detection
        )
        self.btn_start.grid(row=0, column=2, padx=5)

        self.btn_stop = tk.Button(
            button_frame,
            text="Stop",
            width=15,
            command=self.stop_detection
        )
        self.btn_stop.grid(row=0, column=3, padx=5)

        self.video_label = tk.Label(self.root, bg="black")
        self.video_label.pack(pady=10)

        self.info_label = tk.Label(
            self.root,
            text="Video: None | Model: None",
            font=("Arial", 10)
        )
        self.info_label.pack(pady=5)

        self.count_label = tk.Label(
            self.root,
            text="Car: 0 | Bus: 0 | Motorbike: 0 | Truck: 0",
            font=("Arial", 14, "bold")
        )
        self.count_label.pack(pady=10)

    def select_video(self):
        self.video_path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*")
            ]
        )

        self.update_info_label()

    def select_model(self):
        self.model_path = filedialog.askopenfilename(
            title="Select YOLO Model",
            filetypes=[
                ("YOLO model", "*.pt"),
                ("All files", "*.*")
            ]
        )

        if self.model_path:
            try:
                self.model = YOLO(self.model_path)
                messagebox.showinfo("Success", "Model loaded successfully.")
            except Exception as e:
                messagebox.showerror("Error", f"Cannot load model:\n{e}")

        self.update_info_label()

    def update_info_label(self):
        video_name = self.video_path if self.video_path else "None"
        model_name = self.model_path if self.model_path else "None"

        self.info_label.config(
            text=f"Video: {video_name} | Model: {model_name}"
        )

    def start_detection(self):
        if not self.video_path:
            messagebox.showwarning("Warning", "Please select a video first.")
            return

        if not self.model:
            messagebox.showwarning("Warning", "Please select a YOLO model first.")
            return

        if self.running:
            return

        self.running = True
        self.vehicle_counts = defaultdict(int)

        thread = threading.Thread(target=self.process_video)
        thread.daemon = True
        thread.start()

    def stop_detection(self):
        self.running = False

    def process_video(self):
        self.cap = cv2.VideoCapture(self.video_path)

        if not self.cap.isOpened():
            messagebox.showerror("Error", "Cannot open video.")
            self.running = False
            return

        while self.running:
            ret, frame = self.cap.read()

            if not ret:
                break

            results = self.model.predict(
                source=frame,
                conf=0.35,
                imgsz=640,
                verbose=False
            )

            annotated_frame = results[0].plot()

            frame_counts = defaultdict(int)

            if results[0].boxes is not None:
                for box in results[0].boxes:
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    frame_counts[class_name] += 1

            self.update_counts(frame_counts)

            display_frame = cv2.resize(annotated_frame, (960, 540))
            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

            img = Image.fromarray(display_frame)
            imgtk = ImageTk.PhotoImage(image=img)

            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

            self.root.update_idletasks()

        if self.cap:
            self.cap.release()

        self.running = False

    def update_counts(self, counts):
        car = counts.get("car", 0)
        bus = counts.get("bus", 0)
        motorbike = counts.get("motorbike", 0)
        truck = counts.get("truck", 0)

        self.count_label.config(
            text=f"Car: {car} | Bus: {bus} | Motorbike: {motorbike} | Truck: {truck}"
        )


if __name__ == "__main__":
    root = tk.Tk()
    app = VehicleDetectionApp(root)
    root.mainloop()