import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
from collections import defaultdict
import threading
import csv
import time
from datetime import datetime
from main_app import VehicleCounter #import lõi xử lý đếm xe từ main_app.py

class VehicleDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Vehicle Counting System")
        self.root.geometry("1100x750")

        self.video_path = None
        self.cap = None
        self.running = False
        self.paused = False

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

        self.btn_start = tk.Button(button_frame, text="Start", width=15, command=self.start_detection)
        self.btn_start.grid(row=0, column=1, padx=5)

        self.btn_pause = tk.Button(button_frame, text="Pause", width=15, command=self.pause_detection)
        self.btn_pause.grid(row=0, column=2, padx=5)

        self.btn_continue = tk.Button(button_frame, text="Continue", width=15, command=self.continue_detection)
        self.btn_continue.grid(row=0, column=3, padx=5)
        
        self.btn_stop = tk.Button(button_frame, text="Stop & Export", width=15, command=self.stop_detection, bg="#ff4c4c", fg="white")
        self.btn_stop.grid(row=0, column=4, padx=5)

        self.video_label = tk.Label(self.root, bg="black")
        self.video_label.pack(pady=10)

        self.info_label = tk.Label(
            self.root,
            text="Video: None | Model: YOLOv8(best.pt)",
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

    def update_info_label(self):
        video_name = self.video_path if self.video_path else "None"

        self.info_label.config(
            text=f"Video: {video_name} | Model: YOLOv8(best.pt)"
        )

    def start_detection(self):
        if not self.video_path:
            messagebox.showwarning("Warning", "Please select a video first.")
            return
        
        if self.running:
            return

        self.running = True
        self.paused = False
        self.counter = VehicleCounter()

        thread = threading.Thread(target=self.process_video)
        thread.daemon = True
        thread.start()

    def stop_detection(self):
        self.running = False
        self.video_label.config(image='')
        self.export_report()

    def pause_detection(self):
        self.paused = True

    def continue_detection(self):
        self.paused = False

    def process_video(self):
        self.cap = cv2.VideoCapture(self.video_path)

        if not self.cap.isOpened():
            messagebox.showerror("Error", "Cannot open video.")
            self.running = False
            return

        while self.running:
            if self.paused:
                time.sleep(0.1)
                continue
            ret, frame = self.cap.read()
            if not ret:
                break

            processed_frame, current_counts = self.counter.ProcessFrame(frame)
            self.root.after(0, self.update_counts, current_counts)

            display_frame = cv2.resize(processed_frame, (960, 540))
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
        print("\n--- KIỂM TRA BỘ NHỚ ---")
        print(dir(self)) 
        print("-----------------------\n")

        car = counts.get("car", 0)
        bus = counts.get("bus", 0)
        motorbike = counts.get("motorbike", 0)
        truck = counts.get("truck", 0)

        self.count_label.config(
            text=f"Car: {car} | Bus: {bus} | Motorbike: {motorbike} | Truck: {truck}"
        )

    def export_report(self):
        if not hasattr(self, 'counter') or not self.counter:
            return
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"Vehicle_Report_{current_time}.csv"

        try:
            with open(filename, mode='w', newline='', encoding='utf-8-sig') as file:
                writer = csv.writer(file)
                writer.writerow(["Loại xe", "Số lượng Tổng lượt"])
                for vehicle_type, count in self.counter.dem_xe_pl.items():
                    writer.writerow([vehicle_type, count])

            messagebox.showinfo("Export Success", f"Đã xuất báo cáo thành công!\nFile: {filename}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Đã xảy ra lỗi khi xuất báo cáo:\n{e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = VehicleDetectionApp(root)
    root.mainloop()