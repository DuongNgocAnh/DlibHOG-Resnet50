import tkinter as tk
from tkinter import messagebox, ttk
from datetime import datetime
import threading
import cv2
from PIL import Image, ImageTk
import os
import diemdanh
import themmoi

class DiemDanhApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Điểm danh hệ thống")
        self.geometry("1120x650")
        self.configure(bg="#1C1A16")
        
        # COLOR
        bg_main = "#1C1A16"
        bg_sub = "#423E34"
        fg_main = "#FFFFFF"
        btn_gray = "#735F32"
        btn_text_color = "#FFFFFF"
        btn_red = "#F44336"
        self.configure(bg=bg_main)

        # LEFT PANEL
        left_frame = tk.Frame(self, width=500, bg=bg_main)
        left_frame.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(left_frame, text="Điểm danh", font=("Arial", 16, "bold"), bg=bg_main, fg=fg_main).pack(pady=5)
        self.clock_label = tk.Label(left_frame, text="00:00:00", font=("Arial", 14), bg=bg_main, fg=fg_main)
        self.clock_label.pack(pady=5)

        self.btn_export = tk.Button(left_frame, text="Xuất file", command=self.export_file,
                                    bg=btn_gray, fg=btn_text_color, font=("Arial", 10, "bold"))
        self.btn_export.pack(pady=5)

        tk.Label(left_frame, text="Log điểm danh:", font=("Arial", 12), bg=bg_main, fg=fg_main).pack(pady=5)
        self.log_text = tk.Text(left_frame, width=40, height=15, bg=bg_sub, fg=fg_main, insertbackground=fg_main)
        self.log_text.pack(fill=tk.BOTH, padx=5)

        tk.Label(left_frame, text="Danh sách sinh viên", font=("Arial", 12), bg=bg_main, fg=fg_main).pack(pady=5)
        self.listbox = tk.Listbox(left_frame, bg="#000000", fg=fg_main, selectbackground="#555555")
        self.listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        train_folder = 'D:/Study/HK7/CV/ProjectCV/DetectFacesWithDlibHoG/Train'
        if os.path.exists(train_folder):
            for name in os.listdir(train_folder):
                self.listbox.insert(tk.END, name)

        # CENTER PANEL
        center_frame = tk.Frame(self, bg=bg_main)
        center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.camera_label = tk.Label(center_frame, bg="#000000", relief="sunken", bd=2)
        self.camera_label.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

        action_frame = tk.Frame(center_frame, bg=bg_main)
        action_frame.pack(pady=5)

        self.btn_onoff = tk.Button(action_frame, text="ON/OFF Camera", command=self.toggle_camera,bg=btn_gray, fg=btn_text_color, font=("Arial", 10, "bold"))
        self.btn_onoff.pack(side=tk.LEFT, padx=5)

        self.btn_diemdanh = tk.Button(action_frame, text="Điểm danh", command=self.start_diemdanh,bg=btn_red, fg="#FFFFFF", font=("Arial", 10, "bold"))
        self.btn_diemdanh.pack(side=tk.LEFT, padx=5)

        self.btn_dung = tk.Button(action_frame, text="Dừng", command=self.stop_diemdanh,bg=btn_gray, fg=btn_text_color, font=("Arial", 10, "bold"))
        self.btn_dung.pack(side=tk.LEFT, padx=5)

        form_frame = tk.Frame(center_frame, bg=bg_main)
        form_frame.pack(pady=5)

        tk.Label(form_frame, text="MSSV:", bg=bg_main, fg=fg_main).pack(side=tk.LEFT, padx=5)
        self.mssv_entry = tk.Entry(form_frame, width=10, bg="#555555", fg=fg_main, insertbackground=fg_main)
        self.mssv_entry.pack(side=tk.LEFT)

        tk.Label(form_frame, text="Họ Tên:", bg=bg_main, fg=fg_main).pack(side=tk.LEFT, padx=5)
        self.name_entry = tk.Entry(form_frame, width=15, bg="#555555", fg=fg_main, insertbackground=fg_main)
        self.name_entry.pack(side=tk.LEFT)

        self.btn_add = tk.Button(form_frame, text="Thêm sinh viên", command=self.add_user,bg=btn_gray, fg=btn_text_color, font=("Arial", 10, "bold"))
        self.btn_add.pack(side=tk.LEFT, padx=5)

        progress_frame = tk.Frame(center_frame, bg=bg_main)
        progress_frame.pack(pady=5)

        self.progress = ttk.Progressbar(progress_frame, length=300, mode='determinate')
        self.progress.pack(side=tk.LEFT, padx=5)

        self.btn_stop_capture = tk.Button(progress_frame, text="Dừng chụp ảnh", command=self.stop_capture,bg=btn_gray, fg=btn_text_color, font=("Arial", 10, "bold"), state=tk.DISABLED)
        self.btn_stop_capture.pack(side=tk.LEFT)

        self.is_running = False
        self.cap = None
        self.stop_capture_flag = False
        self.timer()


    def timer(self):
        now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        self.clock_label.config(text=now)
        self.after(1000, self.timer)

    def toggle_camera(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            threading.Thread(target=self.update_camera, daemon=True).start()
        else:
            self.cap.release()
            self.cap = None
            self.camera_label.config(image='')

    def update_camera(self):
        while self.cap and self.cap.isOpened() and not self.is_running:
            ret, frame = self.cap.read()
            if ret:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                self.camera_label.imgtk = imgtk
                self.camera_label.config(image=imgtk)

    def start_diemdanh(self):
        self.is_running = True
        model_path = 'D:/Study/HK7/CV/ProjectCV/DetectFacesWithDlibHoG/model/resnet50_finetune.h5'
        class_labels = {0: '21060451_NguyenHungAnh', 1: '21073141_LePhuHao', 2: '21075071_NguyenHanhBaoAn', 3: '21090261_DuongNgocAnh', 4: '21115461_TuanAnhTran'}
        threading.Thread(target=diemdanh.diem_danh, args=(self, model_path, class_labels), daemon=True).start()

    def stop_diemdanh(self):
        self.is_running = False

    def export_file(self):
        log_data = self.log_text.get("1.0", tk.END)
        if not log_data.strip():
            messagebox.showinfo("Thông báo", "Không có log để xuất.")
            return
        today_str = datetime.now().strftime("%d-%m-%Y")
        folder = "logs"
        os.makedirs(folder, exist_ok=True)
        file_path = os.path.join(folder, f"diemdanh_log_{today_str}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(log_data)
        messagebox.showinfo("Thông báo", f"Đã lưu log tại:\n{file_path}")

    def add_user(self):
        mssv = self.mssv_entry.get().strip()
        name = self.name_entry.get().strip()
        if not mssv or not name:
            messagebox.showwarning("Thông báo", "Vui lòng nhập đầy đủ MSSV và Họ Tên!")
            return
        if not mssv.isdigit() or len(mssv) != 8:
            messagebox.showwarning("Thông báo", "MSSV phải gồm đúng 8 chữ số!")
            return
        folder_name = f"{mssv}_{name}"
        train_folder = f"D:/Study/HK7/CV/ProjectCV/DetectFacesWithDlibHoG/Train/{folder_name}"
        test_folder = f"D:/Study/HK7/CV/ProjectCV/DetectFacesWithDlibHoG/Test/{folder_name}"
        if os.path.exists(train_folder):
            messagebox.showwarning("Thông báo", f"MSSV {mssv} đã tồn tại!")
            return
        os.makedirs(train_folder)
        os.makedirs(test_folder)
        self.listbox.insert(tk.END, folder_name)
        messagebox.showwarning("end", f"Đã thêm sinh viên: {folder_name}\n")
        self.log_text.insert("end", f"Đã thêm sinh viên: {folder_name}\n")
        self.mssv_entry.delete(0, tk.END)
        self.name_entry.delete(0, tk.END)
        threading.Thread(target=themmoi.capture_faces, args=(self, train_folder, test_folder), daemon=True).start()

    def stop_capture(self):
        self.stop_capture_flag = True

if __name__ == "__main__":
    app = DiemDanhApp()
    app.mainloop()
