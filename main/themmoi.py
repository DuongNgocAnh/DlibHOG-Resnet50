from tkinter import messagebox
import cv2
import os
import numpy as np
import _dlib_pybind11 as dlib # type: ignore
import shutil
from PIL import Image, ImageTk

def capture_faces(app, train_folder, test_folder, max_images=400):
    app.stop_capture_flag = False
    app.progress['value'] = 0
    app.btn_stop_capture.config(state="normal")

    cap = cv2.VideoCapture(0)
    detector = dlib.get_frontal_face_detector()
    count = 0

    while count < max_images and not app.stop_capture_flag:
        ret, frame = cap.read()
        if not ret: continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            face_crop = frame[y:y + h, x:x + w]
            if face_crop.size == 0: continue
            face_resized = cv2.resize(face_crop, (224, 224))

            save_folder = train_folder if count < int(max_images * 0.8) else test_folder
            save_path = os.path.join(save_folder, f"{count}.jpg")
            cv2.imwrite(save_path, face_resized)
            count += 1

            percent = (count / max_images) * 100
            app.progress['value'] = percent
            app.update_idletasks()

        cv2.imshow("Them sinh vien moi", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            app.stop_capture_flag = True
            break

    cap.release()
    cv2.destroyAllWindows()
    app.btn_stop_capture.config(state="disabled")

    if app.stop_capture_flag:
        delete_folders(train_folder, test_folder)
        messagebox.showwarning("end", f"Đã xóa sinh viên {os.path.basename(train_folder)}\n")
        app.log_text.insert("end", f"Đã xóa sinh viên {os.path.basename(train_folder)}\n")
        app.log_text.see("end")
    else:
        messagebox.showwarning("end", f"Đã chụp {count}/{max_images} ảnh.\n")
        app.log_text.insert("end", f"Đã chụp {count}/{max_images} ảnh.\n")
        app.log_text.see("end")

def delete_folders(train_folder, test_folder):
    if os.path.exists(train_folder):
        shutil.rmtree(train_folder)
    if os.path.exists(test_folder):
        shutil.rmtree(test_folder)
