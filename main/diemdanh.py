import cv2
import numpy as np
import time
from datetime import datetime
from keras.models import load_model # type: ignore
import _dlib_pybind11 as dlib # type: ignore
from PIL import Image, ImageTk

def diem_danh(app, model_path, class_labels):
    model = load_model(model_path)
    detector = dlib.get_frontal_face_detector()
    last_logged = {}

    app.current_display_name = "Unknown"
    last_predict_time = 0

    while app.is_running and app.cap and app.cap.isOpened():
        ret, frame = app.cap.read()
        if not ret: continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        now_time = time.time()

        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(frame, (x, y), (x + w, y + h), (173, 255, 47), 2)

            if now_time - last_predict_time >= 5:
                face_crop = frame[y:y + h, x:x + w]
                if face_crop.size == 0: continue
                face_resized = cv2.resize(face_crop, (224, 224))
                face_array = face_resized / 255.0
                face_array = np.expand_dims(face_array, axis=0)

                pred = model.predict(face_array)
                class_idx = np.argmax(pred)
                class_name = class_labels.get(class_idx, "Unknown")
                app.current_display_name = class_name
                last_predict_time = now_time

                timestamp = datetime.now().strftime("%H:%M:%S")
                if class_name not in last_logged or now_time - last_logged[class_name] >= 10:
                    app.log_text.insert("end", f"{class_name} lúc {timestamp}\n")
                    last_logged[class_name] = now_time

            cv2.putText(frame, app.current_display_name, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        app.camera_label.imgtk = imgtk
        app.camera_label.config(image=imgtk)
