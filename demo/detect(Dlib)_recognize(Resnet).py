import cv2
import numpy as np
from keras.models import load_model # type: ignore
import _dlib_pybind11 as dlib # type: ignore
import time
import csv

print(dlib.__version__)

# Load model
model = load_model('D:/Study/HK7/CV/ProjectCV/DetectFacesWithDlibHoG/model/resnet50_finetune.h5')
class_labels = {0: "NguyenHungAnh", 1: "LePhuHao", 2: "NguyenHanhBaoAn", 3: "DuongNgocAnh", 4: "TuanAnhTran"}

detector = dlib.get_frontal_face_detector()

# Ghi log CSV
log_file = open('diemdanh_log.csv', mode='w', newline='', encoding='utf-8')
csv_writer = csv.writer(log_file)
csv_writer.writerow(['Time', 'Name'])

# Biến lưu thời gian log cuối của mỗi người
last_logged_time = {}

cap = cv2.VideoCapture(0)

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()

        cv2.rectangle(frame, (x, y), (x + w, y + h), (173,255,47), 2)

        face_crop = frame[y:y + h, x:x + w]
        face_resized = cv2.resize(face_crop, (224, 224))
        face_array = face_resized / 255.0
        face_array = np.expand_dims(face_array, axis=0)

        pred = model.predict(face_array)
        class_idx = np.argmax(pred)
        class_name = class_labels[class_idx]

        # Vẽ tên
        cv2.putText(frame, class_name, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

        # Check thời gian log
        current_time = time.time()
        if class_name not in last_logged_time or (current_time - last_logged_time[class_name] >= 10):
            csv_writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), class_name])
            log_file.flush()
            last_logged_time[class_name] = current_time
            print(f"Đã log {class_name} lúc {time.strftime('%H:%M:%S')}")

    # Hiển thị FPS
    fps = 1.0 / (time.time() - prev_time)
    prev_time = time.time()
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (173,255,47), 1)

    cv2.imshow("Diem Danh", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
log_file.close()
cv2.destroyAllWindows()
