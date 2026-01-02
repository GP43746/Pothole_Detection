# detector.py
from ultralytics import YOLO
import cv2
import os
import socket
import time
from datetime import datetime
import threading
import json

# CONFIG
PHONE_URL = "http://192.168.31.36:4747/video"  # DroidCam or IP Webcam
MODEL_PATH = "yolov8n.pt"  # person detection (change to your pothole.pt later)
TCP_PORT = 8888
SAVE_INTERVAL = 5

os.makedirs("detections", exist_ok=True)
SHARED_FILE = "latest_detection.json"

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "dummy"

model = YOLO(MODEL_PATH)

current_lat = "Waiting..."
current_lon = ""
gps_lock = threading.Lock()

# FIXED GPS THREAD (buffer handling + robust parsing)
def gps_thread():
    global current_lat, current_lon
    server = socket.socket()
    server.bind(('', TCP_PORT))
    server.listen(1)
    print(f"Listening for ESP32 on port {TCP_PORT}...")

    conn, addr = server.accept()
    print("ESP32 connected wirelessly from", addr)

    buffer = ""
    while True:
        try:
            data = conn.recv(1024)
            if not data:
                print("ESP32 disconnected — reconnecting...")
                conn.close()
                conn, addr = server.accept()
                print("ESP32 reconnected from", addr)
                buffer = ""
                continue

            buffer += data.decode('ascii', errors='replace')

            while '\r\n' in buffer or '\n' in buffer:
                line, buffer = buffer.split('\n', 1) if '\n' in buffer else buffer.split('\r\n', 1)
                line = line.strip()
                if line.startswith('$GNGGA') or line.startswith('$GPGGA'):
                    parts = line.split(',')
                    if len(parts) >= 15 and parts[6] != '0' and parts[2] and parts[4]:
                        try:
                            # Latitude
                            lat = float(parts[2][:2]) + float(parts[2][2:])/60
                            if parts[3] == 'S': lat = -lat
                            # Longitude
                            lon = float(parts[4][:3]) + float(parts[4][3:])/60
                            if parts[5] == 'W': lon = -lon

                            with gps_lock:
                                current_lat = round(lat, 6)
                                current_lon = round(lon, 6)

                            print(f"GPS LOCK → {current_lat}, {current_lon}")

                        except Exception as e:
                            print("GPS parse error:", e)

        except Exception as e:
            print("Connection error:", e)
            time.sleep(2)

threading.Thread(target=gps_thread, daemon=True).start()

cap = cv2.VideoCapture(PHONE_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print("DETECTOR RUNNING – Wireless GPS + Detection + Save")

last_save = 0
detection_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        cap = cv2.VideoCapture(PHONE_URL)
        continue

    frame = cv2.resize(frame, (640, 480))
    results = model(frame, conf=0.25, verbose=False).__next__()

    detected = False
    highest_conf = 0.0

    for box in results.boxes:
        if int(box.cls) == 0:
            detected = True
            conf = box.conf.item()
            if conf > highest_conf:
                highest_conf = conf
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(frame, f"Person {conf:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    if detected and (time.time() - last_save > SAVE_INTERVAL):
        detection_id += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_path = f"detections/{timestamp}.jpg"
        cv2.imwrite(img_path, frame)

        with gps_lock:
            lat = current_lat
            lon = current_lon

        data = {
            "id": detection_id,
            "time": datetime.now().strftime("%H:%M:%S"),
            "lat": lat,
            "lon": lon,
            "conf": round(highest_conf, 3),
            "image": os.path.abspath(img_path)
        }
        with open(SHARED_FILE, "w") as f:
            json.dump(data, f)

        print(f"SAVED #{detection_id} | GPS: {lat}, {lon} | Conf: {highest_conf:.2f}")
        last_save = time.time()

    cv2.imshow("LIVE FEED – Wireless GPS", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()