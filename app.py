import os
import cv2
import csv
import time
import random
import logging
import threading
import urllib.request
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, render_template, Response, jsonify, send_file, redirect, url_for, request
from ultralytics import YOLO

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(BASE_DIR, "hazard_logs.csv")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
os.makedirs(TEMPLATES_DIR, exist_ok=True)

# Initial baseline counts
BASE_POTHOLES = 42
BASE_GARBAGE = 19

# Global real-time GPS coordinates (updated dynamically via browser/device)
current_live_gps = {"lat": 16.1809, "lon": 81.1303}

# 1. Initialize Log File
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Hazard_Type", "Severity", "Confidence", "Latitude", "Longitude"])

# 2. Camera Streaming Thread
class FastWebcamStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()
            time.sleep(0.01)

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

camera_stream = FastWebcamStream(src=0).start()

# 3. Model Initializations & Auto-Download
POTHOLE_WEIGHT_FILE = os.path.join(BASE_DIR, "best.pt")
GARBAGE_WEIGHT_FILE = os.path.join(BASE_DIR, "garbage_best.pt")

if not os.path.exists(POTHOLE_WEIGHT_FILE) or os.path.getsize(POTHOLE_WEIGHT_FILE) <= 1024:
    print("📥 Downloading pre-trained pothole model...")
    try:
        url = "https://huggingface.co/peterhdd/pothole-detection-yolov8/resolve/main/best.pt"
        urllib.request.urlretrieve(url, POTHOLE_WEIGHT_FILE)
        print("✅ Downloaded best.pt successfully!")
    except Exception as e:
        print(f"⚠️ Auto-download failed: {e}")

try:
    if os.path.exists(POTHOLE_WEIGHT_FILE) and os.path.getsize(POTHOLE_WEIGHT_FILE) > 1024:
        pothole_model = YOLO(POTHOLE_WEIGHT_FILE)
        print("✅ Loaded Custom Pothole Weights (best.pt)")
    else:
        raise ValueError("File is missing or empty.")
except Exception as e:
    print(f"⚠️ Error loading best.pt: {e}. Falling back to yolov8n.pt")
    pothole_model = YOLO("yolov8n.pt")

try:
    if os.path.exists(GARBAGE_WEIGHT_FILE) and os.path.getsize(GARBAGE_WEIGHT_FILE) > 1024:
        garbage_model = YOLO(GARBAGE_WEIGHT_FILE)
        print("✅ Loaded Custom Garbage Pile Weights (garbage_best.pt)")
    else:
        raise ValueError("File is missing or empty.")
except Exception as e:
    print(f"⚠️ Error loading garbage_best.pt: {e}. Falling back to yolov8n.pt")
    garbage_model = YOLO("yolov8n.pt")

base_model = YOLO("yolov8n.pt")
last_logged_time = 0

def generate_frames():
    global last_logged_time

    while True:
        frame = camera_stream.read()
        if frame is None:
            time.sleep(0.05)
            continue

        h_frame, w_frame, _ = frame.shape
        annotated_frame = frame.copy()
        current_time = time.time()
        detected_hazards = []

        # --- A. DETECT OPERATOR / PERSON TO PREVENT FALSE ALARMS ---
        base_results = base_model(frame, conf=0.35, imgsz=320, verbose=False)
        person_boxes = []
        for box in base_results[0].boxes:
            cls_name = base_model.names[int(box.cls[0])].lower()
            if cls_name == "person":
                person_boxes.append(list(map(int, box.xyxy[0])))

        def inside_person(box_coords):
            bx1, by1, bx2, by2 = box_coords
            bcx, bcy = (bx1 + bx2) // 2, (by1 + by2) // 2
            for px1, py1, px2, py2 in person_boxes:
                if px1 <= bcx <= px2 and py1 <= bcy <= py2:
                    return True
            return False

        # --- B. HIGH-ACCURACY POTHOLE DETECTION ---
        p_res = pothole_model(frame, conf=0.42, imgsz=416, verbose=False)
        is_fallback_model = (pothole_model.model.names.get(0) == "person")

        for box in p_res[0].boxes:
            cls_id = int(box.cls[0])
            if is_fallback_model and cls_id == 0:
                continue

            coords = list(map(int, box.xyxy[0]))
            x1, y1, x2, y2 = coords
            box_w = x2 - x1
            box_h = y2 - y1

            if y1 < int(h_frame * 0.35):
                continue
            if box_h > (box_w * 2.2):
                continue
            if inside_person(coords):
                continue

            conf = float(box.conf[0])
            area = box_w * box_h
            detected_hazards.append({
                "type": "Pothole",
                "bbox": coords,
                "conf": conf,
                "area": area
            })

        # --- C. CUSTOM GARBAGE PILE DETECTION ---
        g_res = garbage_model(frame, conf=0.48, imgsz=640, verbose=False)
        for box in g_res[0].boxes:
            coords = list(map(int, box.xyxy[0]))
            if inside_person(coords):
                continue

            x1, y1, x2, y2 = coords
            box_width = x2 - x1
            box_height = y2 - y1
            area = box_width * box_height

            if y1 < int(h_frame * 0.25):
                continue

            conf = float(box.conf[0])
            detected_hazards.append({
                "type": "Garbage Pile",
                "bbox": coords,
                "conf": conf,
                "area": area
            })

        # --- D. DRAW ANNOTATIONS ---
        for hazard in detected_hazards:
            h_type = hazard["type"]
            conf = hazard["conf"]
            x1, y1, x2, y2 = hazard["bbox"]

            color = (0, 0, 255) if h_type == "Pothole" else (0, 165, 255)

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            label_text = f"{h_type} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated_frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(annotated_frame, label_text, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # --- E. CSV TELEMETRY LOGGING USING LIVE DETECTED GPS ---
        if detected_hazards and (current_time - last_logged_time > 1.2):
            top = max(detected_hazards, key=lambda x: x["conf"])
            area = top["area"]
            conf = top["conf"]

            severity = "CRITICAL" if (area > 20000 or conf > 0.70) else ("MODERATE" if area > 8000 else "MINOR")
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Uses live detected coordinates from client browser
            lat = round(current_live_gps["lat"] + random.uniform(-0.00003, 0.00003), 6)
            lon = round(current_live_gps["lon"] + random.uniform(-0.00003, 0.00003), 6)

            with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, top["type"], severity, f"{conf:.2f}", lat, lon])

            last_logged_time = current_time

        ret, buffer = cv2.imencode(".jpg", annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
        if not ret:
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

# --- Routes ---

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        return redirect(url_for("dashboard_view"))
    return render_template("login.html")

@app.route("/dashboard")
def dashboard_view():
    return render_template("dashboard.html")

@app.route("/camera")
def camera_view():
    return render_template("camera.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

# --- API Endpoints ---

@app.route("/api/update_gps", methods=["POST"])
def update_gps():
    global current_live_gps
    data = request.get_json(silent=True) or {}
    if "lat" in data and "lon" in data:
        try:
            current_live_gps["lat"] = float(data["lat"])
            current_live_gps["lon"] = float(data["lon"])
        except ValueError:
            pass
    return jsonify({"status": "success", "gps": current_live_gps})

@app.route("/api/metrics")
def get_metrics():
    pothole_detected = 0
    garbage_detected = 0

    if os.path.exists(LOG_FILE):
        try:
            df = pd.read_csv(LOG_FILE, on_bad_lines="skip")
            df.columns = df.columns.str.strip()
            if not df.empty:
                pothole_detected = int((df["Hazard_Type"].astype(str).str.upper() == "POTHOLE").sum())
                garbage_detected = int((df["Hazard_Type"].astype(str).str.upper().str.contains("GARBAGE")).sum())
        except Exception as e:
            print(f"Error reading metrics from CSV: {e}")

    return jsonify({
        "potholes": BASE_POTHOLES + pothole_detected,
        "garbage": BASE_GARBAGE + garbage_detected
    })

@app.route("/api/logs")
def get_logs():
    if os.path.exists(LOG_FILE):
        try:
            df = pd.read_csv(LOG_FILE, on_bad_lines="skip")
            df.columns = df.columns.str.strip()
            if not df.empty:
                return jsonify({
                    "total_count": len(df),
                    "pothole_count": int((df["Hazard_Type"].astype(str).str.upper() == "POTHOLE").sum()),
                    "garbage_count": int((df["Hazard_Type"].astype(str).str.upper().str.contains("GARBAGE")).sum()),
                    "critical_count": int((df["Severity"].astype(str).str.upper() == "CRITICAL").sum()),
                    "logs": df.tail(30).to_dict(orient="records")
                })
        except Exception as e:
            print(f"Error reading CSV: {e}")
    return jsonify({"total_count": 0, "pothole_count": 0, "garbage_count": 0, "critical_count": 0, "logs": []})

@app.route("/api/potholes")
def get_potholes_table():
    records = []
    if os.path.exists(LOG_FILE):
        try:
            df = pd.read_csv(LOG_FILE, on_bad_lines="skip")
            df.columns = df.columns.str.strip()
            if not df.empty:
                potholes_df = df[df["Hazard_Type"].astype(str).str.upper() == "POTHOLE"]
                for _, row in potholes_df.iterrows():
                    try:
                        conf_val = float(row.get("Confidence", 0.85)) * 100
                        conf_str = f"{conf_val:.1f}%"
                    except Exception:
                        conf_str = f"{row.get('Confidence', '85')}%"

                    records.append({
                        "timestamp": str(row.get("Timestamp", "N/A")),
                        "bus_id": "AP 123",
                        "severity": str(row.get("Severity", "MODERATE")),
                        "confidence": conf_str,
                        "lat": str(row.get("Latitude", current_live_gps["lat"])),
                        "lng": str(row.get("Longitude", current_live_gps["lon"]))
                    })
        except Exception as e:
            print(f"Error reading potholes table: {e}")
    return jsonify(records)

@app.route("/download/csv")
def download_csv():
    if os.path.exists(LOG_FILE):
        return send_file(LOG_FILE, as_attachment=True)
    return "No CSV found", 404

if __name__ == "__main__":
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)