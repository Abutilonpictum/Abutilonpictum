from flask import Flask, render_template, jsonify, Response
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import argparse
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ==================== Configuration ====================
model_path = "best.pt"
video_path = "parking_video.mp4"
label_path = Path('dataset') / 'labels' / 'parking_video.txt'
conf = 0.4
imgsz = 640
iou_threshold = 0.6
TOTAL_SPOTS = 41
PORT = 5000
# ========================================================

parking_status = {
    "total": TOTAL_SPOTS,
    "occupied": 0,
    "available": TOTAL_SPOTS,
    "rate": 0.0,
    "spots": [{"id": i+1, "occupied": False, "confidence": 0.0} for i in range(TOTAL_SPOTS)]
}

model = None
parking_areas = []
use_camera = False
cap = None

def parse_args():
    global use_camera
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', action='store_true')
    use_camera = parser.parse_args().camera

def load_parking_areas(label_path, w, h, class_id=0):
    areas = []
    if not label_path.exists():
        print(f"coordinate file missing: {label_path}")
        return areas

    with open(label_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        parts = list(map(float, line.strip().split()))
        if len(parts) < 5 or int(parts[0]) != class_id:
            continue

        # restore YOLO normalized coordinates to pixel coordinates
        cx, cy, pw, ph = parts[1]*w, parts[2]*h, parts[3]*w, parts[4]*h
        x1, y1 = int(cx - pw/2), int(cy - ph/2)
        x2, y2 = int(cx + pw/2), int(cy + ph/2)
        
        # save both rectangle (for drawing) and polygon (for precise calculation)
        areas.append({
            "rect": (x1, y1, x2, y2),
            "poly": np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        })
    print(f"✅ successfully loaded {len(areas)} parking spaces")
    return areas

def overlap(park_poly, car_box):
    try:
        x1, y1, x2, y2 = map(int, car_box)
        car_poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        
        # calculate the area of the vehicle
        car_area = (x2 - x1) * (y2 - y1)
        if car_area <= 0: return 0.0

        # use OpenCV to calculate the intersection area of the polygon
        inter_area = cv2.intersectConvexConvex(park_poly, car_poly)[0]
        return inter_area / car_area if inter_area > 0 else 0.0
    except Exception as e:
        return 0.0

def init():
    global model, parking_areas, cap
    print("--- initializing system ---")
    model = YOLO(model_path)
    
    source = 0 if use_camera else video_path
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"error: cannot open video source {source}")
        return

    # loop to get the width and height of the video, to prevent some video stream loading delay
    w = h = 0
    for _ in range(10): 
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w > 0: break
        cv2.waitKey(10)

    if w == 0:
        print("error: cannot get the width and height of the video, please check if the video file is damaged")
        return
        
    print(f"successfully loaded the video: {w}x{h}")
    parking_areas = load_parking_areas(label_path, w, h)
    print(f"successfully converted the parking space coordinates, total {len(parking_areas)} parking spaces")

def process_frame(frame):
    if frame is None: return None
    
    # execute detection
    results = model(frame, conf=conf, imgsz=imgsz, verbose=False, half=True)
    
    # core fix: get OBB boxes first, if not, get normal boxes
    car_boxes = []
    if hasattr(results[0], 'obb') and results[0].obb is not None and len(results[0].obb) > 0:
        car_boxes = [ob.xyxy[0] for ob in results[0].obb]
    elif results[0].boxes is not None and len(results[0].boxes) > 0:
        car_boxes = [bx.xyxy[0] for bx in results[0].boxes]

    # iterate over the parking spaces to determine the occupancy status
    occupied_count = 0
    for i in range(min(TOTAL_SPOTS, len(parking_areas))):
        park = parking_areas[i]
        max_overlap = 0.0
        
        # calculate the maximum overlap ratio between the parking space and all detected cars
        for car_box in car_boxes:
            ratio = overlap(park["poly"], car_box)
            if ratio > max_overlap:
                max_overlap = ratio

        # determination logic
        is_occupied = max_overlap > iou_threshold
        parking_status["spots"][i]["occupied"] = bool(is_occupied)
        parking_status["spots"][i]["confidence"] = float(max_overlap)
        
        if is_occupied:
            occupied_count += 1

        # draw debug information on the screen
        color = (0, 0, 255) if is_occupied else (0, 255, 0)
        px1, py1, px2, py2 = park["rect"]
        cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)
        cv2.putText(frame, f"{i+1}", (px1, py1+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # !!! extremely important: synchronize the global JSON data, for the front-end interface call !!!
    parking_status["occupied"] = occupied_count
    parking_status["available"] = len(parking_areas) - occupied_count
    parking_status["rate"] = occupied_count / len(parking_areas) if len(parking_areas) > 0 else 0
    
    # overlay the original YOLO detection boxes and return
    return results[0].plot(img=frame)

def gen():
    print("--- video stream thread started ---")
    while True:
        success, frame = cap.read()
        if not success:
            if not use_camera:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else: break
        
        try:
            processed_frame = process_frame(frame)
            if processed_frame is None: continue
            
            ret, jpg = cv2.imencode('.jpg', processed_frame)
            if not ret: continue
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n')
        except Exception as e:
            print(f"error: occurred while processing the frame: {e}")
            continue

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/parking/status')
def status():
    return jsonify(parking_status)

if __name__ == '__main__':
    parse_args()
    init()
    app.run(host='0.0.0.0', port=PORT, debug=True, threaded=True)