from flask import Flask, render_template, Response
from ultralytics import YOLO
import cv2
import math
import pygame
import time

app = Flask(__name__)
cap = None
model = None
classNames = [
    "Excavator", "Gloves", "Hardhat", "Ladder", "Mask", "NO-Hardhat", "NO-Mask", "NO-Safety Vest",
    "Person", "SUV", "Safety Cone", "Safety Vest", "bus", "dump truck", "fire hydrant", "machinery",
    "mini-van", "sedan", "semi", "trailer", "truck and trailer", "truck", "van", "vehicle", "wheel loader", "No Gloves"
]

# Define colors for each class
class_colors = {
    "Excavator": (255, 0, 0),        # Red
    "Gloves": (0, 255, 0),            # Green
    "Hardhat": (0, 0, 255),           # Blue
    "Ladder": (255, 255, 0),          # Yellow
    "Mask": (255, 0, 255),            # Magenta
    "NO-Hardhat": (0, 255, 255),      # Cyan
    "NO-Mask": (255, 140, 0),         # Orange
    "NO-Safety Vest": (128, 0, 128),  # Purple
    "Person": (255, 69, 0),           # Orange Red
    "SUV": (0, 255, 255),             # Aqua
    "Safety Cone": (255, 215, 0),     # Gold
    "Safety Vest": (0, 128, 128),     # Teal
    "bus": (128, 0, 0),                # Maroon
    "dump truck": (0, 128, 128),       # Olive
    "fire hydrant": (128, 128, 0),     # Navy
    "machinery": (128, 128, 128),      # Gray
    "mini-van": (0, 0, 128),           # Navy
    "sedan": (0, 128, 0),              # Green
    "semi": (0, 0, 128),               # Navy
    "trailer": (128, 128, 0),          # Olive
    "truck and trailer": (128, 128, 0), # Olive
    "truck": (128, 0, 128),            # Purple
    "van": (0, 128, 128),              # Teal
    "vehicle": (0, 0, 128),            # Navy
    "wheel loader": (128, 0, 0),       # Maroon
    "No Gloves": (128, 0, 128)         # Purple
}


def init_camera():
    global cap
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

def init_model():
    global model
    model = YOLO('best (1).pt')

def init_sound():
    pygame.mixer.init()
    pygame.mixer.music.load("mixkit-classic-short-alarm-993.wav")

@app.route('/')
def admin():
    if cap is not None:
        cap.release()
    return render_template('admin.html')

def gen(camera):
    init_sound()
    alarm_active_hardhat = False
    alarm_active_mask = False
    alarm_sound_interval = 5  # seconds
    last_alarm_time_hardhat = 0
    last_alarm_time_mask = 0

    while True:
        success, img = camera.read()
        results = model(img, stream=True)

        for r in results:
            boxes = r.boxes

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                confidence = round(float(box.conf[0]), 2)
                cls = int(box.cls[0])
                class_name = classNames[cls]

                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1

                if class_name in class_colors:
                    color = class_colors[class_name]
                else:
                    color = (255, 255, 255)  # Default color for unknown classes

                thickness = 2

                if class_name == "NO-Hardhat":
                    if not alarm_active_hardhat:
                        if time.time() - last_alarm_time_hardhat >= alarm_sound_interval:
                            pygame.mixer.music.load("mixkit-classic-short-alarm-993.wav")
                            pygame.mixer.music.play()
                            last_alarm_time_hardhat = time.time()
                            alarm_active_hardhat = True
                    else:
                        if time.time() - last_alarm_time_hardhat >= alarm_sound_interval:
                            pygame.mixer.music.stop()
                            alarm_active_hardhat = False

                elif class_name == "NO-Mask":
                    if not alarm_active_mask:
                        if time.time() - last_alarm_time_mask >= alarm_sound_interval + 1:
                            pygame.mixer.music.load("mixkit-data-scaner-2847.wav")
                            pygame.mixer.music.play()
                            last_alarm_time_mask = time.time()
                            alarm_active_mask = True
                    else:
                        if time.time() - last_alarm_time_mask >= alarm_sound_interval + 1:
                            pygame.mixer.music.stop()
                            alarm_active_mask = False

                text = f"{class_name}: {confidence}"
                cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(img, text, org, font, fontScale, color, thickness)

            ret, frame = cv2.imencode('.jpg', img)
            if not ret:
                break

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame.tobytes() + b'\r\n')

@app.route('/camera')
def camera():
    init_camera()
    init_model()
    return render_template('camera.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen(cap), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
