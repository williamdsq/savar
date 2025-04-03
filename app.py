from flask import Flask, Response, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import cvzone
from datetime import datetime, timedelta
import os

app = Flask(__name__)
CORS(app)

# Carrega modelo YOLOv8
model = YOLO('yolov8n.pt')

# Classe 0 = pessoa, classe 67 = celular (COCO)
PERSON_CLASS_ID = 0
CELLPHONE_CLASS_ID = 67

# Inicializa webcam
cap = cv2.VideoCapture("rtsp://admin:2i7xw458@192.168.0.105:554/cam/realmonitor?channel=1&subtype=0")

# Controle de alerta e tempo
alert_triggered = False
alert_start_time = None

def gen_frames():
    global alert_triggered, alert_start_time
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1080, 720))

        results = model(frame)[0]

        alert_triggered = False
        person_boxes = []
        cellphone_boxes = []

        for box in results.boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            if cls == PERSON_CLASS_ID:
                person_boxes.append((x1, y1, x2, y2))
                label = "Pessoa"
                color = (0, 255, 0)
            elif cls == CELLPHONE_CLASS_ID:
                cellphone_boxes.append((x1, y1, x2, y2))
                label = "Celular"
                color = (0, 0, 255)
            else:
                continue

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cvzone.putTextRect(frame, f'{label}', (x1, y1 - 10),
                               colorR=color, scale=1, thickness=1)

        # Verifica se há celular dentro de uma área de pessoa
        for (x1p, y1p, x2p, y2p) in person_boxes:
            for (x1c, y1c, x2c, y2c) in cellphone_boxes:
                if (x1c >= x1p and y1c >= y1p and x2c <= x2p and y2c <= y2p):
                    alert_triggered = True

                    # Cria pasta de imagens se não existir
                    os.makedirs("imagens", exist_ok=True)

                    # Inicia cronômetro se for primeiro frame com alerta
                    if alert_start_time is None:
                        alert_start_time = datetime.now()
                    else:
                        elapsed = datetime.now() - alert_start_time
                        if elapsed > timedelta(seconds=3):
                            filename = f'imagens/alert_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg'
                            cv2.imwrite(filename, frame)
                            with open("log.txt", "a") as f:
                                f.write(f"Alerta: Celular detectado com pessoa - {datetime.now()}\n")
                            alert_start_time = None
                else:
                    alert_start_time = None

        # Codifica e envia frame para o navegador
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/check_alert')
def check_alert():
    global alert_triggered
    return jsonify(alert=alert_triggered)

@app.route('/get_logs')
def get_logs():
    logs = []
    try:
        with open("log.txt", "r") as f:
            for line in f:
                if "Alerta:" in line:
                    parts = line.split(" - ")
                    logs.append({"message": parts[0], "timestamp": parts[1].strip()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify(logs=logs)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
