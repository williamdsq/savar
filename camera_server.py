from flask import Flask, Response
import cv2

app = Flask(__name__)
cap = cv2.VideoCapture("rtsp://admin:2i7xw458@192.168.0.105:554/cam/realmonitor?channel=1&subtype=0")

def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
