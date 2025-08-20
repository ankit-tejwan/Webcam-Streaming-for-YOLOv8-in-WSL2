# 🎥 Webcam Streaming for YOLOv8 in WSL2

## 🚨 Problem
When using **Ubuntu inside WSL2** on Windows 10/11, external devices like webcams are **not supported directly**.  
Running apps like `cheese`, OpenCV, or YOLO will give errors like:

Error during camera setup: No device found

<img width="1743" height="1049" alt="Screenshot 2025-08-19 171129" src="https://github.com/user-attachments/assets/a001efce-9527-4d43-8e96-7245b733b47d" />


## 💡 Solution
Use a **two-step workaround**:
1. Run a **Flask streaming server** on the Windows host to capture the webcam.
<img width="1408" height="894" alt="Screenshot 2025-08-19 171213" src="https://github.com/user-attachments/assets/1c24aed6-e26d-48eb-b111-077647bbf6a6" />



## ⚙️ Step A: Webcam Streaming Server (Windows host)
Create `webcam_server.py` on **Windows**:

```python
from flask import Flask, Response
import cv2

app = Flask(__name__)
cap = cv2.VideoCapture(0)  # Windows webcam

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
```
#######################################################################################
Run it:

python webcam_server.py


2. Connect to that stream inside WSL2 and run **YOLOv8 inference** in real time.

Now your webcam is accessible at:
👉 http://<windows_ip>:8080/video

<img width="1527" height="1024" alt="Screenshot 2025-08-19 171352" src="https://github.com/user-attachments/assets/7f6357ef-129e-4da0-9491-76c767c7e87d" />

########################################################################################
🐧 Step B: YOLOv8 Inference (inside WSL2)

Create yolo_stream.py in Ubuntu WSL2:

```
from ultralytics import YOLO
import cv2
import torch

# Load YOLOv8 model
model = YOLO("yolov8s.pt")

# Open MJPEG stream from Windows host
cap = cv2.VideoCapture("http://<windows_ip>:8080/video")

def infer_and_display(cap):
    if not cap.isOpened():
        print("❌ Couldn't open stream.")
        return

    print("✅ Stream opened.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to grab frame.")
            break

        # Run YOLOv8 detection
        results = model.predict(
            source=frame,
            imgsz=448,
            conf=0.5,
            iou=0.5,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            verbose=False
        )

        # Annotate frame
        annotated_frame = results[0].plot()

        # Display
        cv2.imshow('YOLOv8 Detection', annotated_frame)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 Quitting...")
            break

    cap.release()
    cv2.destroyAllWindows()
```
###########################################
# Run inference
infer_and_display(cap)
Run it:
python yolo_stream.py

################## ✅ Results ######################


Webcam runs on Windows host.

YOLOv8 inference runs inside WSL2 (Ubuntu).

Works in real time.

###############################################
