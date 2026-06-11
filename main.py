import cv2
import os
import time
import numpy as np
import argparse
from collections import Counter

# ---------- MODEL PATHS ----------

BASE_DIR = "models"

FACE_PROTO = os.path.join(BASE_DIR, "deploy.prototxt")
FACE_MODEL = os.path.join(BASE_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

AGE_PROTO = os.path.join(BASE_DIR, "age_deploy.prototxt")
AGE_MODEL = os.path.join(BASE_DIR, "age_net.caffemodel")

# ---------- CONSTANTS ----------

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

age_list = [
'(0-2)',
'(4-6)',
'(8-12)',
'(15-20)',
'(21-24)',
'(25-32)',
'(38-43)',
'(48-53)',
'(60-100)'
]

# ---------- LOAD MODELS ----------

face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)

def get_face_boxes_with_shoulders(net, frame, conf_threshold=0.7):
h, w = frame.shape[:2]

```
blob = cv2.dnn.blobFromImage(
    frame,
    1.0,
    (300, 300),
    [104, 117, 123],
    swapRB=False,
    crop=False
)

net.setInput(blob)
detections = net.forward()

boxes = []

for i in range(detections.shape[2]):
    conf = float(detections[0, 0, i, 2])

    if conf > conf_threshold:
        x1 = int(detections[0, 0, i, 3] * w)
        y1 = int(detections[0, 0, i, 4] * h)
        x2 = int(detections[0, 0, i, 5] * w)
        y2 = int(detections[0, 0, i, 6] * h)

        box_width = x2 - x1
        box_height = y2 - y1

        expand_x = int(box_width * 0.2)
        expand_y_up = int(box_height * 0.15)
        expand_y_down = int(box_height * 0.6)

        x1 = max(0, x1 - expand_x)
        y1 = max(0, y1 - expand_y_up)
        x2 = min(w - 1, x2 + expand_x)
        y2 = min(h - 1, y2 + expand_y_down)

        boxes.append((x1, y1, x2, y2))

return boxes
```

def enhance_image(face):
face_yuv = cv2.cvtColor(face, cv2.COLOR_BGR2YUV)
face_yuv[:, :, 0] = cv2.equalizeHist(face_yuv[:, :, 0])
return cv2.cvtColor(face_yuv, cv2.COLOR_YUV2BGR)

def predict_age_for_face(face_img):
if face_img is None or face_img.size == 0:
return None, 0.0

```
face_img = enhance_image(cv2.resize(face_img, (227, 227)))

blob = cv2.dnn.blobFromImage(
    face_img,
    1.0,
    (227, 227),
    MODEL_MEAN_VALUES,
    swapRB=False
)

age_net.setInput(blob)
preds = age_net.forward()

idx = preds[0].argmax()
confidence = float(preds[0][idx])

return idx, confidence
```

def webcam_age_detection(duration_seconds=3):

```
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam.")
    return

print(f"Capturing webcam for {duration_seconds} seconds...")

start = time.time()
preds = []
last_frame = None

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame.")
        break

    boxes = get_face_boxes_with_shoulders(face_net, frame)

    if len(boxes) > 0:

        x1, y1, x2, y2 = boxes[0]

        face_crop = frame[y1:y2, x1:x2]

        idx, prob = predict_age_for_face(face_crop)

        if idx is not None:

            preds.append(idx)

            label = f"{age_list[idx]} ({prob*100:.1f}%)"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

    else:
        cv2.putText(
            frame,
            "No face detected",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    last_frame = frame.copy()

    cv2.imshow("Age Detection (Capturing...)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if time.time() - start > duration_seconds:
        break

cap.release()
cv2.destroyAllWindows()

if not preds:
    print("No face detected during capture.")
    return

final_idx = Counter(preds).most_common(1)[0][0]
final_label = age_list[final_idx]

print(f"\nFinal Predicted Age Range: {final_label}")

if last_frame is not None:

    result = last_frame.copy()

    cv2.rectangle(result, (10, 10), (360, 70), (0, 0, 0), -1)

    cv2.putText(
        result,
        f"Final Age Range: {final_label}",
        (20, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    cv2.imshow("Final Prediction", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
```

if **name** == "**main**":

```
parser = argparse.ArgumentParser(
    description="Age Detection using Webcam"
)

parser.add_argument(
    "--webcam",
    action="store_true",
    help="Run age detection through webcam"
)

parser.add_argument(
    "--duration",
    type=int,
    default=3,
    help="Duration in seconds"
)

args = parser.parse_args()

if args.webcam:
    webcam_age_detection(duration_seconds=args.duration)
else:
    print("Use --webcam to start webcam detection")
    print("Example:")
    print("python main.py --webcam --duration 4")
```
