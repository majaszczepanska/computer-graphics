from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Enhance contrast (CLAHE) and reduce noise before pose inference."""
    denoised = cv2.GaussianBlur(frame, (3, 3), 0)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_ch)

    enhanced_lab = cv2.merge((l_eq, a_ch, b_ch))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


# load pretrained YOLOv8 pose estimation model
model = YOLO("yolov8n-pose.pt")

video_path = Path(__file__).parent / "squat_sample.mp4"
cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    raise FileNotFoundError(f"Video file not found: '{video_path}'!")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    processed_frame = preprocess_frame(frame)
    results = model.predict(processed_frame, conf=0.5, verbose=False)

    annotated_frame = results[0].plot()
    cv2.imshow("YOLOv8 Pose Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()