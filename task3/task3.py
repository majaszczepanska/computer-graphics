from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO


# COCO keypoint indices:
# Left side:  shoulder=5, elbow=7, wrist=9, hip=11, knee=13, ankle=15
# Right side: shoulder=6, elbow=8, wrist=10, hip=12, knee=14, ankle=16
EXERCISE_CONFIG = {
    "squat": {
        "joints": (11, 13, 15),   # left hip, knee, ankle
        "down_angle": 95.0,
        "up_angle": 160.0,
    },
    "bicep_curl": {
        "joints": (5, 7, 9),      # left shoulder, elbow, wrist
        "down_angle": 50.0,
        "up_angle": 155.0,
    },
}


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Denoise and apply CLAHE contrast enhancement for better keypoint detection."""
    denoised = cv2.GaussianBlur(frame, (3, 3), 0)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_ch)

    enhanced_lab = cv2.merge((l_eq, a_ch, b_ch))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Calculate angle ABC (in degrees) between three 2D points."""
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return float(np.degrees(angle))


def run_rep_counter(exercise: str = "squat", input_name: str = "squat_sample.mp4"):
    cfg = EXERCISE_CONFIG[exercise]
    idx_a, idx_b, idx_c = cfg["joints"]

    model = YOLO("yolov8n-pose.pt")

    video_path = Path(__file__).parent / input_name
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Video file not found: '{video_path}'!")

    rep_count = 0
    stage = "UP"

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = preprocess_frame(frame)
        results = model.predict(processed_frame, conf=0.5, verbose=False)
        annotated = results[0].plot()

        if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
            kpts = results[0].keypoints.xy[0].cpu().numpy()
            confs = results[0].keypoints.conf[0].cpu().numpy()

            if confs[idx_a] > 0.5 and confs[idx_b] > 0.5 and confs[idx_c] > 0.5:
                pt_a, pt_b, pt_c = kpts[idx_a], kpts[idx_b], kpts[idx_c]
                angle = calculate_angle(pt_a, pt_b, pt_c)

                if angle < cfg["down_angle"]:
                    stage = "DOWN"
                elif angle > cfg["up_angle"] and stage == "DOWN":
                    stage = "UP"
                    rep_count += 1

                cv2.putText(
                    annotated,
                    f"{int(angle)} deg",
                    tuple(pt_b.astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )

        cv2.rectangle(annotated, (15, 15), (310, 105), (20, 20, 20), -1)
        cv2.putText(annotated, f"EXERCISE: {exercise.upper()}", (25, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)
        cv2.putText(annotated, f"REPS: {rep_count}  |  STAGE: {stage}", (25, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        cv2.imshow("Exercise Repetition Counter", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_rep_counter(exercise="squat", input_name="squat_sample.mp4")