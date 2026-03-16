"""
main.py
-------
Runs real-time gesture prediction from webcam using:
1) MediaPipe Hand Landmarker
2) Trained Random Forest model from trainer.py

Usage:
	py main.py
	py main.py --model models/gesture_rf.joblib --camera 0 --threshold 0.70
    py main.py --model models/gesture_rf.joblib --camera 0 --threshold 0.70 --window 5 --volume_cooldown 0.60 
"""

from __future__ import annotations

import argparse
import os
import time
import urllib.request
from collections import Counter, deque

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pyautogui
from joblib import load
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarkerOptions

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"


def ensure_landmarker_model() -> None:
	if not os.path.exists(MODEL_PATH):
		print("Downloading hand landmarker model...")
		urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Live webcam gesture recognition")
	parser.add_argument("--model", type=str, default="models/gesture_rf.joblib",
						help="Path to trained model artifact")
	parser.add_argument("--camera", type=int, default=0,
						help="Camera index (default: 0)")
	parser.add_argument("--threshold", type=float, default=0.70,
						help="Confidence threshold for accepting predictions")
	parser.add_argument("--window", type=int, default=5,
						help="Smoothing window size in frames")
	parser.add_argument("--volume_cooldown", type=float, default=0.60,
						help="Minimum seconds between volume key presses")
	return parser.parse_args()


def normalize(landmarks: list[float]) -> list[float]:
	coords = np.array(landmarks, dtype=np.float32).reshape(21, 2)
	coords -= coords[0]
	max_val = float(np.max(np.abs(coords)))
	if max_val > 0:
		coords /= max_val
	return coords.flatten().tolist()


def extract_landmarks(image_bgr: np.ndarray, detector: vision.HandLandmarker) -> list[float] | None:
	image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
	mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
	result = detector.detect(mp_image)
	if not result.hand_landmarks:
		return None
	hand = result.hand_landmarks[0]
	return [coord for lm in hand for coord in (lm.x, lm.y)]


def majority_vote(labels: deque[str]) -> str:
	if not labels:
		return "unknown"
	return Counter(labels).most_common(1)[0][0]


def main() -> None:
	args = parse_args()
	ensure_landmarker_model()

	artifact = load(args.model)
	model = artifact["model"]
	feature_columns = artifact["feature_columns"]
	class_names = artifact["class_names"]

	options = HandLandmarkerOptions(
		base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
		num_hands=1,
		min_hand_detection_confidence=0.4,
		min_hand_presence_confidence=0.4,
		min_tracking_confidence=0.4,
	)

	cap = cv2.VideoCapture(args.camera)
	if not cap.isOpened():
		raise RuntimeError(f"Cannot open camera {args.camera}")

	history: deque[str] = deque(maxlen=max(1, args.window))
	last_volume_action_ts = 0.0
	last_volume_action = "none"

	with vision.HandLandmarker.create_from_options(options) as detector:
		while True:
			ok, frame = cap.read()
			if not ok:
				break

			frame = cv2.flip(frame, 1)
			landmarks = extract_landmarks(frame, detector)

			raw_label = "unknown"
			confidence = 0.0

			if landmarks is not None:
				features = normalize(landmarks)
				sample_df = pd.DataFrame([features], columns=feature_columns)
				probs = model.predict_proba(sample_df)[0]
				best_idx = int(np.argmax(probs))
				confidence = float(probs[best_idx])
				candidate = class_names[best_idx]
				raw_label = candidate if confidence >= args.threshold else "unknown"

			history.append(raw_label)
			smooth_label = majority_vote(history)

			now = time.time()
			if now - last_volume_action_ts >= max(0.0, args.volume_cooldown):
				if smooth_label == "like":
					pyautogui.press("volumeup")
					last_volume_action_ts = now
					last_volume_action = "UP"
				elif smooth_label == "dislike":
					pyautogui.press("volumedown")
					last_volume_action_ts = now
					last_volume_action = "DOWN"

			cv2.rectangle(frame, (0, 0), (frame.shape[1], 80), (20, 20, 20), -1)
			cv2.putText(frame, f"Prediction: {smooth_label}", (12, 30),
						cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 0), 2)
			cv2.putText(frame, f"Confidence: {confidence:.2f}", (12, 62),
						cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
			cv2.putText(frame, f"Volume action: {last_volume_action}", (300, 62),
						cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
			cv2.putText(frame, "Press Q to quit", (frame.shape[1] - 180, frame.shape[0] - 12),
						cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

			cv2.imshow("Gesture Predictor", frame)
			key = cv2.waitKey(1) & 0xFF
			if key == ord("q"):
				break

	cap.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
