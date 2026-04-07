"""
main.py
-------
Runs real-time gesture prediction from webcam using:
1) MediaPipe Hand Landmarker
2) Trained Random Forest model from trainer.py

Usage:
	py main.py --model models/gesture_rf.joblib --camera 0 --threshold 0.70 --window 5 --volume_cooldown 0.60 --play_pause_cooldown 2.00
"""

from __future__ import annotations

import argparse
import os
import time
import urllib.request
from collections import Counter, deque
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pyautogui
import pygetwindow as gw
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
						help="Minimum seconds between volume up/down presses")
	parser.add_argument("--play_pause_cooldown", type=float, default=1.50,
						help="Minimum seconds between palm play/pause presses")
	return parser.parse_args()


def normalize(landmarks: list[float]) -> list[float]:
	coords = np.array(landmarks, dtype=np.float32).reshape(21, 2)
	coords -= coords[0]
	max_val = float(np.max(np.abs(coords)))
	if max_val > 0:
		coords /= max_val
	return coords.flatten().tolist()


def wrist_xy(hand_landmarks: list) -> np.ndarray:
	return np.array([hand_landmarks[0].x, hand_landmarks[0].y], dtype=np.float32)


def hand_area(hand_landmarks: list) -> float:
	xs = [lm.x for lm in hand_landmarks]
	ys = [lm.y for lm in hand_landmarks]
	return float((max(xs) - min(xs)) * (max(ys) - min(ys)))


def hand_bbox(hand_landmarks: list, frame_shape: tuple[int, int, int]) -> tuple[int, int, int, int]:
	h, w = frame_shape[:2]
	xs = [lm.x for lm in hand_landmarks]
	ys = [lm.y for lm in hand_landmarks]
	x1 = max(0, int(min(xs) * w))
	y1 = max(0, int(min(ys) * h))
	x2 = min(w - 1, int(max(xs) * w))
	y2 = min(h - 1, int(max(ys) * h))

	# Pad to a square so it is easy to see which hand is interpreted.
	box_w = x2 - x1
	box_h = y2 - y1
	side = max(box_w, box_h)
	cx = (x1 + x2) // 2
	cy = (y1 + y2) // 2
	half = max(1, side // 2)

	sx1 = max(0, cx - half)
	sy1 = max(0, cy - half)
	sx2 = min(w - 1, cx + half)
	sy2 = min(h - 1, cy + half)
	return sx1, sy1, sx2, sy2


def extract_landmarks(
	image_bgr: np.ndarray,
	detector: Any,
	tracked_wrist: np.ndarray | None,
) -> tuple[list[float] | None, np.ndarray | None, tuple[int, int, int, int] | None]:
	image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
	mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
	result = detector.detect(mp_image)
	if not result.hand_landmarks:
		return None, None, None

	hands = result.hand_landmarks
	if len(hands) == 1:
		hand = hands[0]
		new_wrist = wrist_xy(hand)
		bbox = hand_bbox(hand, image_bgr.shape)
		return [coord for lm in hand for coord in (lm.x, lm.y)], new_wrist, bbox

	# Keep tracking one hand across frames when two are visible.
	if tracked_wrist is not None:
		closest_idx = int(np.argmin([
			np.linalg.norm(wrist_xy(hand) - tracked_wrist) for hand in hands
		]))
		hand = hands[closest_idx]
	else:
		largest_idx = int(np.argmax([hand_area(hand) for hand in hands]))
		hand = hands[largest_idx]

	new_wrist = wrist_xy(hand)
	bbox = hand_bbox(hand, image_bgr.shape)
	return [coord for lm in hand for coord in (lm.x, lm.y)], new_wrist, bbox


def majority_vote(labels: deque[str]) -> str:
	if not labels:
		return "unknown"
	return Counter(labels).most_common(1)[0][0]


def minimize_spotify_window() -> bool:
	# Finds a Spotify window by title and minimizes it.
	try:
		windows = [w for w in gw.getWindowsWithTitle("Spotify") if w.title]
		if not windows:
			return False
		windows[0].minimize()
		return True
	except Exception:
		return False


def main() -> None:
	args = parse_args()
	ensure_landmarker_model()

	artifact = load(args.model)
	model = artifact["model"]
	feature_columns = artifact["feature_columns"]
	class_names = artifact["class_names"]

	options = HandLandmarkerOptions(
		base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
		num_hands=2,
		min_hand_detection_confidence=0.4,
		min_hand_presence_confidence=0.4,
		min_tracking_confidence=0.4,
	)

	cap = cv2.VideoCapture(args.camera)
	if not cap.isOpened():
		raise RuntimeError(f"Cannot open camera {args.camera}")

	history: deque[str] = deque(maxlen=max(1, args.window))
	last_palm_action_ts = 0.0
	last_fist_action_ts = 0.0
	last_volume_action_ts = 0.0
	last_media_action = "none"
	tracked_wrist: np.ndarray | None = None

	with vision.HandLandmarker.create_from_options(options) as detector:
		while True:
			ok, frame = cap.read()
			if not ok:
				break

			frame = cv2.flip(frame, 1)
			landmarks, tracked_wrist, selected_bbox = extract_landmarks(frame, detector, tracked_wrist)

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

			if selected_bbox is not None:
				x1, y1, x2, y2 = selected_bbox
				cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
				cv2.putText(frame, "Tracked hand", (x1, max(18, y1 - 8)),
							cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

			history.append(raw_label)
			smooth_label = majority_vote(history)

			now = time.time()
			palm_cooldown = max(0.0, args.play_pause_cooldown)
			if smooth_label == "palm" and now - last_palm_action_ts >= palm_cooldown:
				# Toggle playback for the active media app (Spotify).
				pyautogui.press("playpause")
				last_palm_action_ts = now
				last_media_action = "PLAY/PAUSE"
			elif smooth_label == "fist" and now - last_fist_action_ts >= max(0.0, args.volume_cooldown):
				minimized = minimize_spotify_window()
				last_fist_action_ts = now
				last_media_action = "MINIMIZE" if minimized else "SPOTIFY NOT FOUND"
			elif smooth_label == "like" and now - last_volume_action_ts >= max(0.0, args.volume_cooldown):
				pyautogui.press("volumeup")
				last_volume_action_ts = now
				last_media_action = "VOLUME UP"
			elif smooth_label == "dislike" and now - last_volume_action_ts >= max(0.0, args.volume_cooldown):
				pyautogui.press("volumedown")
				last_volume_action_ts = now
				last_media_action = "VOLUME DOWN"

			cv2.rectangle(frame, (0, 0), (frame.shape[1], 80), (20, 20, 20), -1)
			cv2.putText(frame, f"Prediction: {smooth_label}", (12, 30),
						cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 0), 2)
			cv2.putText(frame, f"Confidence: {confidence:.2f}", (12, 62),
						cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
			cv2.putText(frame, f"Media action: {last_media_action}", (300, 62),
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
