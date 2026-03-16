"""
verify_landmarks.py
-------------------
Processes all images in a dataset folder, draws MediaPipe hand landmarks
on each one, and saves the annotated versions to a separate output folder.

Usage:
    python verify_landmarks.py
    python verify_landmarks.py --dataset ./custom_dataset --output ./verified
"""

import cv2
import argparse
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarkerOptions
from pathlib import Path
from tqdm import tqdm
import urllib.request
import os

MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading hand landmarker model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")


HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

def draw_landmarks(image, hand_landmarks_list):
    h, w = image.shape[:2]
    for hand_landmarks in hand_landmarks_list:
        points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
        for start, end in HAND_CONNECTIONS:
            cv2.line(image, points[start], points[end], (255, 255, 0), 2)
        for pt in points:
            cv2.circle(image, pt, 4, (0, 255, 0), -1)


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def process_dataset(dataset_path: str, output_path: str):
    ensure_model()

    dataset_root = Path(dataset_path)
    output_root  = Path(output_path)

    class_dirs = sorted([d for d in dataset_root.iterdir() if d.is_dir()])
    if not class_dirs:
        raise FileNotFoundError(f"No class subfolders found in '{dataset_root}'")

    options = HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        num_hands=1,
        min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )

    total_ok   = 0
    total_miss = 0

    with vision.HandLandmarker.create_from_options(options) as detector:
        for class_dir in class_dirs:
            label    = class_dir.name
            save_dir = output_root / label
            save_dir.mkdir(parents=True, exist_ok=True)

            image_paths = [p for p in class_dir.iterdir()
                           if p.suffix.lower() in VALID_EXTENSIONS]

            ok, miss = 0, 0

            for img_path in tqdm(image_paths, desc=f"  {label}", unit="img"):
                image_bgr = cv2.imread(str(img_path))
                if image_bgr is None:
                    miss += 1
                    continue

                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                result    = detector.detect(mp_image)

                annotated = image_bgr.copy()

                if result.hand_landmarks:
                    draw_landmarks(annotated, result.hand_landmarks)
                    ok += 1
                else:
                    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 36), (0, 0, 180), -1)
                    cv2.putText(annotated, "NO HAND DETECTED", (8, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    miss += 1

                cv2.imwrite(str(save_dir / img_path.name), annotated)

            total_ok   += ok
            total_miss += miss
            rate = 100 * ok / max(ok + miss, 1)
            print(f"    {label}: {ok}/{ok+miss} detected ({rate:.1f}%)")

    print("Verification Complete")
    overall = 100 * total_ok / max(total_ok + total_miss, 1)
    print(f"  Detected : {total_ok}")
    print(f"  Missed   : {total_miss}")
    print(f"  Overall  : {overall:.1f}%")
    print(f"\n  Annotated images saved to: '{output_root}/'")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="custom_dataset")
    parser.add_argument("--output",  type=str, default="verified_landmarks")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_dataset(args.dataset, args.output)