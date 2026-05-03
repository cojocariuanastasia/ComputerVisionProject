"""
preprocessor.py
---------------
Downloads a gesture dataset from Kaggle, extracts MediaPipe hand landmarks
from each image, and saves the resulting feature vectors to a CSV file.

Each row in the CSV: label + 42 floats (x, y) for 21 landmarks.

Usage:
    python preprocessor.py --custom ./custom_dataset ./cv_project/my_data2 --samples_per_class 500
    python preprocessor.py --dataset_path "C:/path/to/hagrid" --custom ./custom_dataset ./other_data
"""

import os
import cv2
import csv
import argparse
import random
import urllib.request
import numpy as np
import kagglehub
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarkerOptions
from pathlib import Path
from tqdm import tqdm


HAGRID_CLASS_MAP = {
    "train_val_fist":    "fist",
    "train_val_palm":    "palm",
    "train_val_like":    "like",
    "train_val_dislike": "dislike",
}

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
HEADER = ["label"] + [f"{axis}_{i}" for i in range(21) for axis in ("x", "y")]

MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmarker model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def build_detector():
    options = HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        num_hands=1,
        min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    return vision.HandLandmarker.create_from_options(options)


def extract_landmarks(image_bgr, detector) -> list | None:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result    = detector.detect(mp_image)
    if not result.hand_landmarks:
        return None
    landmarks = result.hand_landmarks[0]
    return [coord for lm in landmarks for coord in (lm.x, lm.y)]


def normalize(landmarks: list) -> list:
    """Center on wrist and scale to [-1, 1] for position invariance."""
    coords = np.array(landmarks).reshape(21, 2)
    coords -= coords[0]
    max_val = np.max(np.abs(coords))
    if max_val > 0:
        coords /= max_val
    return coords.flatten().tolist()


def count_images(folder: Path) -> int:
    return len([p for p in folder.iterdir() if p.suffix.lower() in VALID_EXTENSIONS])


def process_folder(folder: Path, label: str, writer, detector, limit: int = None) -> tuple[int, int]:
    image_paths = [p for p in folder.iterdir() if p.suffix.lower() in VALID_EXTENSIONS]

    if limit:
        random.shuffle(image_paths)
        image_paths = image_paths[:limit]

    ok, skip = 0, 0
    for img_path in tqdm(image_paths, desc=f"  {label}", unit="img"):
        image = cv2.imread(str(img_path))
        if image is None:
            skip += 1
            continue
        landmarks = extract_landmarks(image, detector)
        if landmarks is None:
            skip += 1
            continue
        writer.writerow([label] + normalize(landmarks))
        ok += 1
    return ok, skip


def process_custom_sources(custom_paths: list, output_csv: str, detector) -> dict:
    """Process multiple custom dataset folders and return per-class image counts."""
    custom_counts = {}

    with open(output_csv, "a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        for source_path in custom_paths:
            root = Path(source_path)
            print(f"\nProcessing custom dataset: {root}")
            class_dirs = [(d, d.name) for d in sorted(root.iterdir()) if d.is_dir()]
            if not class_dirs:
                print(f"WARNING: no class subfolders found in '{root}', skipping.")
                continue
            for folder, label in class_dirs:
                ok, skip = process_folder(folder, label, writer, detector)
                custom_counts[label] = custom_counts.get(label, 0) + ok
                print(f"    {label}: {ok} saved ({skip} skipped)")

    return custom_counts


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path",      type=str, default=None)
    parser.add_argument("--kaggle_slug",       type=str, default="innominate817/hagrid-sample-120k-384p")
    parser.add_argument("--output",            type=str, default="data/gestures_dataset.csv")
    parser.add_argument("--custom",            type=str, nargs="+", default=None,
                        help="One or more custom dataset paths to merge")
    parser.add_argument("--samples_per_class", type=int, default=500,
                        help="Total samples per class across all sources (default: 500)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ensure_model()
    os.makedirs(Path(args.output).parent, exist_ok=True)

    if os.path.exists(args.output):
        os.remove(args.output)

    with open(args.output, "w", newline="") as f:
        csv.writer(f).writerow(HEADER)

    custom_counts = {}

    with build_detector() as detector:
        # Process all custom sources first
        if args.custom:
            custom_counts = process_custom_sources(args.custom, args.output, detector)

        # Fill remaining quota from HaGRID
        hagrid_limits = {
            folder: max(0, args.samples_per_class - custom_counts.get(label, 0))
            for folder, label in HAGRID_CLASS_MAP.items()
        }

        print("\nProcessing HaGRID")
        print(f"  Quota remaining per class: { {v: hagrid_limits[k] for k, v in HAGRID_CLASS_MAP.items()} }")

        dataset_path = args.dataset_path or kagglehub.dataset_download(args.kaggle_slug)
        dataset_root = Path(dataset_path)

        with open(args.output, "a", newline="") as csv_file:
            writer = csv.writer(csv_file)
            for folder, label in HAGRID_CLASS_MAP.items():
                folder_path = dataset_root / folder
                if not folder_path.exists():
                    print(f"  WARNING: '{folder}' not found in HaGRID, skipping.")
                    continue
                limit = hagrid_limits[folder]
                if limit == 0:
                    print(f"  {label}: quota met by custom data, skipping HaGRID.")
                    continue
                ok, skip = process_folder(folder_path, label, writer, detector, limit=limit)
                print(f"    {label}: {ok} saved from HaGRID")

    print("\nDone. Ready for trainer.py")