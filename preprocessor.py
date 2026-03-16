"""
preprocessor.py
---------------
Downloads a gesture dataset from Kaggle, extracts MediaPipe hand landmarks
from each image, and saves the resulting feature vectors to a CSV file.

Each row in the CSV: label + 42 floats (x, y) for 21 landmarks.

Usage:
    python preprocessor.py
    python preprocessor.py --custom ./custom_dataset --samples_per_class 500
"""

import os
import cv2
import csv
import argparse
import random
import urllib.request
import numpy as np
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


def process_dataset(dataset_path: str, output_csv: str, class_map: dict = None, limit: int = None):
    dataset_root = Path(dataset_path)

    if class_map:
        class_dirs = [(dataset_root / folder, label)
                      for folder, label in class_map.items()
                      if (dataset_root / folder).exists()]
    else:
        class_dirs = [(d, d.name) for d in sorted(dataset_root.iterdir()) if d.is_dir()]

    if not class_dirs:
        raise FileNotFoundError(f"No valid class folders found in '{dataset_root}'")

    total_ok, total_skip = 0, 0
    stats = {}

    with open(output_csv, "a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        with build_detector() as detector:
            for folder, label in class_dirs:
                ok, skip = process_folder(folder, label, writer, detector, limit=limit)
                stats[label] = (ok, skip)
                total_ok   += ok
                total_skip += skip

    print("\n" + "=" * 50)
    print(f"{'Class':<20} {'Saved':>8} {'Skipped':>10}")
    print("-" * 50)
    for label, (ok, skip) in stats.items():
        print(f"{label:<20} {ok:>8} {skip:>10}")
    print("-" * 50)
    print(f"{'TOTAL':<20} {total_ok:>8} {total_skip:>10}")
    print(f"\nCSV saved to: {output_csv}")


def download_kaggle(slug: str) -> str:
    try:
        import kagglehub
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'kagglehub'. Install it with: py -m pip install kagglehub"
        ) from exc

    print(f"Downloading '{slug}' from Kaggle...")
    path = kagglehub.dataset_download(slug)
    print(f"Dataset ready at: {path}\n")
    return path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path",      type=str, default=None)
    parser.add_argument("--kaggle_slug",       type=str, default="innominate817/hagrid-sample-120k-384p")
    parser.add_argument("--output",            type=str, default="data/gestures_dataset.csv")
    parser.add_argument("--custom",            type=str, default=None,
                        help="Custom dataset path to merge (e.g. ./custom_dataset)")
    parser.add_argument("--samples_per_class", type=int, default=500,
                        help="Total samples per class across all sources (default: 500)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(Path(args.output).parent, exist_ok=True)

    if os.path.exists(args.output):
        os.remove(args.output)

    with open(args.output, "w", newline="") as f:
        csv.writer(f).writerow(HEADER)

    # Count custom samples already collected per class
    custom_counts = {}
    if args.custom:
        print("\n--- Processing custom dataset (priority) ---")
        process_dataset(args.custom, args.output, class_map=None, limit=None)
        for d in Path(args.custom).iterdir():
            if d.is_dir():
                custom_counts[d.name] = len([p for p in d.iterdir()
                                             if p.suffix.lower() in VALID_EXTENSIONS])

    # Fill remaining quota from HaGRID
    hagrid_limits = {
        folder: max(0, args.samples_per_class - custom_counts.get(label, 0))
        for folder, label in HAGRID_CLASS_MAP.items()
    }

    print("\n--- Processing HaGRID ---")
    print(f"  Fetching from HaGRID per class: { {k: v for k, v in hagrid_limits.items()} }")

    dataset_path = args.dataset_path or download_kaggle(args.kaggle_slug)

    dataset_root = Path(dataset_path)
    with open(args.output, "a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        with build_detector() as detector:
            for folder, label in HAGRID_CLASS_MAP.items():
                folder_path = dataset_root / folder
                if not folder_path.exists():
                    print(f"  WARNING: folder '{folder}' not found in HaGRID, skipping.")
                    continue
                limit = hagrid_limits[folder]
                if limit == 0:
                    print(f"  {label}: quota already met by custom dataset, skipping HaGRID.")
                    continue
                ok, skip = process_folder(folder_path, label, writer, detector, limit=limit)
                print(f"    {label}: {ok} saved from HaGRID")

    print("\nDone. Ready for trainer.py")