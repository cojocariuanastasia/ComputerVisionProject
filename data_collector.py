"""
data_collector.py
-----------------
Guides the user through recording hand gesture images via webcam.
Saves labeled images ready to be merged with any external dataset (e.g. HaGRID)
and processed by preprocessor.py.

Output structure:
    custom_dataset/
        fist/        -> neutral / rest
        palm/        -> play / pause
        like/        -> volume up
        dislike/     -> volume down

Usage:
    python data_collector.py
    python data_collector.py --output ./my_data --samples 100
"""

import cv2
import time
import argparse
from pathlib import Path


GESTURES = {
    "fist":    "FIST   - closed hand, all fingers curled (neutral/rest)",
    "palm":    "PALM   - open hand, all fingers extended (play/pause)",
    "like":    "LIKE   - thumbs up, rest of fingers curled (volume up)",
    "dislike": "DISLIKE - thumbs down, rest of fingers curled (volume down)",
}

COLOR_GREEN  = (0, 220, 0)
COLOR_YELLOW = (0, 220, 220)
COLOR_RED    = (0, 0, 220)
COLOR_WHITE  = (255, 255, 255)
COLOR_BG     = (30, 30, 30)


def draw_text(frame, text, pos, color=COLOR_WHITE, scale=0.7, thickness=2):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def draw_hud(frame, gesture_name, description, count, target, state):
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), COLOR_BG, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    draw_text(frame, f"Gesture: {gesture_name.upper()}", (15, 28), COLOR_YELLOW, 0.8, 2)
    draw_text(frame, description, (15, 55), COLOR_WHITE, 0.55, 1)
    draw_text(frame, f"Saved: {count}/{target}", (15, 78), COLOR_GREEN, 0.6, 2)

    if state == "countdown":
        draw_text(frame, "Get ready...", (w // 2 - 90, h // 2), COLOR_YELLOW, 1.2, 3)
    elif state == "recording":
        cv2.circle(frame, (w - 30, 25), 12, COLOR_RED, -1)
        draw_text(frame, "REC", (w - 65, 33), COLOR_RED, 0.65, 2)
    elif state == "done":
        draw_text(frame, "DONE! Press any key...", (w // 2 - 160, h // 2), COLOR_GREEN, 1.0, 2)
    elif state == "waiting":
        draw_text(frame, "Press SPACE to start", (w // 2 - 140, h // 2), COLOR_YELLOW, 1.0, 2)
        draw_text(frame, "Q = quit", (w // 2 - 55, h // 2 + 50), COLOR_WHITE, 0.7, 1)

    draw_text(frame, "Q = quit early", (15, h - 12), COLOR_WHITE, 0.5, 1)


def wait_for_start(cap, gesture_name, description):
    """Show a waiting screen inside the OpenCV window — no blocking input()."""
    while True:
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            draw_hud(frame, gesture_name, description, 0, 0, "waiting")
            cv2.imshow("Data Collector", frame)
        key = cv2.waitKey(30) & 0xFF
        if key == ord(' '):
            return True
        if key == ord('q'):
            return False


def collect_gesture(cap, gesture_name, description, save_dir, target, countdown_sec=3):
    save_dir.mkdir(parents=True, exist_ok=True)

    existing = len(list(save_dir.glob("*.jpg")))
    count = 0
    h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Countdown phase
    deadline = time.time() + countdown_sec
    while time.time() < deadline:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        remaining = int(deadline - time.time()) + 1
        draw_hud(frame, gesture_name, description, count, target, "countdown")
        draw_text(frame, str(remaining), (w // 2 - 20, h // 2 + 80), COLOR_YELLOW, 3.0, 6)

        cv2.imshow("Data Collector", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False

    # Recording phase
    while count < target:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        filename = save_dir / f"{existing + count:04d}.jpg"
        cv2.imwrite(str(filename), frame)
        count += 1

        draw_hud(frame, gesture_name, description, count, target, "recording")
        cv2.imshow("Data Collector", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False

    # Done phase, keep event loop alive
    while True:
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            draw_hud(frame, gesture_name, description, count, target, "done")
            cv2.imshow("Data Collector", frame)
        key = cv2.waitKey(30) & 0xFF
        if key != 255:
            break
        if cv2.getWindowProperty("Data Collector", cv2.WND_PROP_VISIBLE) < 1:
            break

    print(f"  [{gesture_name}] saved {count} images -> {save_dir}")
    return True


def parse_args():
    parser = argparse.ArgumentParser(description="Record custom gesture images via webcam.")
    parser.add_argument("--output",   type=str, default="custom_dataset",
                        help="Root folder for saved images (default: custom_dataset)")
    parser.add_argument("--samples",  type=int, default=100,
                        help="Images to capture per gesture (default: 100)")
    parser.add_argument("--camera",   type=int, default=0,
                        help="Camera index (default: 0)")
    parser.add_argument("--countdown", type=int, default=3,
                        help="Countdown seconds before each gesture (default: 3)")
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = Path(args.output)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print(f"\nData Collector — {args.samples} images/gesture → '{output_root}'")
    print("=" * 55)
    print("Press SPACE in the camera window to start each gesture.")
    print("Press Q to quit at any time.")

    for gesture_name, description in GESTURES.items():
        print(f"\n→ Next: {gesture_name.upper()} — {description}")

        # Wait for SPACE inside the OpenCV window (no blocking input())
        ok = wait_for_start(cap, gesture_name, description)
        if not ok:
            print("\nAborted by user.")
            break

        ok = collect_gesture(
            cap=cap,
            gesture_name=gesture_name,
            description=description,
            save_dir=output_root / gesture_name,
            target=args.samples,
            countdown_sec=args.countdown,
        )
        if not ok:
            print("\nAborted by user.")
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"Collection complete. Images saved to '{output_root}/'")


if __name__ == "__main__":
    main()