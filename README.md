# Computer Vision Project

cd e:/CV/ComputerVisionProject-main

py trainer.py --data data/gestures_dataset.csv --model models/gesture_rf.joblib

py main.py --model models/gesture_rf.joblib --camera 0 --threshold 0.70 --window 5