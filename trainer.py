"""
trainer.py
----------
Trains a Random Forest classifier on gesture landmark CSV data.

Input CSV format:
	label,x_0,y_0,...,x_20,y_20

Outputs:
	- Trained model artifact (.joblib)
	- Evaluation metrics printed to console

Usage:
	py trainer.py
	py trainer.py --data data/gestures_dataset.csv --model models/gesture_rf.joblib
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train Random Forest for gesture classification")
	parser.add_argument("--data", type=str, default="data/gestures_dataset.csv",
						help="Path to training CSV")
	parser.add_argument("--model", type=str, default="models/gesture_rf.joblib",
						help="Path to save trained model artifact")
	parser.add_argument("--test_size", type=float, default=0.2,
						help="Fraction of data for test split (default: 0.2)")
	parser.add_argument("--random_state", type=int, default=42,
						help="Random seed (default: 42)")
	parser.add_argument("--n_estimators", type=int, default=300,
						help="Number of trees (default: 300)")
	parser.add_argument("--max_depth", type=int, default=20,
						help="Max tree depth (default: 20)")
	parser.add_argument("--min_samples_leaf", type=int, default=2,
						help="Minimum samples per leaf (default: 2)")
	return parser.parse_args()


def load_dataset(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
	if not csv_path.exists():
		raise FileNotFoundError(f"Training CSV not found: {csv_path}")

	df = pd.read_csv(csv_path)
	if "label" not in df.columns:
		raise ValueError("CSV must contain a 'label' column")

	# Use all numeric landmark columns as features.
	x = df.drop(columns=["label"])
	y = df["label"]

	if x.empty or y.empty:
		raise ValueError("Dataset is empty or missing feature rows")

	return x, y


def main() -> None:
	args = parse_args()
	csv_path = Path(args.data)
	model_path = Path(args.model)

	x, y = load_dataset(csv_path)

	x_train, x_test, y_train, y_test = train_test_split(
		x,
		y,
		test_size=args.test_size,
		random_state=args.random_state,
		stratify=y,
	)

	model = RandomForestClassifier(
		n_estimators=args.n_estimators,
		max_depth=args.max_depth,
		min_samples_leaf=args.min_samples_leaf,
		class_weight="balanced",
		random_state=args.random_state,
		n_jobs=-1,
	)

	model.fit(x_train, y_train)

	y_pred = model.predict(x_test)

	print("\nClassification report:")
	print(classification_report(y_test, y_pred, digits=4))

	labels = sorted(y.unique().tolist())
	cm = confusion_matrix(y_test, y_pred, labels=labels)
	cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in labels],
						 columns=[f"pred_{l}" for l in labels])
	print("Confusion matrix:")
	print(cm_df)

	artifact = {
		"model": model,
		"feature_columns": x.columns.tolist(),
		"class_names": labels,
	}

	model_path.parent.mkdir(parents=True, exist_ok=True)
	dump(artifact, model_path)
	print(f"\nModel saved to: {model_path}")


if __name__ == "__main__":
	main()
