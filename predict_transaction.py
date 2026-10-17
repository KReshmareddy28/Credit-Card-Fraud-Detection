"""Predict whether a transaction is suspicious using the trained demo model."""
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / "outputs" / "fraud_detection_model.joblib"

def main():
    if not MODEL_FILE.exists():
        print("Model not found. Run: python main.py")
        return

    saved = joblib.load(MODEL_FILE)
    model = saved["model"]
    columns = saved["columns"]

    print("Enter transaction values. This is a synthetic educational demo.")
    values = {}
    for col in columns:
        raw = input(f"{col}: ").strip()
        try:
            values[col] = float(raw)
        except ValueError:
            print("Please enter a numeric value for every feature.")
            return

    row = pd.DataFrame([[values[c] for c in columns]], columns=columns)
    pred = int(model.predict(row)[0])
    prob = float(model.predict_proba(row)[0][1])
    print("\nDemo prediction:", "POTENTIALLY FRAUDULENT" if pred else "LIKELY LEGITIMATE")
    print(f"Model fraud score: {prob:.1%}")
    print("This score is not a verified real-world fraud probability.")

if __name__ == "__main__":
    main()
