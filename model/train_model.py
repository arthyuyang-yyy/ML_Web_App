from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "ufos.csv"
MODEL_PATH = PROJECT_ROOT / "model" / "ufo_model_bundle.pkl"

FEATURES = ["Seconds", "Latitude", "Longitude"]


def load_and_clean_data() -> tuple[pd.DataFrame, pd.Series, LabelEncoder]:
    raw = pd.read_csv(DATA_PATH)

    ufos = pd.DataFrame(
        {
            "Seconds": raw["duration (seconds)"],
            "Country": raw["country"],
            "Latitude": raw["latitude"],
            "Longitude": raw["longitude"],
        }
    )

    ufos = ufos.dropna()
    ufos["Seconds"] = pd.to_numeric(ufos["Seconds"], errors="coerce")
    ufos["Latitude"] = pd.to_numeric(ufos["Latitude"], errors="coerce")
    ufos["Longitude"] = pd.to_numeric(ufos["Longitude"], errors="coerce")
    ufos = ufos.dropna()
    ufos = ufos[(ufos["Seconds"] >= 1) & (ufos["Seconds"] <= 60)]

    encoder = LabelEncoder()
    y = pd.Series(encoder.fit_transform(ufos["Country"]), name="Country")
    x = ufos[FEATURES]

    return x, y, encoder


def train() -> dict:
    x, y, encoder = load_and_clean_data()

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=0,
        stratify=y,
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)

    print(classification_report(y_test, predictions, target_names=encoder.classes_))
    print(f"Accuracy: {accuracy:.4f}")

    bundle = {
        "model": model,
        "label_encoder": encoder,
        "features": FEATURES,
        "accuracy": accuracy,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as f:
        pickle.dump(bundle, f)

    print(f"Saved model bundle to {MODEL_PATH}")
    return bundle


if __name__ == "__main__":
    train()

