from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "model" / "ufo_model_bundle.pkl"


def main() -> None:
    with MODEL_PATH.open("rb") as f:
        bundle = pickle.load(f)

    sample = np.array([[50, 44, -12]])
    prediction = bundle["model"].predict(sample)[0]
    country = bundle["label_encoder"].inverse_transform([prediction])[0]

    print(f"Sample prediction: {country.upper()} (class {prediction})")
    print(f"Model accuracy: {bundle.get('accuracy'):.4f}")


if __name__ == "__main__":
    main()

