#!/usr/bin/env python3
"""inference.py – batch prediction for Flight Deviance Detection

Changes (2025‑07‑28)
--------------------
* **Output now contains exactly three columns** in the CSV: `ID`, `Tail`,
  `Deviance_rate`.
* Rows are **sorted descending by `Deviance_rate`**.
* Still auto‑aligns feature columns to those used during training.

Example
~~~~~~~
python inference.py \
    --model models/RandomForestClassifier.pkl \
    --input data/plane_features_reduced.csv \
    --output outputs/rf_deviance_probability.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

BATCH_SIZE_DEFAULT = 4096
META_COLUMNS = ["ID", "Tail"]  # always keep these for output


def read_csv(csv_path: str | Path):
    """Read full CSV into a DataFrame."""
    return pd.read_csv(csv_path)


def split_features(df: pd.DataFrame, drop_cols: Sequence[str] | None = None):
    """Return X (features only) after dropping *drop_cols*."""
    if drop_cols:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return df


def align_columns(X: pd.DataFrame, expected: Sequence[str]) -> pd.DataFrame:
    """Re‑order *X* to match *expected*; add missing cols (0) & drop extras."""
    for col in expected:
        if col not in X.columns:
            X[col] = 0
    X = X[list(expected)]  # drop extras & order exactly
    return X


def batch_predict(model, X: pd.DataFrame, class_idx: int, batch_size: int):
    """Predict in mini‑batches with tqdm."""
    out = []
    for start in tqdm(range(0, len(X), batch_size), desc="Inference", unit="rows"):
        chunk = X.iloc[start : start + batch_size]
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(chunk)[:, class_idx]
        else:
            p = model.predict(chunk)
        out.append(p)
    return np.concatenate(out)


def run(model_path: str | Path, input_csv: str | Path, output_csv: str | Path,
        prob_class: int, batch_size: int):
    # 1) Load items
    model = joblib.load(model_path)
    df = read_csv(input_csv)

    # 2) Prepare X with same features used in training
    expected_cols = getattr(model, "feature_names_in_", None)
    if expected_cols is None:
        raise RuntimeError("Model lacks feature_names_in_. Re‑train with sklearn >=1.0")

    X = split_features(df.copy(), META_COLUMNS)  # drop meta cols
    X = align_columns(X, expected_cols)

    # 3) Predict probabilities
    probs = batch_predict(model, X, class_idx=prob_class, batch_size=batch_size)

    # 4) Assemble output DataFrame
    out_df = pd.DataFrame({
        "ID": df["ID"],
        "Tail": df["Tail"],
        "Deviance_rate": probs,
    }).sort_values("Deviance_rate", ascending=False)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    print(f"✅ Saved predictions → {output_csv}  (rows: {len(out_df)})")


def main():
    p = argparse.ArgumentParser(description="Batch inference – outputs ID, Tail, Deviance_rate (sorted)")
    p.add_argument("--model", default="models/RandomForestClassifier.pkl", help="Path to trained .pkl model")
    p.add_argument("--input", required=True, help="Input CSV with raw features—including ID & Tail columns")
    p.add_argument("--output", default="outputs/rf_deviance_probability.csv", help="Destination CSV")
    p.add_argument("--prob-class", type=int, default=1, help="Class index that represents 'deviance'")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT, help="Batch size for prediction")
    args = p.parse_args()

    run(args.model, args.input, args.output, args.prob_class, args.batch_size)


if __name__ == "__main__":
    main()
