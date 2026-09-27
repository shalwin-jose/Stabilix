"""Train the LHM V2 session-grouped workload/instability classifier."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

from lhm_v2.features import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "stabilix_lhm_v2_training.csv"
MODEL = ROOT / "models" / "stabilix_lhm_v2.joblib"


def train():
    if not DATA.exists():
        raise FileNotFoundError(f"Run preprocess_lhm.py first: {DATA}")
    df = pd.read_csv(DATA)
    missing = [c for c in FEATURE_COLUMNS if c not in df]
    if missing:
        raise ValueError(f"Training data is missing shared features: {missing}")
    X = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    X = X.replace([float("inf"), float("-inf")], np.nan)
    y = df["Label"].astype(str)
    groups = df["Session_ID"].astype(str)
    if y.nunique() < 2:
        raise ValueError("At least two distinct labels are required to train")
    counts = y.value_counts()
    class_weights = {
        label: len(y) / (y.nunique() * count) * (2.0 if label == "Stable_Idle__No_Imminent_Instability" else 1.0)
        for label, count in counts.items()
    }

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("classifier", RandomForestClassifier(
            n_estimators=500, max_features=0.5, min_samples_leaf=3,
            class_weight=class_weights, n_jobs=-1, random_state=42,
        )),
    ])

    # Hold out whole recordings so adjacent overlapping windows never leak
    # between train and validation partitions.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    tr, va = next(splitter.split(X, y, groups))
    model.fit(X.iloc[tr], y.iloc[tr])
    predictions = model.predict(X.iloc[va])
    print("Session-held-out validation report (rare crash labels need more captures):")
    print(classification_report(y.iloc[va], predictions, zero_division=0))

    # Refit on all labeled captures after reporting the held-out estimate.
    model.fit(X, y)
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_columns": FEATURE_COLUMNS,
                 "window_seconds": 10.0, "label_classes": sorted(y.unique())}, MODEL)
    print(f"Saved model: {MODEL}\nRows: {len(df)} | Sessions: {groups.nunique()} | Features: {len(FEATURE_COLUMNS)}")
    print("Classes:", ", ".join(sorted(y.unique())))


if __name__ == "__main__":
    train()
