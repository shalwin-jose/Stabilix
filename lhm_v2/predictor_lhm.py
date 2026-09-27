"""Model loading and prediction for the LHM V2 feature contract."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from lhm_v2.features import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "stabilix_lhm_v2.joblib"


class LhmPredictor:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Train the LHM model first: {MODEL_PATH}")
        artifact = joblib.load(MODEL_PATH)
        if artifact.get("feature_columns") != FEATURE_COLUMNS:
            raise ValueError("Saved model features do not match the live feature contract")
        self.model = artifact["model"]

    def predict(self, features):
        missing = [name for name in FEATURE_COLUMNS if name not in features]
        if missing:
            raise ValueError(f"Feature contract mismatch; missing {len(missing)} columns: {missing[:8]}")
        for required in ("CPU_Total_Load_Avg", "GPU_Core_Load_Avg", "GPU_Core_Temperature_Avg"):
            value = features.get(required)
            try:
                valid = np.isfinite(float(value))
            except (TypeError, ValueError):
                valid = False
            if not valid:
                raise ValueError(
                    f"Required live sensor feature {required} is missing. "
                    "Prediction suppressed because imputed missing telemetry is unreliable."
                )
        X = pd.DataFrame([[features[name] for name in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
        X = X.apply(pd.to_numeric, errors="coerce").replace([float("inf"), float("-inf")], np.nan)
        probabilities = self.model.predict_proba(X)[0]
        classes = list(self.model.named_steps["classifier"].classes_)
        probs = {str(k): float(v) for k, v in zip(classes, probabilities)}
        label = max(probs, key=probs.get)
        confidence = probs[label]
        risk = sum(p * (100.0 if name.endswith("__Crash_Imminent") else 45.0 if name.endswith("__Approaching_Instability") else 0.0) for name, p in probs.items())
        return {"label": label, "probabilities": probs,
                "confidence": confidence * 100.0, "risk": min(100.0, risk)}
