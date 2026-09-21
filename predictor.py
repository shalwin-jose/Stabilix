import os

import joblib
import pandas as pd


MODEL_PATH = os.path.join(
    "models",
    "stabilix_rf_v1.joblib",
)


FEATURE_COLUMNS = [
    "CPU_Usage_Avg",
    "CPU_Usage_Std",
    "CPU_Performance_Avg",
    "CPU_Performance_Latest",
    "CPU_Performance_Rate",
    "RAM_Usage_Avg",
    "RAM_Usage_Std",
    "RAM_Usage_Latest",
    "Disk_Read_Avg",
    "Disk_Write_Avg",
    "GPU_Usage_Avg",
    "GPU_Usage_Std",
    "GPU_Usage_Latest",
    "GPU_Memory_Usage_Avg",
    "GPU_Memory_Usage_Latest",
    "GPU_Temperature_Avg",
    "GPU_Temperature_Latest",
    "GPU_Temperature_Rate",
    "GPU_Clock_Avg",
    "GPU_Clock_Latest",
    "GPU_Clock_Rate",
    # GPU_Memory_Used_Avg / GPU_Memory_Used_Latest removed: raw MB
    # values that were session-leaky, not workload signal. See
    # train_model.py for the full explanation.
]


DISPLAY_NAMES = {
    "Stable": "Stable",
    "Normal_Workload": "Normal Workload",
    "Medium_Workload": "Medium Workload",
    "Heavy_CPU": "Heavy CPU",
    "Heavy_GPU": "Heavy GPU",
    "Application_Crash_Imminent": "Application Crash Imminent",
    "Instability_Imminent": "Instability Imminent",
}


# This is a stability-risk heuristic, NOT a probability.
# Heavy workload alone is not treated as imminent instability.
RISK_WEIGHTS = {
    "Stable": 0.00,
    "Normal_Workload": 0.00,
    "Medium_Workload": 0.05,
    "Heavy_CPU": 0.10,
    "Heavy_GPU": 0.10,
    "Application_Crash_Imminent": 0.85,
    "Instability_Imminent": 1.00,
}


class Predictor:

    def __init__(self):
        print("\n" + "=" * 70)
        print("STABILIX ML MODEL")
        print("=" * 70)
        print("Loading:", MODEL_PATH)

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                "Stabilix model not found:\n"
                f"{MODEL_PATH}\n\n"
                "Run train_model.py first."
            )

        self.model = joblib.load(MODEL_PATH)

        self.classifier = self._get_classifier()

        feature_count = getattr(
            self.classifier,
            "n_features_in_",
            None,
        )

        if feature_count is not None and feature_count != len(FEATURE_COLUMNS):
            raise ValueError(
                f"Model expects {feature_count} features, "
                f"but Stabilix V1 requires {len(FEATURE_COLUMNS)}."
            )

        classes = getattr(self.classifier, "classes_", None)
        if classes is None:
            raise ValueError("Loaded model has no classes_ attribute.")

        self.classes = [str(value) for value in classes]

        print("Model loaded successfully.")
        print(f"Features: {len(FEATURE_COLUMNS)}")
        print("Classes:")
        for cls in self.classes:
            print(" -", cls)

        print("=" * 70)

    def _get_classifier(self):
        if hasattr(self.model, "named_steps"):
            if "classifier" in self.model.named_steps:
                return self.model.named_steps["classifier"]

            if "rf" in self.model.named_steps:
                return self.model.named_steps["rf"]

        return self.model

    def _prepare_input(self, features):
        if features is None:
            raise ValueError("Predictor received None.")

        missing = [
            feature
            for feature in FEATURE_COLUMNS
            if feature not in features
        ]

        if missing:
            raise ValueError(
                "Missing ML features:\n"
                + "\n".join(missing)
            )

        X = pd.DataFrame(
            [{
                feature: features[feature]
                for feature in FEATURE_COLUMNS
            }],
            columns=FEATURE_COLUMNS,
        )

        for column in FEATURE_COLUMNS:
            X[column] = pd.to_numeric(
                X[column],
                errors="coerce",
            )

        X = X.replace(
            [float("inf"), float("-inf")],
            pd.NA,
        )

        # The saved model already contains a median imputer.
        # Keep NaNs here so that the pipeline performs the same
        # preprocessing used during training.
        return X

    def _calculate_risk(self, probabilities):
        """
        Convert class probabilities into a stability-risk score.

        This is intentionally different from the old hard-coded
        "risk = 45 if Heavy_CPU" behavior.

        A Heavy GPU workload is not automatically unstable.
        Imminent event classes carry most of the risk.
        """
        risk = 0.0

        for label, probability in probabilities.items():
            weight = RISK_WEIGHTS.get(label, 0.0)
            risk += (probability / 100.0) * weight * 100.0

        return round(max(0.0, min(100.0, risk)), 1)

    def predict(self, features):
        X = self._prepare_input(features)

        prediction = self.model.predict(X)[0]
        label = str(prediction)

        if not hasattr(self.model, "predict_proba"):
            raise ValueError(
                "Loaded Stabilix model does not support predict_proba()."
            )

        probability_values = self.model.predict_proba(X)[0]

        probabilities = {
            str(cls): round(float(probability) * 100.0, 2)
            for cls, probability
            in zip(self.classifier.classes_, probability_values)
        }

        confidence = max(probabilities.values()) if probabilities else 0.0
        risk = self._calculate_risk(probabilities)

        return {
            "label": label,
            "state": DISPLAY_NAMES.get(label, label),
            "confidence": round(confidence, 1),
            "risk": risk,
            "probabilities": probabilities,
        }