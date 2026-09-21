import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import classification_report


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data/processed/stabilix_processed_v2.csv"

MODEL_DIR = "models"
MODEL_FILE = os.path.join(
    MODEL_DIR,
    "stabilix_rf_v1.joblib"
)


# ============================================================
# ORIGINAL 23 STABILIX FEATURES
# ============================================================

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

    # GPU_Memory_Used_Avg / GPU_Memory_Used_Latest intentionally
    # removed. They're raw megabytes (not percentage) and were
    # nearly CONSTANT within any one recording session - so instead
    # of measuring workload, they were measuring "which of the 6
    # recording sessions is this", based on whatever background apps
    # happened to hold GPU memory that day. They ranked as the #1
    # and #2 most important features in the old model and were
    # directly responsible for idle machines being misread as
    # Heavy_CPU whenever their live GPU memory baseline happened to
    # be higher than the training rig's idle baseline.
    # GPU_Memory_Usage_Avg/Latest (percentage, above) is the
    # non-leaky version of this signal and is kept.
]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("STABILIX - ORIGINAL 23 FEATURE MODEL")
print("=" * 70)

print("\nLoading:")
print(INPUT_FILE)

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"\nDataset not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print(
    f"\nDataset shape: {df.shape}"
)


# ============================================================
# CHECK FEATURES
# ============================================================

missing_features = [
    feature
    for feature in FEATURE_COLUMNS
    if feature not in df.columns
]

if missing_features:

    print("\nMissing features:")

    for feature in missing_features:
        print(" -", feature)

    raise ValueError(
        "\nRequired 23 features are missing."
    )


# ============================================================
# CHECK LABEL
# ============================================================

if "Label" not in df.columns:

    raise ValueError(
        "Label column not found."
    )


# ============================================================
# PREPARE X AND Y
# ============================================================

X = df[FEATURE_COLUMNS].copy()

y = df["Label"].astype(str)

EXPECTED_CLASSES = {
    "Stable",
    "Normal_Workload",
    "Medium_Workload",
    "Heavy_CPU",
    "Heavy_GPU",
    "Application_Crash_Imminent",
    "Instability_Imminent",
}

missing_classes = EXPECTED_CLASSES - set(y.unique())

if missing_classes:
    raise ValueError(
        "Training dataset is missing expected classes: "
        + ", ".join(sorted(missing_classes))
    )


# ============================================================
# CLEAN NUMERIC DATA
# ============================================================

for column in FEATURE_COLUMNS:

    X[column] = pd.to_numeric(
        X[column],
        errors="coerce"
    )


X = X.replace(
    [float("inf"), float("-inf")],
    pd.NA
)


# ============================================================
# DISPLAY LABEL DISTRIBUTION
# ============================================================

print("\n" + "-" * 70)
print("LABEL DISTRIBUTION")
print("-" * 70)

for label, count in y.value_counts().items():

    print(
        f"{label:<30} {count:>6}"
    )


# ============================================================
# CREATE RANDOM FOREST
# ============================================================

print("\n" + "-" * 70)
print("CREATING RANDOM FOREST")
print("-" * 70)

model = Pipeline([

    (
        "imputer",
        SimpleImputer(
            strategy="median"
        )
    ),

    (
        "classifier",
        RandomForestClassifier(

            n_estimators=400,

            max_features="sqrt",

            min_samples_leaf=2,

            class_weight="balanced_subsample",

            random_state=42,

            n_jobs=-1
        )
    )
])


# ============================================================
# HONEST PERFORMANCE CHECK (before trusting this model)
# ============================================================
#
# class_weight="balanced_subsample" on the RandomForest already
# rebalances every bootstrap sample internally. We used to ALSO
# multiply in a manual per-row sample_weight (compute_sample_weight),
# which double-applied the correction. For classes like
# Instability_Imminent (11 rows) that meant each row got ~120x
# weight from sample_weight, AND got rebalanced again inside every
# tree's bootstrap - so the forest could carve out leaves that
# memorize those 11 specific rows instead of learning a pattern
# that generalizes to real (slightly different) telemetry.
#
# Fix: rely on class_weight="balanced_subsample" alone, and
# actually measure per-class recall on held-out sessions (not
# held-out rows, which would leak - rows from the same recording
# session are highly correlated) before trusting the model.

print("\n" + "-" * 70)
print("CROSS-VALIDATED PERFORMANCE (grouped by Session_ID)")
print("-" * 70)

n_sessions = df["Session_ID"].nunique()
n_splits = min(5, n_sessions)

if n_splits < 2:
    print(
        "Not enough distinct sessions for a grouped CV report "
        "(need at least 2). Skipping."
    )
else:
    group_kfold = GroupKFold(n_splits=n_splits)

    cv_predictions = cross_val_predict(
        model,
        X,
        y,
        cv=group_kfold,
        groups=df["Session_ID"],
        n_jobs=-1,
    )

    print(
        classification_report(
            y,
            cv_predictions,
            zero_division=0,
        )
    )

    print(
        "Treat the rows above for Application_Crash_Imminent and "
        "Instability_Imminent with real skepticism - with only "
        f"{int((y == 'Application_Crash_Imminent').sum())} and "
        f"{int((y == 'Instability_Imminent').sum())} examples "
        "respectively, high precision/recall here can still be "
        "an artifact of too little data rather than a model that "
        "will catch a real, novel crash."
    )


# ============================================================
# TRAIN FINAL MODEL ON ALL DATA
# ============================================================

print("\nTraining final model on all data...")

model.fit(X, y)


# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

joblib.dump(
    model,
    MODEL_FILE
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("STABILIX RF V1 TRAINING COMPLETE")
print("=" * 70)

print(
    "\nModel saved to:"
)

print(
    MODEL_FILE
)

print(
    f"\nNumber of features: "
    f"{len(FEATURE_COLUMNS)}"
)

print("\nFeatures:")

for i, feature in enumerate(
    FEATURE_COLUMNS,
    start=1
):

    print(
        f"{i:02d}. {feature}"
    )

print("\nClasses:")

for cls in model.classes_:

    print(
        " -",
        cls
    )

print("\nDone.")