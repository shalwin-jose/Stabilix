import os
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data/processed/stabilix_processed_v2.csv"

OUTPUT_FILE = "data/processed/stabilix_training_v3.csv"

EPSILON = 1e-6


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("STABILIX FEATURE ENGINEERING V2")
print("=" * 70)

if not os.path.exists(INPUT_FILE):

    print("\nERROR:")
    print(f"File not found: {INPUT_FILE}")

    raise SystemExit(1)


df = pd.read_csv(INPUT_FILE)

print("\nLoaded:")
print(f"Rows    : {len(df)}")
print(f"Columns : {len(df.columns)}")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_ratio(a, b):

    return a / (b.abs() + EPSILON)


def add_deviation(
    dataframe,
    latest_column,
    average_column,
    new_column
):

    if (
        latest_column in dataframe.columns
        and average_column in dataframe.columns
    ):

        dataframe[new_column] = (
            dataframe[latest_column]
            - dataframe[average_column]
        )


def add_absolute_rate(
    dataframe,
    rate_column,
    new_column
):

    if rate_column in dataframe.columns:

        dataframe[new_column] = (
            dataframe[rate_column]
            .abs()
        )


def add_relative_rate(
    dataframe,
    rate_column,
    reference_column,
    new_column
):

    if (
        rate_column in dataframe.columns
        and reference_column in dataframe.columns
    ):

        dataframe[new_column] = (
            dataframe[rate_column]
            / (
                dataframe[reference_column].abs()
                + EPSILON
            )
        )


# ============================================================
# 1. LATEST VS AVERAGE DEVIATIONS
# ============================================================

print("\nAdding latest-vs-average deviation features...")


add_deviation(
    df,
    "CPU_Usage_Latest",
    "CPU_Usage_Avg",
    "CPU_Usage_Deviation"
)


add_deviation(
    df,
    "CPU_Performance_Latest",
    "CPU_Performance_Avg",
    "CPU_Performance_Deviation"
)


add_deviation(
    df,
    "RAM_Usage_Latest",
    "RAM_Usage_Avg",
    "RAM_Usage_Deviation"
)


add_deviation(
    df,
    "GPU_Usage_Latest",
    "GPU_Usage_Avg",
    "GPU_Usage_Deviation"
)


add_deviation(
    df,
    "GPU_Memory_Usage_Latest",
    "GPU_Memory_Usage_Avg",
    "GPU_Memory_Usage_Deviation"
)


add_deviation(
    df,
    "GPU_Temperature_Latest",
    "GPU_Temperature_Avg",
    "GPU_Temperature_Deviation"
)


add_deviation(
    df,
    "GPU_Clock_Latest",
    "GPU_Clock_Avg",
    "GPU_Clock_Deviation"
)


add_deviation(
    df,
    "GPU_Memory_Used_Latest",
    "GPU_Memory_Used_Avg",
    "GPU_Memory_Used_Deviation"
)


# ============================================================
# 2. ABSOLUTE RATE FEATURES
# ============================================================

print("Adding absolute-rate features...")


add_absolute_rate(
    df,
    "CPU_Performance_Rate",
    "CPU_Performance_Rate_Abs"
)


add_absolute_rate(
    df,
    "GPU_Temperature_Rate",
    "GPU_Temperature_Rate_Abs"
)


add_absolute_rate(
    df,
    "GPU_Clock_Rate",
    "GPU_Clock_Rate_Abs"
)


# ============================================================
# 3. RELATIVE RATE FEATURES
# ============================================================

print("Adding relative-rate features...")


add_relative_rate(
    df,
    "CPU_Performance_Rate",
    "CPU_Performance_Avg",
    "CPU_Performance_Relative_Rate"
)


add_relative_rate(
    df,
    "GPU_Temperature_Rate",
    "GPU_Temperature_Avg",
    "GPU_Temperature_Relative_Rate"
)


add_relative_rate(
    df,
    "GPU_Clock_Rate",
    "GPU_Clock_Avg",
    "GPU_Clock_Relative_Rate"
)


# ============================================================
# 4. RELATIVE VOLATILITY
# ============================================================

print("Adding relative-volatility features...")


if (
    "CPU_Usage_Std" in df.columns
    and "CPU_Usage_Avg" in df.columns
):

    df["CPU_Usage_CV"] = safe_ratio(
        df["CPU_Usage_Std"],
        df["CPU_Usage_Avg"]
    )


if (
    "RAM_Usage_Std" in df.columns
    and "RAM_Usage_Avg" in df.columns
):

    df["RAM_Usage_CV"] = safe_ratio(
        df["RAM_Usage_Std"],
        df["RAM_Usage_Avg"]
    )


if (
    "GPU_Usage_Std" in df.columns
    and "GPU_Usage_Avg" in df.columns
):

    df["GPU_Usage_CV"] = safe_ratio(
        df["GPU_Usage_Std"],
        df["GPU_Usage_Avg"]
    )


# ============================================================
# 5. GPU BEHAVIOUR FEATURES
# ============================================================

print("Adding GPU behaviour features...")


# How much GPU clock changes relative to its current level.
if (
    "GPU_Clock_Rate" in df.columns
    and "GPU_Clock_Avg" in df.columns
):

    df["GPU_Clock_Change_Ratio"] = (
        df["GPU_Clock_Rate"]
        / (
            df["GPU_Clock_Avg"].abs()
            + EPSILON
        )
    )


# GPU usage relative to clock.
if (
    "GPU_Usage_Avg" in df.columns
    and "GPU_Clock_Avg" in df.columns
):

    df["GPU_Usage_Clock_Ratio"] = (
        df["GPU_Usage_Avg"]
        / (
            df["GPU_Clock_Avg"].abs()
            + EPSILON
        )
    )


# GPU memory usage relative to GPU utilization.
if (
    "GPU_Memory_Usage_Avg" in df.columns
    and "GPU_Usage_Avg" in df.columns
):

    df["GPU_Memory_Usage_Load_Ratio"] = (
        df["GPU_Memory_Usage_Avg"]
        / (
            df["GPU_Usage_Avg"].abs()
            + EPSILON
        )
    )


# ============================================================
# 6. GPU THERMAL BEHAVIOUR
# ============================================================

print("Adding GPU thermal behaviour features...")


if (
    "GPU_Temperature_Rate" in df.columns
    and "GPU_Temperature_Avg" in df.columns
):

    df["GPU_Temperature_Change_Ratio"] = (
        df["GPU_Temperature_Rate"]
        / (
            df["GPU_Temperature_Avg"].abs()
            + EPSILON
        )
    )


# Temperature change relative to GPU activity.
if (
    "GPU_Temperature_Rate" in df.columns
    and "GPU_Usage_Avg" in df.columns
):

    df["Temperature_Usage_Ratio"] = (
        df["GPU_Temperature_Rate"]
        / (
            df["GPU_Usage_Avg"].abs()
            + EPSILON
        )
    )


# ============================================================
# 7. CPU / GPU RELATIONSHIP
# ============================================================

print("Adding CPU/GPU interaction features...")


if (
    "CPU_Usage_Avg" in df.columns
    and "GPU_Usage_Avg" in df.columns
):

    df["CPU_GPU_Load_Gap"] = (
        df["CPU_Usage_Avg"]
        - df["GPU_Usage_Avg"]
    )


if (
    "CPU_Usage_Avg" in df.columns
    and "GPU_Usage_Avg" in df.columns
):

    df["CPU_GPU_Load_Ratio"] = (
        df["CPU_Usage_Avg"]
        / (
            df["GPU_Usage_Avg"].abs()
            + EPSILON
        )
    )


# ============================================================
# 8. RATE INTERACTION
# ============================================================

print("Adding rate interaction features...")


if (
    "GPU_Clock_Rate" in df.columns
    and "GPU_Temperature_Rate" in df.columns
):

    df["GPU_Clock_Temperature_Rate"] = (
        df["GPU_Clock_Rate"]
        * df["GPU_Temperature_Rate"]
    )


if (
    "GPU_Usage_Std" in df.columns
    and "GPU_Temperature_Rate_Abs" in df.columns
):

    df["GPU_Volatility_Temperature_Change"] = (
        df["GPU_Usage_Std"]
        * df["GPU_Temperature_Rate_Abs"]
    )


# ============================================================
# 9. SIGNED CHANGE FLAGS
# ============================================================

print("Adding directional change indicators...")


if "GPU_Clock_Rate" in df.columns:

    df["GPU_Clock_Dropping"] = (
        df["GPU_Clock_Rate"] < 0
    ).astype(int)


if "GPU_Temperature_Rate" in df.columns:

    df["GPU_Temperature_Rising"] = (
        df["GPU_Temperature_Rate"] > 0
    ).astype(int)


if "CPU_Performance_Rate" in df.columns:

    df["CPU_Performance_Dropping"] = (
        df["CPU_Performance_Rate"] < 0
    ).astype(int)


# ============================================================
# 10. CLEAN NUMERICAL VALUES
# ============================================================

print("Cleaning numerical values...")


numeric_columns = df.select_dtypes(
    include=[np.number]
).columns


df[numeric_columns] = (
    df[numeric_columns]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
)


# Don't invent values for completely unavailable
# original features. The five empty CPU temperature/
# frequency features are already absent from V1.
#
# For newly generated features, replace any numerical
# NaN/inf resulting from division with 0.
#
# This only handles mathematical edge cases.

new_columns = [
    c for c in df.columns
    if c not in pd.read_csv(INPUT_FILE, nrows=1).columns
]


for column in new_columns:

    if column in df.columns:

        df[column] = (
            df[column]
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .fillna(0)
        )


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("V2 PREPROCESSING COMPLETE")
print("=" * 70)

print(
    f"\nOriginal columns : "
    f"{len(pd.read_csv(INPUT_FILE, nrows=1).columns)}"
)

print(
    f"New columns      : "
    f"{len(df.columns)}"
)

print(
    f"New features     : "
    f"{len(new_columns)}"
)

print(
    f"Rows             : "
    f"{len(df)}"
)

print(
    "\nNew features:"
)

for feature in new_columns:

    print(
        f"  + {feature}"
    )

print(
    f"\nSaved to:"
)

print(
    OUTPUT_FILE
)

print(
    "\nDone."
)