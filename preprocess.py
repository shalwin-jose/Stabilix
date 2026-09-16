import os
import glob
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

OUTPUT_FILE = os.path.join(
    PROCESSED_DIR,
    "stabilix_processed_v1.csv"
)

WINDOW_SECONDS = 10
STRIDE_SECONDS = 1


# ============================================================
# COLUMN NORMALIZATION
# ============================================================

COLUMN_MAP = {
    # Timestamp/session information
    "Timestamp": "Timestamp",
    "Session_ID": "Session_ID",
    "Session_Name": "Session_Name",
    "Elapsed_Time": "Elapsed_Time",
    "Sample_Number": "Sample_Number",

    # CPU
    "CPU_Usage": "CPU_Usage_Percent",
    "CPU_Usage_Percent": "CPU_Usage_Percent",

    "CPU_Performance": "CPU_Performance_Percent",
    "CPU_Performance_Percent": "CPU_Performance_Percent",

    "CPU_Frequency_MHz": "CPU_Frequency_MHz",
    "CPU_Max_Frequency_MHz": "CPU_Max_Frequency_MHz",
    "CPU_Temperature_C": "CPU_Temperature_C",

    # RAM
    "RAM_Usage": "RAM_Usage_Percent",
    "RAM_Usage_Percent": "RAM_Usage_Percent",

    # Disk
    "Disk_Read_MBps": "Disk_Read_MBps",
    "Disk_Write_MBps": "Disk_Write_MBps",

    "Disk_Read": "Disk_Read_MBps",
    "Disk_Write": "Disk_Write_MBps",

    # GPU
    "GPU_Usage": "GPU_Usage_Percent",
    "GPU_Usage_Percent": "GPU_Usage_Percent",

    "GPU_Memory_Usage": "GPU_Memory_Usage_Percent",
    "GPU_Memory_Usage_Percent": "GPU_Memory_Usage_Percent",

    "GPU_Memory_Used_MB": "GPU_Memory_Used_MB",
    "GPU_Memory_Total_MB": "GPU_Memory_Total_MB",

    "GPU_Temperature": "GPU_Temperature_C",
    "GPU_Temperature_C": "GPU_Temperature_C",

    "GPU_Clock": "GPU_Graphics_Clock_MHz",
    "GPU_Graphics_Clock_MHz": "GPU_Graphics_Clock_MHz",

    # Battery
    "Battery_Percentage": "Battery_Percentage",
    "Charging": "Charging",
}


# ============================================================
# LABEL MAPPING
# ============================================================

def get_workload_label(filename):
    """
    Assigns the known workload label from the recording name.

    This is contextual labeling for v1.
    It is NOT based on arbitrary telemetry thresholds.
    """

    name = filename.lower()

    if "idle" in name:
        return "Stable"

    if "browsing" in name:
        return "Normal_Workload"

    if "mediumwork" in name:
        return "Medium_Workload"

    if "heavycpu" in name:
        return "Heavy_CPU"

    if "gamingmoderate" in name:
        return "Heavy_GPU"

    if "gamingheavy" in name:
        return "Heavy_GPU"

    if "gamingmain" in name:
        return "Heavy_GPU"

    if "gaming" in name:
        return "Heavy_GPU"

    if "stress" in name:
        return "Heavy_GPU"

    return "Unknown"


def is_confirmed_crash_file(filename):
    """
    The increasingstress recording was confirmed by the user
    to have ended in a black screen requiring a forced restart.
    """

    name = filename.lower()

    return "increasingstress" in name


# ============================================================
# DATA CLEANING
# ============================================================

def normalize_columns(df):
    """
    Converts old and new collector column names
    into one common schema.
    """

    rename_dict = {}

    for column in df.columns:
        if column in COLUMN_MAP:
            rename_dict[column] = COLUMN_MAP[column]

    df = df.rename(columns=rename_dict)

    return df


def clean_data(df):
    """
    Basic cleaning:
    - converts telemetry columns to numeric
    - converts -1 sentinel values to NaN
    - removes completely invalid rows
    """

    numeric_columns = [
        "CPU_Usage_Percent",
        "CPU_Performance_Percent",
        "CPU_Frequency_MHz",
        "CPU_Max_Frequency_MHz",
        "CPU_Temperature_C",
        "RAM_Usage_Percent",
        "Disk_Read_MBps",
        "Disk_Write_MBps",
        "GPU_Usage_Percent",
        "GPU_Memory_Usage_Percent",
        "GPU_Memory_Used_MB",
        "GPU_Memory_Total_MB",
        "GPU_Temperature_C",
        "GPU_Graphics_Clock_MHz",
        "Battery_Percentage",
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

            # -1 means unavailable in some old recordings.
            df[column] = df[column].replace(-1, np.nan)

    # Sort by time if available
    if "Timestamp" in df.columns:

        df["Timestamp"] = pd.to_datetime(
            df["Timestamp"],
            errors="coerce"
        )

        df = df.sort_values("Timestamp")

    # Remove rows without telemetry at all
    telemetry_columns = [
        column
        for column in numeric_columns
        if column in df.columns
    ]

    if telemetry_columns:

        df = df.dropna(
            subset=telemetry_columns,
            how="all"
        )

    return df.reset_index(drop=True)


# ============================================================
# FEATURE FUNCTIONS
# ============================================================

def mean_value(window, column):
    if column not in window.columns:
        return np.nan

    return window[column].mean()


def std_value(window, column):
    if column not in window.columns:
        return np.nan

    return window[column].std()


def latest_value(window, column):
    if column not in window.columns:
        return np.nan

    return window[column].iloc[-1]


def rate_of_change(window, column):
    """
    Calculates change per second.

    Correct denominator:
        number of intervals = n - 1
    """

    if column not in window.columns:
        return np.nan

    values = window[column].dropna()

    if len(values) < 2:
        return np.nan

    first = values.iloc[0]
    last = values.iloc[-1]

    return (last - first) / (len(values) - 1)


# ============================================================
# WINDOW FEATURE EXTRACTION
# ============================================================

def extract_features(window):

    features = {}

    # --------------------------------------------------------
    # CPU
    # --------------------------------------------------------

    features["CPU_Usage_Avg"] = mean_value(
        window,
        "CPU_Usage_Percent"
    )

    features["CPU_Usage_Std"] = std_value(
        window,
        "CPU_Usage_Percent"
    )

    features["CPU_Performance_Avg"] = mean_value(
        window,
        "CPU_Performance_Percent"
    )

    features["CPU_Performance_Latest"] = latest_value(
        window,
        "CPU_Performance_Percent"
    )

    features["CPU_Performance_Rate"] = rate_of_change(
        window,
        "CPU_Performance_Percent"
    )

    features["CPU_Frequency_Avg"] = mean_value(
        window,
        "CPU_Frequency_MHz"
    )

    features["CPU_Frequency_Latest"] = latest_value(
        window,
        "CPU_Frequency_MHz"
    )

    features["CPU_Temperature_Avg"] = mean_value(
        window,
        "CPU_Temperature_C"
    )

    features["CPU_Temperature_Latest"] = latest_value(
        window,
        "CPU_Temperature_C"
    )

    features["CPU_Temperature_Rate"] = rate_of_change(
        window,
        "CPU_Temperature_C"
    )

    # --------------------------------------------------------
    # RAM
    # --------------------------------------------------------

    features["RAM_Usage_Avg"] = mean_value(
        window,
        "RAM_Usage_Percent"
    )

    features["RAM_Usage_Std"] = std_value(
        window,
        "RAM_Usage_Percent"
    )

    features["RAM_Usage_Latest"] = latest_value(
        window,
        "RAM_Usage_Percent"
    )

    # --------------------------------------------------------
    # DISK
    # --------------------------------------------------------

    features["Disk_Read_Avg"] = mean_value(
        window,
        "Disk_Read_MBps"
    )

    features["Disk_Write_Avg"] = mean_value(
        window,
        "Disk_Write_MBps"
    )

    # --------------------------------------------------------
    # GPU
    # --------------------------------------------------------

    features["GPU_Usage_Avg"] = mean_value(
        window,
        "GPU_Usage_Percent"
    )

    features["GPU_Usage_Std"] = std_value(
        window,
        "GPU_Usage_Percent"
    )

    features["GPU_Usage_Latest"] = latest_value(
        window,
        "GPU_Usage_Percent"
    )

    features["GPU_Memory_Usage_Avg"] = mean_value(
        window,
        "GPU_Memory_Usage_Percent"
    )

    features["GPU_Memory_Usage_Latest"] = latest_value(
        window,
        "GPU_Memory_Usage_Percent"
    )

    features["GPU_Temperature_Avg"] = mean_value(
        window,
        "GPU_Temperature_C"
    )

    features["GPU_Temperature_Latest"] = latest_value(
        window,
        "GPU_Temperature_C"
    )

    features["GPU_Temperature_Rate"] = rate_of_change(
        window,
        "GPU_Temperature_C"
    )

    features["GPU_Clock_Avg"] = mean_value(
        window,
        "GPU_Graphics_Clock_MHz"
    )

    features["GPU_Clock_Latest"] = latest_value(
        window,
        "GPU_Graphics_Clock_MHz"
    )

    features["GPU_Clock_Rate"] = rate_of_change(
        window,
        "GPU_Graphics_Clock_MHz"
    )

    # --------------------------------------------------------
    # GPU MEMORY
    # --------------------------------------------------------

    features["GPU_Memory_Used_Avg"] = mean_value(
        window,
        "GPU_Memory_Used_MB"
    )

    features["GPU_Memory_Used_Latest"] = latest_value(
        window,
        "GPU_Memory_Used_MB"
    )

    return features


# ============================================================
# CREATE WINDOWS
# ============================================================

def create_windows(df):

    windows = []

    if "Timestamp" in df.columns:

        timestamps = df["Timestamp"]

        for start_time in timestamps:

            end_time = start_time + pd.Timedelta(
                seconds=WINDOW_SECONDS
            )

            window = df[
                (df["Timestamp"] >= start_time)
                &
                (df["Timestamp"] < end_time)
            ]

            if len(window) >= WINDOW_SECONDS:

                windows.append(window)

    else:

        # Fallback for recordings without timestamps
        window_size = WINDOW_SECONDS
        stride = STRIDE_SECONDS

        for start in range(
            0,
            len(df) - window_size + 1,
            stride
        ):

            window = df.iloc[
                start:start + window_size
            ]

            windows.append(window)

    return windows


# ============================================================
# PROCESS ONE SESSION
# ============================================================

def process_file(filepath):

    filename = os.path.basename(filepath)

    print(f"\nProcessing: {filename}")

    try:
        df = pd.read_csv(filepath)

    except Exception as e:
        print(f"Could not read file: {e}")
        return []

    # Normalize schema
    df = normalize_columns(df)

    # Clean
    df = clean_data(df)

    if len(df) < WINDOW_SECONDS:

        print("Not enough samples for a 10-second window.")

        return []

    workload_label = get_workload_label(filename)

    confirmed_crash = is_confirmed_crash_file(filename)

    windows = create_windows(df)

    processed_rows = []

    total_windows = len(windows)

    for index, window in enumerate(windows):

        features = extract_features(window)

        # ----------------------------------------------------
        # Workload label
        # ----------------------------------------------------

        label = workload_label

        # ----------------------------------------------------
        # Crash labeling
        # ----------------------------------------------------
        #
        # For the confirmed crash recording:
        # final 10 seconds of telemetry are marked
        # Instability_Imminent.
        #
        # We do NOT label the entire session.
        #

        if confirmed_crash:

            last_sample_time = df["Timestamp"].iloc[-1]

            window_end = window["Timestamp"].iloc[-1]

            seconds_to_end = (
                last_sample_time - window_end
            ).total_seconds()

            if seconds_to_end <= 10:

                label = "Instability_Imminent"

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        row = {

            "Session_ID":
                window["Session_ID"].iloc[0]
                if "Session_ID" in window.columns
                else filename,

            "Source_File":
                filename,

            "Window_Start":
                window["Timestamp"].iloc[0]
                if "Timestamp" in window.columns
                else np.nan,

            "Window_End":
                window["Timestamp"].iloc[-1]
                if "Timestamp" in window.columns
                else np.nan,

            "Window_Samples":
                len(window),

            "Label":
                label,

        }

        row.update(features)

        processed_rows.append(row)

    print(
        f"  Samples: {len(df)}"
    )

    print(
        f"  Windows: {len(processed_rows)}"
    )

    print(
        f"  Label: {workload_label}"
    )

    return processed_rows


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        PROCESSED_DIR,
        exist_ok=True
    )

    files = glob.glob(
        os.path.join(
            RAW_DIR,
            "*.csv"
        )
    )

    if not files:

        print(
            f"No CSV files found in {RAW_DIR}"
        )

        return

    print(
        f"Found {len(files)} CSV files."
    )

    all_rows = []

    for filepath in files:

        rows = process_file(filepath)

        all_rows.extend(rows)

    if not all_rows:

        print(
            "No processed windows were generated."
        )

        return

    processed_df = pd.DataFrame(
        all_rows
    )

    # --------------------------------------------------------
    # Final cleanup
    # --------------------------------------------------------

    # Remove completely empty feature rows
    feature_columns = [
        column
        for column in processed_df.columns
        if column not in [
            "Session_ID",
            "Source_File",
            "Window_Start",
            "Window_End",
            "Window_Samples",
            "Label",
        ]
    ]

    processed_df = processed_df.dropna(
        subset=feature_columns,
        how="all"
    )

    # Sort by session and time
    processed_df = processed_df.sort_values(
        [
            "Session_ID",
            "Window_Start"
        ]
    ).reset_index(drop=True)

    # Save
    processed_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n===================================")
    print("PREPROCESSING COMPLETE")
    print("===================================")

    print(
        f"Processed windows: {len(processed_df)}"
    )

    print(
        f"Features: {len(feature_columns)}"
    )

    print("\nLabel distribution:")

    print(
        processed_df["Label"]
        .value_counts()
        .to_string()
    )

    print(
        f"\nSaved to:\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()