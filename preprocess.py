import os
import glob
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
EVENTS_FILE = "data/events.csv"

OUTPUT_FILE = os.path.join(
    PROCESSED_DIR,
    "stabilix_processed_v2.csv"
)

WINDOW_SECONDS = 10
STRIDE_SECONDS = 1

# Populated in main() from data/events.csv.
EVENT_MAP = {}


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


def load_event_map():
    """
    Loads confirmed events from data/events.csv.

    The CSV should contain:
        Source_File,Event

    Supported events:
        Application_Crash
        System_Crash

    Files not listed in events.csv are treated as having no
    confirmed event.
    """

    if not os.path.exists(EVENTS_FILE):
        print(f"Event file not found: {EVENTS_FILE}")
        print("Continuing with no confirmed crash events.")
        return {}

    try:
        events_df = pd.read_csv(EVENTS_FILE)

        required_columns = {"Source_File", "Event"}
        missing = required_columns - set(events_df.columns)

        if missing:
            print(
                f"Warning: events.csv is missing columns: {sorted(missing)}"
            )
            return {}

        allowed_events = {
            "Application_Crash",
            "System_Crash"
        }

        event_map = {}

        for _, row in events_df.iterrows():
            source_file = str(row["Source_File"]).strip()
            event = str(row["Event"]).strip()

            if not source_file or source_file.lower() == "nan":
                continue

            if event not in allowed_events:
                print(
                    f"Warning: ignoring unsupported event '{event}' "
                    f"for '{source_file}'"
                )
                continue

            event_map[source_file] = event

        return event_map

    except Exception as e:
        print(f"Could not read events file: {e}")
        return {}


def get_event_for_file(filename, event_map):
    """
    Returns the confirmed event for a raw CSV file.

    Files absent from events.csv have no confirmed event.
    """
    return event_map.get(filename, None)


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

    event = get_event_for_file(filename, EVENT_MAP)

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
        # Event-proximity labeling
        # ----------------------------------------------------
        #
        # Only the final ~10 seconds before a CONFIRMED event
        # receive an event-imminent label.
        #
        # Application_Crash:
        #     Application_Crash_Imminent
        #
        # System_Crash:
        #     Instability_Imminent
        #
        # Earlier windows retain their normal workload label.
        #

        if event is not None:

            if "Timestamp" in df.columns:

                last_sample_time = df["Timestamp"].iloc[-1]
                window_end = window["Timestamp"].iloc[-1]

                seconds_to_end = (
                    last_sample_time - window_end
                ).total_seconds()

            else:
                # Fallback for recordings without timestamps.
                seconds_to_end = (
                    total_windows - 1 - index
                )

            if seconds_to_end <= 10:

                if event == "Application_Crash":
                    label = "Application_Crash_Imminent"

                elif event == "System_Crash":
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

            "Event":
                event if event is not None else "None",

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

    global EVENT_MAP

    os.makedirs(
        PROCESSED_DIR,
        exist_ok=True
    )

    EVENT_MAP = load_event_map()

    print(
        f"Loaded {len(EVENT_MAP)} confirmed event records "
        f"from {EVENTS_FILE}."
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

    # Make sure files without a recorded event use the explicit
    # string "None" instead of pandas NaN.
    processed_df["Event"] = (
        processed_df["Event"]
        .fillna("None")
        .astype(str)
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
            "Event",
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

    print("\nEvent distribution:")

    print(
        processed_df["Event"]
        .value_counts()
        .to_string()
    )

    print(
        f"\nSaved to:\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()