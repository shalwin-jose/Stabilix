"""Shared LibreHardwareMonitor sensor mapping and 10-second feature contract.

Both batch preprocessing and live telemetry call this module. Keep the
canonical sensor names and statistic order stable after training a model.
"""
from collections import deque
import re

import numpy as np

WINDOW_SECONDS = 10.0
SAMPLE_PERIOD_SECONDS = 0.2

# Canonical signal names map to the end of LHM's exported Sensor_* column.
# Patterns intentionally avoid machine-name prefixes and DIMM identifiers.
SENSOR_PATTERNS = {
    "CPU_Total_Load": r"Load_CPU_Total_Load$",
    "CPU_Core_Max_Load": r"Load_CPU_Core_Max_Load$",
    "CPU_Package_Power": r"Powers_CPU_Package_Power$",
    "CPU_Cores_Power": r"Powers_CPU_Cores_Power$",
    "CPU_Package_Temperature": r"Temperatures_CPU_Package_Temperature$",
    "CPU_Core_Average_Temperature": r"Temperatures_Core_Average_Temperature$",
    "CPU_Core_Max_Temperature": r"Temperatures_Core_Max_Temperature$",
    "CPU_Core_Clock": r"Clocks_CPU_Core_\d+_Clock$",
    "GPU_Core_Clock": r"NVIDIA.*Clocks_GPU_Core_Clock$",
    "GPU_Memory_Clock": r"NVIDIA.*Clocks_GPU_Memory_Clock$",
    "GPU_Core_Temperature": r"NVIDIA.*Temperatures_GPU_Core_Temperature$",
    "GPU_Hot_Spot_Temperature": r"NVIDIA.*Temperatures_GPU_Hot_Spot_Temperature$",
    "GPU_Memory_Junction_Temperature": r"NVIDIA.*Temperatures_GPU_Memory_Junction_Temperature$",
    "GPU_Core_Load": r"NVIDIA.*Load_GPU_Core_Load$",
    "GPU_Memory_Controller_Load": r"NVIDIA.*Load_GPU_Memory_Controller_Load$",
    "GPU_Memory_Load": r"NVIDIA.*Load_GPU_Memory_Load$",
    "GPU_Package_Power": r"NVIDIA.*Powers_GPU_Package_Power$",
    "GPU_Core_Voltage": r"NVIDIA.*Voltages_GPU_Core_Voltage_Voltage$",
    "GPU_VRAM_Used": r"NVIDIA.*Data_GPU_Memory_Used_SmallData$",
    "RAM_Load": r"Total_Memory_Load_Memory_Load$",
    "RAM_Used": r"Total_Memory_Data_Memory_Used_Data$",
    "VMem_Load": r"Virtual_Memory_Load_Memory_Load$",
    "VMem_Used": r"Virtual_Memory_Data_Memory_Used_Data$",
    "Disk_Read_Rate": r"Unknown_Throughput_Read_Rate_Throughput$",
    "Disk_Write_Rate": r"Unknown_Throughput_Write_Rate_Throughput$",
    "WiFi_Download_Rate": r"Wi-Fi_Throughput_Download_Speed_Throughput$",
    "WiFi_Upload_Rate": r"Wi-Fi_Throughput_Upload_Speed_Throughput$",
    "Battery_Voltage": r"A32-K55_Voltages_Voltage_Voltage$",
}
STATISTICS = ("Avg", "Std", "Min", "Max", "Latest", "Rate")
FEATURE_COLUMNS = [f"{sensor}_{stat}" for sensor in SENSOR_PATTERNS for stat in STATISTICS]


def canonicalize_columns(frame):
    """Return timestamps, session metadata and a canonical numeric sensor frame."""
    import pandas as pd

    out = pd.DataFrame(index=frame.index)
    timestamp_col = next((c for c in ("Unix_Time", "Timestamp") if c in frame), None)
    if timestamp_col == "Unix_Time":
        out["_time"] = pd.to_numeric(frame[timestamp_col], errors="coerce")
    elif timestamp_col:
        out["_time"] = pd.to_datetime(frame[timestamp_col], errors="coerce").astype("int64") / 1e9
        out.loc[pd.to_datetime(frame[timestamp_col], errors="coerce").isna(), "_time"] = np.nan
    else:
        raise ValueError("LHM CSV needs Unix_Time or Timestamp")

    for key in ("Session_ID", "Workload_Label", "Session_Description", "Sample_Index"):
        if key in frame:
            out[key] = frame[key]

    cols = list(frame.columns)
    for sensor, pattern in SENSOR_PATTERNS.items():
        matches = [c for c in cols if re.search(pattern, c, re.IGNORECASE)]
        # Aggregate per-core clocks when LHM supplies individual channels.
        if matches:
            vals = frame[matches].apply(pd.to_numeric, errors="coerce")
            out[sensor] = vals.max(axis=1) if sensor in ("CPU_Core_Max_Load", "CPU_Core_Max_Temperature", "GPU_Core_Temperature", "GPU_Hot_Spot_Temperature", "GPU_Memory_Junction_Temperature") else vals.mean(axis=1)
        else:
            out[sensor] = np.nan
    return out


class FeatureEngine:
    """Streaming time-based feature window; same implementation used in batch."""
    def __init__(self, window_seconds=WINDOW_SECONDS, min_samples=3):
        self.window_seconds = float(window_seconds)
        self.min_samples = int(min_samples)
        self.samples = deque()

    def reset(self):
        self.samples.clear()

    def add_sample(self, timestamp, sensors):
        row = {"_time": float(timestamp)}
        for name in SENSOR_PATTERNS:
            try:
                value = float(sensors.get(name, np.nan))
                row[name] = value if np.isfinite(value) else np.nan
            except (TypeError, ValueError):
                row[name] = np.nan
        if self.samples and row["_time"] <= self.samples[-1]["_time"]:
            return None
        self.samples.append(row)
        cutoff = row["_time"] - self.window_seconds
        while len(self.samples) > 1 and self.samples[1]["_time"] <= cutoff:
            self.samples.popleft()
        if len(self.samples) < self.min_samples or row["_time"] - self.samples[0]["_time"] < self.window_seconds * 0.90:
            return None
        return self.compute()

    def compute(self):
        rows = list(self.samples)
        times = np.asarray([r["_time"] for r in rows], dtype=float)
        feats = {}
        for sensor in SENSOR_PATTERNS:
            vals = np.asarray([r[sensor] for r in rows], dtype=float)
            mask = np.isfinite(vals)
            x, y = times[mask], vals[mask]
            prefix = f"{sensor}_"
            if not len(y):
                stats = [np.nan] * 6
            else:
                # Center time first: Unix timestamps are around 1e9 seconds,
                # and fitting directly against them loses slope precision.
                if len(y) > 1 and np.ptp(x) > 0:
                    centered_x = x - x.mean()
                    centered_y = y - y.mean()
                    rate = float(np.dot(centered_x, centered_y) / np.dot(centered_x, centered_x))
                else:
                    rate = np.nan
                stats = [float(np.mean(y)), float(np.std(y, ddof=1)) if len(y) > 1 else 0.0,
                         float(np.min(y)), float(np.max(y)), float(y[-1]), rate]
            feats.update(zip((prefix + s for s in STATISTICS), stats))
        return feats


def engineer_frame(canonical_frame, window_seconds=WINDOW_SECONDS):
    """Build one feature row per sample, always restarting at session boundaries."""
    engine = FeatureEngine(window_seconds=window_seconds)
    records = []
    for _, row in canonical_frame.iterrows():
        sensors = {name: row.get(name, np.nan) for name in SENSOR_PATTERNS}
        features = engine.add_sample(row["_time"], sensors)
        records.append(features if features is not None else {k: np.nan for k in FEATURE_COLUMNS})
    return __import__("pandas").DataFrame(records, index=canonical_frame.index, columns=FEATURE_COLUMNS)
