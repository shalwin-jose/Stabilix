"""Convert raw LHM captures to session-safe, shared-contract training rows."""
from pathlib import Path

import pandas as pd

from lhm_v2.features import FEATURE_COLUMNS, WINDOW_SECONDS, canonicalize_columns, engineer_frame

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "lhm_raw"
OUTPUT = ROOT / "data" / "processed" / "stabilix_lhm_v2_training.csv"
CRASH_DESCRIPTIONS = {"rapidcrash", "crash", "crashheavy", "crashtest"}


def workload_state(label, filename):
    text = str(label or "").strip().lower()
    name = filename.lower()
    if "idle" in text or name.startswith("idle"):
        return "Stable_Idle"
    if "cpu" in text or name.startswith("cpuheavy"):
        return "CPU_Heavy"
    if "gpu" in text or "gaming" in text or "gpu" in name or "game" in name:
        return "GPU_Heavy"
    if "ram" in name or "heavyram" in name:
        return "RAM_Heavy"
    if "network" in text or "network" in name:
        return "Network_Heavy"
    if "normal" in text:
        return "Normal_Usage"
    return "Normal_Usage"


def is_crash_session(path, raw, state, duration):
    desc = str(raw["Session_Description"].iloc[0] if "Session_Description" in raw else "").strip().lower()
    name = path.stem.lower()
    if desc in CRASH_DESCRIPTIONS or "crashed" in name:
        return True
    # Per project capture notes: short GPU/game runs ended when the game crashed.
    return state == "GPU_Heavy" and duration < 300.0


def label_trend(times, crash, crash_horizon=10.0, approach_horizon=30.0):
    trend = pd.Series("No_Imminent_Instability", index=times.index, dtype="object")
    if crash:
        end = float(times.iloc[-1])
        remaining = end - times
        trend.loc[remaining <= crash_horizon] = "Crash_Imminent"
        trend.loc[(remaining > crash_horizon) & (remaining <= crash_horizon + approach_horizon)] = "Approaching_Instability"
    return trend


def build_dataset():
    records = []
    paths = sorted(RAW_DIR.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No LHM captures in {RAW_DIR}")
    for path in paths:
        raw = pd.read_csv(path, low_memory=False)
        canonical = canonicalize_columns(raw)
        canonical = canonical.sort_values("_time").dropna(subset=["_time"]).reset_index(drop=True)
        if len(canonical) < 3:
            continue
        duration = float(canonical["_time"].iloc[-1] - canonical["_time"].iloc[0])
        state = workload_state(canonical.get("Workload_Label", pd.Series([""])).iloc[0], path.name)
        crash = is_crash_session(path, canonical, state, duration)
        features = engineer_frame(canonical, WINDOW_SECONDS)
        ready = features.notna().any(axis=1)
        features = features.loc[ready].reset_index(drop=True)
        times = canonical.loc[ready, "_time"].reset_index(drop=True)
        trend = label_trend(times, crash)
        part = features
        part.insert(0, "Session_ID", path.stem)
        part.insert(1, "Source_File", path.name)
        # RAM-heavy is defined by measured memory pressure, regardless of the
        # capture filename/workload metadata. Use each window's latest reading
        # so windows above 80% become RAM-heavy training examples.
        ram_load = pd.to_numeric(part["RAM_Load_Latest"], errors="coerce")
        states = pd.Series(state, index=part.index, dtype="object")
        high_ram = ram_load >= 80.0
        # Do not keep filename-derived RAM_Heavy labels below the threshold;
        # treat those windows as ordinary usage until the measured threshold
        # is crossed. This keeps the class definition consistent both ways.
        states.loc[(states == "RAM_Heavy") & ~high_ram] = "Normal_Usage"
        states.loc[high_ram] = "RAM_Heavy"
        # Filename-derived GPU labels also need matching sensor evidence.
        # Require both the window average and latest GPU load to reach 10%;
        # otherwise the workload is not GPU-heavy in that window. Keep its
        # instability trend, which is labeled independently below.
        gpu_avg = pd.to_numeric(part["GPU_Core_Load_Avg"], errors="coerce")
        gpu_latest = pd.to_numeric(part["GPU_Core_Load_Latest"], errors="coerce")
        low_gpu = (gpu_avg < 10.0) | (gpu_latest < 10.0) | gpu_avg.isna() | gpu_latest.isna()
        states.loc[(states == "GPU_Heavy") & low_gpu] = "Normal_Usage"
        states.loc[(state == "GPU_Heavy") & ~low_gpu] = "GPU_Heavy"
        # Network-heavy captures must also show sustained traffic on the
        # network sensors; a filename alone is not enough to train this class.
        wifi_down_avg = pd.to_numeric(part["WiFi_Download_Rate_Avg"], errors="coerce")
        wifi_up_avg = pd.to_numeric(part["WiFi_Upload_Rate_Avg"], errors="coerce")
        wifi_down_latest = pd.to_numeric(part["WiFi_Download_Rate_Latest"], errors="coerce")
        wifi_up_latest = pd.to_numeric(part["WiFi_Upload_Rate_Latest"], errors="coerce")
        network_avg = pd.concat([wifi_down_avg, wifi_up_avg], axis=1).max(axis=1)
        network_latest = pd.concat([wifi_down_latest, wifi_up_latest], axis=1).max(axis=1)
        low_network = ((network_avg < 50.0) | (network_latest < 50.0)
                       | network_avg.isna() | network_latest.isna())
        states.loc[(states == "Network_Heavy") & low_network] = "Normal_Usage"
        # Give genuinely quiet windows an explicit idle class even when the
        # recording was tagged as normal usage. Require CPU below 10%, low GPU,
        # non-heavy RAM, and no sustained Wi-Fi traffic.
        cpu_avg = pd.to_numeric(part["CPU_Total_Load_Avg"], errors="coerce")
        cpu_latest = pd.to_numeric(part["CPU_Total_Load_Latest"], errors="coerce")
        low_cpu = (cpu_avg < 10.0) & (cpu_latest < 15.0)
        low_gpu_for_idle = (gpu_avg < 5.0) & (gpu_latest < 5.0)
        low_wifi = (network_avg < 50.0) & (network_latest < 50.0)
        stable_idle = low_cpu & low_gpu_for_idle & (ram_load < 80.0) & low_wifi
        stable_idle = stable_idle.fillna(False)
        states.loc[stable_idle] = "Stable_Idle"
        # RAM remains the stronger workload label when both thresholds are met.
        states.loc[high_ram] = "RAM_Heavy"
        part.insert(2, "State", states)
        part.insert(3, "Trend", trend)
        part.insert(4, "Label", part["State"] + "__" + part["Trend"])
        part.insert(5, "Elapsed_Seconds", times - times.iloc[0])
        records.append(part)
        print(f"{path.name}: {len(part)} windows, {state}, crash={crash}, duration={duration:.1f}s")
    result = pd.concat(records, ignore_index=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False)
    print(f"\nSaved {len(result)} rows x {len(FEATURE_COLUMNS)} engineered features to {OUTPUT}")
    print("Labels:\n" + result.Label.value_counts().to_string())
    return result


if __name__ == "__main__":
    build_dataset()
