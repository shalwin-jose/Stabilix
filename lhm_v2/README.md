# Stabilix LHM V2

This package is the LibreHardwareMonitor based pipeline. It uses the same
canonical sensor map and timestamp-based feature engine for training captures
and live telemetry.

## Build the training data and model

From the Stabilix project directory, run:

```powershell
python -m pip install -r .\requirements-lhm-v2.txt
python -m lhm_v2.preprocess_lhm
python -m lhm_v2.train_lhm
```

Raw captures are in `data/lhm_raw`; preprocessing writes
`data/processed/stabilix_lhm_v2_training.csv`; training writes
`models/stabilix_lhm_v2.joblib`. Each session is processed independently. A
10-second rolling window uses actual sample timestamps and creates 168
features (28 LHM signals × average, standard deviation, minimum, maximum,
latest value, and per-second slope).

## Live overlay

Start Stabilix with `python main.py`. In LibreHardwareMonitor, enable
**Options → Remote Web Server → Run**. The overlay polls the server's
`/data.json` endpoint and calls the same `FeatureEngine` used during offline
preprocessing. By default it connects to `http://127.0.0.1:8085`; set the
`STABILIX_LHM_URL` environment variable if the server runs on another host or
port. Stabilix suppresses predictions until CPU load, GPU load, and GPU
temperature are all present, so the model cannot silently turn missing
telemetry into a workload label.

## Label assumptions

The workload state starts from each capture's `Workload_Label` and filename.
Any 10-second window whose latest `RAM_Load` is at least 80% is labeled
`RAM_Heavy`, regardless of the capture's original workload. Windows inherited
from a RAM-heavy capture but below 80% are labeled `Normal_Usage`, so the
training class never includes below-threshold RAM examples. A GPU-heavy capture
window is labeled `GPU_Heavy` only when both its 10-second average and latest
GPU load are at least 10%; lower-activity windows are labeled `Normal_Usage`.
Network-heavy capture windows require both the 10-second average and latest
Wi-Fi download/upload maximum to reach 50; otherwise they are labeled
`Normal_Usage`. The overlay displays live Wi-Fi readings alongside CPU, GPU,
RAM, and temperature. A window is labeled `Stable_Idle` when CPU average is
below 10% and latest is below 15%, GPU average/latest are below 5%, RAM is
below 80%, and Wi-Fi average/latest are both below 50. This relabels quiet windows even when their
recording was tagged as normal usage.
The instability trend is retained independently of that workload relabeling.
Other windows use the capture label. The stable trend class is named
`No_Imminent_Instability` in training and displayed as “No imminent
instability.” The overlay requires a class to win at least three of the last
five predictions before changing the displayed label, smoothing brief
one- or two-window model errors while keeping the risk score live.

Sessions with crash descriptions (`rapidcrash`, `crashheavy`, `crash`,
`crashtest`), names containing `crashed`, and short GPU/game sessions under
five minutes are labeled as ending in an application crash. The last 10
seconds are `Crash_Imminent`; the preceding 30 seconds are
`Approaching_Instability`. This uses the project capture notes, not a crash
threshold inferred from sensor values. It assumes the logger's final sample
is close to the crash; longer-term validation needs more confirmed crash
captures and exact event timestamps.
