from collections import deque
from math import isnan


WINDOW_SIZE = 10


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
    # GPU_Memory_Used_Avg / GPU_Memory_Used_Latest removed - see
    # train_model.py for why (session-leaky raw MB values).
]


def _values(samples, key):
    return [float(sample.get(key, 0.0) or 0.0) for sample in samples]


def _mean(values):
    if not values:
        return 0.0
    return sum(values) / len(values)


def _sample_std(values):
    # pandas Series.std() uses ddof=1 by default.
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return variance ** 0.5


def _latest(samples, key):
    if not samples:
        return 0.0
    return float(samples[-1].get(key, 0.0) or 0.0)


def _rate(samples, key):
    """
    Same definition as preprocess.py:
        (last - first) / (n - 1)
    """
    values = [
        float(sample.get(key, 0.0) or 0.0)
        for sample in samples
    ]

    if len(values) < 2:
        return 0.0

    return (values[-1] - values[0]) / (len(values) - 1)


class SlidingWindow:
    """
    Maintains the same 10-sample window and feature definitions
    used by preprocess.py.

    At the normal 1 Hz collection rate this corresponds to the
    same 10 samples used by the training windows.
    """

    def __init__(self, window_size=WINDOW_SIZE):
        self.window = deque(maxlen=window_size)

    def add_sample(self, sample):
        if not isinstance(sample, dict):
            raise TypeError("SlidingWindow expects a telemetry dictionary.")
        self.window.append(sample)

    def is_ready(self):
        return len(self.window) == self.window.maxlen

    def get_window(self):
        return list(self.window)

    def compute_features(self):
        if not self.is_ready():
            return None

        samples = list(self.window)

        cpu_usage = _values(samples, "CPU_Usage_Percent")
        cpu_perf = _values(samples, "CPU_Performance_Percent")
        ram = _values(samples, "RAM_Usage_Percent")
        disk_read = _values(samples, "Disk_Read_MBps")
        disk_write = _values(samples, "Disk_Write_MBps")
        gpu_usage = _values(samples, "GPU_Usage_Percent")
        gpu_memory_usage = _values(
            samples,
            "GPU_Memory_Usage_Percent",
        )
        gpu_temp = _values(samples, "GPU_Temperature_C")
        gpu_clock = _values(
            samples,
            "GPU_Graphics_Clock_MHz",
        )

        features = {
            "CPU_Usage_Avg": _mean(cpu_usage),
            "CPU_Usage_Std": _sample_std(cpu_usage),
            "CPU_Performance_Avg": _mean(cpu_perf),
            "CPU_Performance_Latest": _latest(
                samples,
                "CPU_Performance_Percent",
            ),
            "CPU_Performance_Rate": _rate(
                samples,
                "CPU_Performance_Percent",
            ),

            "RAM_Usage_Avg": _mean(ram),
            "RAM_Usage_Std": _sample_std(ram),
            "RAM_Usage_Latest": _latest(
                samples,
                "RAM_Usage_Percent",
            ),

            "Disk_Read_Avg": _mean(disk_read),
            "Disk_Write_Avg": _mean(disk_write),

            "GPU_Usage_Avg": _mean(gpu_usage),
            "GPU_Usage_Std": _sample_std(gpu_usage),
            "GPU_Usage_Latest": _latest(
                samples,
                "GPU_Usage_Percent",
            ),

            "GPU_Memory_Usage_Avg": _mean(gpu_memory_usage),
            "GPU_Memory_Usage_Latest": _latest(
                samples,
                "GPU_Memory_Usage_Percent",
            ),

            "GPU_Temperature_Avg": _mean(gpu_temp),
            "GPU_Temperature_Latest": _latest(
                samples,
                "GPU_Temperature_C",
            ),
            "GPU_Temperature_Rate": _rate(
                samples,
                "GPU_Temperature_C",
            ),

            "GPU_Clock_Avg": _mean(gpu_clock),
            "GPU_Clock_Latest": _latest(
                samples,
                "GPU_Graphics_Clock_MHz",
            ),
            "GPU_Clock_Rate": _rate(
                samples,
                "GPU_Graphics_Clock_MHz",
            ),
        }

        # Guarantee exact schema and order.
        return {
            feature: float(features[feature])
            for feature in FEATURE_COLUMNS
        }