from collections import deque


class SlidingWindow:

    def __init__(self, window_size=10):

        self.window = deque(maxlen=window_size)

    def add_sample(self, sample):

        self.window.append(sample)

    def is_ready(self):

        return len(self.window) == self.window.maxlen

    def get_window(self):

        return list(self.window)

    def compute_features(self):

        if not self.is_ready():
            return None

        samples = list(self.window)

        # -------------------------
        # Rolling Averages
        # -------------------------

        cpu_avg = sum(s["cpu_usage"] for s in samples) / len(samples)

        gpu_avg = sum(s["gpu_usage"] for s in samples) / len(samples)

        ram_avg = sum(s["ram_usage"] for s in samples) / len(samples)

        disk_read_avg = sum(s["disk_read"] for s in samples) / len(samples)

        disk_write_avg = sum(s["disk_write"] for s in samples) / len(samples)

        gpu_mem_avg = sum(s["gpu_memory_percent"] for s in samples) / len(samples)

        # -------------------------
        # Current Values
        # -------------------------

        latest = samples[-1]

        cpu_perf = latest["cpu_performance"]

        gpu_temp = latest["gpu_temp"]

        gpu_clock = latest["gpu_clock"]

        # -------------------------
        # Rate of Change
        # -------------------------

        cpu_perf_rate = (
            samples[-1]["cpu_performance"]
            - samples[0]["cpu_performance"]
        ) / len(samples)

        gpu_temp_rate = (
            samples[-1]["gpu_temp"]
            - samples[0]["gpu_temp"]
        ) / len(samples)

        gpu_clock_rate = (
            samples[-1]["gpu_clock"]
            - samples[0]["gpu_clock"]
        ) / len(samples)

        return {

            # Rolling averages

            "cpu_avg": round(cpu_avg,2),

            "gpu_avg": round(gpu_avg,2),

            "ram_avg": round(ram_avg,2),

            "disk_read_avg": round(disk_read_avg,2),

            "disk_write_avg": round(disk_write_avg,2),

            "gpu_memory_avg": round(gpu_mem_avg,2),

            # Current values

            "cpu_performance": round(cpu_perf,2),

            "gpu_temperature": gpu_temp,

            "gpu_clock": gpu_clock,

            # Rates

            "cpu_perf_rate": round(cpu_perf_rate,2),

            "gpu_temp_rate": round(gpu_temp_rate,2),

            "gpu_clock_rate": round(gpu_clock_rate,2)

        }