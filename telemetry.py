import time

import psutil

# Optional Windows Performance Counter support.
try:
    import win32pdh

    PDH_AVAILABLE = True
except Exception as e:
    win32pdh = None
    PDH_AVAILABLE = False
    print("PDH unavailable:", e)

# Optional NVIDIA NVML support.
try:
    import pynvml

    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except Exception as e:
    pynvml = None
    NVML_AVAILABLE = False
    print("NVIDIA NVML unavailable:", e)


class Telemetry:
    """
    Live telemetry collector.

    The runtime schema is intentionally aligned with the raw telemetry
    used by preprocess.py:

        CPU_Usage_Percent
        CPU_Performance_Percent
        RAM_Usage_Percent
        Disk_Read_MBps
        Disk_Write_MBps
        GPU_Usage_Percent
        GPU_Memory_Usage_Percent
        GPU_Memory_Used_MB
        GPU_Memory_Total_MB
        GPU_Temperature_C
        GPU_Graphics_Clock_MHz

    Extra display-only values (CPU frequency, CPU temperature, battery)
    are also returned.
    """

    def __init__(self, profile="hybrid"):
        self.profile = profile

        self.previous_time = time.time()
        self.previous_disk = psutil.disk_io_counters()

        self.pdh_query = None
        self.pdh_counter = None

        # True only once we've actually confirmed a working PDH
        # counter. CPU_Performance_* features were trained on real
        # "% Processor Performance" PDH readings; the frequency-ratio
        # fallback below is a DIFFERENT signal (different scale and
        # dynamics), so silently sliding into it makes 3 of the 23
        # model features out-of-distribution without anyone noticing.
        self.using_pdh_fallback = True

        if PDH_AVAILABLE:
            try:
                self.pdh_query = win32pdh.OpenQuery()
                self.pdh_counter = win32pdh.AddCounter(
                    self.pdh_query,
                    r"\\Processor Information(_Total)\\% Processor Performance",
                )
                win32pdh.CollectQueryData(self.pdh_query)
                self.using_pdh_fallback = False
            except Exception as e:
                self.pdh_query = None
                self.pdh_counter = None
                print("PDH initialization failed:", e)

        if self.using_pdh_fallback:
            print("\n" + "!" * 70)
            print("WARNING: CPU_Performance_* is running on the FREQUENCY-RATIO")
            print("FALLBACK, not the PDH counter used to train the model.")
            print("Predictions may be unreliable until this is resolved")
            print("(check that pywin32 is installed and PDH counters are")
            print("accessible on this machine).")
            print("!" * 70 + "\n")

        self.gpu_handle = None
        self.last_gpu = {
            "GPU_Usage_Percent": 0.0,
            "GPU_Memory_Usage_Percent": 0.0,
            "GPU_Memory_Used_MB": 0.0,
            "GPU_Memory_Total_MB": 0.0,
            "GPU_Temperature_C": 0.0,
            "GPU_Graphics_Clock_MHz": 0.0,
        }

        self.gpu_error_count = 0
        self._refresh_gpu_handle()

        # psutil's first cpu_percent() call is only a baseline.
        psutil.cpu_percent(interval=None)

    # ------------------------------------------------------------
    # GPU
    # ------------------------------------------------------------

    def _refresh_gpu_handle(self):
        if not NVML_AVAILABLE:
            self.gpu_handle = None
            return False

        try:
            count = pynvml.nvmlDeviceGetCount()
            if count <= 0:
                self.gpu_handle = None
                return False

            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            return True
        except Exception as e:
            self.gpu_handle = None
            print("Could not refresh NVIDIA GPU handle:", e)
            return False

    def _get_gpu(self):
        if self.gpu_handle is None:
            self._refresh_gpu_handle()

        if self.gpu_handle is None:
            return dict(self.last_gpu)

        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            memory = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            temp = pynvml.nvmlDeviceGetTemperature(
                self.gpu_handle,
                pynvml.NVML_TEMPERATURE_GPU,
            )
            clock = pynvml.nvmlDeviceGetClockInfo(
                self.gpu_handle,
                pynvml.NVML_CLOCK_GRAPHICS,
            )

            total = float(memory.total)
            used = float(memory.used)

            memory_percent = (
                (used / total) * 100.0
                if total > 0
                else 0.0
            )

            self.last_gpu = {
                "GPU_Usage_Percent": float(util.gpu),
                "GPU_Memory_Usage_Percent": memory_percent,
                "GPU_Memory_Used_MB": used / 1024.0 / 1024.0,
                "GPU_Memory_Total_MB": total / 1024.0 / 1024.0,
                "GPU_Temperature_C": float(temp),
                "GPU_Graphics_Clock_MHz": float(clock),
            }

            self.gpu_error_count = 0
            return dict(self.last_gpu)

        except Exception as e:
            self.gpu_error_count += 1
            print(
                f"GPU telemetry warning "
                f"(attempt {self.gpu_error_count}): {e}"
            )

            # The GPU may have temporarily disappeared after a
            # driver/application state change. Try to reacquire it.
            self._refresh_gpu_handle()

            return dict(self.last_gpu)

    # ------------------------------------------------------------
    # CPU performance
    # ------------------------------------------------------------

    def _get_cpu_performance(self, cpu_frequency):
        """
        Use the same Windows PDH counter used by the training collector.

        If PDH is unavailable, fall back to CPU current/max frequency.
        The fallback is explicitly a fallback because it is not identical
        to the training collector's PDH measurement.
        """
        if self.pdh_query is not None and self.pdh_counter is not None:
            try:
                win32pdh.CollectQueryData(self.pdh_query)
                _, value = win32pdh.GetFormattedCounterValue(
                    self.pdh_counter,
                    win32pdh.PDH_FMT_LONG,
                )
                self.using_pdh_fallback = False
                return float(value)
            except Exception as e:
                print("PDH CPU performance warning (falling back):", e)
                self.using_pdh_fallback = True

        if cpu_frequency and cpu_frequency.max:
            return (
                float(cpu_frequency.current)
                / float(cpu_frequency.max)
            ) * 100.0

        return 0.0

    # ------------------------------------------------------------
    # CPU temperature
    # ------------------------------------------------------------

    def _get_cpu_temperature(self):
        """
        CPU temperature is display-only in V1.

        The original 23-feature model does not use CPU temperature,
        so failure to obtain it must never affect prediction.
        """
        return None

    # ------------------------------------------------------------
    # Main sample
    # ------------------------------------------------------------

    def get_sample(self):
        now = time.time()
        elapsed = now - self.previous_time
        if elapsed <= 0:
            elapsed = 0.001

        cpu_usage = psutil.cpu_percent(interval=None)
        cpu_frequency = psutil.cpu_freq()
        cpu_performance = self._get_cpu_performance(cpu_frequency)

        ram_usage = psutil.virtual_memory().percent

        current_disk = psutil.disk_io_counters()
        read_speed = (
            current_disk.read_bytes
            - self.previous_disk.read_bytes
        ) / 1024.0 / 1024.0 / elapsed

        write_speed = (
            current_disk.write_bytes
            - self.previous_disk.write_bytes
        ) / 1024.0 / 1024.0 / elapsed

        self.previous_disk = current_disk
        self.previous_time = now

        gpu = self._get_gpu()

        battery = psutil.sensors_battery()
        if battery is None:
            battery_percent = 100.0
            charging = False
        else:
            battery_percent = float(battery.percent)
            charging = bool(battery.power_plugged)

        cpu_frequency_current = (
            float(cpu_frequency.current)
            if cpu_frequency
            else 0.0
        )

        cpu_temperature = self._get_cpu_temperature()

        return {
            # Runtime keys used by SlidingWindow.
            "CPU_Usage_Percent": float(cpu_usage),
            "CPU_Performance_Percent": float(cpu_performance),
            "RAM_Usage_Percent": float(ram_usage),
            "Disk_Read_MBps": float(read_speed),
            "Disk_Write_MBps": float(write_speed),
            **gpu,

            # Display-only values.
            "CPU_Frequency_MHz": cpu_frequency_current,
            "CPU_Temperature_C": cpu_temperature,
            "Battery_Percentage": battery_percent,
            "Charging": charging,

            # Diagnostic flag: True means CPU_Performance_Percent came
            # from the frequency-ratio fallback, not the PDH counter
            # the model was trained on.
            "CPU_Performance_Is_Fallback": self.using_pdh_fallback,

            # Backward-compatible short names used by the UI.
            "cpu_usage": float(cpu_usage),
            "cpu_performance": float(cpu_performance),
            "ram_usage": float(ram_usage),
            "disk_read": float(read_speed),
            "disk_write": float(write_speed),
            "gpu_usage": gpu["GPU_Usage_Percent"],
            "gpu_memory_percent": gpu["GPU_Memory_Usage_Percent"],
            "gpu_memory_used": gpu["GPU_Memory_Used_MB"],
            "gpu_memory_total": gpu["GPU_Memory_Total_MB"],
            "gpu_temp": gpu["GPU_Temperature_C"],
            "gpu_clock": gpu["GPU_Graphics_Clock_MHz"],
            "battery": battery_percent,
            "charging": charging,
        }