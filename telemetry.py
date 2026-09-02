import psutil
import time

try:
    from pynvml import *
    nvmlInit()
    GPU_AVAILABLE = True
    handle = nvmlDeviceGetHandleByIndex(0)
except:
    GPU_AVAILABLE = False


class Telemetry:

    def __init__(self):
        self.previous_time = time.time()
        self.previous_disk = psutil.disk_io_counters()

    def get_sample(self):

        # ---------------- CPU ----------------

        cpu_usage = psutil.cpu_percent(interval=None)

        cpu_freq = psutil.cpu_freq()

        if cpu_freq:
            cpu_performance = (
                cpu_freq.current / cpu_freq.max
            ) * 100
        else:
            cpu_performance = 0

        # ---------------- RAM ----------------

        ram = psutil.virtual_memory()

        ram_usage = ram.percent

        # ---------------- Disk ----------------

        current_disk = psutil.disk_io_counters()

        current_time = time.time()

        elapsed = current_time - self.previous_time

        read_speed = (
            current_disk.read_bytes -
            self.previous_disk.read_bytes
        ) / 1024 / 1024 / elapsed

        write_speed = (
            current_disk.write_bytes -
            self.previous_disk.write_bytes
        ) / 1024 / 1024 / elapsed

        self.previous_disk = current_disk
        self.previous_time = current_time

        # ---------------- GPU ----------------

        if GPU_AVAILABLE:

            util = nvmlDeviceGetUtilizationRates(handle)

            memory = nvmlDeviceGetMemoryInfo(handle)

            temp = nvmlDeviceGetTemperature(
                handle,
                NVML_TEMPERATURE_GPU
            )

            clock = nvmlDeviceGetClockInfo(
                handle,
                NVML_CLOCK_GRAPHICS
            )

            gpu_usage = util.gpu

            gpu_memory_percent = (
                memory.used /
                memory.total
            ) * 100

            gpu_memory_used = memory.used / 1024 / 1024

            gpu_memory_total = memory.total / 1024 / 1024

            gpu_temp = temp

            gpu_clock = clock

        else:

            gpu_usage = 0
            gpu_memory_percent = 0
            gpu_memory_used = 0
            gpu_memory_total = 0
            gpu_temp = 0
            gpu_clock = 0

        # ---------------- Battery ----------------

        battery = psutil.sensors_battery()

        if battery:

            battery_percent = battery.percent
            charging = battery.power_plugged

        else:

            battery_percent = 100
            charging = False

        return {

            "cpu_usage": cpu_usage,

            "cpu_performance": cpu_performance,

            "ram_usage": ram_usage,

            "disk_read": round(read_speed,2),

            "disk_write": round(write_speed,2),

            "gpu_usage": gpu_usage,

            "gpu_memory_percent": round(gpu_memory_percent,2),

            "gpu_memory_used": round(gpu_memory_used,2),

            "gpu_memory_total": round(gpu_memory_total,2),

            "gpu_temp": gpu_temp,

            "gpu_clock": gpu_clock,

            "battery": battery_percent,

            "charging": charging

        }