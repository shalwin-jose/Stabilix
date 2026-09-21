import csv
import time
from datetime import datetime

import psutil
import win32pdh
from pynvml import *

# ==========================================================
# SETTINGS
# ==========================================================

INTERVAL = 1.0  # seconds

SESSION_NAME = input("Enter Session Name (e.g. idle, gaming, youtube): ").strip()

FILE_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

SESSION_ID = f"{SESSION_NAME}_{FILE_TIMESTAMP}"

CSV_FILE = f"{SESSION_ID}.csv"

# ==========================================================
# NVIDIA INITIALIZATION
# ==========================================================

try:
    nvmlInit()
    gpu = nvmlDeviceGetHandleByIndex(0)

except Exception as e:
    print("Failed to initialize NVIDIA NVML.")
    print(e)
    exit()

# ==========================================================
# WINDOWS PERFORMANCE COUNTER
# ==========================================================

query = win32pdh.OpenQuery()

counter = win32pdh.AddCounter(
    query,
    r"\Processor Information(_Total)\% Processor Performance"
)

# Prime counter
win32pdh.CollectQueryData(query)
time.sleep(1)

# Used if Windows Performance Counter temporarily fails
previous_cpu_perf = 100

# ==========================================================
# CSV HEADER
# ==========================================================

header = [

    "Sample_Number",
    "Timestamp",
    "Elapsed_Time_Seconds",
    "Session_ID",

    "CPU_Usage_Percent",
    "CPU_Performance_Percent",

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
    "Charging"

]

# ==========================================================
# START COLLECTION
# ==========================================================

with open(CSV_FILE, "w", newline="") as f:

    writer = csv.writer(f)
    writer.writerow(header)

    start_time = time.time()

    sample_number = 1

    prev_disk = psutil.disk_io_counters()

    print("\n======================================================")
    print("      Thermal Telemetry Collector v1.0")
    print("======================================================")
    print(f"Session Name : {SESSION_NAME}")
    print(f"Session ID   : {SESSION_ID}")
    print(f"Output File  : {CSV_FILE}")
    print(f"Sampling     : {INTERVAL:.1f} Sample / Second")
    print("Press CTRL+C to Stop")
    print("======================================================\n")

    try:

        while True:

            loop_start = time.time()

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            elapsed = round(loop_start - start_time, 2)

            # ==================================================
            # CPU
            # ==================================================

            cpu_usage = psutil.cpu_percent(interval=None)

            try:

                win32pdh.CollectQueryData(query)

                _, cpu_perf = win32pdh.GetFormattedCounterValue(
                    counter,
                    win32pdh.PDH_FMT_LONG
                )

                previous_cpu_perf = cpu_perf

            except Exception:

                cpu_perf = previous_cpu_perf

            # ==================================================
            # RAM
            # ==================================================

            ram = psutil.virtual_memory().percent

            # ==================================================
            # DISK
            # ==================================================

            current_disk = psutil.disk_io_counters()

            read_speed = (
                current_disk.read_bytes -
                prev_disk.read_bytes
            ) / (1024 * 1024 * INTERVAL)

            write_speed = (
                current_disk.write_bytes -
                prev_disk.write_bytes
            ) / (1024 * 1024 * INTERVAL)

            prev_disk = current_disk

            # ==================================================
            # GPU
            # ==================================================

            util = nvmlDeviceGetUtilizationRates(gpu)

            mem = nvmlDeviceGetMemoryInfo(gpu)

            gpu_temp = nvmlDeviceGetTemperature(
                gpu,
                NVML_TEMPERATURE_GPU
            )

            gpu_clock = nvmlDeviceGetClockInfo(
                gpu,
                NVML_CLOCK_GRAPHICS
            )

            # ==================================================
            # BATTERY
            # ==================================================

            battery = psutil.sensors_battery()

            if battery is not None:
                battery_percent = battery.percent
                charging = battery.power_plugged
            else:
                battery_percent = -1
                charging = False

            # ==================================================
            # SAVE ROW
            # ==================================================

            writer.writerow([

                sample_number,

                timestamp,

                elapsed,

                SESSION_ID,

                round(cpu_usage, 2),

                cpu_perf,

                round(ram, 2),

                round(read_speed, 3),

                round(write_speed, 3),

                util.gpu,

                util.memory,

                round(mem.used / (1024 ** 2), 2),

                round(mem.total / (1024 ** 2), 2),

                gpu_temp,

                gpu_clock,

                battery_percent,

                charging

            ])

            f.flush()

            print(
                f"{sample_number:6d} | "
                f"{elapsed:7.1f}s | "
                f"CPU {cpu_usage:5.1f}% | "
                f"Perf {cpu_perf:3d}% | "
                f"GPU {util.gpu:3d}% | "
                f"Temp {gpu_temp:2d}°C | "
                f"Clock {gpu_clock:4d} MHz"
            )

            sample_number += 1

            elapsed_loop = time.time() - loop_start

            if elapsed_loop < INTERVAL:
                time.sleep(INTERVAL - elapsed_loop)

    except KeyboardInterrupt:

        print("\n======================================================")
        print("Collection Stopped")
        print("======================================================")

    finally:

        nvmlShutdown()
        win32pdh.CloseQuery(query)

        print(f"\nDataset saved successfully: {CSV_FILE}")