"""Read live values from LibreHardwareMonitor's Remote Web Server JSON API."""
import json
import math
import os
import re
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

from lhm_v2.features import SENSOR_PATTERNS

DEFAULT_SERVER_URL = os.environ.get("STABILIX_LHM_URL", "http://127.0.0.1:8085").rstrip("/")

def _key(value):
    return " ".join(str(value or "").lower().replace("_", " ").replace("/", " ").replace("-", " ").split())


def _matches(sensor, words, excludes=()):
    text = " ".join(_key(sensor.get(k, "")) for k in ("Name", "Identifier", "Parent", "SensorType"))
    return all(w in text for w in words) and not any(w in text for w in excludes)


def _parse_lhm_value(value):
    """Parse numeric or unit-formatted values exactly like monitorv2.py."""
    if value is None:
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", ".")
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text)
    return float(match.group(0)) if match else float("nan")


class LhmTelemetry:
    """Poll LHM's JSON tree and map it to the canonical CSV sensor schema."""
    RULES = {
        "CPU_Total_Load": (("cpu total",), ("thread",)),
        "CPU_Core_Max_Load": (("cpu core max",), ()),
        "CPU_Package_Power": (("cpu package", "power"), ()),
        "CPU_Cores_Power": (("cpu cores", "power"), ()),
        "CPU_Package_Temperature": (("cpu package", "temperature"), ()),
        "CPU_Core_Average_Temperature": (("core average", "temperature"), ()),
        "CPU_Core_Max_Temperature": (("core max", "temperature"), ()),
        "CPU_Core_Clock": (("cpu core", "clock"), ("bus",)),
        "GPU_Core_Clock": (("gpu core", "clock"), ()),
        "GPU_Memory_Clock": (("gpu memory", "clock"), ()),
        "GPU_Core_Temperature": (("gpu core", "temperature"), ()),
        "GPU_Hot_Spot_Temperature": (("hot spot", "temperature"), ()),
        "GPU_Memory_Junction_Temperature": (("memory junction", "temperature"), ()),
        "GPU_Core_Load": (("gpu core", "load"), ("memory", "video", "bus")),
        "GPU_Memory_Controller_Load": (("memory controller", "load"), ()),
        "GPU_Memory_Load": (("gpu memory", "load"), ("controller",)),
        "GPU_Package_Power": (("gpu package", "power"), ()),
        "GPU_Core_Voltage": (("gpu core", "voltage"), ()),
        "GPU_VRAM_Used": (("gpu memory", "used"), ("free",)),
        # Scope these to LHM's Total Memory hardware node. Without this,
        # "GPU Memory" sensors also satisfy the generic memory/load rule.
        "RAM_Load": (("total memory", "load"), ("virtual", "gpu")),
        "RAM_Used": (("total memory", "used"), ("virtual", "gpu")),
        "VMem_Load": (("virtual memory", "load"), ()),
        "VMem_Used": (("virtual memory", "used"), ()),
        "Disk_Read_Rate": (("read", "rate"), ()),
        "Disk_Write_Rate": (("write", "rate"), ()),
        "WiFi_Download_Rate": (("wi fi", "download"), ()),
        "WiFi_Upload_Rate": (("wi fi", "upload"), ()),
        "Battery_Voltage": (("a32 k55", "voltage"), ()),
    }

    def __init__(self, server_url=DEFAULT_SERVER_URL):
        if len(self.RULES) != len(SENSOR_PATTERNS):
            raise RuntimeError("Live sensor map and offline LHM schema differ")
        self.server_url = server_url.rstrip("/")
        self._read_tree()  # fail early with an actionable URL if server is off
        self.started = time.time()

    def _read_tree(self):
        request = Request(self.server_url + "/data.json", headers={"Accept": "application/json", "Cache-Control": "no-cache"})
        try:
            with urlopen(request, timeout=2.0) as response:
                return json.loads(response.read().decode("utf-8"))
        except (OSError, URLError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "Cannot read LibreHardwareMonitor's Remote Web Server. In LHM, open "
                "Options > Remote Web Server and turn Run on. Then check that "
                f"{self.server_url}/data.json opens in a browser. Details: {exc}"
            ) from exc

    @staticmethod
    def _flatten(node, parents=()):
        here = parents + (str(node.get("Text", "")),)
        result = []
        if node.get("SensorId") and node.get("Type"):
            result.append({
                "Name": str(node.get("Text", "")),
                "Identifier": str(node.get("SensorId", "")),
                "Parent": " ".join(parents),
                "SensorType": str(node.get("Type", "")),
                # monitorv2.py trained the raw data from formatted Value,
                # so parse the same field/units for live feature parity.
                "Value": node.get("Value"),
            })
        for child in node.get("Children", []) or []:
            result.extend(LhmTelemetry._flatten(child, here))
        return result

    def sample(self):
        root = self._read_tree()
        sensors = self._flatten(root)
        out = {name: float("nan") for name in SENSOR_PATTERNS}
        buckets = {name: [] for name in SENSOR_PATTERNS}
        for sensor in sensors:
            try:
                val = _parse_lhm_value(sensor["Value"])
            except (TypeError, ValueError, AttributeError):
                continue
            if not math.isfinite(val):
                continue
            for canonical, (words, excludes) in self.RULES.items():
                if _matches(sensor, words, excludes):
                    buckets[canonical].append(val)
        for name, values in buckets.items():
            if values:
                # Keep aggregation identical to canonicalize_columns: core
                # clocks are averaged; max channels use their maximum.
                out[name] = max(values) if name in ("CPU_Core_Max_Load", "CPU_Core_Max_Temperature", "GPU_Core_Temperature", "GPU_Hot_Spot_Temperature", "GPU_Memory_Junction_Temperature") else sum(values) / len(values)
        required = ("CPU_Total_Load", "GPU_Core_Load", "GPU_Core_Temperature")
        missing = [name for name in required if not math.isfinite(out[name])]
        if missing:
            relevant = [s for s in sensors if any(
                token in _key(s["Name"] + " " + s["Parent"])
                for token in ("gpu", "cpu total", "memory", "temperature", "load")
            )]
            available = "; ".join(
                f"{s['Name']} [{s['SensorType']}] id={s['Identifier']} value={s['Value']}"
                for s in relevant[:30]
            ) or "no CPU/GPU/load/temperature sensor nodes found"
            raise RuntimeError(
                "LHM server responded, but Stabilix could not map required live "
                f"sensors: {', '.join(missing)}. Relevant sensor nodes: {available}"
            )
        return time.time(), out
