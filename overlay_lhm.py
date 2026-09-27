"""Compact, expandable Stabilix overlay backed by the LHM V2 pipeline."""
from collections import Counter, deque
import math
import tkinter as tk

import customtkinter as ctk

from lhm_v2.features import SENSOR_PATTERNS
from lhm_v2.predictor_lhm import LhmPredictor
from lhm_v2.sliding_window_lhm import LhmSlidingWindow
from lhm_v2.telemetry_lhm import LhmTelemetry

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

BG = "#111418"
PANEL = "#171b21"
MUTED = "#8b949e"
BLUE = "#58a6ff"


class LiveGraph(ctk.CTkFrame):
    """A small rolling line graph with a fixed or data-driven vertical scale."""

    def __init__(self, master, title, colors, unit="", fixed_max=None, series_names=None):
        super().__init__(master, fg_color=PANEL, corner_radius=8)
        self.title = title
        self.colors = colors
        self.unit = unit
        self.fixed_max = fixed_max
        self.series_names = series_names or []
        self.history = deque(maxlen=90)

        heading = ctk.CTkFrame(self, fg_color="transparent")
        heading.pack(fill="x", padx=10, pady=(6, 0))
        ctk.CTkLabel(heading, text=title, font=("Segoe UI", 11, "bold"), text_color="#e6edf3").pack(side="left")
        self.value_label = ctk.CTkLabel(heading, text="--", font=("Consolas", 10), text_color=MUTED)
        self.value_label.pack(side="right")

        if self.series_names:
            legend = "   ".join(self.series_names)
            ctk.CTkLabel(self, text=legend, font=("Segoe UI", 8), text_color=MUTED, anchor="w").pack(fill="x", padx=10)

        self.canvas = tk.Canvas(self, height=22, bg=PANEL, highlightthickness=0, bd=0)
        self.canvas.pack(fill="x", padx=8, pady=(0, 6))
        self.canvas.bind("<Configure>", lambda _event: self._draw_graph())

    def add(self, values, display):
        numeric = []
        for value in values:
            try:
                number = float(value)
                numeric.append(number if math.isfinite(number) else 0.0)
            except (TypeError, ValueError):
                numeric.append(0.0)
        self.history.append(tuple(numeric))
        self.value_label.configure(text=display)
        self._draw_graph()

    def _draw_graph(self):
        canvas = self.canvas
        canvas.delete("all")
        width = max(4, canvas.winfo_width())
        height = max(4, canvas.winfo_height())
        pad = 3
        for fraction in (0.33, 0.66):
            y = pad + (height - pad * 2) * fraction
            canvas.create_line(0, y, width, y, fill="#252b33", width=1)
        if len(self.history) < 2:
            return

        if self.fixed_max is not None:
            scale_max = float(self.fixed_max)
        else:
            peak = max((max(point) for point in self.history), default=1.0)
            scale_max = max(10.0, peak * 1.15)
        usable_height = height - pad * 2
        denominator = max(1, len(self.history) - 1)
        for series_index, color in enumerate(self.colors):
            points = []
            for index, sample in enumerate(self.history):
                value = sample[series_index] if series_index < len(sample) else 0.0
                x = pad + (width - pad * 2) * index / denominator
                y = height - pad - min(1.0, max(0.0, value / scale_max)) * usable_height
                points.extend((x, y))
            if len(points) >= 4:
                canvas.create_line(*points, fill=color, width=1.5, smooth=True, splinesteps=12)


class StabilixLhmOverlay(ctk.CTk):
    COLLAPSED_WIDTH = 380  # approximately 10 cm at 96 DPI
    COLLAPSED_HEIGHT = 76  # approximately 2 cm at 96 DPI
    EXPANDED_WIDTH = 400
    EXPANDED_HEIGHT = 440
    SCREEN_MARGIN = 14
    TOP_MARGIN = 18

    def __init__(self):
        super().__init__()
        self.title("Stabilix V2 • LHM")
        self.resizable(False, False)
        self.attributes("-topmost", True)
        self.configure(fg_color=BG)
        self.expanded = False
        self.prediction_history = deque(maxlen=5)
        self.displayed_label = None
        self.graphs = {}

        self.shell = ctk.CTkFrame(self, fg_color=BG, corner_radius=12, border_width=1, border_color="#30363d")
        self.shell.pack(fill="both", expand=True)

        self.header = ctk.CTkFrame(self.shell, fg_color="#191e25", corner_radius=10, height=66)
        self.header.pack(fill="x", padx=4, pady=4)
        self.header.pack_propagate(False)
        self.title_label = ctk.CTkLabel(self.header, text="STABILIX  /  LHM", font=("Segoe UI", 14, "bold"), text_color="#e6edf3")
        self.title_label.place(x=12, y=3)
        self.toggle_button = ctk.CTkButton(
            self.header, text="+", width=30, height=28, corner_radius=8,
            font=("Segoe UI", 18, "bold"), command=self.toggle_expanded,
        )
        self.toggle_button.place(relx=1.0, x=-8, y=3, anchor="ne")

        self.compact_status = ctk.CTkLabel(
            self.header, text="Connecting to LibreHardwareMonitor…",
            font=("Segoe UI", 11, "bold"), text_color="#d29922", anchor="w",
        )
        self.compact_status.place(x=13, y=28, relwidth=0.82)
        self.compact_metrics = ctk.CTkLabel(
            self.header, text="Waiting for live sensors", font=("Segoe UI", 9),
            text_color=MUTED, anchor="w",
        )
        self.compact_metrics.place(x=13, y=47, relwidth=0.82)
        for widget in (self.header, self.title_label, self.compact_status, self.compact_metrics):
            widget.bind("<Button-1>", self._expand_from_header)

        self.details = ctk.CTkFrame(self.shell, fg_color="transparent")
        self.state_label = ctk.CTkLabel(
            self.details, text="Building sensor history…", font=("Segoe UI", 14, "bold"),
            text_color="#d29922", anchor="w",
        )
        self.state_label.pack(fill="x", padx=12, pady=(2, 0))
        self.risk_label = ctk.CTkLabel(
            self.details, text="", font=("Segoe UI", 10), text_color=MUTED,
            anchor="w", wraplength=360, justify="left",
        )
        self.risk_label.pack(fill="x", padx=12, pady=(0, 6))

        graph_specs = (
            ("CPU load", "#3fb950", "%", 100, None),
            ("GPU load", BLUE, "%", 100, None),
            ("RAM use", "#d29922", "%", 100, None),
            ("GPU temperature", "#f85149", "°C", 110, None),
            ("Wi-Fi traffic", (BLUE, "#bc8cff"), "", None, ("↓ Download     ↑ Upload",)),
        )
        self.graph_panel = ctk.CTkFrame(self.shell, fg_color="transparent")
        for title, colors, unit, scale, names in graph_specs:
            graph = LiveGraph(self.graph_panel, title, colors if isinstance(colors, tuple) else (colors,), unit, scale, names)
            graph.pack(fill="x", padx=8, pady=3)
            self.graphs[title] = graph

        self.footer = ctk.CTkLabel(
            self.shell,
            text=f"{len(SENSOR_PATTERNS)} live signals  •  10-second prediction window",
            font=("Segoe UI", 9), text_color="#6e7681",
        )

        self._place_window(expanded=False)
        self.after(100, self.collect_sample)
        try:
            self.telemetry = LhmTelemetry()
            self.predictor = LhmPredictor()
            self.window = LhmSlidingWindow()
        except Exception as exc:
            self.compact_status.configure(text="LHM V2 unavailable", text_color="#f85149")
            self.compact_metrics.configure(text=str(exc))
            self.state_label.configure(text="LHM V2 unavailable", text_color="#f85149")
            self.risk_label.configure(text=str(exc))

    def _place_window(self, expanded):
        width = self.EXPANDED_WIDTH if expanded else self.COLLAPSED_WIDTH
        height = self.EXPANDED_HEIGHT if expanded else self.COLLAPSED_HEIGHT
        screen_width = self.winfo_screenwidth()
        x = max(0, screen_width - width - self.SCREEN_MARGIN)
        self.geometry(f"{width}x{height}+{x}+{self.TOP_MARGIN}")

    def _expand_from_header(self, _event=None):
        if not self.expanded:
            self.toggle_expanded()

    def toggle_expanded(self):
        self.expanded = not self.expanded
        if self.expanded:
            self.compact_status.place_forget()
            self.compact_metrics.place_forget()
            self.details.pack(fill="x", padx=4)
            self.graph_panel.pack(fill="both", expand=True, padx=2)
            self.footer.pack(side="bottom", pady=(0, 6))
            self.toggle_button.configure(text="−")
        else:
            self.details.pack_forget()
            self.graph_panel.pack_forget()
            self.footer.pack_forget()
            self.compact_status.place(x=13, y=28, relwidth=0.82)
            self.compact_metrics.place(x=13, y=47, relwidth=0.82)
            self.toggle_button.configure(text="+")
        self._place_window(self.expanded)

    @staticmethod
    def _format_workload(label):
        workload, _trend = label.split("__", 1)
        return workload.replace("_", " ")

    @staticmethod
    def _format_trend(label):
        _workload, trend = label.split("__", 1)
        return {
            "No_Imminent_Instability": "No imminent instability",
            "Approaching_Instability": "Instability approaching",
            "Crash_Imminent": "Crash imminent",
        }.get(trend, trend.replace("_", " "))

    def collect_sample(self):
        if not hasattr(self, "telemetry"):
            self.after(200, self.collect_sample)
            return
        try:
            now, sensors = self.telemetry.sample()
            features = self.window.add_sample(now, sensors)
            cpu = sensors.get("CPU_Total_Load")
            gpu = sensors.get("GPU_Core_Load")
            ram = sensors.get("RAM_Load")
            temp = sensors.get("GPU_Core_Temperature")
            wifi_down = sensors.get("WiFi_Download_Rate")
            wifi_up = sensors.get("WiFi_Upload_Rate")

            self.graphs["CPU load"].add((cpu,), f"{cpu:.1f}%")
            self.graphs["GPU load"].add((gpu,), f"{gpu:.1f}%")
            self.graphs["RAM use"].add((ram,), f"{ram:.1f}%")
            self.graphs["GPU temperature"].add((temp,), f"{temp:.1f}°C")
            self.graphs["Wi-Fi traffic"].add((wifi_down, wifi_up), f"↓ {wifi_down:.1f}   ↑ {wifi_up:.1f}")
            self.compact_metrics.configure(
                text=f"CPU {cpu:.1f}%  •  GPU {gpu:.1f}%  •  RAM {ram:.1f}%  •  Wi-Fi ↓{wifi_down:.1f} ↑{wifi_up:.1f}"
            )

            if features is None:
                elapsed = max(0.0, now - self.window.samples[0]["_time"]) if self.window.samples else 0
                self.compact_status.configure(text="Collecting sensor history…", text_color="#d29922")
                self.state_label.configure(text="Building 10-second sensor history…", text_color="#d29922")
                self.risk_label.configure(text=f"{min(10.0, elapsed):.1f} / 10 seconds")
            else:
                result = self.predictor.predict(features)
                self.prediction_history.append((result["label"], result["confidence"]))
                votes = Counter(label for label, _ in self.prediction_history)
                candidate, count = votes.most_common(1)[0]
                # A label needs three of five votes to replace the displayed state.
                if count >= 3:
                    self.displayed_label = candidate
                color = "#f85149" if result["risk"] >= 60 else "#d29922" if result["risk"] >= 25 else "#3fb950"
                if self.displayed_label is None:
                    self.compact_status.configure(text=f"Stabilizing prediction… ({count}/3)", text_color="#d29922")
                    self.compact_metrics.configure(text=f"Risk {result['risk']:.0f}%")
                    self.state_label.configure(text="Stabilizing prediction…", text_color="#d29922")
                    self.risk_label.configure(text=f"Risk {result['risk']:.0f}%")
                else:
                    workload = self._format_workload(self.displayed_label)
                    trend = self._format_trend(self.displayed_label)
                    matching = [confidence for label, confidence in self.prediction_history if label == self.displayed_label]
                    confidence = sum(matching) / len(matching)
                    self.compact_status.configure(text=f"{workload}  •  {trend}", text_color=color)
                    self.compact_metrics.configure(text=f"Risk {result['risk']:.0f}%  •  Model confidence {confidence:.0f}%")
                    self.state_label.configure(text=f"Workload: {workload}", text_color=color)
                    self.risk_label.configure(
                        text=f"Trend: {trend}   •   Risk {result['risk']:.0f}%   •   Model confidence {confidence:.0f}%"
                    )
        except Exception as exc:
            self.compact_status.configure(text="Live telemetry error", text_color="#f85149")
            self.compact_metrics.configure(text=str(exc))
            self.state_label.configure(text="Live telemetry error", text_color="#f85149")
            self.risk_label.configure(text=str(exc))
        self.after(200, self.collect_sample)


if __name__ == "__main__":
    app = StabilixLhmOverlay()
    app.mainloop()
