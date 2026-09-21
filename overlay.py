import customtkinter as ctk

from telemetry import Telemetry
from sliding_window import SlidingWindow
from predictor import Predictor


# ============================================================
# APPEARANCE
# ============================================================

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# ============================================================
# COLORS
# ============================================================

BG = "#111418"
CARD = "#191d23"
CARD_HOVER = "#222831"

TEXT = "#f0f2f5"
MUTED = "#8b949e"

GREEN = "#3fb950"
YELLOW = "#d29922"
RED = "#f85149"
BLUE = "#58a6ff"


# ============================================================
# WINDOW SIZE
# ============================================================

COMPACT_WIDTH = 340
COMPACT_HEIGHT = 72

EXPANDED_WIDTH = 850
EXPANDED_HEIGHT = 570


# ============================================================
# STABILIX OVERLAY
# ============================================================

class StabilixOverlay(ctk.CTk):

    def __init__(self, profile="hybrid"):

        super().__init__()

        # ----------------------------------------------------
        # Profile
        # ----------------------------------------------------

        self.profile = profile

        # ----------------------------------------------------
        # Backend
        # ----------------------------------------------------

        self.telemetry = Telemetry(profile=profile)
        self.window = SlidingWindow(window_size=10)
        self.predictor = Predictor()

        # ----------------------------------------------------
        # State
        # ----------------------------------------------------

        self.expanded = False

        self.drag_start_x = 0
        self.drag_start_y = 0
        self.dragging = False

        # Tracks whether we've already warned about the PDH fallback,
        # so we nag once per state change instead of every second.
        self._warned_pdh_fallback = False

        # ----------------------------------------------------
        # Window
        # ----------------------------------------------------

        self.title("Stabilix")

        self.geometry(
            f"{COMPACT_WIDTH}x{COMPACT_HEIGHT}"
        )

        self.overrideredirect(True)

        self.attributes(
            "-topmost",
            True
        )

        self.attributes(
            "-alpha",
            0.96
        )

        self.configure(
            fg_color=BG
        )

        # ----------------------------------------------------
        # Position
        # ----------------------------------------------------

        self.position_compact()

        # ----------------------------------------------------
        # Build UI
        # ----------------------------------------------------

        self.build_ui()

        # ----------------------------------------------------
        # Start monitoring
        # ----------------------------------------------------

        self.after(
            200,
            self.update_dashboard
        )

    # ========================================================
    # POSITIONING
    # ========================================================

    def position_compact(self):

        screen_width = self.winfo_screenwidth()

        x = (
            screen_width
            - COMPACT_WIDTH
            - 15
        )

        y = 20

        self.geometry(
            f"{COMPACT_WIDTH}x{COMPACT_HEIGHT}+{x}+{y}"
        )

    # --------------------------------------------------------

    def position_expanded(self):

        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        x = (
            screen_width
            - EXPANDED_WIDTH
        ) // 2

        y = (
            screen_height
            - EXPANDED_HEIGHT
        ) // 2

        self.geometry(
            f"{EXPANDED_WIDTH}x{EXPANDED_HEIGHT}+{x}+{y}"
        )

    # ========================================================
    # BUILD UI
    # ========================================================

    def build_ui(self):

        self.main_frame = ctk.CTkFrame(
            self,
            fg_color=BG,
            corner_radius=18
        )

        self.main_frame.pack(
            fill="both",
            expand=True
        )

        self.build_compact()
        self.build_expanded()

        self.bind_drag_handlers()

    # ========================================================
    # COMPACT VIEW
    # ========================================================

    def build_compact(self):

        self.compact_frame = ctk.CTkFrame(
            self.main_frame,
            fg_color=CARD,
            corner_radius=18
        )

        self.compact_frame.pack(
            fill="both",
            expand=True
        )

        self.compact_title = ctk.CTkLabel(
            self.compact_frame,
            text="🛡  STABILIX",
            font=("Segoe UI", 14, "bold"),
            text_color=TEXT
        )

        self.compact_title.pack(
            side="left",
            padx=(15, 10)
        )

        self.compact_status = ctk.CTkLabel(
            self.compact_frame,
            text="● Starting...",
            font=("Segoe UI", 13, "bold"),
            text_color=MUTED
        )

        self.compact_status.pack(
            side="left"
        )

        self.compact_info = ctk.CTkLabel(
            self.compact_frame,
            text="",
            font=("Segoe UI", 11),
            text_color=MUTED
        )

        self.compact_info.pack(
            side="right",
            padx=(5, 15)
        )

    # ========================================================
    # EXPANDED VIEW
    # ========================================================

    def build_expanded(self):

        self.expanded_frame = ctk.CTkFrame(
            self.main_frame,
            fg_color=BG,
            corner_radius=18
        )

        # ----------------------------------------------------
        # Header
        # ----------------------------------------------------

        self.header = ctk.CTkFrame(
            self.expanded_frame,
            fg_color=CARD,
            corner_radius=15
        )

        self.header.pack(
            fill="x",
            padx=15,
            pady=15
        )

        self.header_title = ctk.CTkLabel(
            self.header,
            text="🛡  STABILIX",
            font=("Segoe UI", 23, "bold"),
            text_color=TEXT
        )

        self.header_title.pack(
            side="left",
            padx=18,
            pady=12
        )

        self.header_subtitle = ctk.CTkLabel(
            self.header,
            text="Real-Time System Stability Monitor",
            font=("Segoe UI", 12),
            text_color=MUTED
        )

        self.header_subtitle.pack(
            side="left"
        )

        profile_text = (
            "HYBRID"
            if self.profile == "hybrid"
            else "ECO"
        )

        self.profile_badge = ctk.CTkLabel(
            self.header,
            text=profile_text,
            font=("Segoe UI", 10, "bold"),
            text_color=(
                BLUE
                if self.profile == "hybrid"
                else GREEN
            )
        )

        self.profile_badge.pack(
            side="right",
            padx=10
        )

        self.collapse_button = ctk.CTkButton(
            self.header,
            text="−",
            width=40,
            height=35,
            font=("Segoe UI", 18, "bold"),
            fg_color=CARD_HOVER,
            hover_color="#303640",
            command=self.collapse_view
        )

        self.collapse_button.pack(
            side="right",
            padx=8
        )

        self.close_button = ctk.CTkButton(
            self.header,
            text="×",
            width=40,
            height=35,
            font=("Segoe UI", 18, "bold"),
            fg_color=CARD_HOVER,
            hover_color="#303640",
            command=self.destroy
        )

        self.close_button.pack(
            side="right",
            padx=(0, 10)
        )

        # ----------------------------------------------------
        # Status Card
        # ----------------------------------------------------

        self.status_card = ctk.CTkFrame(
            self.expanded_frame,
            fg_color=CARD,
            corner_radius=15
        )

        self.status_card.pack(
            fill="x",
            padx=15,
            pady=(0, 12)
        )

        self.status_label = ctk.CTkLabel(
            self.status_card,
            text="Collecting Telemetry...",
            font=("Segoe UI", 25, "bold"),
            text_color=TEXT
        )

        self.status_label.pack(
            side="left",
            padx=20,
            pady=18
        )

        self.risk_label = ctk.CTkLabel(
            self.status_card,
            text="Risk: --",
            font=("Segoe UI", 14),
            text_color=MUTED
        )

        self.risk_label.pack(
            side="right",
            padx=20
        )

        # ----------------------------------------------------
        # Metric Cards
        # ----------------------------------------------------

        self.metrics_frame = ctk.CTkFrame(
            self.expanded_frame,
            fg_color="transparent"
        )

        self.metrics_frame.pack(
            fill="x",
            padx=10,
            pady=(0, 12)
        )

        self.cpu_value = self.create_metric(
            self.metrics_frame,
            "CPU USAGE"
        )

        self.gpu_value = self.create_metric(
            self.metrics_frame,
            "GPU USAGE"
        )

        self.ram_value = self.create_metric(
            self.metrics_frame,
            "RAM USAGE"
        )

        # ----------------------------------------------------
        # Hardware Details
        # ----------------------------------------------------

        self.details_card = ctk.CTkFrame(
            self.expanded_frame,
            fg_color=CARD,
            corner_radius=15
        )

        self.details_card.pack(
            fill="x",
            padx=15,
            pady=(0, 12)
        )

        self.details_title = ctk.CTkLabel(
            self.details_card,
            text="HARDWARE TELEMETRY",
            font=("Segoe UI", 11, "bold"),
            text_color=MUTED
        )

        self.details_title.pack(
            anchor="w",
            padx=18,
            pady=(12, 5)
        )

        self.details_frame = ctk.CTkFrame(
            self.details_card,
            fg_color="transparent"
        )

        self.details_frame.pack(
            fill="x",
            padx=18,
            pady=(0, 14)
        )

        self.cpu_temp_label = self.create_detail(
            self.details_frame,
            "CPU TEMP",
            "-- °C"
        )

        self.gpu_temp_label = self.create_detail(
            self.details_frame,
            "GPU TEMP",
            "-- °C"
        )

        self.cpu_clock_label = self.create_detail(
            self.details_frame,
            "CPU CLOCK",
            "-- MHz"
        )

        self.gpu_clock_label = self.create_detail(
            self.details_frame,
            "GPU CLOCK",
            "-- MHz"
        )

        self.battery_label = self.create_detail(
            self.details_frame,
            "BATTERY",
            "-- %"
        )

        # ----------------------------------------------------
        # Live Monitoring
        # ----------------------------------------------------

        self.live_card = ctk.CTkFrame(
            self.expanded_frame,
            fg_color=CARD,
            corner_radius=15
        )

        self.live_card.pack(
            fill="both",
            expand=True,
            padx=15,
            pady=(0, 15)
        )

        self.live_title = ctk.CTkLabel(
            self.live_card,
            text="LIVE MONITORING",
            font=("Segoe UI", 11, "bold"),
            text_color=MUTED
        )

        self.live_title.pack(
            anchor="w",
            padx=18,
            pady=(12, 5)
        )

        self.live_info = ctk.CTkLabel(
            self.live_card,
            text="Starting telemetry...",
            font=("Segoe UI", 13),
            text_color=TEXT
        )

        self.live_info.pack(
            anchor="w",
            padx=18,
            pady=10
        )

        self.confidence_label = ctk.CTkLabel(
            self.live_card,
            text="Prediction confidence: --",
            font=("Segoe UI", 12),
            text_color=MUTED
        )

        self.confidence_label.pack(
            anchor="w",
            padx=18
        )

        self.profile_info = ctk.CTkLabel(
            self.live_card,
            text=self.get_profile_description(),
            font=("Segoe UI", 11),
            text_color=MUTED
        )

        self.profile_info.pack(
            anchor="w",
            padx=18,
            pady=(8, 0)
        )

    # ========================================================
    # CREATE METRIC
    # ========================================================

    def create_metric(self, parent, title):

        card = ctk.CTkFrame(
            parent,
            fg_color=CARD,
            corner_radius=15
        )

        card.pack(
            side="left",
            fill="both",
            expand=True,
            padx=5
        )

        title_label = ctk.CTkLabel(
            card,
            text=title,
            font=("Segoe UI", 11, "bold"),
            text_color=MUTED
        )

        title_label.pack(
            anchor="w",
            padx=15,
            pady=(12, 0)
        )

        value_label = ctk.CTkLabel(
            card,
            text="--",
            font=("Segoe UI", 24, "bold"),
            text_color=TEXT
        )

        value_label.pack(
            anchor="w",
            padx=15,
            pady=(3, 12)
        )

        return value_label

    # ========================================================
    # CREATE DETAIL
    # ========================================================

    def create_detail(
        self,
        parent,
        title,
        value
    ):

        label = ctk.CTkLabel(
            parent,
            text=f"{title}\n{value}",
            font=("Segoe UI", 13),
            text_color=TEXT
        )

        label.pack(
            side="left",
            expand=True
        )

        return label

    # ========================================================
    # PROFILE DESCRIPTION
    # ========================================================

    def get_profile_description(self):

        if self.profile == "hybrid":

            return (
                "Profile: Hybrid • "
                "CPU + NVIDIA GPU telemetry"
            )

        return (
            "Profile: Eco • "
            "CPU-focused monitoring"
        )

    # ========================================================
    # VIEW FUNCTIONS
    # ========================================================

    def expand_view(self):

        if self.expanded:
            return

        self.expanded = True

        self.compact_frame.pack_forget()

        self.expanded_frame.pack(
            fill="both",
            expand=True
        )

        self.position_expanded()

    # --------------------------------------------------------

    def collapse_view(self):

        if not self.expanded:
            return

        self.expanded = False

        self.expanded_frame.pack_forget()

        self.compact_frame.pack(
            fill="both",
            expand=True
        )

        self.position_compact()

    # --------------------------------------------------------

    def toggle_view(self):

        if self.expanded:
            self.collapse_view()

        else:
            self.expand_view()

    # ========================================================
    # DRAGGING
    # ========================================================

    def start_drag(self, event):

        self.drag_start_x = (
            event.x_root
            - self.winfo_x()
        )

        self.drag_start_y = (
            event.y_root
            - self.winfo_y()
        )

        self.dragging = False

    # --------------------------------------------------------

    def perform_drag(self, event):

        new_x = (
            event.x_root
            - self.drag_start_x
        )

        new_y = (
            event.y_root
            - self.drag_start_y
        )

        current_x = self.winfo_x()
        current_y = self.winfo_y()

        if (
            abs(new_x - current_x) > 2
            or
            abs(new_y - current_y) > 2
        ):

            self.dragging = True

        self.geometry(
            f"+{new_x}+{new_y}"
        )

    # --------------------------------------------------------

    def end_drag(self, event):

        if not self.dragging:

            self.toggle_view()

        self.dragging = False

    # ========================================================
    # DRAG BINDINGS
    # ========================================================

    def bind_drag(self, widget):

        widget.bind(
            "<Button-1>",
            self.start_drag
        )

        widget.bind(
            "<B1-Motion>",
            self.perform_drag
        )

        widget.bind(
            "<ButtonRelease-1>",
            self.end_drag
        )

    # --------------------------------------------------------

    def bind_drag_handlers(self):

        compact_widgets = [

            self.compact_frame,
            self.compact_title,
            self.compact_status,
            self.compact_info
        ]

        for widget in compact_widgets:

            self.bind_drag(widget)

        expanded_widgets = [

            self.expanded_frame,
            self.header,
            self.header_title,
            self.header_subtitle,
            self.profile_badge,

            self.status_card,
            self.status_label,
            self.risk_label,

            self.metrics_frame,

            self.details_card,
            self.details_title,
            self.details_frame,

            self.cpu_temp_label,
            self.gpu_temp_label,
            self.cpu_clock_label,
            self.gpu_clock_label,
            self.battery_label,

            self.live_card,
            self.live_title,
            self.live_info,
            self.confidence_label,
            self.profile_info
        ]

        for widget in expanded_widgets:

            self.bind_drag(widget)

    # ========================================================
    # LIVE UPDATE
    # ========================================================

    def update_dashboard(self):

        try:

            # ------------------------------------------------
            # Telemetry
            # ------------------------------------------------

            sample = self.telemetry.get_sample()

            self.window.add_sample(
                sample
            )

            # ------------------------------------------------
            # PDH fallback warning
            # ------------------------------------------------

            is_fallback = sample.get(
                "CPU_Performance_Is_Fallback",
                False
            )

            if is_fallback and not self._warned_pdh_fallback:

                self._warned_pdh_fallback = True

                print(
                    "\n[Stabilix] WARNING: predictions are running with "
                    "CPU_Performance_Percent on the frequency-ratio "
                    "fallback, not the PDH counter the model was "
                    "trained on. Treat the workload state/risk shown "
                    "right now with skepticism.\n"
                )

            elif not is_fallback:

                self._warned_pdh_fallback = False

            # ------------------------------------------------
            # Current values
            # ------------------------------------------------

            cpu = sample.get(
                "cpu_usage",
                0
            )

            gpu = sample.get(
                "gpu_usage",
                0
            )

            ram = sample.get(
                "ram_usage",
                0
            )

            gpu_temp = sample.get(
                "gpu_temp",
                0
            )

            gpu_clock = sample.get(
                "gpu_clock",
                0
            )

            battery = sample.get(
                "battery",
                0
            )

            # ------------------------------------------------
            # Metric cards
            # ------------------------------------------------

            self.cpu_value.configure(
                text=f"{cpu:.1f}%"
            )

            self.gpu_value.configure(
                text=f"{gpu:.1f}%"
            )

            self.ram_value.configure(
                text=f"{ram:.1f}%"
            )

            # ------------------------------------------------
            # Hardware details
            # ------------------------------------------------

            self.gpu_temp_label.configure(
                text=f"GPU TEMP\n{gpu_temp} °C"
            )

            self.gpu_clock_label.configure(
                text=f"GPU CLOCK\n{gpu_clock} MHz"
            )

            self.battery_label.configure(
                text=f"BATTERY\n{battery:.0f}%"
            )

            self.cpu_temp_label.configure(
                text="CPU TEMP\n-- °C"
            )

            self.cpu_clock_label.configure(
                text="CPU CLOCK\n-- MHz"
            )

            # =================================================
            # ML PREDICTION
            # =================================================

            if self.window.is_ready():

                features = (
                    self.window.compute_features()
                )

                prediction = (
                    self.predictor.predict(
                        features
                    )
                )

                # ------------------------------------------------
                # IMPORTANT DEBUG
                # ------------------------------------------------

                raw_label = prediction.get(
                    "label",
                    "UNKNOWN"
                )

                state = prediction.get(
                    "state",
                    raw_label
                )

                risk = prediction.get(
                    "risk",
                    0
                )

                confidence = prediction.get(
                    "confidence",
                    0
                )

                probabilities = prediction.get(
                    "probabilities",
                    {}
                )

                print("\n")
                print("=" * 70)
                print("OVERLAY RECEIVED ML RESULT")
                print("=" * 70)

                print(
                    f"Raw ML label : {raw_label}"
                )

                print(
                    f"Display state: {state}"
                )

                print(
                    f"Risk         : {risk}%"
                )

                print(
                    f"Confidence   : {confidence}%"
                )

                print(
                    f"PDH fallback : {is_fallback}"
                )

                print(
                    "\nRaw features fed to the model:"
                )

                for feature_name, feature_value in features.items():

                    print(
                        f"  {feature_name:<28} {feature_value:>10.3f}"
                    )

                print(
                    "\nClass probabilities:"
                )

                for label, probability in sorted(
                    probabilities.items(),
                    key=lambda item: item[1],
                    reverse=True
                ):

                    print(
                        f"  {label:<32} "
                        f"{probability:6.2f}%"
                    )

                print("=" * 70)

                # --------------------------------------------
                # Risk level
                # --------------------------------------------

                if risk < 30:

                    icon = "🟢"
                    status_color = GREEN

                elif risk < 70:

                    icon = "🟡"
                    status_color = YELLOW

                else:

                    icon = "🔴"
                    status_color = RED

                # --------------------------------------------
                # Compact
                # --------------------------------------------

                self.compact_status.configure(
                    text=f"{icon} {state}",
                    text_color=status_color
                )

                fallback_tag = " ⚠ FALLBACK" if is_fallback else ""

                self.compact_info.configure(
                    text=(
                        f"Risk {risk}%"
                        f"  |  "
                        f"GPU {gpu_temp}°C"
                        f"{fallback_tag}"
                    )
                )

                # --------------------------------------------
                # Expanded
                # --------------------------------------------

                self.status_label.configure(
                    text=f"{icon}  {state}",
                    text_color=status_color
                )

                self.risk_label.configure(
                    text=f"Risk: {risk}%"
                )

                confidence_text = (
                    f"Prediction confidence: {confidence}%"
                )

                if is_fallback:
                    confidence_text += (
                        "  •  ⚠ CPU_Performance on fallback "
                        "(predictions may be unreliable)"
                    )

                self.confidence_label.configure(
                    text=confidence_text
                )

                self.live_info.configure(
                    text=(
                        f"CPU {cpu:.1f}%    "
                        f"GPU {gpu:.1f}%    "
                        f"RAM {ram:.1f}%    "
                        f"GPU Temperature "
                        f"{gpu_temp}°C"
                    )
                )

            # ------------------------------------------------
            # Building window
            # ------------------------------------------------

            else:

                samples = len(
                    self.window.window
                )

                self.compact_status.configure(
                    text="● Collecting...",
                    text_color=MUTED
                )

                self.compact_info.configure(
                    text=f"{samples}/10"
                )

                self.status_label.configure(
                    text="Collecting Telemetry...",
                    text_color=TEXT
                )

                self.risk_label.configure(
                    text=(
                        "Building prediction "
                        f"window: {samples}/10"
                    )
                )

                self.confidence_label.configure(
                    text="Prediction confidence: --"
                )

                self.live_info.configure(
                    text=(
                        "Waiting for enough "
                        "telemetry samples..."
                    )
                )

        except Exception as e:

            # ------------------------------------------------
            # Error state
            # ------------------------------------------------

            print("\n" + "=" * 60)
            print("STABILIX UPDATE ERROR")
            print("=" * 60)

            import traceback

            traceback.print_exc()

            print("=" * 60 + "\n")

            self.compact_status.configure(
                text="⚠ Error",
                text_color=RED
            )

            self.compact_info.configure(
                text="Telemetry"
            )

            self.status_label.configure(
                text="⚠ Telemetry Error",
                text_color=RED
            )

            self.risk_label.configure(
                text="Error"
            )

            self.confidence_label.configure(
                text="Prediction confidence: --"
            )

            self.live_info.configure(
                text=str(e)
            )

        # ----------------------------------------------------
        # Continue monitoring
        # ----------------------------------------------------

        self.after(
            1000,
            self.update_dashboard
        )


# ============================================================
# DIRECT START
# ============================================================

if __name__ == "__main__":

    app = StabilixOverlay(
        profile="hybrid"
    )

    app.mainloop()