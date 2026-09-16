import customtkinter as ctk


# ============================================================
# APPEARANCE
# ============================================================

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# ============================================================
# COLORS
# ============================================================

BG = "#0d1117"
CARD = "#161b22"
CARD_HOVER = "#21262d"

TEXT = "#f0f2f5"
MUTED = "#8b949e"

GREEN = "#3fb950"
BLUE = "#58a6ff"


# ============================================================
# SETUP WINDOW
# ============================================================

class SetupWindow(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.selected_profile = None

        # ----------------------------------------------------
        # Window
        # ----------------------------------------------------

        self.title("Stabilix")
        self.geometry("720x560")
        self.resizable(False, False)

        self.configure(
            fg_color=BG
        )

        # ----------------------------------------------------
        # Main frame
        # ----------------------------------------------------

        self.main_frame = ctk.CTkFrame(
            self,
            fg_color=BG,
            corner_radius=0
        )

        self.main_frame.pack(
            fill="both",
            expand=True,
            padx=45,
            pady=35
        )

        # ----------------------------------------------------
        # Logo
        # ----------------------------------------------------

        self.logo = ctk.CTkLabel(
            self.main_frame,
            text="🛡",
            font=("Segoe UI", 42)
        )

        self.logo.pack(
            pady=(5, 0)
        )

        # ----------------------------------------------------
        # Title
        # ----------------------------------------------------

        self.title_label = ctk.CTkLabel(
            self.main_frame,
            text="STABILIX",
            font=("Segoe UI", 34, "bold"),
            text_color=TEXT
        )

        self.title_label.pack(
            pady=(0, 3)
        )

        # ----------------------------------------------------
        # Subtitle
        # ----------------------------------------------------

        self.subtitle_label = ctk.CTkLabel(
            self.main_frame,
            text="Real-Time System Stability Monitor",
            font=("Segoe UI", 14),
            text_color=MUTED
        )

        self.subtitle_label.pack(
            pady=(0, 25)
        )

        # ----------------------------------------------------
        # Description
        # ----------------------------------------------------

        self.description = ctk.CTkLabel(
            self.main_frame,
            text=(
                "Monitor hardware telemetry in real time and "
                "detect potential system stability risks."
            ),
            font=("Segoe UI", 13),
            text_color=MUTED
        )

        self.description.pack(
            pady=(0, 25)
        )

        # ----------------------------------------------------
        # Profile title
        # ----------------------------------------------------

        profile_title = ctk.CTkLabel(
            self.main_frame,
            text="SELECT MONITORING PROFILE",
            font=("Segoe UI", 12, "bold"),
            text_color=MUTED
        )

        profile_title.pack(
            anchor="w",
            pady=(0, 10)
        )

        # ----------------------------------------------------
        # Profile container
        # ----------------------------------------------------

        profile_frame = ctk.CTkFrame(
            self.main_frame,
            fg_color="transparent"
        )

        profile_frame.pack(
            fill="x"
        )

        # ----------------------------------------------------
        # Hybrid card
        # ----------------------------------------------------

        self.hybrid_button = ctk.CTkButton(
            profile_frame,
            text=(
                "🖥  HYBRID MODE\n"
                "CPU + NVIDIA GPU"
            ),
            width=295,
            height=85,
            corner_radius=14,
            fg_color=CARD,
            hover_color=CARD_HOVER,
            border_width=2,
            border_color=BG,
            text_color=TEXT,
            font=("Segoe UI", 15, "bold"),
            command=lambda: self.select_profile("hybrid")
        )

        self.hybrid_button.pack(
            side="left",
            expand=True,
            fill="x",
            padx=(0, 6)
        )

        # ----------------------------------------------------
        # Eco card
        # ----------------------------------------------------

        self.eco_button = ctk.CTkButton(
            profile_frame,
            text=(
                "💻  ECO MODE\n"
                "CPU-focused monitoring"
            ),
            width=295,
            height=85,
            corner_radius=14,
            fg_color=CARD,
            hover_color=CARD_HOVER,
            border_width=2,
            border_color=BG,
            text_color=TEXT,
            font=("Segoe UI", 15, "bold"),
            command=lambda: self.select_profile("eco")
        )

        self.eco_button.pack(
            side="left",
            expand=True,
            fill="x",
            padx=(6, 0)
        )

        # ----------------------------------------------------
        # Status
        # ----------------------------------------------------

        self.status_label = ctk.CTkLabel(
            self.main_frame,
            text="Select a profile to continue",
            font=("Segoe UI", 12),
            text_color=MUTED
        )

        self.status_label.pack(
            pady=(18, 10)
        )

        # ----------------------------------------------------
        # Start button
        # ----------------------------------------------------

        self.start_button = ctk.CTkButton(
            self.main_frame,
            text="START MONITORING  →",
            width=420,
            height=52,
            corner_radius=12,
            fg_color=BLUE,
            hover_color="#79b8ff",
            text_color="#ffffff",
            font=("Segoe UI", 15, "bold"),
            state="disabled",
            command=self.start_monitoring
        )

        self.start_button.pack(
            pady=(5, 0)
        )

        # ----------------------------------------------------
        # Footer
        # ----------------------------------------------------

        footer = ctk.CTkLabel(
            self.main_frame,
            text="Stabilix • Hardware Telemetry Intelligence",
            font=("Segoe UI", 10),
            text_color="#586069"
        )

        footer.pack(
            side="bottom",
            pady=(15, 0)
        )

    # ========================================================
    # PROFILE SELECTION
    # ========================================================

    def select_profile(self, profile):

        self.selected_profile = profile

        # Reset borders
        self.hybrid_button.configure(
            border_color=BG
        )

        self.eco_button.configure(
            border_color=BG
        )

        # ----------------------------------------------------
        # Hybrid
        # ----------------------------------------------------

        if profile == "hybrid":

            self.hybrid_button.configure(
                border_color=BLUE
            )

            self.status_label.configure(
                text="Hybrid Mode selected • CPU + NVIDIA GPU",
                text_color=TEXT
            )

        # ----------------------------------------------------
        # Eco
        # ----------------------------------------------------

        elif profile == "eco":

            self.eco_button.configure(
                border_color=GREEN
            )

            self.status_label.configure(
                text="Eco Mode selected • CPU-focused monitoring",
                text_color=TEXT
            )

        self.start_button.configure(
            state="normal"
        )

    # ========================================================
    # START MONITORING
    # ========================================================

    def start_monitoring(self):

        if self.selected_profile is None:
            return

        # Save profile before destroying window
        profile = self.selected_profile

        self.destroy()

        # Import only when monitoring starts
        from overlay import StabilixOverlay

        app = StabilixOverlay(
            profile=profile
        )

        app.mainloop()


# ============================================================
# DIRECT START
# ============================================================

if __name__ == "__main__":

    app = SetupWindow()

    app.mainloop()