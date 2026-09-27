"""Launcher for the LHM-backed Stabilix V2 overlay."""
import customtkinter as ctk


BG = "#0d1117"
CARD = "#151a21"
MUTED = "#8b949e"
GREEN = "#3fb950"


class SetupWindow(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Stabilix V2")
        self.geometry("560x450")
        self.resizable(False, False)
        self.configure(fg_color=BG)

        shell = ctk.CTkFrame(self, fg_color=BG)
        shell.pack(fill="both", expand=True, padx=28, pady=22)

        brand_row = ctk.CTkFrame(shell, fg_color="transparent")
        brand_row.pack(fill="x", pady=(8, 22))
        ctk.CTkLabel(
            brand_row, text="▰", font=("Segoe UI Symbol", 22, "bold"), text_color="#24a8ff",
        ).pack(side="left", padx=(0, 8))
        ctk.CTkLabel(
            brand_row, text="STABILIX  /  V2", font=("Segoe UI", 15, "bold"), text_color="#e6edf3",
        ).pack(side="left")
        ctk.CTkLabel(
            brand_row, text="LIVE SYSTEM MONITOR", font=("Segoe UI", 9, "bold"), text_color=MUTED,
        ).pack(side="right", pady=3)

        ctk.CTkLabel(
            shell, text="Your system, at a glance.", font=("Segoe UI", 25, "bold"),
            text_color="#f0f6fc", anchor="w",
        ).pack(fill="x", pady=(0, 5))
        ctk.CTkLabel(
            shell, text="Connect LibreHardwareMonitor to start live tracking.",
            font=("Segoe UI", 12), text_color=MUTED, anchor="w",
        ).pack(fill="x", pady=(0, 20))

        self._step(shell, "1", "Open LibreHardwareMonitor", "Keep it running in the background.")
        self._step(shell, "2", "Enable its local sensor feed", "Options  →  Remote Web Server  →  Run")

        self.start_button = ctk.CTkButton(
            shell, text="Start monitoring", width=504, height=48, corner_radius=10,
            font=("Segoe UI", 14, "bold"), fg_color="#1683d8", hover_color="#1473bd",
            command=self.start_monitoring,
        )
        self.start_button.pack(fill="x", pady=(17, 9))

        ctk.CTkLabel(
            shell, text="The compact monitor opens at the top-right. Click + to expand the live graphs.",
            font=("Segoe UI", 10), text_color=MUTED, wraplength=500,
        ).pack(fill="x", pady=(3, 0))

    @staticmethod
    def _step(parent, number, title, detail):
        card = ctk.CTkFrame(parent, fg_color=CARD, corner_radius=10)
        card.pack(fill="x", pady=5)
        ctk.CTkLabel(
            card, text=number, width=30, height=30, corner_radius=15,
            fg_color="#202a34", text_color=GREEN, font=("Segoe UI", 12, "bold"),
        ).pack(side="left", padx=12, pady=12)
        copy = ctk.CTkFrame(card, fg_color="transparent")
        copy.pack(side="left", fill="x", expand=True, pady=9)
        ctk.CTkLabel(copy, text=title, font=("Segoe UI", 12, "bold"), text_color="#e6edf3", anchor="w").pack(fill="x")
        ctk.CTkLabel(copy, text=detail, font=("Segoe UI", 10), text_color=MUTED, anchor="w").pack(fill="x", pady=(2, 0))

    def start_monitoring(self):
        self.destroy()
        from overlay_lhm import StabilixLhmOverlay
        app = StabilixLhmOverlay()
        app.mainloop()


if __name__ == "__main__":
    app = SetupWindow()
    app.mainloop()
