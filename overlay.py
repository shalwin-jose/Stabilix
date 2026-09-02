import customtkinter as ctk
from telemetry import Telemetry
from sliding_window import SlidingWindow
from predictor import Predictor

# -----------------------------------
# Initialize Backend
# -----------------------------------

telemetry = Telemetry()
window = SlidingWindow()
predictor = Predictor()

# -----------------------------------
# Window
# -----------------------------------

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.overrideredirect(True)
app.attributes("-topmost", True)
app.attributes("-alpha", 0.92)

WIDTH = 330
HEIGHT = 60

screen_width = app.winfo_screenwidth()

x = screen_width - WIDTH - 5
y = 5

app.geometry(f"{WIDTH}x{HEIGHT}+{x}+{y}")

# -----------------------------------
# Frame
# -----------------------------------

frame = ctk.CTkFrame(
    app,
    fg_color="#202020",
    corner_radius=18
)

frame.pack(fill="both", expand=True)

# -----------------------------------
# Labels
# -----------------------------------

status_label = ctk.CTkLabel(
    frame,
    text="Collecting Data...",
    font=("Segoe UI",18,"bold")
)

status_label.pack(pady=(6,0))

info_label = ctk.CTkLabel(
    frame,
    text="Waiting for 10 samples...",
    font=("Segoe UI",13)
)

info_label.pack()

# -----------------------------------
# Dragging
# -----------------------------------

drag_x = 0
drag_y = 0

def start_drag(event):
    global drag_x, drag_y
    drag_x = event.x_root - app.winfo_x()
    drag_y = event.y_root - app.winfo_y()

def dragging(event):
    x = event.x_root - drag_x
    y = event.y_root - drag_y
    app.geometry(f"+{x}+{y}")

for widget in [frame, status_label, info_label]:
    widget.bind("<Button-1>", start_drag)
    widget.bind("<B1-Motion>", dragging)

# -----------------------------------
# Live Update
# -----------------------------------

def update():

    sample = telemetry.get_sample()

    window.add_sample(sample)

    if window.is_ready():

        features = window.compute_features()

        prediction = predictor.predict(features)

        state = prediction["state"]
        risk = prediction["risk"]

        temp = features["gpu_temperature"]

        # ---------------- Colors ----------------

        if risk < 30:

            icon = "🟢"

        elif risk < 70:

            icon = "🟡"

        else:

            icon = "🔴"

        status_label.configure(
            text=f"{icon} {state}"
        )

        info_label.configure(
            text=f"Risk: {risk}%   |   GPU Temp: {temp}°C"
        )

    else:

        status_label.configure(
            text="Collecting Telemetry..."
        )

        info_label.configure(
            text=f"{len(window.window)}/10 Samples"
        )

    app.after(1000, update)

# -----------------------------------

update()

app.mainloop()