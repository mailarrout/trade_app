import tkinter as tk
from datetime import datetime
import pytz
from ctypes import windll, Structure, c_int, byref

# --- Theme Colors ---
ACCENT_COLOR = "#0ac759"  # Green
CLOSE_COLOR = "#ff5555"   # Red

# --- Window size ---
WINDOW_WIDTH = 200
WINDOW_HEIGHT = 50

# --- Get screen dimensions ---
root = tk.Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# --- Get Taskbar Info ---
class RECT(Structure):
    _fields_ = [("left", c_int), ("top", c_int),
                ("right", c_int), ("bottom", c_int)]

taskbar_rect = RECT()
hwnd_tb = windll.user32.FindWindowW("Shell_TrayWnd", None)
windll.user32.GetWindowRect(hwnd_tb, byref(taskbar_rect))
taskbar_height = taskbar_rect.bottom - taskbar_rect.top

# --- Logical coordinate system (X:1-10, Y:0 bottom) ---
x_logical = 7
y_logical = 0  # bottom

x_pixel = int((x_logical - 1) / 9 * screen_width)
y_pixel = screen_height - taskbar_height - int(y_logical / 10 * screen_height)

# --- Configure window ---
root.overrideredirect(True)         # No title bar
root.attributes("-topmost", True)   # Always on top
root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}+{x_pixel}+{y_pixel}")

# --- Transparency for Windows ---
root.config(bg="black")
root.wm_attributes("-transparentcolor", "black")

# --- Click-through ---
hwnd = windll.user32.GetParent(root.winfo_id())
def set_click_through(enable=True):
    styles = windll.user32.GetWindowLongW(hwnd, -20)
    if enable:
        windll.user32.SetWindowLongW(hwnd, -20, styles | 0x80000 | 0x20)
    else:
        windll.user32.SetWindowLongW(hwnd, -20, styles & ~0x20)
set_click_through(True)

# --- Clock + Close button ---
clock_frame = tk.Frame(root, bg="black")
clock_frame.pack()

time_label = tk.Label(clock_frame, font=("Helvetica", 20, "bold"),
                      fg=ACCENT_COLOR, bg="black")
time_label.pack(side="left")

close_button = tk.Label(clock_frame, text="✖", font=("Helvetica", 10, "bold"),
                        fg=CLOSE_COLOR, bg="black", cursor="hand2")
close_button.pack(side="left", padx=(10,0))
close_button.bind("<Button-1>", lambda e: root.destroy())

# --- Dragging behavior ---
def start_move(event):
    root.x = event.x
    root.y = event.y

def stop_move(event):
    root.x = None
    root.y = None

def do_move(event):
    x = root.winfo_x() + (event.x - root.x)
    y = root.winfo_y() + (event.y - root.y)
    root.geometry(f"+{x}+{y}")

for widget in (clock_frame, time_label):
    widget.bind("<ButtonPress-1>", start_move)
    widget.bind("<ButtonRelease-1>", stop_move)
    widget.bind("<B1-Motion>", do_move)

# --- Clock update ---
ist = pytz.timezone('Asia/Kolkata')
def update_clock():
    now = datetime.now(ist)
    time_label.config(text=now.strftime("%H:%M:%S"))
    root.after(1000, update_clock)

# --- Always on top enforcement ---
def enforce_topmost():
    root.attributes("-topmost", True)
    root.after(2000, enforce_topmost)  # reapply every 2s

update_clock()
enforce_topmost()
root.mainloop()
