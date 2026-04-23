"""
Basler GigE Camera Tool
=======================
Tkinter GUI for discovering, configuring and streaming a Basler acA2040-25gc
(or any other GigE Vision / USB3 Vision Basler camera) via pypylon.

Features
--------
* Device discovery on all network interfaces (including unreachable/mis-IP'd cameras).
* Force a temporary IP (ForceIP) for cameras currently outside your subnet.
* Program a Persistent IP that survives reboot.
* Issue a DeviceReset (reboot) command.
* Connect / disconnect.
* Dashboard: Exposure, Gain, Pixel Format, AOI/ROI, Trigger mode.
* Live continuous grab with FPS counter.
* Software trigger single-shot with "Save Image" button.

Tested against: pypylon 3.x, Basler acA-family (GigE), Windows 11, Python 3.10+.

Author: generated for Nick
"""

from __future__ import annotations

import os
import sys
import time
import queue
import threading
import traceback
from datetime import datetime
from typing import Optional, List, Dict, Any

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

try:
    import numpy as np
    from PIL import Image, ImageTk
    from pypylon import pylon, genicam
except ImportError as e:
    print("Missing dependency:", e)
    print("Run:  pip install -r requirements.txt")
    sys.exit(1)

# Optional AI detector (ultralytics YOLOv8). The GUI works without it; the
# AI panel will show a helpful message if it's not installed.
try:
    from detector import Detector, COCO_CLASSES, DEFAULT_CLASSES
    _DETECTOR_AVAILABLE = True
except Exception as _det_err:
    Detector = None  # type: ignore
    COCO_CLASSES = []  # type: ignore
    DEFAULT_CLASSES = []  # type: ignore
    _DETECTOR_AVAILABLE = False
    _DET_IMPORT_ERROR = str(_det_err)

# Optional Cognex DataMan tab (TCP/DMCC). No Cognex SDK install required.
try:
    from cognex_tab import CognexTab
    _COGNEX_AVAILABLE = True
except Exception as _cog_err:
    CognexTab = None  # type: ignore
    _COGNEX_AVAILABLE = False
    _COG_IMPORT_ERROR = str(_cog_err)

# Optional Keyence SR-X tab (TCP command + data stream). No Keyence SDK needed.
try:
    from keyence_tab import KeyenceTab
    _KEYENCE_AVAILABLE = True
except Exception as _key_err:
    KeyenceTab = None  # type: ignore
    _KEYENCE_AVAILABLE = False
    _KEY_IMPORT_ERROR = str(_key_err)

# Optional 3D Mapping tab (uses Basler frames + MiDaS depth + matplotlib 3D).
try:
    from mapping3d_tab import Mapping3DTab
    _MAP3D_AVAILABLE = True
except Exception as _map_err:
    Mapping3DTab = None  # type: ignore
    _MAP3D_AVAILABLE = False
    _MAP3D_IMPORT_ERROR = str(_map_err)




# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def mac_bytes_to_str(mac_int_or_str) -> str:
    """pypylon returns MAC as hex string like '0030530A1B2C'. Pretty-print it."""
    s = str(mac_int_or_str).replace(":", "").replace("-", "").upper()
    if len(s) == 12:
        return ":".join(s[i:i + 2] for i in range(0, 12, 2))
    return str(mac_int_or_str)


def _get_node(node_map, name: str):
    """Safely get a node by name. Returns None if it doesn't exist."""
    try:
        node = node_map.GetNode(name)
        if node is None:
            return None
        return node
    except Exception:
        return None


def first_existing_node(node_map, *names):
    """Return the first node (by name) that actually exists on this camera."""
    for n in names:
        node = _get_node(node_map, n)
        if node is not None:
            return node, n
    return None, None


def safe_get(node_map, name: str, default=None):
    """Read a GenICam node if it exists and is readable, else return default."""
    node = _get_node(node_map, name)
    if node is None:
        return default
    try:
        if not genicam.IsReadable(node):
            return default
        if hasattr(node, "GetValue"):
            return node.GetValue()
        if hasattr(node, "ToString"):
            return node.ToString()
    except Exception:
        pass
    return default


def safe_set(node_map, name: str, value):
    """Write a GenICam node if it exists and is writable."""
    node = _get_node(node_map, name)
    if node is None:
        raise RuntimeError(f"Node '{name}' not found on camera.")
    if not genicam.IsWritable(node):
        raise RuntimeError(f"Node '{name}' is not writable in current state.")
    if hasattr(node, "FromString") and isinstance(value, str):
        node.FromString(value)
    else:
        node.SetValue(value)


def _iou(a, b) -> float:
    """Intersection-over-union of two (x1,y1,x2,y2) boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0



# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------
class BaslerApp(tk.Tk):
    CONVERTER_FMT = "BGR8packed"  # for display via OpenCV-style BGR numpy arrays
    DISPLAY_MAX_W = 900
    DISPLAY_MAX_H = 600

    def __init__(self):
        super().__init__()
        self.title("Machine Vision Tool — Basler + Cognex DataMan + Keyence SR-X")

        self.geometry("1700x900")
        self.minsize(1200, 720)


        # --- State ---
        self.tl_factory = pylon.TlFactory.GetInstance()
        self.discovered: List[pylon.DeviceInfo] = []
        self.camera: Optional[pylon.InstantCamera] = None
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        self.grab_thread: Optional[threading.Thread] = None
        self.grab_stop = threading.Event()
        self.frame_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=2)
        self.last_frame: Optional[np.ndarray] = None      # raw sensor frame
        self.last_display: Optional[np.ndarray] = None    # what's on the canvas
        self.fps = 0.0
        self._fps_tick = time.time()
        self._fps_count = 0

        # --- AI detector state ---
        self.detector = Detector() if _DETECTOR_AVAILABLE else None
        self.ai_enabled = tk.BooleanVar(value=False)
        self.ai_conf = tk.DoubleVar(value=0.35)
        self.ai_every_n = tk.IntVar(value=2)  # run detector every Nth frame
        self._ai_frame_counter = 0
        self._ai_last_detections: List[dict] = []
        self._ai_selected_idx: Optional[int] = None   # clicked-on detection
        # Canvas-to-image mapping recorded in _paint_to_canvas, used by click
        # handler to translate screen coords back into image coords.
        # tuple: (img_w, img_h, draw_x, draw_y, scale, canvas_w, canvas_h)
        self._canvas_map: Optional[tuple] = None
        self.class_vars: Dict[str, tk.BooleanVar] = {}

        self._build_ui()
        self._log("Ready. Click 'Discover' to find cameras.")
        self.after(33, self._ui_tick)  # ~30Hz UI update

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        # ---- Main notebook: Basler tab + Cognex DataMan tab ----
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True)

        basler_page = ttk.Frame(self.notebook)
        self.notebook.add(basler_page, text="  Basler  (GigE / USB3)  ")

        if _COGNEX_AVAILABLE:
            self.cognex_tab = CognexTab(self.notebook, log_cb=self._log)
            self.notebook.add(self.cognex_tab, text="  Cognex DataMan  (DMCC / TCP)  ")
        else:
            self.cognex_tab = None
            placeholder = ttk.Frame(self.notebook)
            tk.Label(placeholder,
                     text=f"Cognex tab unavailable: {_COG_IMPORT_ERROR}",
                     fg="red", padx=12, pady=12).pack()
            self.notebook.add(placeholder, text="  Cognex DataMan  ")

        if _KEYENCE_AVAILABLE:
            self.keyence_tab = KeyenceTab(self.notebook, log_cb=self._log)
            self.notebook.add(self.keyence_tab,
                              text="  Keyence SR-X  (TCP ASCII)  ")
        else:
            self.keyence_tab = None
            placeholder2 = ttk.Frame(self.notebook)
            tk.Label(placeholder2,
                     text=f"Keyence tab unavailable: {_KEY_IMPORT_ERROR}",
                     fg="red", padx=12, pady=12).pack()
            self.notebook.add(placeholder2, text="  Keyence SR-X  ")

        # ---- 3D Mapping tab ----
        # Feeds from the Basler live stream via a callback so it doesn't
        # need to own the camera.
        if _MAP3D_AVAILABLE:
            self.map3d_tab = Mapping3DTab(
                self.notebook,
                get_frame_cb=self._get_latest_frame,
                log_cb=self._log,
            )
            self.notebook.add(self.map3d_tab,
                              text="  3D Mapping  (depth -> point cloud)  ")
        else:
            self.map3d_tab = None
            placeholder3 = ttk.Frame(self.notebook)
            tk.Label(placeholder3,
                     text=f"3D Mapping tab unavailable: {_MAP3D_IMPORT_ERROR}"
                          "\n\nInstall:  pip install matplotlib torch",
                     fg="red", padx=12, pady=12, justify="left"
                     ).pack(anchor="w")
            self.notebook.add(placeholder3, text="  3D Mapping  ")

        root_pane = ttk.PanedWindow(basler_page, orient="horizontal")

        root_pane.pack(fill="both", expand=True)

        # --- Column 1: Discovery + Network + Connection + Log ---
        left = ttk.Frame(root_pane, width=420)
        root_pane.add(left, weight=0)

        disc_frame = ttk.LabelFrame(left, text="1. Discovery")
        disc_frame.pack(fill="x", padx=8, pady=6)

        ttk.Button(disc_frame, text="Discover Cameras",
                   command=self.discover).pack(fill="x", padx=6, pady=4)

        cols = ("model", "serial", "ip", "mac", "status")
        self.tree = ttk.Treeview(disc_frame, columns=cols, show="headings", height=6)
        for c, w in zip(cols, (140, 100, 110, 130, 80)):
            self.tree.heading(c, text=c.upper())
            self.tree.column(c, width=w, anchor="w")
        self.tree.pack(fill="x", padx=6, pady=4)
        self.tree.bind("<<TreeviewSelect>>", self._on_select_device)

        net_frame = ttk.LabelFrame(left, text="2. Network Configuration (GigE only)")
        net_frame.pack(fill="x", padx=8, pady=6)

        self.var_ip = tk.StringVar()
        self.var_mask = tk.StringVar(value="255.255.255.0")
        self.var_gw = tk.StringVar(value="0.0.0.0")

        for row, (lbl, var) in enumerate([("IP", self.var_ip),
                                          ("Subnet", self.var_mask),
                                          ("Gateway", self.var_gw)]):
            ttk.Label(net_frame, text=lbl).grid(row=row, column=0, sticky="e", padx=4, pady=2)
            ttk.Entry(net_frame, textvariable=var, width=18).grid(
                row=row, column=1, sticky="w", padx=4, pady=2)

        ttk.Button(net_frame, text="Force IP (temporary)",
                   command=self.force_ip).grid(row=3, column=0, columnspan=2,
                                               sticky="ew", padx=4, pady=2)
        ttk.Button(net_frame, text="Write Persistent IP (survives reboot)",
                   command=self.set_persistent_ip).grid(row=4, column=0, columnspan=2,
                                                        sticky="ew", padx=4, pady=2)
        ttk.Button(net_frame, text="Reboot Camera",
                   command=self.reboot_camera).grid(row=5, column=0, columnspan=2,
                                                    sticky="ew", padx=4, pady=2)

        conn_frame = ttk.LabelFrame(left, text="3. Connection")
        conn_frame.pack(fill="x", padx=8, pady=6)
        self.btn_connect = ttk.Button(conn_frame, text="Connect",
                                      command=self.connect_selected)
        self.btn_connect.pack(fill="x", padx=6, pady=2)
        self.btn_disconnect = ttk.Button(conn_frame, text="Disconnect",
                                         command=self.disconnect, state="disabled")
        self.btn_disconnect.pack(fill="x", padx=6, pady=2)

        self.lbl_conn_status = ttk.Label(conn_frame, text="Not connected",
                                         foreground="gray")
        self.lbl_conn_status.pack(padx=6, pady=2)

        # Log window
        log_frame = ttk.LabelFrame(left, text="Log")
        log_frame.pack(fill="both", expand=True, padx=8, pady=6)
        self.log_txt = scrolledtext.ScrolledText(log_frame, height=10, state="disabled",
                                                 wrap="word", font=("Consolas", 9))
        self.log_txt.pack(fill="both", expand=True, padx=4, pady=4)

        # --- Column 2: Dashboard + Video controls + AI panel (scrollable) ---
        mid_outer = ttk.Frame(root_pane)
        root_pane.add(mid_outer, weight=1)

        mid_canvas = tk.Canvas(mid_outer, borderwidth=0, highlightthickness=0,
                               width=520)
        mid_vsb = ttk.Scrollbar(mid_outer, orient="vertical",
                                command=mid_canvas.yview)
        mid_canvas.configure(yscrollcommand=mid_vsb.set)
        mid_vsb.pack(side="right", fill="y")
        mid_canvas.pack(side="left", fill="both", expand=True)
        right = ttk.Frame(mid_canvas)
        mid_window = mid_canvas.create_window((0, 0), window=right, anchor="nw")

        def _mid_resize(_e=None):
            mid_canvas.configure(scrollregion=mid_canvas.bbox("all"))
            # Make the inner frame match the canvas width so widgets fill.
            mid_canvas.itemconfigure(mid_window, width=mid_canvas.winfo_width())
        right.bind("<Configure>", _mid_resize)
        mid_canvas.bind("<Configure>", _mid_resize)

        # Mouse-wheel scrolling over the middle column.
        def _on_mousewheel(e):
            mid_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
        mid_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        dash = ttk.LabelFrame(right, text="4. Dashboard (available after connect)")
        dash.pack(fill="x", padx=8, pady=6)

        # Exposure
        ttk.Label(dash, text="Exposure (µs)").grid(row=0, column=0, sticky="e", padx=4, pady=3)
        self.var_exp = tk.DoubleVar(value=10000.0)
        self.ent_exp = ttk.Entry(dash, textvariable=self.var_exp, width=12)
        self.ent_exp.grid(row=0, column=1, sticky="w", padx=4)
        ttk.Button(dash, text="Apply", command=self.apply_exposure).grid(row=0, column=2, padx=2)

        # Gain
        ttk.Label(dash, text="Gain (dB)").grid(row=0, column=3, sticky="e", padx=4)
        self.var_gain = tk.DoubleVar(value=0.0)
        self.ent_gain = ttk.Entry(dash, textvariable=self.var_gain, width=8)
        self.ent_gain.grid(row=0, column=4, sticky="w", padx=4)
        ttk.Button(dash, text="Apply", command=self.apply_gain).grid(row=0, column=5, padx=2)

        # Pixel format
        ttk.Label(dash, text="Pixel Format").grid(row=1, column=0, sticky="e", padx=4, pady=3)
        self.cmb_pf = ttk.Combobox(dash, width=18, state="readonly")
        self.cmb_pf.grid(row=1, column=1, columnspan=2, sticky="w", padx=4)
        self.cmb_pf.bind("<<ComboboxSelected>>", lambda e: self.apply_pixel_format())

        # Trigger mode
        ttk.Label(dash, text="Trigger").grid(row=1, column=3, sticky="e", padx=4)
        self.cmb_trig = ttk.Combobox(dash, width=14, state="readonly",
                                     values=("Off (Continuous)", "Software"))
        self.cmb_trig.current(0)
        self.cmb_trig.grid(row=1, column=4, sticky="w", padx=4)
        self.cmb_trig.bind("<<ComboboxSelected>>", lambda e: self.apply_trigger_mode())

        # ROI
        roi_f = ttk.Frame(dash)
        roi_f.grid(row=2, column=0, columnspan=6, sticky="ew", pady=(8, 2))
        self.var_w = tk.IntVar(value=0)
        self.var_h = tk.IntVar(value=0)
        self.var_ox = tk.IntVar(value=0)
        self.var_oy = tk.IntVar(value=0)
        for i, (lbl, var) in enumerate([("W", self.var_w), ("H", self.var_h),
                                        ("OffsetX", self.var_ox), ("OffsetY", self.var_oy)]):
            ttk.Label(roi_f, text=lbl).grid(row=0, column=i * 2, sticky="e", padx=3)
            ttk.Entry(roi_f, textvariable=var, width=8).grid(row=0, column=i * 2 + 1, sticky="w")
        ttk.Button(roi_f, text="Apply ROI", command=self.apply_roi).grid(row=0, column=8, padx=8)
        ttk.Button(roi_f, text="Max ROI", command=self.reset_roi).grid(row=0, column=9, padx=4)

        # Video controls
        vc = ttk.LabelFrame(right, text="5. Video")
        vc.pack(fill="x", padx=8, pady=6)
        self.btn_live = ttk.Button(vc, text="Start Live", command=self.start_live, state="disabled")
        self.btn_live.pack(side="left", padx=6, pady=4)
        self.btn_stop = ttk.Button(vc, text="Stop", command=self.stop_grab, state="disabled")
        self.btn_stop.pack(side="left", padx=6, pady=4)
        self.btn_trig = ttk.Button(vc, text="Software Trigger (1 shot)",
                                   command=self.software_trigger, state="disabled")
        self.btn_trig.pack(side="left", padx=6, pady=4)
        self.btn_save = ttk.Button(vc, text="Save Last Frame",
                                   command=self.save_last_frame, state="disabled")
        self.btn_save.pack(side="left", padx=6, pady=4)

        self.lbl_fps = ttk.Label(vc, text="FPS: --")
        self.lbl_fps.pack(side="right", padx=10)

        # ---- AI / ML Detection panel ----
        ai = ttk.LabelFrame(right, text="6. AI / ML Pattern Detection (YOLOv8)")
        ai.pack(fill="x", padx=8, pady=6)

        if not _DETECTOR_AVAILABLE:
            tk.Label(
                ai,
                text=("ultralytics not installed. Run:  "
                      "python -m pip install ultralytics   to enable AI."),
                fg="red", anchor="w", padx=8, pady=6,
            ).pack(fill="x")
        else:
            top_row = ttk.Frame(ai)
            top_row.pack(fill="x", padx=6, pady=4)
            ttk.Checkbutton(top_row, text="Enable AI overlay",
                            variable=self.ai_enabled,
                            command=self._on_ai_toggle).pack(side="left", padx=4)
            ttk.Label(top_row, text="Confidence:").pack(side="left", padx=(16, 2))
            ttk.Scale(top_row, from_=0.10, to=0.90, orient="horizontal",
                      variable=self.ai_conf, length=140,
                      command=lambda *_: self._sync_detector_conf()
                      ).pack(side="left")
            self.lbl_conf = ttk.Label(top_row, text="0.35", width=5)
            self.lbl_conf.pack(side="left", padx=4)
            ttk.Label(top_row, text="Infer every N frames:").pack(side="left", padx=(16, 2))
            ttk.Spinbox(top_row, from_=1, to=10, width=4,
                        textvariable=self.ai_every_n).pack(side="left")
            self.lbl_ai_status = ttk.Label(top_row, text="(model not loaded)",
                                           foreground="gray")
            self.lbl_ai_status.pack(side="right", padx=6)

            # Scrollable grid of class-toggle checkbuttons
            cls_container = ttk.Frame(ai)
            cls_container.pack(fill="x", padx=6, pady=(2, 4))
            ttk.Label(cls_container, text="Detect classes:").grid(
                row=0, column=0, columnspan=8, sticky="w", pady=(2, 4))

            # Lay out all 80 COCO classes in a compact 8-column grid.
            for i, name in enumerate(COCO_CLASSES):
                var = tk.BooleanVar(value=(name in DEFAULT_CLASSES))
                self.class_vars[name] = var
                ttk.Checkbutton(
                    cls_container, text=name, variable=var,
                    command=self._sync_detector_classes
                ).grid(row=1 + i // 8, column=i % 8, sticky="w", padx=2)

            btn_row = ttk.Frame(ai)
            btn_row.pack(fill="x", padx=6, pady=(0, 6))
            ttk.Button(btn_row, text="Select All",
                       command=lambda: self._select_classes(True)
                       ).pack(side="left", padx=3)
            ttk.Button(btn_row, text="Clear All",
                       command=lambda: self._select_classes(False)
                       ).pack(side="left", padx=3)
            ttk.Button(btn_row, text="Presets: People & Devices",
                       command=lambda: self._preset_classes(DEFAULT_CLASSES)
                       ).pack(side="left", padx=3)

        # ---- 7. GenICam command console ----
        cmd = ttk.LabelFrame(right,
                             text="7. Command Console (GenICam GET / SET / EXEC)")
        cmd.pack(fill="x", padx=8, pady=6)

        # Operation + node + value
        line1 = ttk.Frame(cmd)
        line1.pack(fill="x", padx=6, pady=4)
        self.var_cmd_op = tk.StringVar(value="GET")
        ttk.Combobox(line1, width=6, state="readonly",
                     values=("GET", "SET", "EXEC"),
                     textvariable=self.var_cmd_op).pack(side="left", padx=3)
        self.var_cmd_node = tk.StringVar()
        ttk.Entry(line1, textvariable=self.var_cmd_node, width=22
                  ).pack(side="left", padx=3)
        ttk.Label(line1, text="=").pack(side="left")
        self.var_cmd_val = tk.StringVar()
        ttk.Entry(line1, textvariable=self.var_cmd_val, width=14
                  ).pack(side="left", padx=3)
        ttk.Button(line1, text="Send",
                   command=self._cmd_send).pack(side="left", padx=3)
        ttk.Button(line1, text="Dump all readable nodes",
                   command=self._cmd_dump_nodes).pack(side="left", padx=3)

        # Quick-pick buttons for common ops
        line2 = ttk.Frame(cmd)
        line2.pack(fill="x", padx=6, pady=(2, 4))
        presets = [
            ("GET DeviceModelName",  "GET", "DeviceModelName",  ""),
            ("GET DeviceSerialNumber","GET","DeviceSerialNumber",""),
            ("GET DeviceTemperature","GET","DeviceTemperature", ""),
            ("GET ExposureTime",     "GET", "ExposureTime",     ""),
            ("GET PixelFormat",      "GET", "PixelFormat",      ""),
            ("EXEC AcquisitionStart","EXEC","AcquisitionStart", ""),
            ("EXEC AcquisitionStop", "EXEC","AcquisitionStop",  ""),
            ("EXEC DeviceReset",     "EXEC","DeviceReset",      ""),
        ]
        for text, op, node, val in presets:
            ttk.Button(line2, text=text, width=24,
                       command=lambda o=op, n=node, v=val:
                       self._cmd_preset(o, n, v)
                       ).pack(side="left", padx=2, pady=2)

        # Dedicated command-log view (separate from general log)
        clog_frame = ttk.Frame(cmd)
        clog_frame.pack(fill="both", expand=True, padx=6, pady=(2, 6))
        self.cmd_log = scrolledtext.ScrolledText(
            clog_frame, height=10, state="disabled",
            wrap="word", font=("Consolas", 9))
        self.cmd_log.pack(fill="both", expand=True)
        # Tag styling for >>> commands vs responses vs errors
        self.cmd_log.tag_configure("cmd",  foreground="#0066cc")
        self.cmd_log.tag_configure("ok",   foreground="#006600")
        self.cmd_log.tag_configure("err",  foreground="#cc0000")

        ctrl_row = ttk.Frame(cmd)
        ctrl_row.pack(fill="x", padx=6, pady=(0, 6))
        ttk.Button(ctrl_row, text="Clear command log",
                   command=self._cmd_clear).pack(side="left", padx=3)
        ttk.Button(ctrl_row, text="Save command log to file…",
                   command=self._cmd_save).pack(side="left", padx=3)

        # --- Column 3: Preview (dedicated resizable pane) ---
        preview_pane = ttk.Frame(root_pane)
        root_pane.add(preview_pane, weight=3)   # takes the most room by default

        self.video_frame = ttk.LabelFrame(preview_pane, text="Preview")
        self.video_frame.pack(fill="both", expand=True, padx=8, pady=6)

        # Viewer options bar (fit vs. 1:1 vs. fill)
        viewer_bar = ttk.Frame(self.video_frame)
        viewer_bar.pack(fill="x", padx=4, pady=(4, 0))
        self.view_mode = tk.StringVar(value="fit")
        ttk.Label(viewer_bar, text="View:").pack(side="left")
        for mode, lbl in (("fit", "Fit"),
                          ("fill", "Fill"),
                          ("1:1", "Actual (1:1)")):
            ttk.Radiobutton(viewer_bar, text=lbl, value=mode,
                            variable=self.view_mode).pack(side="left", padx=3)

        self.canvas = tk.Canvas(self.video_frame, bg="black",
                                highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=4, pady=4)
        # Redraw when the preview pane is resized.
        self.canvas.bind("<Configure>",
                         lambda _e: self._redraw_current() if self.last_frame is not None else None)
        # Left-click a detection box to select it; click empty space to clear.
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self._tk_img = None  # keep reference

        # Selection info bar under the preview
        self.lbl_selection = ttk.Label(
            self.video_frame,
            text="Click any detection box in the preview to select it.",
            foreground="gray",
            anchor="w",
        )
        self.lbl_selection.pack(fill="x", padx=6, pady=(0, 4))

    # --------------------------------------------------------------- logging
    def _log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}\n"
        self.log_txt.configure(state="normal")
        self.log_txt.insert("end", line)
        self.log_txt.see("end")
        self.log_txt.configure(state="disabled")

    def _err(self, msg: str, exc: Exception | None = None):
        detail = f"{msg}"
        if exc is not None:
            detail += f"\n{type(exc).__name__}: {exc}"
        self._log("ERROR: " + detail.replace("\n", " | "))
        if exc is not None:
            # Log full traceback to the log pane so we can see which line failed.
            self.log_txt.configure(state="normal")
            self.log_txt.insert("end", traceback.format_exc() + "\n")
            self.log_txt.see("end")
            self.log_txt.configure(state="disabled")
        messagebox.showerror("Error", detail)

    # ------------------------------------------------------------- discovery
    def discover(self):
        self._log("Enumerating GigE + USB Basler devices...")
        self.tree.delete(*self.tree.get_children())
        self.discovered.clear()

        try:
            # EnumerateAllDevices finds cameras even with mismatched subnets.
            devices = self.tl_factory.EnumerateDevices()
        except Exception as e:
            self._err("Enumeration failed", e)
            return

        if not devices:
            self._log("No devices found. Check cables, PoE/24V, and that pylon "
                      "drivers are installed.")
            return

        for d in devices:
            model = d.GetModelName()
            serial = d.GetSerialNumber()
            ip = d.GetIpAddress() if d.IsIpAddressAvailable() else "-"
            mac = mac_bytes_to_str(d.GetMacAddress()) if d.IsMacAddressAvailable() else "-"
            # Reachability heuristic: GigE devices expose a subnet mask
            status = "OK"
            try:
                if d.GetDeviceClass() == "BaslerGigE":
                    # If the host can't reach it, pylon still lists it, mark it.
                    if hasattr(d, "IsPersistentIpActive"):
                        pass
                    # Newer pypylon exposes IsReachable via a helper; fallback:
                    subnet = d.GetSubnetAddress() if d.IsSubnetAddressAvailable() else ""
                    if subnet == "" or ip == "0.0.0.0":
                        status = "Unreachable"
            except Exception:
                pass

            self.discovered.append(d)
            self.tree.insert("", "end", values=(model, serial, ip, mac, status))

        self._log(f"Found {len(devices)} device(s).")

    def _on_select_device(self, _evt=None):
        sel = self.tree.selection()
        if not sel:
            return
        idx = self.tree.index(sel[0])
        d = self.discovered[idx]
        if d.IsIpAddressAvailable():
            self.var_ip.set(d.GetIpAddress())

    def _get_selected_device(self) -> Optional[pylon.DeviceInfo]:
        sel = self.tree.selection()
        if not sel:
            messagebox.showwarning("No selection", "Select a camera from the list first.")
            return None
        return self.discovered[self.tree.index(sel[0])]

    # ------------------------------------------------- network (GigE only)
    def force_ip(self):
        d = self._get_selected_device()
        if d is None:
            return
        if d.GetDeviceClass() != "BaslerGigE":
            self._err("ForceIP is only available for GigE cameras.")
            return
        ip, mask, gw = self.var_ip.get(), self.var_mask.get(), self.var_gw.get()
        mac = d.GetMacAddress()
        try:
            # pylon.GigeTransportLayer.ForceIp expects MAC as int or string
            gtl = self.tl_factory.CreateTl("BaslerGigE")
            gtl.ForceIp(mac, ip, mask, gw)
            self._log(f"ForceIp sent: MAC={mac_bytes_to_str(mac)}  IP={ip}  "
                      f"Mask={mask}  GW={gw}  (temporary, lost on reboot)")
            self.discover()
        except Exception as e:
            self._err("ForceIp failed", e)

    def set_persistent_ip(self):
        """Write StaticIP into the camera's non-volatile memory."""
        d = self._get_selected_device()
        if d is None:
            return
        if d.GetDeviceClass() != "BaslerGigE":
            self._err("Persistent IP is only available for GigE cameras.")
            return
        ip, mask, gw = self.var_ip.get(), self.var_mask.get(), self.var_gw.get()

        # Must open the camera to write the persistent IP nodes.
        cam = None
        try:
            cam = pylon.InstantCamera(self.tl_factory.CreateDevice(d))
            cam.Open()
            nm = cam.GetNodeMap()
            # Enable persistent IP configuration
            safe_set(nm, "GevCurrentIPConfigurationPersistentIP", True)
            safe_set(nm, "GevCurrentIPConfigurationDHCP", False)
            safe_set(nm, "GevPersistentIPAddress", ip)
            safe_set(nm, "GevPersistentSubnetMask", mask)
            safe_set(nm, "GevPersistentDefaultGateway", gw)
            self._log(f"Persistent IP written: {ip}/{mask} gw={gw}. "
                      "Reboot camera for it to apply.")
        except Exception as e:
            self._err("Writing persistent IP failed", e)
        finally:
            if cam and cam.IsOpen():
                cam.Close()

    def reboot_camera(self):
        """Issue DeviceReset. Works for GigE and USB3."""
        d = self._get_selected_device()
        if d is None:
            return
        if not messagebox.askyesno("Reboot?",
                                   "Send DeviceReset to the camera?\n"
                                   "It will drop the connection for ~10-30 seconds."):
            return
        if self.camera and self.camera.IsOpen():
            self.disconnect()
        cam = None
        try:
            cam = pylon.InstantCamera(self.tl_factory.CreateDevice(d))
            cam.Open()
            nm = cam.GetNodeMap()
            node = nm.GetNode("DeviceReset")
            if node is None:
                raise RuntimeError("Camera has no DeviceReset command.")
            node.Execute()
            self._log("DeviceReset executed. Re-run Discover in ~20 seconds.")
        except Exception as e:
            # DeviceReset typically throws because the connection is torn down.
            # That's expected behavior -- don't treat it as a hard error.
            msg = str(e)
            if "device has been removed" in msg.lower() or "timeout" in msg.lower():
                self._log("DeviceReset executed (connection dropped as expected).")
            else:
                self._err("DeviceReset failed", e)
        finally:
            try:
                if cam and cam.IsOpen():
                    cam.Close()
            except Exception:
                pass

    # ------------------------------------------------------------ connection
    def connect_selected(self):
        d = self._get_selected_device()
        if d is None:
            return
        # Close any previous camera cleanly first.
        if self.camera is not None:
            try:
                if self.camera.IsOpen():
                    self.camera.Close()
            except Exception:
                pass
            self.camera = None

        try:
            self.camera = pylon.InstantCamera(self.tl_factory.CreateDevice(d))
            self.camera.Open()
        except Exception as e:
            self.camera = None
            self._err("Connect failed (Open)", e)
            return

        # Bigger ring buffer to absorb network hiccups (GigE underruns).
        try:
            self.camera.MaxNumBuffer.Value = 30
        except Exception:
            try:
                self.camera.MaxNumBuffer = 30
            except Exception:
                pass  # harmless if it fails

        # Auto-tune GigE network parameters to prevent "incompletely grabbed"
        # buffer errors. Best-effort -- safe on USB cameras too (no-op).
        try:
            self._tune_gige(verbose=True)
        except Exception as e:
            self._log(f"GigE tuning skipped: {e}")

        # Dashboard population is best-effort -- do NOT fail the whole connect
        # if an individual node is missing.
        try:
            self._populate_dashboard()
        except Exception as e:
            self._log(f"WARNING: dashboard partially populated: {e}")

        self.lbl_conn_status.configure(
            text=f"Connected: {d.GetModelName()} ({d.GetSerialNumber()})",
            foreground="green")
        for b in (self.btn_disconnect, self.btn_live, self.btn_trig, self.btn_save):
            b.configure(state="normal")
        self.btn_connect.configure(state="disabled")
        self._log(f"Opened {d.GetModelName()} / {d.GetSerialNumber()}")

    def disconnect(self):
        self.stop_grab()
        try:
            if self.camera and self.camera.IsOpen():
                self.camera.Close()
        except Exception:
            pass
        self.camera = None
        self.lbl_conn_status.configure(text="Not connected", foreground="gray")
        self.btn_connect.configure(state="normal")
        for b in (self.btn_disconnect, self.btn_live, self.btn_stop,
                  self.btn_trig, self.btn_save):
            b.configure(state="disabled")
        self._log("Disconnected.")

    # --------------------------------------------------- GigE tuning
    def _tune_gige(self, verbose: bool = False):
        """Optimize GigE streaming to eliminate 'incompletely grabbed' errors.

        These settings are the standard Basler-recommended workaround for
        buffer underruns on Windows hosts that aren't configured for jumbo
        frames. Safe to call on USB3 cameras (nodes simply won't exist).
        """
        if not self.camera or not self.camera.IsOpen():
            return
        nm = self.camera.GetNodeMap()

        # Try to probe the host NIC MTU. If jumbo frames are off (typical
        # Windows default = 1500), cap packet size at 1500; otherwise use
        # the camera's max (usually 8192 or 9000).
        changes = []

        # --- 1. Packet size ---------------------------------------------------
        try:
            ps = _get_node(nm, "GevSCPSPacketSize")
            if ps is not None and genicam.IsWritable(ps):
                # Ask pylon to probe the safe max via its built-in helper if
                # available, otherwise fall back to a conservative 1500.
                try:
                    # pylon 7+: camera.GevSCPSPacketSize has a max reflecting
                    # what the path can carry after negotiation.
                    max_ps = ps.GetMax()
                except Exception:
                    max_ps = 1500
                # Prefer 1500 (safe on any Windows NIC without jumbo frames).
                # If user has enabled jumbo frames the NIC will accept more,
                # but 1500 is universally safe.
                target = 1500 if max_ps >= 1500 else int(max_ps)
                try:
                    ps.SetValue(target)
                    changes.append(f"PacketSize={target}")
                except Exception as e:
                    changes.append(f"PacketSize failed: {e}")
        except Exception:
            pass

        # --- 2. Inter-packet delay -------------------------------------------
        # Higher = slower stream but fewer lost packets. 1000 ticks is a
        # good starting value for a single 1 GbE link with one camera.
        try:
            ipd = _get_node(nm, "GevSCPD")
            if ipd is not None and genicam.IsWritable(ipd):
                try:
                    ipd.SetValue(1000)
                    changes.append("InterPacketDelay=1000")
                except Exception as e:
                    changes.append(f"IPD failed: {e}")
        except Exception:
            pass

        # --- 3. Frame retention / stream transfer timeout --------------------
        try:
            ft = _get_node(nm, "GevSCFTD")  # Frame Transmission Delay
            if ft is not None and genicam.IsWritable(ft):
                try:
                    ft.SetValue(0)
                    changes.append("FrameTransmissionDelay=0")
                except Exception:
                    pass
        except Exception:
            pass

        # --- 4. Heartbeat timeout (keep connection alive during debug) -------
        try:
            hb = _get_node(nm, "GevHeartbeatTimeout")
            if hb is not None and genicam.IsWritable(hb):
                try:
                    hb.SetValue(5000)
                    changes.append("Heartbeat=5000ms")
                except Exception:
                    pass
        except Exception:
            pass

        # --- 5. Limit frame rate (back off sensor if grab can't keep up) -----
        # Only cap if AcquisitionFrameRateEnable exists. Don't enable it if
        # the user already disabled it -- we just ensure a sane cap.
        try:
            en = _get_node(nm, "AcquisitionFrameRateEnable")
            fr, fr_name = first_existing_node(nm, "AcquisitionFrameRate",
                                              "AcquisitionFrameRateAbs")
            if en is not None and fr is not None and genicam.IsWritable(en):
                try:
                    en.SetValue(True)
                    fr.SetValue(15.0)  # 15 fps is comfortable on stock Windows
                    changes.append(f"{fr_name}=15 fps cap")
                except Exception:
                    pass
        except Exception:
            pass

        # --- 6. Stream grabber: max transfer size, resend, num buffers -------
        # StreamGrabberParams live on the stream's own node map, not the
        # device node map. They're accessed via camera.StreamGrabber.
        try:
            sg = getattr(self.camera, "StreamGrabber", None)
            if sg is not None:
                # Number of buffers queued to the NIC driver
                try:
                    sg.MaxNumBuffer.SetValue(30)
                    changes.append("StreamBuffers=30")
                except Exception:
                    pass
                # Enable resend requests (recovers dropped packets over UDP)
                try:
                    sg.EnableResend.SetValue(True)
                    changes.append("Resend=ON")
                except Exception:
                    pass
        except Exception:
            pass

        if verbose:
            if changes:
                self._log("GigE tuning: " + ", ".join(changes))
            else:
                self._log("GigE tuning: (no GigE nodes found -- probably USB "
                          "camera or already optimal).")

    # ---------------------------------------------------------- dashboard
    def _populate_dashboard(self):
        """Populate dashboard widgets from the camera. Each field is
        populated independently -- a missing node on this firmware must
        NOT abort the whole connect."""
        if not self.camera:
            return
        nm = self.camera.GetNodeMap()

        # Exposure (float µs) -- handle ExposureTime (SFNC 2.x) vs ExposureTimeAbs (1.x)
        try:
            exp_node, exp_name = first_existing_node(nm, "ExposureTime", "ExposureTimeAbs")
            if exp_node is not None and genicam.IsReadable(exp_node):
                self.var_exp.set(float(exp_node.GetValue()))
                self._log(f"Exposure node: {exp_name} = {self.var_exp.get():.1f} µs")
            else:
                self._log("Exposure node not available on this camera.")
        except Exception as e:
            self._log(f"Exposure read failed: {e}")

        # Gain -- SFNC 2.x uses 'Gain', older firmwares use 'GainRaw'
        try:
            gain_node, gain_name = first_existing_node(nm, "Gain", "GainRaw")
            if gain_node is not None and genicam.IsReadable(gain_node):
                self.var_gain.set(float(gain_node.GetValue()))
                self._log(f"Gain node: {gain_name} = {self.var_gain.get()}")
            else:
                self._log("Gain node not available on this camera.")
        except Exception as e:
            self._log(f"Gain read failed: {e}")

        # Pixel format enum
        try:
            pf = _get_node(nm, "PixelFormat")
            if pf is not None:
                values = []
                try:
                    for e in pf.GetEntries():
                        try:
                            if genicam.IsAvailable(e):
                                values.append(e.GetSymbolic())
                        except Exception:
                            continue
                except Exception:
                    pass
                if values:
                    self.cmb_pf.configure(values=values)
                try:
                    self.cmb_pf.set(pf.ToString())
                except Exception:
                    pass
        except Exception as e:
            self._log(f"PixelFormat read failed: {e}")

        # ROI
        try:
            self.var_w.set(int(safe_get(nm, "Width", 0) or 0))
            self.var_h.set(int(safe_get(nm, "Height", 0) or 0))
            self.var_ox.set(int(safe_get(nm, "OffsetX", 0) or 0))
            self.var_oy.set(int(safe_get(nm, "OffsetY", 0) or 0))
        except Exception as e:
            self._log(f"ROI read failed: {e}")

        # Trigger
        try:
            tm = safe_get(nm, "TriggerMode")
            self.cmb_trig.set("Software" if tm == "On" else "Off (Continuous)")
        except Exception as e:
            self._log(f"TriggerMode read failed: {e}")

    def apply_exposure(self):
        if not self.camera:
            return
        try:
            nm = self.camera.GetNodeMap()
            node, _ = first_existing_node(nm, "ExposureTime", "ExposureTimeAbs")
            if node is None:
                raise RuntimeError("No ExposureTime/ExposureTimeAbs node on this camera.")
            node.SetValue(float(self.var_exp.get()))
            self._log(f"Exposure set to {self.var_exp.get():.1f} µs.")
        except Exception as e:
            self._err("Set exposure failed", e)

    def apply_gain(self):
        if not self.camera:
            return
        try:
            nm = self.camera.GetNodeMap()
            node, _ = first_existing_node(nm, "Gain", "GainRaw")
            if node is None:
                raise RuntimeError("No Gain/GainRaw node on this camera.")
            node.SetValue(float(self.var_gain.get()))
            self._log(f"Gain set to {self.var_gain.get()}.")
        except Exception as e:
            self._err("Set gain failed", e)

    def apply_pixel_format(self):
        if not self.camera:
            return
        val = self.cmb_pf.get()
        if not val:
            return
        was_grabbing = self.camera.IsGrabbing()
        if was_grabbing:
            self.stop_grab()
        try:
            safe_set(self.camera.GetNodeMap(), "PixelFormat", val)
            self._log(f"PixelFormat -> {val}")
        except Exception as e:
            self._err("Set pixel format failed", e)
        if was_grabbing:
            self.start_live()

    def apply_roi(self):
        if not self.camera:
            return
        was_grabbing = self.camera.IsGrabbing()
        if was_grabbing:
            self.stop_grab()
        try:
            nm = self.camera.GetNodeMap()
            # Order matters: shrink first, then reposition, then re-grow.
            safe_set(nm, "OffsetX", 0)
            safe_set(nm, "OffsetY", 0)
            safe_set(nm, "Width", int(self.var_w.get()))
            safe_set(nm, "Height", int(self.var_h.get()))
            safe_set(nm, "OffsetX", int(self.var_ox.get()))
            safe_set(nm, "OffsetY", int(self.var_oy.get()))
            self._log(f"ROI applied: {self.var_w.get()}x{self.var_h.get()} "
                      f"@ ({self.var_ox.get()},{self.var_oy.get()})")
        except Exception as e:
            self._err("Apply ROI failed", e)
        if was_grabbing:
            self.start_live()

    def reset_roi(self):
        if not self.camera:
            return
        was_grabbing = self.camera.IsGrabbing()
        if was_grabbing:
            self.stop_grab()
        try:
            nm = self.camera.GetNodeMap()
            wmax = nm.GetNode("Width").GetMax()
            hmax = nm.GetNode("Height").GetMax()
            safe_set(nm, "OffsetX", 0)
            safe_set(nm, "OffsetY", 0)
            safe_set(nm, "Width", wmax)
            safe_set(nm, "Height", hmax)
            self.var_w.set(wmax)
            self.var_h.set(hmax)
            self.var_ox.set(0)
            self.var_oy.set(0)
            self._log(f"ROI reset to max {wmax}x{hmax}.")
        except Exception as e:
            self._err("Max ROI failed", e)
        if was_grabbing:
            self.start_live()

    def apply_trigger_mode(self):
        if not self.camera:
            return
        mode = self.cmb_trig.get()
        try:
            nm = self.camera.GetNodeMap()
            if mode.startswith("Off"):
                safe_set(nm, "TriggerMode", "Off")
                self._log("TriggerMode = Off (continuous acquisition).")
            else:
                safe_set(nm, "TriggerSelector", "FrameStart")
                safe_set(nm, "TriggerMode", "On")
                safe_set(nm, "TriggerSource", "Software")
                self._log("TriggerMode = Software (use 'Software Trigger' button).")
        except Exception as e:
            self._err("Set trigger mode failed", e)

    # --------------------------------------------------------- acquisition
    def start_live(self):
        if not self.camera:
            return
        if self.camera.IsGrabbing():
            return
        try:
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            self.grab_stop.clear()
            self.grab_thread = threading.Thread(target=self._grab_loop, daemon=True)
            self.grab_thread.start()
            self.btn_live.configure(state="disabled")
            self.btn_stop.configure(state="normal")
            self._log("Live grab started.")
        except Exception as e:
            self._err("Start grab failed", e)

    def stop_grab(self):
        self.grab_stop.set()
        try:
            if self.camera and self.camera.IsGrabbing():
                self.camera.StopGrabbing()
        except Exception:
            pass
        if self.grab_thread:
            self.grab_thread.join(timeout=2.0)
        self.grab_thread = None
        self.btn_stop.configure(state="disabled")
        if self.camera and self.camera.IsOpen():
            self.btn_live.configure(state="normal")

    def software_trigger(self):
        """Switch to SW trigger (if not already) and fire one."""
        if not self.camera:
            return
        try:
            nm = self.camera.GetNodeMap()
            if safe_get(nm, "TriggerMode") != "On":
                self.cmb_trig.set("Software")
                self.apply_trigger_mode()
            if not self.camera.IsGrabbing():
                self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
                self.grab_stop.clear()
                self.grab_thread = threading.Thread(target=self._grab_loop, daemon=True)
                self.grab_thread.start()
                self.btn_stop.configure(state="normal")
            # Wait for trigger ready
            if self.camera.WaitForFrameTriggerReady(1000, pylon.TimeoutHandling_ThrowException):
                self.camera.ExecuteSoftwareTrigger()
                self._log("Software trigger fired.")
        except Exception as e:
            self._err("Software trigger failed", e)

    def _grab_loop(self):
        """Runs in a background thread. Pushes frames to the UI via a queue."""
        while not self.grab_stop.is_set() and self.camera and self.camera.IsGrabbing():
            try:
                result = self.camera.RetrieveResult(500, pylon.TimeoutHandling_Return)
            except Exception as e:
                self._log(f"Grab exception: {e}")
                break
            # RetrieveResult with TimeoutHandling_Return returns an INVALID
            # grab-result object on timeout (not None). Accessing it without
            # first checking IsValid() raises "Cannot access NULL pointer".
            if result is None:
                continue
            try:
                valid = False
                try:
                    valid = bool(result.IsValid())
                except Exception:
                    valid = False
                if not valid:
                    # Timeout / no frame available -- just try again.
                    continue

                if result.GrabSucceeded():
                    img = self.converter.Convert(result).GetArray()  # HxWx3 BGR
                    try:
                        self.frame_queue.put_nowait(img)
                    except queue.Full:
                        # drop oldest
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                        try:
                            self.frame_queue.put_nowait(img)
                        except queue.Full:
                            pass
                else:
                    try:
                        code = result.ErrorCode
                        desc = result.ErrorDescription
                        self._log(f"Grab failed: {code} {desc}")
                    except Exception:
                        self._log("Grab failed: (unknown error)")
            finally:
                try:
                    result.Release()
                except Exception:
                    pass

    # ----------------------------------------------------------- UI tick
    def _ui_tick(self):
        try:
            frame = None
            while True:
                frame = self.frame_queue.get_nowait()
        except queue.Empty:
            pass

        if frame is not None:
            self.last_frame = frame
            self._display(frame)
            self._fps_count += 1

        now = time.time()
        if now - self._fps_tick >= 1.0:
            self.fps = self._fps_count / (now - self._fps_tick)
            self._fps_count = 0
            self._fps_tick = now
            self.lbl_fps.configure(text=f"FPS: {self.fps:5.2f}")

        self.after(33, self._ui_tick)

    def _display(self, bgr: np.ndarray):
        # ----- AI inference (optional, runs every N frames) -----
        display_bgr = bgr
        ran_inference = False
        if (self.ai_enabled.get() and _DETECTOR_AVAILABLE
                and self.detector is not None):
            self._ai_frame_counter += 1
            n = max(1, int(self.ai_every_n.get() or 1))
            if self._ai_frame_counter % n == 0:
                if not self.detector.is_ready:
                    # Lazy-load the model on first use (downloads yolov8n.pt).
                    self._log("Loading YOLOv8n model (first run downloads ~6 MB)…")
                    try:
                        self.lbl_ai_status.configure(text="loading model...",
                                                     foreground="orange")
                        self.update_idletasks()
                    except Exception:
                        pass
                    self.detector.ensure_loaded()
                    if self.detector.is_ready:
                        self._sync_detector_classes()
                        self._sync_detector_conf()
                        self._log("YOLOv8n model loaded.")
                        try:
                            self.lbl_ai_status.configure(text="ready",
                                                         foreground="green")
                        except Exception:
                            pass
                    else:
                        self._log(f"Model load failed: {self.detector.load_error}")
                        try:
                            self.lbl_ai_status.configure(
                                text=f"error: {self.detector.load_error}",
                                foreground="red")
                        except Exception:
                            pass
                        self.ai_enabled.set(False)

                if self.detector.is_ready:
                    # Run inference on the RAW frame (not annotated) so we
                    # can re-draw cleanly with our own selection highlight.
                    _, dets = self.detector.infer(bgr)
                    self._reconcile_selection(dets)
                    self._ai_last_detections = dets
                    ran_inference = True
                    try:
                        cnt = len(dets)
                        by_class: Dict[str, int] = {}
                        for d in dets:
                            by_class[d["cls"]] = by_class.get(d["cls"], 0) + 1
                        if by_class:
                            summary = ", ".join(
                                f"{k}:{v}" for k, v in sorted(by_class.items())
                            )
                            self.lbl_ai_status.configure(
                                text=f"{cnt} det · {summary}",
                                foreground="green")
                        else:
                            self.lbl_ai_status.configure(
                                text="0 detections",
                                foreground="gray")
                    except Exception:
                        pass

            # Whether we just inferred or are on a skip frame, draw the
            # cached detections (with selection highlight) onto the frame.
            if self._ai_last_detections:
                display_bgr = self._draw_detections(bgr, self._ai_last_detections)
        else:
            # AI off -> no detection boxes; clear any selection.
            if self._ai_selected_idx is not None:
                self._ai_selected_idx = None
                self._ai_last_detections = []
                self._update_selection_label()

        self.last_display = display_bgr
        self._paint_to_canvas(display_bgr)

    # ---------- canvas scaling / painting ----------
    def _paint_to_canvas(self, bgr: np.ndarray):
        """Resize `bgr` to the current canvas size per self.view_mode, then draw.

        Also records a canvas<->image mapping so mouse clicks can be back-
        projected into image coordinates.

        view_mode:
          - 'fit'  : preserve aspect, upscale/downscale to fully show the image
          - 'fill' : preserve aspect, upscale/downscale to fill + crop excess
          - '1:1'  : show at native pixel size (centered; may overflow canvas)
        """
        h, w = bgr.shape[:2]
        cw = max(self.canvas.winfo_width(), 100)
        ch = max(self.canvas.winfo_height(), 100)

        mode = self.view_mode.get() if hasattr(self, "view_mode") else "fit"
        if mode == "1:1":
            nw, nh = w, h
            scale = 1.0
        elif mode == "fill":
            scale = max(cw / w, ch / h)
            nw, nh = max(int(w * scale), 1), max(int(h * scale), 1)
        else:  # "fit"
            scale = min(cw / w, ch / h)
            nw, nh = max(int(w * scale), 1), max(int(h * scale), 1)

        rgb = bgr[:, :, ::-1]
        # Use BILINEAR for smoother upscale; it's still fast enough for live video.
        resample = Image.BILINEAR if (nw != w or nh != h) else Image.NEAREST
        img = Image.fromarray(rgb).resize((nw, nh), resample)
        self._tk_img = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        # anchor="center": top-left of the image lands at (cw/2 - nw/2, ch/2 - nh/2)
        draw_x = cw / 2 - nw / 2
        draw_y = ch / 2 - nh / 2
        self.canvas.create_image(cw // 2, ch // 2,
                                 image=self._tk_img, anchor="center")

        # Record the mapping so _on_canvas_click can convert back.
        self._canvas_map = (w, h, draw_x, draw_y, scale, cw, ch)

        # Paint a crisp canvas-space highlight around the selected detection
        # (on top of the already-burned-in image boxes). This keeps the
        # selection visible even at 1:1 when the box might be clipped.
        if (self._ai_selected_idx is not None
                and self._ai_last_detections
                and 0 <= self._ai_selected_idx < len(self._ai_last_detections)):
            d = self._ai_last_detections[self._ai_selected_idx]
            x1, y1, x2, y2 = d["xyxy"]
            cx1 = draw_x + x1 * scale
            cy1 = draw_y + y1 * scale
            cx2 = draw_x + x2 * scale
            cy2 = draw_y + y2 * scale
            # Thick yellow outline
            self.canvas.create_rectangle(cx1, cy1, cx2, cy2,
                                         outline="#FFFF00", width=4)
            # Corner ticks for emphasis
            tick = 14
            for (px, py, dx1, dy1, dx2, dy2) in (
                (cx1, cy1,  0,  0,  tick,  0),   # TL -
                (cx1, cy1,  0,  0,  0,  tick),   # TL |
                (cx2, cy1, -tick, 0,  0,  0),    # TR -
                (cx2, cy1,  0,  0,  0,  tick),   # TR |
                (cx1, cy2,  0, -tick, 0,  0),    # BL |
                (cx1, cy2,  0,  0,  tick, 0),    # BL -
                (cx2, cy2,  0, -tick, 0,  0),    # BR |
                (cx2, cy2, -tick, 0,  0,  0),    # BR -
            ):
                self.canvas.create_line(px + dx1, py + dy1,
                                        px + dx2, py + dy2,
                                        fill="#FFFF00", width=4)
            # Label with class, confidence, and box size
            label = f"{d['cls']}  {d['conf']*100:.1f}%  ({x2-x1}x{y2-y1}px)"
            tx = cx1
            ty = max(cy1 - 18, 4)
            self.canvas.create_rectangle(tx, ty,
                                         tx + 9 * len(label), ty + 16,
                                         fill="#FFFF00", outline="")
            self.canvas.create_text(tx + 4, ty + 8, anchor="w",
                                    text=label, fill="black",
                                    font=("Segoe UI", 9, "bold"))

    def _redraw_current(self):
        """Re-paint the last displayed frame (called on canvas resize)."""
        if self.last_display is not None:
            self._paint_to_canvas(self.last_display)
        elif self.last_frame is not None:
            self._paint_to_canvas(self.last_frame)

    # ------------------------------------------------ AI helper methods
    def _draw_detections(self, bgr: np.ndarray, dets: list) -> np.ndarray:
        """Burn the detection boxes into a copy of `bgr` and return it.
        The currently selected detection gets a thicker yellow box (the
        canvas-space overlay in _paint_to_canvas makes it unmistakable)."""
        import cv2  # local import to avoid module-level dependency at startup
        out = bgr.copy()
        for i, d in enumerate(dets):
            x1, y1, x2, y2 = d["xyxy"]
            label = d["cls"]
            conf = d["conf"]
            is_sel = (i == self._ai_selected_idx)
            color = (0, 255, 255) if is_sel else (0, 255, 0)  # BGR: yellow / green
            thick = 4 if is_sel else 2
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
            txt = f"{label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX,
                                          0.5, 1)
            cv2.rectangle(out, (x1, y1 - th - 6),
                          (x1 + tw + 4, y1), color, -1)
            cv2.putText(out, txt, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0) if is_sel else (255, 255, 255),
                        1, cv2.LINE_AA)
        return out

    # ----- keep compatibility with any external callers that used the old name
    _redraw_detections = _draw_detections

    # ------------------------------------------------- selection handling
    def _on_canvas_click(self, event):
        """User clicked the preview canvas. If they hit a detection box,
        select it. If they clicked empty space, clear the selection."""
        if self._canvas_map is None or not self._ai_last_detections:
            return
        img_w, img_h, draw_x, draw_y, scale, cw, ch = self._canvas_map

        # Convert canvas -> image pixel
        ix = (event.x - draw_x) / max(scale, 1e-6)
        iy = (event.y - draw_y) / max(scale, 1e-6)

        # Click must be inside the image area
        if not (0 <= ix <= img_w and 0 <= iy <= img_h):
            self._clear_selection()
            return

        # Find the smallest box that contains the click (so overlapping
        # children can be selected instead of a giant parent).
        hit_idx = None
        hit_area = None
        for i, d in enumerate(self._ai_last_detections):
            x1, y1, x2, y2 = d["xyxy"]
            if x1 <= ix <= x2 and y1 <= iy <= y2:
                area = (x2 - x1) * (y2 - y1)
                if hit_area is None or area < hit_area:
                    hit_area = area
                    hit_idx = i

        if hit_idx is None:
            self._clear_selection()
            return

        self._ai_selected_idx = hit_idx
        self._update_selection_label()
        # Re-paint right away so the highlight is visible even on slow FPS.
        self._redraw_current()

    def _clear_selection(self):
        if self._ai_selected_idx is not None:
            self._ai_selected_idx = None
            self._update_selection_label()
            self._redraw_current()

    def _update_selection_label(self):
        if (self._ai_selected_idx is None
                or not self._ai_last_detections
                or self._ai_selected_idx >= len(self._ai_last_detections)):
            try:
                self.lbl_selection.configure(
                    text="Click any detection box in the preview to select it.",
                    foreground="gray")
            except Exception:
                pass
            return
        d = self._ai_last_detections[self._ai_selected_idx]
        x1, y1, x2, y2 = d["xyxy"]
        w = x2 - x1
        h = y2 - y1
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        txt = (f"Selected: {d['cls']}   confidence {d['conf']*100:.1f}%   "
               f"box {w}×{h}px   center ({cx},{cy})   "
               f"bbox ({x1},{y1})-({x2},{y2})")
        try:
            self.lbl_selection.configure(text=txt, foreground="black")
        except Exception:
            pass

    def _reconcile_selection(self, new_dets: list):
        """When a fresh inference produces a new detection list, try to keep
        the currently selected object selected by matching to the closest
        new box of the same class (IoU-based greedy match)."""
        if self._ai_selected_idx is None or not self._ai_last_detections:
            return
        old = self._ai_last_detections[self._ai_selected_idx]
        best = None
        best_iou = 0.0
        for i, d in enumerate(new_dets):
            if d["cls"] != old["cls"]:
                continue
            iou = _iou(old["xyxy"], d["xyxy"])
            if iou > best_iou:
                best_iou = iou
                best = i
        if best is not None and best_iou >= 0.2:
            self._ai_selected_idx = best
        else:
            # Lost the tracked object -> clear selection silently
            self._ai_selected_idx = None
        self._update_selection_label()

    def _on_ai_toggle(self):
        """Called when the 'Enable AI overlay' checkbox changes."""
        if not _DETECTOR_AVAILABLE:
            return
        if self.ai_enabled.get():
            self._log("AI overlay enabled. Model will load on first frame.")
        else:
            self._ai_last_detections = []
            self._log("AI overlay disabled.")
            try:
                self.lbl_ai_status.configure(text="disabled", foreground="gray")
            except Exception:
                pass

    def _sync_detector_classes(self):
        if not _DETECTOR_AVAILABLE or self.detector is None:
            return
        names = [n for n, v in self.class_vars.items() if v.get()]
        self.detector.set_classes(names)

    def _sync_detector_conf(self):
        if not _DETECTOR_AVAILABLE or self.detector is None:
            return
        c = float(self.ai_conf.get())
        self.detector.set_conf(c)
        try:
            self.lbl_conf.configure(text=f"{c:.2f}")
        except Exception:
            pass

    def _select_classes(self, on: bool):
        for v in self.class_vars.values():
            v.set(on)
        self._sync_detector_classes()

    def _preset_classes(self, names: list):
        wanted = set(names)
        for n, v in self.class_vars.items():
            v.set(n in wanted)
        self._sync_detector_classes()

    # --------------------------------------------- command console
    def _cmd_append(self, text: str, tag: str = "") -> None:
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        line = f"[{ts}] {text}\n"
        self.cmd_log.configure(state="normal")
        if tag:
            self.cmd_log.insert("end", line, tag)
        else:
            self.cmd_log.insert("end", line)
        self.cmd_log.see("end")
        self.cmd_log.configure(state="disabled")

    def _cmd_preset(self, op: str, node: str, value: str = "") -> None:
        """Fill the command fields from a preset button and execute it."""
        self.var_cmd_op.set(op)
        self.var_cmd_node.set(node)
        self.var_cmd_val.set(value)
        self._cmd_send()

    def _cmd_send(self) -> None:
        """Execute the currently-entered GET / SET / EXEC command and log
        both the command and the camera's response to the command log."""
        op = self.var_cmd_op.get().strip().upper()
        name = self.var_cmd_node.get().strip()
        value = self.var_cmd_val.get().strip()

        if not name:
            self._cmd_append("(no node name)", "err")
            return

        if self.camera is None or not self.camera.IsOpen():
            self._cmd_append(f">>> {op} {name}" + (f" = {value}" if value else ""),
                             "cmd")
            self._cmd_append("    ERROR: not connected -- click Connect first.",
                             "err")
            return

        # Echo the command (with arrow + arguments) in the log
        if op == "SET":
            self._cmd_append(f">>> {op} {name} = {value}", "cmd")
        else:
            self._cmd_append(f">>> {op} {name}", "cmd")

        nm = self.camera.GetNodeMap()
        node = _get_node(nm, name)
        if node is None:
            self._cmd_append(f"    ERROR: node '{name}' does not exist on this "
                             "camera.", "err")
            return

        try:
            if op == "GET":
                if not genicam.IsReadable(node):
                    self._cmd_append(f"    ERROR: node is not readable in "
                                     "current state.", "err")
                    return
                # Value rendering
                val_repr = ""
                try:
                    v = node.GetValue()
                    val_repr = repr(v)
                except Exception:
                    try:
                        val_repr = node.ToString()
                    except Exception as e:
                        raise RuntimeError(
                            f"can't read value: {e}") from e

                # Extra detail for enumerations: list available symbolics
                try:
                    entries = list(node.GetEntries())
                    syms = []
                    for e in entries:
                        try:
                            if genicam.IsAvailable(e):
                                syms.append(e.GetSymbolic())
                        except Exception:
                            continue
                    if syms:
                        val_repr += f"   [available: {', '.join(syms)}]"
                except Exception:
                    pass

                # Extra detail for numbers: min/max/unit
                try:
                    if hasattr(node, "GetMin") and hasattr(node, "GetMax"):
                        unit = ""
                        try:
                            unit = node.GetUnit() or ""
                        except Exception:
                            pass
                        val_repr += (f"   [min={node.GetMin()}, "
                                     f"max={node.GetMax()}"
                                     + (f", unit={unit}" if unit else "")
                                     + "]")
                except Exception:
                    pass

                self._cmd_append(f"    {val_repr}", "ok")

            elif op == "SET":
                if not genicam.IsWritable(node):
                    self._cmd_append("    ERROR: node is not writable in "
                                     "current state.", "err")
                    return
                # For enum / string nodes, FromString is appropriate.
                # For ints/floats, try numeric cast first then fall back to FromString.
                applied = False
                if value != "":
                    # Try numeric paths first -- the node's own SetValue knows
                    # int vs float and will raise if it doesn't match.
                    try:
                        if "." in value:
                            node.SetValue(float(value))
                            applied = True
                        else:
                            try:
                                node.SetValue(int(value))
                                applied = True
                            except Exception:
                                # Boolean nodes accept 'true'/'false'
                                if value.lower() in ("true", "false"):
                                    node.SetValue(value.lower() == "true")
                                    applied = True
                    except Exception:
                        pass
                if not applied:
                    # Fall back to string ("FromString" works for enums,
                    # strings, even some numerics on certain nodes)
                    if hasattr(node, "FromString"):
                        node.FromString(value)
                    else:
                        node.SetValue(value)
                self._cmd_append(f"    OK: {name} = {value!r}", "ok")

            elif op == "EXEC":
                if not hasattr(node, "Execute"):
                    self._cmd_append("    ERROR: node is not a command "
                                     "(no Execute method).", "err")
                    return
                node.Execute()
                self._cmd_append(f"    OK: executed {name}", "ok")

            else:
                self._cmd_append(f"    ERROR: unknown op '{op}' (use GET/SET/EXEC).",
                                 "err")

        except genicam.GenericException as e:
            self._cmd_append(f"    GenICam error: {e}", "err")
        except Exception as e:
            self._cmd_append(f"    {type(e).__name__}: {e}", "err")

    def _cmd_dump_nodes(self) -> None:
        """Enumerate every readable node on the camera and dump its value."""
        if self.camera is None or not self.camera.IsOpen():
            self._cmd_append(">>> dump all readable nodes", "cmd")
            self._cmd_append("    ERROR: not connected.", "err")
            return
        self._cmd_append(">>> dump all readable nodes", "cmd")
        nm = self.camera.GetNodeMap()
        try:
            nodes = nm.GetNodes()
        except Exception as e:
            self._cmd_append(f"    ERROR: {e}", "err")
            return

        count = 0
        for node in nodes:
            try:
                name = node.GetName()
            except Exception:
                continue
            # Skip non-leaf / non-value nodes
            try:
                if not genicam.IsReadable(node):
                    continue
            except Exception:
                continue
            # Skip command / category nodes (they don't have values)
            try:
                if hasattr(node, "GetPrincipalInterfaceType"):
                    itype = node.GetPrincipalInterfaceType()
                    # intfIString / intfIInteger / intfIFloat / intfIEnumeration /
                    # intfIBoolean all have readable values. intfICommand /
                    # intfICategory / intfIRegister do not.
                    skipped = {genicam.intfICommand,
                               genicam.intfICategory,
                               genicam.intfIRegister}
                    if itype in skipped:
                        continue
            except Exception:
                pass
            try:
                val = node.ToString()
            except Exception:
                continue
            self._cmd_append(f"    {name} = {val}", "ok")
            count += 1
        self._cmd_append(f"    ({count} readable nodes)", "cmd")

    def _cmd_clear(self) -> None:
        self.cmd_log.configure(state="normal")
        self.cmd_log.delete("1.0", "end")
        self.cmd_log.configure(state="disabled")

    def _cmd_save(self) -> None:
        fn = filedialog.asksaveasfilename(
            defaultextension=".txt",
            initialfile=f"basler_cmdlog_{datetime.now():%Y%m%d_%H%M%S}.txt",
            filetypes=[("Text", "*.txt"), ("Log", "*.log")])
        if not fn:
            return
        try:
            with open(fn, "w", encoding="utf-8") as f:
                f.write(self.cmd_log.get("1.0", "end"))
            self._log(f"Command log saved to {fn}")
        except Exception as e:
            self._err("Save command log failed", e)

    # ---------------------------------------------- 3D-mapping frame source
    def _get_latest_frame(self):
        """Callback used by Mapping3DTab. Returns the most recent BGR frame
        or None if no frame is available yet. Does NOT block."""
        return self.last_frame

    # ----------------------------------------------------------- save
    def save_last_frame(self):
        if self.last_frame is None:
            messagebox.showinfo("Nothing to save", "Grab a frame first.")
            return
        # If AI is on, save the annotated view; otherwise save the raw frame.
        src = (self.last_display if (self.ai_enabled.get()
                                     and self.last_display is not None)
               else self.last_frame)
        fn = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile=f"basler_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("TIFF", "*.tif")])
        if not fn:
            return
        try:
            rgb = src[:, :, ::-1]
            Image.fromarray(rgb).save(fn)
            self._log(f"Saved {fn}")
        except Exception as e:
            self._err("Save failed", e)

    # --------------------------------------------------------- shutdown
    def _on_close(self):
        try:
            self.stop_grab()
            self.disconnect()
        except Exception:
            pass
        # Also close the Cognex tab's TCP session if open.
        try:
            if getattr(self, "cognex_tab", None) is not None:
                self.cognex_tab.shutdown()
        except Exception:
            pass
        # Also close the Keyence tab.
        try:
            if getattr(self, "keyence_tab", None) is not None:
                self.keyence_tab.shutdown()
        except Exception:
            pass
        # Also stop the live 3D mapping (closes Open3D window + worker).
        try:
            if getattr(self, "map3d_tab", None) is not None:
                self.map3d_tab.shutdown()
        except Exception:
            pass
        self.destroy()




# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        app = BaslerApp()
        app.mainloop()
    except Exception as e:
        traceback.print_exc()
        messagebox.showerror("Fatal error", f"{type(e).__name__}: {e}")
