"""
3D Mapping tab for the Basler GUI
---------------------------------
Builds a live 3D map from the Basler camera stream using monocular depth
estimation (MiDaS) and Open3D.

How it works
------------
* A background worker thread pulls the most recent BGR frame from the Basler
  via a callback supplied by the main app.
* Each frame is turned into a dense depth map with MiDaS (torch.hub, cached).
* Pixels are back-projected with user-configurable intrinsics into an Open3D
  PointCloud.
* New clouds are voxel-downsampled and optionally ICP-aligned to the running
  map so you can physically walk the camera around and the map keeps growing
  / stays coherent.
* An Open3D visualization window runs on its own thread and is refreshed
  every frame, giving you a real 3D scene you can rotate, pan, zoom with
  the mouse.

Dependencies
------------
* open3d (required for the live 3D window)        -  pip install open3d
* torch + MiDaS from torch.hub (recommended)      - comes with ultralytics
* numpy, cv2, tkinter
"""

from __future__ import annotations

import os
import time
import threading
import traceback
from datetime import datetime
from typing import Callable, List, Optional

import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


# --- Optional: Open3D -----------------------------------------------------
_O3D_OK = False
_O3D_IMPORT_ERR: Optional[str] = None
try:
    import open3d as o3d  # noqa: F401
    _O3D_OK = True
except Exception as _e:
    _O3D_IMPORT_ERR = f"{type(_e).__name__}: {_e}"


# --- Optional torch / MiDaS ----------------------------------------------
_TORCH_OK = False
_MIDAS_LOAD_ERR: Optional[str] = None
try:
    import torch  # noqa: F401
    _TORCH_OK = True
except Exception as _e:
    _MIDAS_LOAD_ERR = f"torch not available: {_e}"


# =====================================================================
# Depth estimator: MiDaS (lazy-loaded, thread-safe for single GUI user)
# =====================================================================
class _DepthEngine:
    def __init__(self):
        self.model = None
        self.transform = None
        self.device = None
        self.last_error: Optional[str] = _MIDAS_LOAD_ERR
        self.ready = False
        self._lock = threading.Lock()

    def load(self, logger: Callable[[str], None]) -> bool:
        with self._lock:
            if self.ready:
                return True
            if not _TORCH_OK:
                self.last_error = "torch is not installed."
                logger(f"Depth engine: {self.last_error}")
                return False
            try:
                import torch
                logger("Depth engine: loading MiDaS_small from torch.hub ...")
                self.device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
                self.model = torch.hub.load(
                    "intel-isl/MiDaS", "MiDaS_small", trust_repo=True
                )
                self.model.to(self.device).eval()
                mt = torch.hub.load("intel-isl/MiDaS", "transforms",
                                    trust_repo=True)
                self.transform = mt.small_transform
                self.ready = True
                logger(f"Depth engine: ready on {self.device}.")
                return True
            except Exception as e:
                self.last_error = f"{type(e).__name__}: {e}"
                logger(f"Depth engine load failed: {self.last_error}")
                self.model = None
                self.transform = None
                self.ready = False
                return False

    def infer(self, rgb: np.ndarray) -> np.ndarray:
        """rgb: HxWx3 uint8 (RGB).  Returns HxW float32 inverse-depth in 0..1
        (larger = closer)."""
        import torch
        with self._lock:
            with torch.no_grad():
                inp = self.transform(rgb).to(self.device)
                pred = self.model(inp)
                pred = torch.nn.functional.interpolate(
                    pred.unsqueeze(1),
                    size=rgb.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze().cpu().numpy()
        mn, mx = float(pred.min()), float(pred.max())
        if mx - mn < 1e-6:
            return np.zeros_like(pred, dtype=np.float32)
        return ((pred - mn) / (mx - mn)).astype(np.float32)


def _fallback_depth(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.float32) / 255.0
    try:
        import cv2
        g = cv2.GaussianBlur(g, (9, 9), 0)
    except Exception:
        pass
    mn, mx = float(g.min()), float(g.max())
    if mx - mn < 1e-6:
        return np.zeros_like(g)
    return (g - mn) / (mx - mn)


# =====================================================================
# Open3D viewer that runs on its own thread
# =====================================================================
class _LiveO3DViewer:
    """
    Owns an Open3D VisualizerWithKeyCallback window on a dedicated thread.
    The capture worker calls `push_cloud(o3d_pcd)` to merge a newly-seen
    point cloud into the running map and signal a refresh.
    """

    def __init__(self, voxel_size: float, window_name: str = "Basler 3D Map"):
        self.voxel_size = voxel_size
        self.window_name = window_name
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._incoming_lock = threading.Lock()
        self._incoming_queue: list = []   # list of (pcd, reset_flag)
        self._merged = None  # type: Optional[o3d.geometry.PointCloud]
        self._needs_update = False

    # -------- lifecycle --------
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self):
        if self.is_running():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        self._thread = None

    # -------- external interface --------
    def reset(self):
        with self._incoming_lock:
            self._incoming_queue.append(("reset", None))

    def push_cloud(self, pcd):
        """pcd is an open3d.geometry.PointCloud to merge into the live map."""
        with self._incoming_lock:
            self._incoming_queue.append(("add", pcd))

    def get_merged(self):
        return self._merged

    # -------- thread body --------
    def _run(self):
        try:
            vis = o3d.visualization.VisualizerWithKeyCallback()
            vis.create_window(window_name=self.window_name,
                              width=1100, height=750)
            opt = vis.get_render_option()
            opt.background_color = np.array([0.05, 0.07, 0.10])
            opt.point_size = 2.0

            # Empty placeholder so the window has something to render.
            self._merged = o3d.geometry.PointCloud()
            vis.add_geometry(self._merged)

            # Add world axis at origin for orientation.
            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
            vis.add_geometry(axis, reset_bounding_box=False)

            first_cloud_seen = False

            while not self._stop.is_set():
                # Drain inbound queue
                with self._incoming_lock:
                    batch = self._incoming_queue
                    self._incoming_queue = []

                changed = False
                for op, payload in batch:
                    if op == "reset":
                        self._merged.clear()
                        first_cloud_seen = False
                        changed = True
                    elif op == "add" and payload is not None:
                        if len(self._merged.points) == 0:
                            # First cloud: take it wholesale
                            self._merged.points = payload.points
                            self._merged.colors = payload.colors
                        else:
                            self._merged += payload
                            try:
                                self._merged = self._merged.voxel_down_sample(
                                    self.voxel_size
                                )
                            except Exception:
                                pass
                        changed = True

                if changed:
                    vis.update_geometry(self._merged)
                    if not first_cloud_seen and len(self._merged.points) > 0:
                        vis.reset_view_point(True)
                        first_cloud_seen = True

                if not vis.poll_events():
                    break
                vis.update_renderer()
                time.sleep(0.01)  # don't spin-lock

            vis.destroy_window()
        except Exception:
            # Keep traceback so the main thread can show it if needed
            traceback.print_exc()


# =====================================================================
# The Tkinter tab
# =====================================================================
class Mapping3DTab(ttk.Frame):
    """Live 3D mapping from the Basler stream, rendered in Open3D."""

    def __init__(self, master,
                 get_frame_cb: Callable[[], Optional[np.ndarray]],
                 log_cb: Callable[[str], None]):
        super().__init__(master)
        self._get_frame = get_frame_cb
        self._log_main = log_cb
        self._engine = _DepthEngine()

        # Open3D live viewer (created on Start)
        self._viewer: Optional[_LiveO3DViewer] = None

        # Capture worker
        self._capture_stop = threading.Event()
        self._capture_thread: Optional[threading.Thread] = None
        self._stats_lock = threading.Lock()
        self._stats = {"frames": 0, "pts": 0, "hz": 0.0, "last": 0.0}

        # Intrinsics (sensible defaults for a 2040x1536 sensor)
        self.var_fx = tk.DoubleVar(value=1500.0)
        self.var_fy = tk.DoubleVar(value=1500.0)
        self.var_cx = tk.DoubleVar(value=-1.0)
        self.var_cy = tk.DoubleVar(value=-1.0)
        self.var_scale = tk.DoubleVar(value=1.0)

        # Sampling & limits
        self.var_stride = tk.IntVar(value=6)
        self.var_min_depth = tk.DoubleVar(value=0.05)
        self.var_max_depth = tk.DoubleVar(value=0.95)

        # Live streaming knobs
        self.var_rate = tk.DoubleVar(value=2.0)      # Hz of capture
        self.var_voxel = tk.DoubleVar(value=0.02)    # meters (map resolution)
        self.var_icp = tk.BooleanVar(value=False)    # align consecutive frames
        self.var_max_points = tk.IntVar(value=500000)

        self._build_ui()
        self._update_engine_status()
        self.after(250, self._poll_stats)

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(fill="both", expand=True, padx=6, pady=6)

        # Header / status banner ---------------------------------------
        banner = ttk.LabelFrame(top, text="3D Mapping — live Basler → Open3D")
        banner.pack(fill="x", pady=(0, 6))

        if not _O3D_OK:
            tk.Label(
                banner, fg="red",
                text=(f"Open3D is not installed: {_O3D_IMPORT_ERR}\n"
                      "Run:  pip install open3d\n"
                      "(The tab needs Open3D for the live 3D window.)"),
                justify="left", padx=10, pady=8
            ).pack(anchor="w")
        else:
            tk.Label(
                banner,
                text=("Click 'Start Live Mapping'. A separate Open3D window "
                      "will open showing the 3D map as it's built from the "
                      "Basler stream. Rotate with left-drag, pan with Shift-"
                      "left-drag, zoom with the scroll wheel."),
                justify="left", padx=10, pady=6,
                foreground="#223", wraplength=900,
            ).pack(anchor="w")

        # Two columns: Controls | Live stats + buttons -----------------
        body = ttk.Frame(top)
        body.pack(fill="both", expand=True)

        left = ttk.Frame(body)
        left.pack(side="left", fill="y", padx=(0, 8))

        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True)

        # --- Streaming controls ---------------------------------------
        stream = ttk.LabelFrame(left, text="Live Stream")
        stream.pack(fill="x", pady=(0, 6))
        ttk.Label(stream, text="Capture rate (Hz)").grid(
            row=0, column=0, sticky="e", padx=4, pady=2)
        ttk.Spinbox(stream, from_=0.2, to=10.0, increment=0.5,
                    textvariable=self.var_rate, width=8).grid(
            row=0, column=1, sticky="w", padx=4, pady=2)

        ttk.Label(stream, text="Map voxel size (m)").grid(
            row=1, column=0, sticky="e", padx=4, pady=2)
        ttk.Spinbox(stream, from_=0.001, to=0.2, increment=0.005,
                    format="%.3f",
                    textvariable=self.var_voxel, width=8).grid(
            row=1, column=1, sticky="w", padx=4, pady=2)

        ttk.Checkbutton(stream, text="ICP-align consecutive frames",
                        variable=self.var_icp).grid(
            row=2, column=0, columnspan=2, sticky="w", padx=4, pady=(6, 2))

        ttk.Label(stream, text="Max points (cap)").grid(
            row=3, column=0, sticky="e", padx=4, pady=2)
        ttk.Spinbox(stream, from_=50000, to=2000000, increment=50000,
                    textvariable=self.var_max_points, width=10).grid(
            row=3, column=1, sticky="w", padx=4, pady=2)

        # --- Intrinsics -----------------------------------------------
        intr = ttk.LabelFrame(left, text="Camera Intrinsics (px)")
        intr.pack(fill="x", pady=(0, 6))
        for r, (lbl, v) in enumerate([
            ("fx", self.var_fx), ("fy", self.var_fy),
            ("cx  (-1 = auto)", self.var_cx),
            ("cy  (-1 = auto)", self.var_cy),
            ("world scale", self.var_scale),
        ]):
            ttk.Label(intr, text=lbl).grid(row=r, column=0, sticky="e",
                                           padx=4, pady=2)
            ttk.Entry(intr, textvariable=v, width=10).grid(
                row=r, column=1, sticky="w", padx=4, pady=2)

        # --- Sampling -------------------------------------------------
        samp = ttk.LabelFrame(left, text="Sampling")
        samp.pack(fill="x", pady=(0, 6))
        ttk.Label(samp, text="Pixel stride").grid(row=0, column=0, sticky="e",
                                                  padx=4, pady=2)
        ttk.Spinbox(samp, from_=1, to=32, textvariable=self.var_stride,
                    width=6).grid(row=0, column=1, sticky="w")
        ttk.Label(samp, text="Min depth (0..1)").grid(
            row=1, column=0, sticky="e", padx=4, pady=2)
        ttk.Entry(samp, textvariable=self.var_min_depth, width=8).grid(
            row=1, column=1, sticky="w")
        ttk.Label(samp, text="Max depth (0..1)").grid(
            row=2, column=0, sticky="e", padx=4, pady=2)
        ttk.Entry(samp, textvariable=self.var_max_depth, width=8).grid(
            row=2, column=1, sticky="w")

        # --- Actions --------------------------------------------------
        act = ttk.LabelFrame(right, text="Actions")
        act.pack(fill="x")

        self.btn_start = ttk.Button(act, text="▶  Start Live Mapping",
                                    command=self.start_live,
                                    state=("disabled" if not _O3D_OK else "normal"))
        self.btn_start.pack(fill="x", padx=8, pady=(8, 3))

        self.btn_stop = ttk.Button(act, text="■  Stop Live Mapping",
                                   command=self.stop_live, state="disabled")
        self.btn_stop.pack(fill="x", padx=8, pady=3)

        ttk.Separator(act, orient="horizontal").pack(fill="x", padx=4, pady=6)

        ttk.Button(act, text="↺  Reset map (keep window open)",
                   command=self.reset_map).pack(fill="x", padx=8, pady=3)
        ttk.Button(act, text="💾  Export map to PLY...",
                   command=self.export_ply).pack(fill="x", padx=8, pady=3)

        # --- Stats ---------------------------------------------------
        stats = ttk.LabelFrame(right, text="Live Stats")
        stats.pack(fill="both", expand=True, pady=(6, 0))
        self.lbl_status = ttk.Label(stats, text="Depth engine: not loaded",
                                    foreground="gray")
        self.lbl_status.pack(anchor="w", padx=8, pady=(8, 2))
        self.lbl_running = ttk.Label(stats, text="State: idle",
                                     foreground="gray")
        self.lbl_running.pack(anchor="w", padx=8, pady=2)
        self.lbl_frames = ttk.Label(stats, text="Frames processed: 0")
        self.lbl_frames.pack(anchor="w", padx=8, pady=2)
        self.lbl_hz = ttk.Label(stats, text="Actual Hz: 0.00")
        self.lbl_hz.pack(anchor="w", padx=8, pady=2)
        self.lbl_points = ttk.Label(stats, text="Map points: 0")
        self.lbl_points.pack(anchor="w", padx=8, pady=2)

        tip = tk.Label(
            stats, justify="left",
            text=("Tip: walk the camera slowly and keep some overlap between "
                  "views. Enable ICP to let Open3D align each new cloud onto "
                  "the running map (requires enough texture/overlap)."),
            fg="#555", wraplength=560, padx=8, pady=6,
        )
        tip.pack(anchor="w")

    # ---------------------------------------------------------- utilities
    def _log(self, msg: str):
        try:
            self._log_main(f"[3D] {msg}")
        except Exception:
            print("[3D]", msg)

    def _update_engine_status(self):
        if not _TORCH_OK:
            self.lbl_status.configure(
                text="Depth engine: torch not installed (using pseudo-depth)",
                foreground="#b37000")
        elif self._engine.ready:
            self.lbl_status.configure(
                text=f"Depth engine: MiDaS ready ({self._engine.device})",
                foreground="green")
        else:
            self.lbl_status.configure(
                text="Depth engine: MiDaS not loaded yet (loads on Start)",
                foreground="gray")

    def _poll_stats(self):
        with self._stats_lock:
            s = dict(self._stats)
        self.lbl_frames.configure(text=f"Frames processed: {s['frames']}")
        self.lbl_hz.configure(text=f"Actual Hz: {s['hz']:.2f}")
        self.lbl_points.configure(text=f"Map points: {s['pts']}")
        running = self._capture_thread is not None and self._capture_thread.is_alive()
        self.lbl_running.configure(
            text=("State: LIVE" if running else "State: idle"),
            foreground=("green" if running else "gray"),
        )
        self.after(250, self._poll_stats)

    # ----------------------------------------------------- live pipeline
    def start_live(self):
        if not _O3D_OK:
            messagebox.showerror("Open3D missing",
                                 "Open3D is required for the live 3D viewer.\n"
                                 "Run:  pip install open3d")
            return
        if self._capture_thread and self._capture_thread.is_alive():
            return

        # Spin up the Open3D window on its own thread.
        self._viewer = _LiveO3DViewer(voxel_size=float(self.var_voxel.get()))
        self._viewer.start()

        # Reset stats
        with self._stats_lock:
            self._stats = {"frames": 0, "pts": 0, "hz": 0.0, "last": 0.0}

        # Start capture worker
        self._capture_stop.clear()
        self._capture_thread = threading.Thread(
            target=self._capture_worker, daemon=True)
        self._capture_thread.start()

        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self._log("Live mapping started. Open3D window is open.")

    def stop_live(self):
        self._capture_stop.set()
        if self._capture_thread:
            self._capture_thread.join(timeout=3.0)
        self._capture_thread = None

        if self._viewer is not None:
            self._viewer.stop()
            self._viewer = None

        self.btn_start.configure(state="normal" if _O3D_OK else "disabled")
        self.btn_stop.configure(state="disabled")
        self._log("Live mapping stopped.")

    def reset_map(self):
        if self._viewer is None:
            return
        self._viewer.reset()
        with self._stats_lock:
            self._stats["pts"] = 0
            self._stats["frames"] = 0

    def export_ply(self):
        if self._viewer is None or self._viewer.get_merged() is None:
            messagebox.showinfo("Empty", "No map to export yet. Start live "
                                "mapping first.")
            return
        pcd = self._viewer.get_merged()
        if len(pcd.points) == 0:
            messagebox.showinfo("Empty", "The map is empty.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".ply",
            filetypes=[("PLY point cloud", "*.ply")],
            initialfile=f"basler_map_{datetime.now():%Y%m%d_%H%M%S}.ply",
        )
        if not path:
            return
        try:
            o3d.io.write_point_cloud(path, pcd, write_ascii=False)
            self._log(f"Exported {len(pcd.points)} pts to {path}")
            messagebox.showinfo("Exported",
                                f"{len(pcd.points)} points\n{path}")
        except Exception as e:
            messagebox.showerror("Export failed", str(e))

    # ------------------------------------------------- capture worker
    def _capture_worker(self):
        """Runs on a background thread: grabs frames, builds point clouds,
        feeds them to the Open3D viewer."""
        # Lazy-load the depth model
        if _TORCH_OK and not self._engine.ready:
            self._engine.load(self._log)
            self.after(0, self._update_engine_status)

        prev_pcd = None            # for ICP
        last_tick = time.time()
        ema_hz = 0.0

        while not self._capture_stop.is_set():
            # Respect user rate
            target_dt = 1.0 / max(0.2, float(self.var_rate.get()))
            now = time.time()
            sleep_for = target_dt - (now - last_tick)
            if sleep_for > 0:
                time.sleep(sleep_for)
            last_tick = time.time()

            frame = self._get_frame()
            if frame is None:
                # No frame yet - keep the loop alive but don't spin
                time.sleep(0.2)
                continue

            try:
                rgb = frame[:, :, ::-1].astype(np.uint8, copy=False)

                if self._engine.ready:
                    depth = self._engine.infer(rgb)
                else:
                    gray = np.mean(rgb, axis=2)
                    depth = _fallback_depth(gray)

                pts, cols = self._backproject(rgb, depth)
                if pts.size == 0:
                    continue

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
                pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))
                voxel = max(1e-4, float(self.var_voxel.get()))
                pcd = pcd.voxel_down_sample(voxel)

                # Optional ICP alignment against the previous frame so a
                # moving camera accumulates into a consistent map.
                if (self.var_icp.get()
                        and prev_pcd is not None
                        and len(pcd.points) > 100
                        and len(prev_pcd.points) > 100):
                    try:
                        threshold = 5 * voxel
                        result = o3d.pipelines.registration.registration_icp(
                            pcd, prev_pcd, threshold,
                            np.eye(4),
                            o3d.pipelines.registration.
                            TransformationEstimationPointToPoint(),
                            o3d.pipelines.registration.
                            ICPConvergenceCriteria(max_iteration=30),
                        )
                        pcd.transform(result.transformation)
                    except Exception as e:
                        self._log(f"ICP skipped: {e}")

                prev_pcd = pcd

                if self._viewer is not None:
                    self._viewer.push_cloud(pcd)
                    merged = self._viewer.get_merged()
                    pts_count = len(merged.points) if merged is not None else 0

                    # Enforce a hard cap on point count (down-sample harder
                    # as we approach the limit).
                    cap = int(self.var_max_points.get())
                    if merged is not None and pts_count > cap:
                        # Signal viewer to voxel-down the merged cloud a bit
                        # more aggressively next cycle by nudging voxel size
                        # transiently. Cheap approach: drop half the points.
                        try:
                            merged_down = merged.voxel_down_sample(
                                voxel * 1.5
                            )
                            # Replace merged cloud atomically
                            with self._viewer._incoming_lock:
                                self._viewer._incoming_queue.append(
                                    ("reset", None))
                                self._viewer._incoming_queue.append(
                                    ("add", merged_down))
                        except Exception:
                            pass
                else:
                    pts_count = 0

                # update stats
                with self._stats_lock:
                    self._stats["frames"] += 1
                    self._stats["pts"] = pts_count
                    dt = time.time() - self._stats["last"]
                    if self._stats["last"] > 0 and dt > 0:
                        inst_hz = 1.0 / dt
                        ema_hz = 0.7 * ema_hz + 0.3 * inst_hz
                        self._stats["hz"] = ema_hz
                    self._stats["last"] = time.time()

            except Exception as e:
                self._log(f"Worker error: {e}")
                self._log(traceback.format_exc())
                time.sleep(0.5)

        self._log("Capture worker exited.")

    # ----------------------------------------------- geometry helpers
    def _backproject(self, rgb: np.ndarray, depth01: np.ndarray):
        h, w = depth01.shape
        stride = max(1, int(self.var_stride.get()))
        fx = float(self.var_fx.get())
        fy = float(self.var_fy.get())
        cx = float(self.var_cx.get())
        cy = float(self.var_cy.get())
        if cx < 0:
            cx = w / 2.0
        if cy < 0:
            cy = h / 2.0
        scale = float(self.var_scale.get())
        d_min = float(self.var_min_depth.get())
        d_max = float(self.var_max_depth.get())

        ys, xs = np.mgrid[0:h:stride, 0:w:stride]
        ds = depth01[ys, xs]
        mask = (ds >= d_min) & (ds <= d_max)

        xs = xs[mask].astype(np.float32)
        ys = ys[mask].astype(np.float32)
        ds = ds[mask].astype(np.float32)

        Z = scale / (ds + 1e-3)
        X = (xs - cx) * Z / fx
        Y = (ys - cy) * Z / fy
        pts = np.stack([X, Y, Z], axis=1)

        cols = rgb[ys.astype(int), xs.astype(int)].astype(np.float32) / 255.0
        return pts, cols

    # --------------------------------------------------------- shutdown
    def shutdown(self):
        try:
            self.stop_live()
        except Exception:
            pass
