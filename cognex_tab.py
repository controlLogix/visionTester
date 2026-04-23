"""
cognex_tab.py
-------------
Tkinter tab for controlling a Cognex DataMan (DM262 and compatible) reader
via the DMCC protocol. Mirrors the layout of the Basler tab:

    [ Discovery / Network ]  |  [ Dashboard / Commands ]  |  [ Result & Image ]

Connects via cognex_dataman.DataManReader (pure Python, TCP/Telnet DMCC).
No Cognex SDK install required.
"""

from __future__ import annotations

import io
import os
import sys
import time
import queue
import threading
import traceback
from datetime import datetime
from typing import Dict, List, Optional

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

from PIL import Image, ImageTk

from cognex_dataman import (
    DataManReader,
    DataManDataChannel,
    TRIGGER_MODES,
    SYMBOLOGIES,
    discover_on_subnet,
)



class CognexTab(ttk.Frame):
    """A self-contained tab frame. Create once and .pack() into a parent."""

    POLL_MS = 400  # how often to poll for decoded results while connected

    def __init__(self, parent, log_cb=None):
        super().__init__(parent)
        self.reader: Optional[DataManReader] = None
        self.dch: Optional[DataManDataChannel] = None  # receive-only mode
        self._poll_after = None
        self._last_result: str = ""
        self._last_image_bytes: Optional[bytes] = None
        self._tk_img = None

        # Optional callback to the main app's log pane. We also keep our own.
        self._external_log = log_cb

        # Two connection modes:
        #   "dmcc"    -> full control over port 23 (requires Telnet enabled)
        #   "channel" -> receive-only stream on port 44444 (enabled by default)
        self.var_mode = tk.StringVar(value="channel")
        self.var_host = tk.StringVar(value="192.168.1.50")
        self.var_port_dmcc = tk.IntVar(value=23)
        self.var_port_dch = tk.IntVar(value=44444)
        self.var_user = tk.StringVar(value="admin")
        self.var_pass = tk.StringVar(value="")
        self.var_trigger = tk.StringVar(value="0 - Single")
        self.var_exposure = tk.IntVar(value=500)       # µs
        self.var_gain = tk.IntVar(value=0)
        self.var_light = tk.IntVar(value=50)           # internal light 0..100
        self.var_aimer = tk.BooleanVar(value=True)
        self.var_autopull = tk.BooleanVar(value=False)
        self.sym_vars: Dict[str, tk.BooleanVar] = {}

        self._build_ui()


    # -------------------------------------------------------------- logging
    def _log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}\n"
        self.log_txt.configure(state="normal")
        self.log_txt.insert("end", line)
        self.log_txt.see("end")
        self.log_txt.configure(state="disabled")
        if self._external_log:
            try:
                self._external_log(f"[Cognex] {msg}")
            except Exception:
                pass

    def _err(self, msg: str, exc: Exception | None = None):
        detail = msg
        if exc is not None:
            detail += f"\n{type(exc).__name__}: {exc}"
        self._log("ERROR: " + detail.replace("\n", " | "))
        if exc is not None:
            self.log_txt.configure(state="normal")
            self.log_txt.insert("end", traceback.format_exc() + "\n")
            self.log_txt.see("end")
            self.log_txt.configure(state="disabled")
        messagebox.showerror("Cognex error", detail)

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        pane = ttk.PanedWindow(self, orient="horizontal")
        pane.pack(fill="both", expand=True)

        # ----- Column 1: Discovery + Connection + Log -----
        left = ttk.Frame(pane, width=420)
        pane.add(left, weight=0)

        disc = ttk.LabelFrame(left, text="1. Discovery (scans local /24)")
        disc.pack(fill="x", padx=8, pady=6)
        ttk.Button(disc, text="Scan network for DataMan readers",
                   command=self._on_discover).pack(fill="x", padx=6, pady=4)
        self.lst_found = tk.Listbox(disc, height=4)
        self.lst_found.pack(fill="x", padx=6, pady=4)
        self.lst_found.bind("<<ListboxSelect>>", self._on_pick_found)

        net = ttk.LabelFrame(left, text="2. Connection")
        net.pack(fill="x", padx=8, pady=6)

        # Mode selector
        mode_row = ttk.Frame(net)
        mode_row.grid(row=0, column=0, columnspan=2, sticky="ew",
                      padx=4, pady=(2, 4))
        ttk.Label(mode_row, text="Mode:").pack(side="left")
        ttk.Radiobutton(mode_row, text="Data Channel (port 44444, no setup)",
                        value="channel", variable=self.var_mode,
                        command=self._on_mode_change).pack(side="left", padx=4)
        ttk.Radiobutton(mode_row, text="DMCC (port 23, full control)",
                        value="dmcc", variable=self.var_mode,
                        command=self._on_mode_change).pack(side="left", padx=4)

        # Host / credentials
        ttk.Label(net, text="Host / IP").grid(row=1, column=0, sticky="e",
                                              padx=4, pady=2)
        ttk.Entry(net, textvariable=self.var_host, width=22
                  ).grid(row=1, column=1, sticky="w", padx=4, pady=2)

        # DMCC-only rows (enabled/disabled by _on_mode_change)
        self.lbl_user = ttk.Label(net, text="User (DMCC)")
        self.lbl_user.grid(row=2, column=0, sticky="e", padx=4, pady=2)
        self.ent_user = ttk.Entry(net, textvariable=self.var_user, width=22)
        self.ent_user.grid(row=2, column=1, sticky="w", padx=4, pady=2)

        self.lbl_pass = ttk.Label(net, text="Password (DMCC)")
        self.lbl_pass.grid(row=3, column=0, sticky="e", padx=4, pady=2)
        self.ent_pass = ttk.Entry(net, textvariable=self.var_pass, width=22,
                                  show="*")
        self.ent_pass.grid(row=3, column=1, sticky="w", padx=4, pady=2)

        self.btn_connect = ttk.Button(net, text="Connect",
                                      command=self._on_connect)
        self.btn_connect.grid(row=4, column=0, columnspan=2, sticky="ew",
                              padx=4, pady=2)
        self.btn_disconnect = ttk.Button(net, text="Disconnect",
                                         command=self._on_disconnect,
                                         state="disabled")
        self.btn_disconnect.grid(row=5, column=0, columnspan=2, sticky="ew",
                                 padx=4, pady=2)
        self.lbl_conn = ttk.Label(net, text="Not connected", foreground="gray")
        self.lbl_conn.grid(row=6, column=0, columnspan=2, sticky="w",
                           padx=6, pady=2)
        self.lbl_mode_hint = ttk.Label(
            net, foreground="#666",
            text=("Data Channel: receive-only stream, works out of the box.\n"
                  "DMCC: full control, requires Telnet enabled on reader."),
            justify="left", wraplength=340)
        self.lbl_mode_hint.grid(row=7, column=0, columnspan=2, sticky="w",
                                 padx=6, pady=(4, 2))

        # Device info box (populated after connect)
        info = ttk.LabelFrame(left, text="3. Device Info")
        info.pack(fill="x", padx=8, pady=6)
        self.info_txt = tk.Text(info, height=7, state="disabled",
                                wrap="word", font=("Consolas", 9))
        self.info_txt.pack(fill="both", expand=True, padx=4, pady=4)

        # Log
        lf = ttk.LabelFrame(left, text="Log")
        lf.pack(fill="both", expand=True, padx=8, pady=6)
        self.log_txt = scrolledtext.ScrolledText(lf, height=8, state="disabled",
                                                 wrap="word",
                                                 font=("Consolas", 9))
        self.log_txt.pack(fill="both", expand=True, padx=4, pady=4)

        # ----- Column 2: Dashboard / Symbologies -----
        mid_outer = ttk.Frame(pane)
        pane.add(mid_outer, weight=1)

        mcanvas = tk.Canvas(mid_outer, borderwidth=0, highlightthickness=0,
                            width=520)
        mvsb = ttk.Scrollbar(mid_outer, orient="vertical", command=mcanvas.yview)
        mcanvas.configure(yscrollcommand=mvsb.set)
        mvsb.pack(side="right", fill="y")
        mcanvas.pack(side="left", fill="both", expand=True)
        mid = ttk.Frame(mcanvas)
        mwin = mcanvas.create_window((0, 0), window=mid, anchor="nw")

        def _mresize(_e=None):
            mcanvas.configure(scrollregion=mcanvas.bbox("all"))
            mcanvas.itemconfigure(mwin, width=mcanvas.winfo_width())
        mid.bind("<Configure>", _mresize)
        mcanvas.bind("<Configure>", _mresize)

        dash = ttk.LabelFrame(mid, text="4. Acquisition")
        dash.pack(fill="x", padx=8, pady=6)

        # Trigger mode
        ttk.Label(dash, text="Trigger mode").grid(row=0, column=0, sticky="e",
                                                  padx=4, pady=3)
        trig_vals = [f"{k} - {v}" for k, v in TRIGGER_MODES.items()]
        self.cmb_trig = ttk.Combobox(dash, state="readonly",
                                     values=trig_vals,
                                     textvariable=self.var_trigger, width=18)
        self.cmb_trig.grid(row=0, column=1, sticky="w", padx=4)
        ttk.Button(dash, text="Apply",
                   command=self._apply_trigger_mode).grid(row=0, column=2)

        # Exposure µs
        ttk.Label(dash, text="Exposure (µs)").grid(row=1, column=0, sticky="e",
                                                   padx=4, pady=3)
        ttk.Spinbox(dash, from_=50, to=100000, increment=50,
                    textvariable=self.var_exposure, width=10
                    ).grid(row=1, column=1, sticky="w", padx=4)
        ttk.Button(dash, text="Apply",
                   command=self._apply_exposure).grid(row=1, column=2)

        # Gain
        ttk.Label(dash, text="Gain").grid(row=2, column=0, sticky="e",
                                          padx=4, pady=3)
        ttk.Spinbox(dash, from_=0, to=255, increment=1,
                    textvariable=self.var_gain, width=10
                    ).grid(row=2, column=1, sticky="w", padx=4)
        ttk.Button(dash, text="Apply",
                   command=self._apply_gain).grid(row=2, column=2)

        # Internal light 0..100
        ttk.Label(dash, text="Light (%)").grid(row=3, column=0, sticky="e",
                                                padx=4, pady=3)
        ttk.Scale(dash, from_=0, to=100, orient="horizontal",
                  variable=self.var_light, length=160,
                  command=lambda *_: None
                  ).grid(row=3, column=1, sticky="w", padx=4)
        ttk.Button(dash, text="Apply",
                   command=self._apply_light).grid(row=3, column=2)

        # Aimer
        ttk.Checkbutton(dash, text="Aimer beam ON",
                        variable=self.var_aimer,
                        command=self._apply_aimer
                        ).grid(row=4, column=0, columnspan=2,
                               sticky="w", padx=4, pady=4)

        # Trigger / result-control buttons
        tc = ttk.LabelFrame(mid, text="5. Trigger & Commands")
        tc.pack(fill="x", padx=8, pady=6)
        self.btn_trigger = ttk.Button(tc, text="Trigger (one read)",
                                      command=self._on_trigger,
                                      state="disabled")
        self.btn_trigger.pack(side="left", padx=6, pady=4)
        self.btn_stop_trig = ttk.Button(tc, text="Stop (continuous)",
                                        command=self._on_stop_trigger,
                                        state="disabled")
        self.btn_stop_trig.pack(side="left", padx=6, pady=4)
        self.btn_pull = ttk.Button(tc, text="Pull Last Image",
                                   command=self._on_pull_image,
                                   state="disabled")
        self.btn_pull.pack(side="left", padx=6, pady=4)
        ttk.Checkbutton(tc, text="Auto-pull image after trigger",
                        variable=self.var_autopull
                        ).pack(side="left", padx=6, pady=4)

        # Raw DMCC console
        raw = ttk.LabelFrame(mid, text="6. Raw DMCC command (advanced)")
        raw.pack(fill="x", padx=8, pady=6)
        cmd_row = ttk.Frame(raw)
        cmd_row.pack(fill="x", padx=4, pady=4)
        self.var_raw = tk.StringVar()
        ttk.Entry(cmd_row, textvariable=self.var_raw
                  ).pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(cmd_row, text="Send",
                   command=self._on_send_raw).pack(side="left", padx=4)
        ttk.Label(raw, text='Examples:   GET DEVICE.TYPE   |   '
                              'SET TRIGGER.TYPE 0   |   TRIGGER ON',
                  foreground="gray").pack(anchor="w", padx=6)

        # Symbologies
        sf = ttk.LabelFrame(mid, text="7. Symbologies (enable/disable)")
        sf.pack(fill="x", padx=8, pady=6)
        for i, name in enumerate(SYMBOLOGIES):
            v = tk.BooleanVar(value=False)
            self.sym_vars[name] = v
            ttk.Checkbutton(sf, text=name, variable=v,
                            command=lambda n=name, var=v: self._apply_symbology(n, var)
                            ).grid(row=i // 4, column=i % 4, sticky="w",
                                   padx=4, pady=2)
        ttk.Button(sf, text="Read current states from reader",
                   command=self._refresh_symbologies
                   ).grid(row=(len(SYMBOLOGIES) // 4) + 1, column=0,
                          columnspan=4, sticky="ew", padx=4, pady=4)

        # ----- Column 3: Results + Image preview -----
        right = ttk.Frame(pane)
        pane.add(right, weight=3)

        res = ttk.LabelFrame(right, text="8. Decoded results (live)")
        res.pack(fill="x", padx=8, pady=6)
        big = ttk.Frame(res)
        big.pack(fill="x", padx=4, pady=4)
        ttk.Label(big, text="Last decoded:",
                  foreground="gray").pack(anchor="w")
        self.lbl_last = ttk.Label(big, text="—",
                                  font=("Segoe UI", 14, "bold"),
                                  foreground="#0066cc")
        self.lbl_last.pack(anchor="w", pady=(0, 4))
        self.lbl_sym = ttk.Label(big, text="",
                                 foreground="gray")
        self.lbl_sym.pack(anchor="w")

        cols = ("time", "symbology", "data")
        self.tree_res = ttk.Treeview(res, columns=cols, show="headings",
                                     height=8)
        for c, w in zip(cols, (100, 120, 500)):
            self.tree_res.heading(c, text=c.upper())
            self.tree_res.column(c, width=w, anchor="w")
        self.tree_res.pack(fill="both", expand=True, padx=4, pady=4)

        btn_row = ttk.Frame(res)
        btn_row.pack(fill="x", padx=4, pady=(0, 4))
        ttk.Button(btn_row, text="Clear log",
                   command=lambda: self.tree_res.delete(*self.tree_res.get_children())
                   ).pack(side="left", padx=4)
        ttk.Button(btn_row, text="Export CSV",
                   command=self._export_csv).pack(side="left", padx=4)

        img = ttk.LabelFrame(right, text="9. Last captured image")
        img.pack(fill="both", expand=True, padx=8, pady=6)
        self.img_canvas = tk.Canvas(img, bg="black", highlightthickness=0)
        self.img_canvas.pack(fill="both", expand=True, padx=4, pady=4)
        self.img_canvas.bind("<Configure>",
                             lambda _e: self._redraw_image())

    # ----------------------------------------------------------- actions
    def _on_discover(self):
        self.lst_found.delete(0, "end")
        self._log("Scanning local /24 for DataMan readers (may take ~15s)...")
        self.update_idletasks()

        def work():
            hits = discover_on_subnet()
            def done():
                if not hits:
                    self._log("No DataMan readers found. Check cable/power "
                              "and that you're on the same subnet.")
                else:
                    self._log(f"Found {len(hits)} candidate(s): {', '.join(hits)}")
                    for ip in hits:
                        self.lst_found.insert("end", ip)
            self.after(0, done)
        threading.Thread(target=work, daemon=True).start()

    def _on_pick_found(self, _evt=None):
        sel = self.lst_found.curselection()
        if sel:
            self.var_host.set(self.lst_found.get(sel[0]))

    def _on_mode_change(self):
        """Enable/disable DMCC-only widgets based on the selected mode."""
        mode = self.var_mode.get()
        dmcc_state = "normal" if mode == "dmcc" else "disabled"
        try:
            self.ent_user.configure(state=dmcc_state)
            self.ent_pass.configure(state=dmcc_state)
            self.lbl_user.configure(foreground="black" if mode == "dmcc" else "gray")
            self.lbl_pass.configure(foreground="black" if mode == "dmcc" else "gray")
        except Exception:
            pass

    def _on_connect(self):
        host = self.var_host.get().strip()
        if not host:
            messagebox.showwarning("Missing host", "Enter an IP address first.")
            return

        mode = self.var_mode.get()
        if mode == "dmcc":
            self._connect_dmcc(host)
        else:
            self._connect_channel(host)

    # ------------------------------------------------------ DMCC connect
    def _connect_dmcc(self, host: str):
        self.reader = DataManReader(host, user=self.var_user.get(),
                                    password=self.var_pass.get())
        try:
            self.reader.connect()
        except Exception as e:
            self.reader = None
            self._err(f"Could not connect to {host} via DMCC (port 23)", e)
            return

        self.lbl_conn.configure(
            text=f"Connected to {host}  (DMCC / port 23)", foreground="green")
        for b in (self.btn_trigger, self.btn_stop_trig, self.btn_pull):
            b.configure(state="normal")
        self.btn_connect.configure(state="disabled")
        self.btn_disconnect.configure(state="normal")
        self._log(f"Connected to DataMan at {host} via DMCC.")

        # Populate device info
        try:
            info = self.reader.get_device_info()
        except Exception as e:
            info = {"error": str(e)}
        self.info_txt.configure(state="normal")
        self.info_txt.delete("1.0", "end")
        for k, v in info.items():
            self.info_txt.insert("end", f"{k:>10}: {v}\n")
        self.info_txt.configure(state="disabled")

        # Read current trigger mode into the combobox
        try:
            tm = self.reader.get_trigger_mode()
            if tm in TRIGGER_MODES:
                self.var_trigger.set(f"{tm} - {TRIGGER_MODES[tm]}")
        except Exception:
            pass

        self._refresh_symbologies()
        self._start_polling()

    # --------------------------------------------------- Data Channel connect
    def _connect_channel(self, host: str):
        port = self.var_port_dch.get()
        self.dch = DataManDataChannel(host, port=port)

        # Callback from the reader thread -> marshal to Tk main thread.
        def _on_result_bg(text: str):
            self.after(0, lambda t=text: self._on_channel_result(t))

        try:
            self.dch.start(_on_result_bg)
        except Exception as e:
            self.dch = None
            self._err(f"Could not connect to {host}:{port} Data Channel", e)
            return

        self.lbl_conn.configure(
            text=f"Connected to {host}  (Data Channel / port {port})",
            foreground="green")
        self.btn_connect.configure(state="disabled")
        self.btn_disconnect.configure(state="normal")
        # DMCC-only buttons stay disabled in this mode
        for b in (self.btn_trigger, self.btn_stop_trig, self.btn_pull):
            b.configure(state="disabled")
        self._log(
            f"Connected to DataMan at {host} on Data Channel port {port}. "
            "Trigger the reader (hardware / button / its own self-trigger) "
            "and decoded results will stream into the list.")

        self.info_txt.configure(state="normal")
        self.info_txt.delete("1.0", "end")
        self.info_txt.insert(
            "end",
            "Mode: Data Channel (receive-only)\n\n"
            "Readings appear in the results panel when the reader\n"
            "decodes a code. To enable full control (trigger/SET/GET),\n"
            "switch to DMCC mode and enable Telnet on the reader.")
        self.info_txt.configure(state="disabled")

    def _on_channel_result(self, text: str):
        """Called on the Tk main thread for each code received from the DCH."""
        t = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.tree_res.insert("", 0, values=(t, "(stream)", text))
        self.lbl_last.configure(text=text)
        self.lbl_sym.configure(text=f"symbology: (stream)   @ {t}")
        self._last_result = text

    def _on_disconnect(self):
        self._stop_polling()
        if self.reader is not None:
            try:
                self.reader.disconnect()
            except Exception:
                pass
            self.reader = None
        if self.dch is not None:
            try:
                self.dch.stop()
            except Exception:
                pass
            self.dch = None
        self.lbl_conn.configure(text="Not connected", foreground="gray")
        for b in (self.btn_trigger, self.btn_stop_trig, self.btn_pull):
            b.configure(state="disabled")
        self.btn_connect.configure(state="normal")
        self.btn_disconnect.configure(state="disabled")
        self._log("Disconnected.")


    # ---------------------------------------------------------- DMCC ops
    def _check_dmcc(self) -> bool:
        """Gate DMCC commands: must be connected in DMCC mode."""
        if self.reader is None or not self.reader.is_connected:
            if self.var_mode.get() != "dmcc":
                self._log("This action requires DMCC mode. Switch the Mode "
                          "radio button at the top of the Connection panel.")
            else:
                self._log("Not connected -- click Connect first.")
            return False
        return True

    def _apply_trigger_mode(self):
        if not self._check_dmcc():
            return
        sel = self.var_trigger.get()
        try:
            mode = int(sel.split(" ")[0])
        except Exception:
            return
        ok, msg = self.reader.set_trigger_mode(mode)
        self._log(f"SET TRIGGER.TYPE {mode} -> {'OK' if ok else 'FAIL'}  "
                  f"({msg})")

    def _apply_exposure(self):
        if not self._check_dmcc():
            return
        ok, msg = self.reader.set_exposure_us(self.var_exposure.get())
        self._log(f"SET CAMERA.EXPOSURE {self.var_exposure.get()} -> "
                  f"{'OK' if ok else 'FAIL'}  ({msg})")

    def _apply_gain(self):
        if not self._check_dmcc():
            return
        ok, msg = self.reader.set_gain(self.var_gain.get())
        self._log(f"SET CAMERA.GAIN {self.var_gain.get()} -> "
                  f"{'OK' if ok else 'FAIL'}  ({msg})")

    def _apply_light(self):
        if not self._check_dmcc():
            return
        ok, msg = self.reader.set_internal_light(self.var_light.get())
        self._log(f"Light={self.var_light.get()}% -> "
                  f"{'OK' if ok else 'FAIL'}  ({msg})")

    def _apply_aimer(self):
        if not self._check_dmcc():
            return
        ok, msg = self.reader.set_aimer(self.var_aimer.get())
        self._log(f"SET LIGHT.AIMER-ENABLED {int(self.var_aimer.get())} -> "
                  f"{'OK' if ok else 'FAIL'}  ({msg})")

    def _apply_symbology(self, name: str, var: tk.BooleanVar):
        if not self._check_dmcc():
            return
        ok, msg = self.reader.set_symbology(name, var.get())
        self._log(f"SET SYMBOL.{name}-ENABLE {int(var.get())} -> "
                  f"{'OK' if ok else 'FAIL'}  ({msg})")

    def _refresh_symbologies(self):
        if not self.reader:
            return
        for name, var in self.sym_vars.items():
            try:
                state = self.reader.get_symbology(name)
                if state is not None:
                    var.set(bool(state))
            except Exception:
                pass
        self._log("Refreshed symbology states from reader.")

    def _on_trigger(self):
        if not self.reader:
            return
        ok, msg = self.reader.trigger()
        self._log(f"Trigger -> {'OK' if ok else msg}")
        if self.var_autopull.get():
            self.after(250, self._on_pull_image)

    def _on_stop_trigger(self):
        if not self.reader:
            return
        ok, msg = self.reader.stop_continuous()
        self._log(f"Stop trigger -> {'OK' if ok else msg}")

    def _on_pull_image(self):
        if not self.reader:
            return
        try:
            blob = self.reader.pull_last_image()
        except Exception as e:
            self._err("pull_last_image failed", e)
            return
        if not blob:
            self._log("No image available (is Image Transfer enabled "
                      "on the reader?).")
            return
        self._last_image_bytes = blob
        self._redraw_image()
        self._log(f"Pulled image, {len(blob)} bytes.")

    def _on_send_raw(self):
        if not self.reader:
            return
        cmd = self.var_raw.get().strip()
        if not cmd:
            return
        try:
            raw = self.reader.send_command(cmd)
        except Exception as e:
            self._err("send_command failed", e)
            return
        self._log(f">>> {cmd}")
        for line in raw.splitlines():
            if line.strip():
                self._log(f"    {line.strip()}")

    # --------------------------------------------------------- polling
    def _start_polling(self):
        self._stop_polling()
        self._poll_after = self.after(self.POLL_MS, self._poll_results)

    def _stop_polling(self):
        if self._poll_after is not None:
            try:
                self.after_cancel(self._poll_after)
            except Exception:
                pass
        self._poll_after = None

    def _poll_results(self):
        if not self.reader or not self.reader.is_connected:
            return
        try:
            data = self.reader.get_last_result()
            if data and data != self._last_result:
                self._last_result = data
                sym = self.reader.get_last_symbology()
                t = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                self.tree_res.insert("", 0, values=(t, sym, data))
                self.lbl_last.configure(text=data)
                self.lbl_sym.configure(
                    text=f"symbology: {sym}   @ {t}")
        except Exception:
            # Intermittent polling errors aren't fatal.
            pass
        self._poll_after = self.after(self.POLL_MS, self._poll_results)

    # --------------------------------------------------------- image view
    def _redraw_image(self):
        if not self._last_image_bytes:
            return
        try:
            img = Image.open(io.BytesIO(self._last_image_bytes))
        except Exception:
            self._log("Received image data but couldn't decode it "
                      "(unknown format).")
            return
        cw = max(self.img_canvas.winfo_width(), 100)
        ch = max(self.img_canvas.winfo_height(), 100)
        iw, ih = img.size
        scale = min(cw / iw, ch / ih)
        nw, nh = max(int(iw * scale), 1), max(int(ih * scale), 1)
        img = img.resize((nw, nh), Image.BILINEAR)
        self._tk_img = ImageTk.PhotoImage(img)
        self.img_canvas.delete("all")
        self.img_canvas.create_image(cw // 2, ch // 2,
                                     image=self._tk_img, anchor="center")

    # --------------------------------------------------------- csv export
    def _export_csv(self):
        items = self.tree_res.get_children()
        if not items:
            messagebox.showinfo("Nothing to export", "No decoded results yet.")
            return
        fn = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialfile=f"dataman_{datetime.now():%Y%m%d_%H%M%S}.csv",
            filetypes=[("CSV", "*.csv")])
        if not fn:
            return
        try:
            import csv
            with open(fn, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(("time", "symbology", "data"))
                for iid in items:
                    w.writerow(self.tree_res.item(iid)["values"])
            self._log(f"Exported {len(items)} rows to {fn}")
        except Exception as e:
            self._err("CSV export failed", e)

    # --------------------------------------------------------- shutdown
    def shutdown(self):
        try:
            self._stop_polling()
            if self.reader and self.reader.is_connected:
                self.reader.disconnect()
        except Exception:
            pass
        try:
            if self.dch and self.dch.is_connected:
                self.dch.stop()
        except Exception:
            pass

