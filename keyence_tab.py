"""
keyence_tab.py
--------------
Tkinter tab for a Keyence SR-X100W (and compatible SR-series) barcode
reader. Mirrors the Cognex tab layout:

    [ Connection / Device Info / Log ] | [ Commands / Raw console ] | [ Results ]

Two connection modes, both work without any Keyence software installed:

  * Data Stream (port 9005, default on SR-X series) -- receive-only listener
    for decoded codes. Works out of the box.
  * Command    (port 9004) -- sends trigger / SET / GET commands to the reader
    and receives OK/ER replies. Requires the reader's Ethernet output type
    to be set to "TCP" in AutoID Navigator.
"""

from __future__ import annotations

import threading
import traceback
from datetime import datetime
from typing import Dict, Optional

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

from keyence_sr import (
    KeyenceSRReader,
    KeyenceSRDataStream,
)


class KeyenceTab(ttk.Frame):
    """Self-contained tab frame for a Keyence SR-X100W."""

    def __init__(self, parent, log_cb=None):
        super().__init__(parent)
        self.reader: Optional[KeyenceSRReader] = None
        self.stream: Optional[KeyenceSRDataStream] = None
        self._external_log = log_cb
        self._last_result = ""

        self.var_mode = tk.StringVar(value="stream")    # "stream" | "command"
        self.var_host = tk.StringVar(value="192.168.0.1")
        self.var_port_cmd = tk.IntVar(value=9004)
        self.var_port_data = tk.IntVar(value=9005)
        self.var_raw = tk.StringVar()

        self._build_ui()

    # ----------------------------------------------------------- logging
    def _log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}\n"
        self.log_txt.configure(state="normal")
        self.log_txt.insert("end", line)
        self.log_txt.see("end")
        self.log_txt.configure(state="disabled")
        if self._external_log:
            try:
                self._external_log(f"[Keyence] {msg}")
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
        messagebox.showerror("Keyence error", detail)

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        pane = ttk.PanedWindow(self, orient="horizontal")
        pane.pack(fill="both", expand=True)

        # ----- Column 1: Connection / Info / Log -----
        left = ttk.Frame(pane, width=420)
        pane.add(left, weight=0)

        net = ttk.LabelFrame(left, text="1. Connection")
        net.pack(fill="x", padx=8, pady=6)

        mode_row = ttk.Frame(net)
        mode_row.grid(row=0, column=0, columnspan=2, sticky="ew",
                      padx=4, pady=(2, 4))
        ttk.Label(mode_row, text="Mode:").pack(side="left")
        ttk.Radiobutton(mode_row, text="Data Stream (port 9005)",
                        value="stream", variable=self.var_mode,
                        command=self._on_mode_change
                        ).pack(side="left", padx=4)
        ttk.Radiobutton(mode_row, text="Command (port 9004, full control)",
                        value="command", variable=self.var_mode,
                        command=self._on_mode_change
                        ).pack(side="left", padx=4)

        for r, (lbl, var) in enumerate([
            ("Host / IP", self.var_host),
            ("Cmd port", self.var_port_cmd),
            ("Data port", self.var_port_data),
        ]):
            ttk.Label(net, text=lbl).grid(row=1 + r, column=0, sticky="e",
                                          padx=4, pady=2)
            ttk.Entry(net, textvariable=var, width=22
                      ).grid(row=1 + r, column=1, sticky="w", padx=4, pady=2)

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
        ttk.Label(
            net, foreground="#666",
            text=("Data Stream: receive-only, works out of the box.\n"
                  "Command: full control. Set Ethernet output to 'TCP' in "
                  "AutoID Navigator."),
            justify="left", wraplength=340
        ).grid(row=7, column=0, columnspan=2, sticky="w",
               padx=6, pady=(4, 2))

        info = ttk.LabelFrame(left, text="2. Device Info")
        info.pack(fill="x", padx=8, pady=6)
        self.info_txt = tk.Text(info, height=6, state="disabled",
                                wrap="word", font=("Consolas", 9))
        self.info_txt.pack(fill="both", expand=True, padx=4, pady=4)

        lf = ttk.LabelFrame(left, text="Log")
        lf.pack(fill="both", expand=True, padx=8, pady=6)
        self.log_txt = scrolledtext.ScrolledText(lf, height=10,
                                                 state="disabled",
                                                 wrap="word",
                                                 font=("Consolas", 9))
        self.log_txt.pack(fill="both", expand=True, padx=4, pady=4)

        # ----- Column 2: Commands -----
        mid = ttk.Frame(pane)
        pane.add(mid, weight=1)

        cmds = ttk.LabelFrame(mid, text="3. Commands (requires Command mode)")
        cmds.pack(fill="x", padx=8, pady=6)

        row1 = ttk.Frame(cmds)
        row1.pack(fill="x", padx=4, pady=4)
        self.btn_trig = ttk.Button(row1, text="Software Trigger (TRG)",
                                   command=self._cmd_trg, state="disabled")
        self.btn_trig.pack(side="left", padx=4)
        self.btn_lon = ttk.Button(row1, text="Start Read Mode (LON)",
                                  command=self._cmd_lon, state="disabled")
        self.btn_lon.pack(side="left", padx=4)
        self.btn_loff = ttk.Button(row1, text="Stop Read Mode (LOFF)",
                                   command=self._cmd_loff, state="disabled")
        self.btn_loff.pack(side="left", padx=4)

        row2 = ttk.Frame(cmds)
        row2.pack(fill="x", padx=4, pady=4)
        self.btn_aimer = ttk.Button(row2, text="Aimer Flash (ATRG)",
                                    command=self._cmd_aimer, state="disabled")
        self.btn_aimer.pack(side="left", padx=4)
        self.btn_info = ttk.Button(row2, text="Refresh Device Info",
                                   command=self._refresh_info,
                                   state="disabled")
        self.btn_info.pack(side="left", padx=4)

        raw = ttk.LabelFrame(mid, text="4. Raw command (advanced)")
        raw.pack(fill="x", padx=8, pady=6)
        cmd_row = ttk.Frame(raw)
        cmd_row.pack(fill="x", padx=4, pady=4)
        ttk.Entry(cmd_row, textvariable=self.var_raw
                  ).pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(cmd_row, text="Send",
                   command=self._cmd_raw).pack(side="left", padx=4)
        ttk.Label(raw, foreground="gray",
                  text=("Examples:   LON   |   LOFF   |   TRG   |   ATRG   |   "
                        "RS,MODEL   |   RS,VERSION   |   RS,PRM,1001")
                  ).pack(anchor="w", padx=6)

        # Quick reference
        ref = ttk.LabelFrame(mid, text="5. Common commands (reference)")
        ref.pack(fill="both", expand=True, padx=8, pady=6)
        ref_txt = tk.Text(ref, wrap="word", font=("Consolas", 9))
        ref_txt.pack(fill="both", expand=True, padx=4, pady=4)
        ref_txt.insert("end",
            "TRG          – Software trigger (one read)\n"
            "LON / LOFF   – Start / stop read-mode\n"
            "ATRG         – Flash aimer for ~2s\n"
            "RS,MODEL     – Get model name\n"
            "RS,VERSION   – Get firmware version\n"
            "RS,SERIAL    – Get serial number\n"
            "RS,PRM,<id>  – Get parameter <id>  (e.g. RS,PRM,1001)\n"
            "WS,PRM,<id>,<val>  – Set parameter <id> to <val>\n"
            "\n"
            "Note: the full parameter map for the SR-X100W is listed in the "
            "'Command Communications' chapter of the SR-X Series user's "
            "manual (Keyence document 96M12920 / 96M13190).\n"
        )
        ref_txt.configure(state="disabled")

        # ----- Column 3: Results -----
        right = ttk.Frame(pane)
        pane.add(right, weight=2)

        res = ttk.LabelFrame(right, text="6. Decoded results (live)")
        res.pack(fill="both", expand=True, padx=8, pady=6)

        big = ttk.Frame(res)
        big.pack(fill="x", padx=4, pady=4)
        ttk.Label(big, text="Last decoded:",
                  foreground="gray").pack(anchor="w")
        self.lbl_last = ttk.Label(big, text="—",
                                  font=("Segoe UI", 14, "bold"),
                                  foreground="#0066cc")
        self.lbl_last.pack(anchor="w", pady=(0, 4))
        self.lbl_meta = ttk.Label(big, text="", foreground="gray")
        self.lbl_meta.pack(anchor="w")

        cols = ("time", "source", "data")
        self.tree_res = ttk.Treeview(res, columns=cols, show="headings",
                                     height=12)
        for c, w in zip(cols, (100, 110, 620)):
            self.tree_res.heading(c, text=c.upper())
            self.tree_res.column(c, width=w, anchor="w")
        self.tree_res.pack(fill="both", expand=True, padx=4, pady=4)

        btns = ttk.Frame(res)
        btns.pack(fill="x", padx=4, pady=(0, 4))
        ttk.Button(btns, text="Clear log",
                   command=lambda: self.tree_res.delete(*self.tree_res.get_children())
                   ).pack(side="left", padx=4)
        ttk.Button(btns, text="Export CSV",
                   command=self._export_csv).pack(side="left", padx=4)

    # ----------------------------------------------------------- handlers
    def _on_mode_change(self):
        pass  # nothing to gray out here, both modes use the same fields

    def _on_connect(self):
        host = self.var_host.get().strip()
        if not host:
            messagebox.showwarning("Missing host", "Enter an IP address first.")
            return
        mode = self.var_mode.get()
        if mode == "command":
            self._connect_command(host)
        else:
            self._connect_stream(host)

    def _connect_command(self, host: str):
        port = self.var_port_cmd.get()
        self.reader = KeyenceSRReader(host, port=port)
        try:
            self.reader.connect()
        except Exception as e:
            self.reader = None
            self._err(f"Could not connect to {host}:{port}", e)
            return

        self.lbl_conn.configure(
            text=f"Connected to {host}  (Command / port {port})",
            foreground="green")
        self.btn_connect.configure(state="disabled")
        self.btn_disconnect.configure(state="normal")
        for b in (self.btn_trig, self.btn_lon, self.btn_loff,
                  self.btn_aimer, self.btn_info):
            b.configure(state="normal")
        self._log(f"Connected to Keyence at {host} on command port {port}.")
        self._refresh_info()

    def _connect_stream(self, host: str):
        port = self.var_port_data.get()
        self.stream = KeyenceSRDataStream(host, port=port)

        def _on_result_bg(text: str):
            self.after(0, lambda t=text: self._on_stream_result(t))

        try:
            self.stream.start(_on_result_bg)
        except Exception as e:
            self.stream = None
            self._err(f"Could not connect to {host}:{port} data port", e)
            return

        self.lbl_conn.configure(
            text=f"Connected to {host}  (Data Stream / port {port})",
            foreground="green")
        self.btn_connect.configure(state="disabled")
        self.btn_disconnect.configure(state="normal")
        self._log(
            f"Connected to Keyence at {host} on data port {port}. "
            "Trigger the reader (I/O / button / configured trigger source) "
            "and decoded codes will stream into the results panel.")
        self.info_txt.configure(state="normal")
        self.info_txt.delete("1.0", "end")
        self.info_txt.insert(
            "end",
            "Mode: Data Stream (receive-only)\n\n"
            "Decoded codes appear in the results panel whenever the\n"
            "reader decodes one. For software trigger / GET / SET,\n"
            "switch to Command mode.\n")
        self.info_txt.configure(state="disabled")

    def _on_stream_result(self, text: str):
        t = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.tree_res.insert("", 0, values=(t, "stream", text))
        self.lbl_last.configure(text=text)
        self.lbl_meta.configure(text=f"received on data port  @ {t}")
        self._last_result = text

    def _on_disconnect(self):
        if self.reader is not None:
            try:
                self.reader.disconnect()
            except Exception:
                pass
            self.reader = None
        if self.stream is not None:
            try:
                self.stream.stop()
            except Exception:
                pass
            self.stream = None
        self.lbl_conn.configure(text="Not connected", foreground="gray")
        self.btn_connect.configure(state="normal")
        self.btn_disconnect.configure(state="disabled")
        for b in (self.btn_trig, self.btn_lon, self.btn_loff,
                  self.btn_aimer, self.btn_info):
            b.configure(state="disabled")
        self._log("Disconnected.")

    # ------------------------------------------------------- commands
    def _require_cmd(self) -> bool:
        if self.reader is None or not self.reader.is_connected:
            self._log("This action requires Command mode. Switch the Mode "
                      "radio button and connect.")
            return False
        return True

    def _cmd_trg(self):
        if not self._require_cmd():
            return
        ok, msg = self.reader.software_trigger()
        self._log(f"TRG -> {'OK' if ok else 'FAIL'}  ({msg!r})")

    def _cmd_lon(self):
        if not self._require_cmd():
            return
        ok, msg = self.reader.trigger()
        self._log(f"LON -> {'OK' if ok else 'FAIL'}  ({msg!r})")

    def _cmd_loff(self):
        if not self._require_cmd():
            return
        ok, msg = self.reader.stop_read_mode()
        self._log(f"LOFF -> {'OK' if ok else 'FAIL'}  ({msg!r})")

    def _cmd_aimer(self):
        if not self._require_cmd():
            return
        ok, msg = self.reader.aimer_pulse()
        self._log(f"ATRG -> {'OK' if ok else 'FAIL'}  ({msg!r})")

    def _refresh_info(self):
        if not self._require_cmd():
            return
        try:
            info = self.reader.get_device_info()
        except Exception as e:
            info = {"error": str(e)}
        self.info_txt.configure(state="normal")
        self.info_txt.delete("1.0", "end")
        for k, v in info.items():
            self.info_txt.insert("end", f"{k:>10}: {v}\n")
        self.info_txt.configure(state="disabled")
        self._log("Device info refreshed.")

    def _cmd_raw(self):
        if not self._require_cmd():
            return
        cmd = self.var_raw.get().strip()
        if not cmd:
            return
        try:
            raw = self.reader.send(cmd)
        except Exception as e:
            self._err("send failed", e)
            return
        self._log(f">>> {cmd}")
        for line in raw.splitlines():
            if line.strip():
                self._log(f"    {line.strip()}")

    # ---------------------------------------------------- CSV export
    def _export_csv(self):
        items = self.tree_res.get_children()
        if not items:
            messagebox.showinfo("Nothing to export", "No decoded results yet.")
            return
        fn = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialfile=f"keyence_{datetime.now():%Y%m%d_%H%M%S}.csv",
            filetypes=[("CSV", "*.csv")])
        if not fn:
            return
        try:
            import csv
            with open(fn, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(("time", "source", "data"))
                for iid in items:
                    w.writerow(self.tree_res.item(iid)["values"])
            self._log(f"Exported {len(items)} rows to {fn}")
        except Exception as e:
            self._err("CSV export failed", e)

    # ---------------------------------------------------- shutdown
    def shutdown(self):
        try:
            if self.reader and self.reader.is_connected:
                self.reader.disconnect()
        except Exception:
            pass
        try:
            if self.stream and self.stream.is_connected:
                self.stream.stop()
        except Exception:
            pass
