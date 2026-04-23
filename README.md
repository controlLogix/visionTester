# visionTester

A Tkinter-based machine-vision utility for Windows that unifies four common
factory devices behind a single UI:

| Tab | Device family | Protocol |
|---|---|---|
| **Basler** | Basler GigE / USB3 cameras (pypylon) | GenICam |
| **Cognex DataMan** | DM262 and compatible barcode readers | DMCC + Data Channel (TCP) |
| **Keyence SR-X** | SR-X100W / SR-X300 / SR-1000 / SR-2000 readers | ASCII command + data stream (TCP) |
| **3D Mapping** | Depth estimation on Basler frames | MiDaS (PyTorch) → point cloud |

![Main window](docs/screenshot.png)

---

## Features

### Basler tab
- Discover all Basler cameras on every NIC (even unreachable ones)
- Force IP / write persistent IP / reboot camera
- Live preview with Fit / Fill / 1:1 scaling, resizable pane
- Exposure, Gain, Pixel Format, ROI, and Trigger mode dashboard
- **GenICam Command Console** — GET / SET / EXEC any node with a preset-button bar and color-coded response log
- **AI/ML overlay** — YOLOv8 object detection (80 COCO classes), click-to-select
- Save last frame to PNG / JPG / TIFF

### Cognex DataMan tab
- Network scan for DataMan readers
- Two connection modes:
  - **Data Channel** (port 44444) — receive-only, works out of the box
  - **DMCC** (port 23) — full control (requires Telnet enabled on the reader)
- Trigger modes, exposure, gain, aimer, lighting
- Symbology enable/disable grid
- Raw DMCC console
- Live results table with CSV export
- Optional last-image pull

### Keyence SR-X tab
- Two connection modes:
  - **Data Stream** (port 9005) — receive-only
  - **Command** (port 9004) — full control (requires Ethernet output set to TCP in AutoID Navigator)
- Software trigger, LON / LOFF / ATRG, refresh device info
- Raw ASCII command console with cheat-sheet
- Live results table with CSV export

### 3D Mapping tab
- Uses the current Basler frame as input (no second camera required)
- Monocular depth estimation via **Intel MiDaS** (auto-downloads model on first use)
- Builds an RGB-colored point cloud and renders it live with matplotlib 3D
- Interactive orbit / zoom / save to `.ply`

---

## Install

**Requirements:** Windows 10/11, 64-bit Python 3.10+, Basler pylon Camera Software Suite (only needed for the Basler tab — install from <https://www.baslerweb.com/en/software/pylon/>).

```powershell
git clone git@github.com:controlLogix/visionTester.git
cd visionTester\basler_tool
python -m pip install -r requirements.txt
```

`requirements.txt` pulls in pypylon, opencv-python, Pillow, numpy, and (for the AI tab) ultralytics / torch / torchvision.

> The Cognex and Keyence tabs are pure-Python TCP clients — **no vendor SDK installed required** for them.

---

## Run

```powershell
python basler_gui.py
```

The main window has four tabs along the top. Each tab is self-contained; disconnecting or crashing one doesn't affect the others.

---

## Typical workflows

### Basler
1. **Discover Cameras** → pick yours in the list
2. If shown *Unreachable*, fix the IP via **Force IP** → **Write Persistent IP** → **Reboot Camera**
3. **Connect** → **Start Live**
4. Use the **Command Console** for anything the dashboard doesn't cover, e.g.:
   - `GET DeviceTemperature`
   - `SET ExposureTime = 15000`
   - `EXEC AcquisitionStart`

### Cognex DataMan (DM262)
1. Pick **Data Channel** mode (works out of the box)
2. Enter the reader's IP → **Connect**
3. Trigger the reader — decoded codes appear in the results panel

### Keyence SR-X100W
1. Pick **Data Stream** mode
2. Enter the reader's IP → **Connect**
3. Trigger the reader (hardware, button, or self-trigger) — decoded codes stream in

---

## Project layout

```
basler_tool/
├── basler_gui.py         # main app (notebook + Basler tab)
├── detector.py           # YOLOv8 wrapper
├── cognex_dataman.py     # DataMan TCP clients
├── cognex_tab.py         # Cognex tab UI
├── keyence_sr.py         # Keyence SR TCP clients
├── keyence_tab.py        # Keyence tab UI
├── mapping3d_tab.py      # MiDaS depth + matplotlib 3D tab
├── requirements.txt
└── README.md
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Basler: "No devices found" | Install pylon runtime; check cable / PoE / 24 V; temporarily disable Windows Firewall on the camera NIC |
| Basler: "incompletely grabbed" | App auto-tunes GigE settings on connect; for the nuclear fix, enable **Jumbo Frames (9014)** on the NIC |
| DataMan: `ConnectionRefusedError 10061` | Telnet is disabled on the reader, or Setup Tool is already connected. Use Data Channel mode, or enable Telnet in Setup Tool |
| Keyence: `ConnectionRefusedError` | Reader Ethernet output is not set to TCP. Open AutoID Navigator once and set output type = TCP, command port = 9004 |
| 3D Mapping: "torch not installed" | `pip install torch torchvision` (the Basler tab still works without it) |
