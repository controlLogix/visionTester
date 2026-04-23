"""
cognex_dataman.py
-----------------
Pure-Python client for the Cognex DataMan family (DM262, DM370, DM475, etc.)
using the documented DMCC (DataMan Control Command) protocol over TCP.

Reference: Cognex DMCC Reference Manual (publicly available PDF).
Default credentials on a fresh DataMan are user "admin" with empty password.

No Cognex SDK install required -- just a network connection to the reader
on the same subnet.

Commands return strings of the form:
    "||S<STATUS> <value>\r\n"        (success, <value> may be empty)
    "||E<ERROR>\r\n"                 (error)

Typical usage:
    r = DataManReader("192.168.1.50")
    r.connect()
    print(r.get_device_info())
    r.set_trigger_mode(0)            # single
    r.trigger()
    print(r.get_last_result())       # '123456789012'
    img = r.pull_last_image()        # bytes (PNG/JPEG depending on firmware)
    r.disconnect()
"""

from __future__ import annotations

import socket
import threading
import time
import ipaddress
from typing import Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# DMCC constants (a small, curated subset -- DataMan supports ~300 commands)
# ---------------------------------------------------------------------------
TRIGGER_MODES = {
    0: "Single",
    1: "Burst",
    2: "Self",
    3: "Presentation",
    4: "Manual",
    5: "Continuous",
    # 6 is "Burst" for some firmwares; 7+ reserved
}

# DMCC symbol/symbology names (SET SYMBOL.<NAME>-ENABLE n).
# This is the practical subset for a DM262 -- more exist on higher-end models.
SYMBOLOGIES: List[str] = [
    "DATAMATRIX", "QR-CODE", "MICRO-QR-CODE", "AZTEC-CODE", "PDF417",
    "MICROPDF417", "MAXICODE", "DOTCODE",
    "CODE128", "CODE39", "CODE93", "CODABAR",
    "UPC-A", "UPC-E", "EAN-13", "EAN-8", "GS1-DATABAR",
    "INTERLEAVED-2OF5", "MSI", "POSTAL",
]


# ---------------------------------------------------------------------------
class DataManReader:
    """DMCC TCP client. Thread-safe for command serialization."""

    DEFAULT_PORT = 23          # Telnet / DMCC
    DEFAULT_USER = "admin"
    DEFAULT_PASS = ""          # factory default is empty

    def __init__(
        self,
        host: str,
        port: int = DEFAULT_PORT,
        user: str = DEFAULT_USER,
        password: str = DEFAULT_PASS,
        timeout: float = 2.5,
    ):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.timeout = timeout

        self._sock: Optional[socket.socket] = None
        self._lock = threading.Lock()
        self._connected = False

    # -------------------------------------------------------------- state
    @property
    def is_connected(self) -> bool:
        return self._connected and self._sock is not None

    # --------------------------------------------------------- connection
    def connect(self) -> None:
        """Open the TCP session and authenticate (if needed).

        DataMan behavior out of the factory:
          - When NO password is set (default), the reader just prints its model
            name as a banner (e.g. b"DM262\r\n") and waits for DMCC commands.
            No username/password prompt is shown.
          - When a password IS set, the reader prints prompts like
            "User:" and "Password:" before accepting DMCC.

        So we try DMCC first; if it doesn't work, fall back to sending creds.

        NOTE: DataMan accepts only ONE concurrent Telnet/DMCC session per
        reader. If you get ConnectionRefusedError:
          * Close any previous instance of this app
          * Close Cognex DataMan Setup Tool
          * Wait 30s or power-cycle the reader, then try again.
        """
        # Always close any stale socket before reconnecting -- a prior failed
        # connect can leave self._sock in a half-open state.
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        self._connected = False

        try:
            self._sock = socket.create_connection(
                (self.host, self.port), timeout=self.timeout
            )
        except ConnectionRefusedError as e:
            raise RuntimeError(
                f"Connection refused by {self.host}:{self.port}. "
                "The DataMan reader accepts only one Telnet/DMCC session "
                "at a time. Common fixes:\n"
                " • Close Cognex DataMan Setup Tool if it's running\n"
                " • Close any previous instance of this app\n"
                " • Wait ~30 seconds (reader may be cooling down after a "
                "dropped session) or power-cycle the reader\n"
                " • Verify Telnet/DMCC is enabled on the reader"
            ) from e
        except (socket.timeout, TimeoutError) as e:
            raise RuntimeError(
                f"Timed out connecting to {self.host}:{self.port}. "
                "Check cable, power, IP, and that the reader is on this "
                "subnet."
            ) from e
        except OSError as e:
            raise RuntimeError(
                f"Network error connecting to {self.host}:{self.port}: {e}"
            ) from e
        self._sock.settimeout(self.timeout)


        # Absorb the initial banner (e.g. "DM262\r\n") so it doesn't get
        # mistaken for a command response.
        banner = self._drain(0.5)

        def _looks_like_dmcc_reply(text: str) -> bool:
            """True if the reply looks like a DMCC response -- either classic
            checksummed (||S/||E), or no-checksum (any non-empty line that
            isn't the "User:"/"Password:" prompt or our own command echo)."""
            if not text:
                return False
            if "||S" in text or "||E" in text:
                return True
            low = text.lower()
            if "user:" in low or "password:" in low or "login" in low:
                return False
            # Any line that has real content and isn't just our command echo
            for line in text.splitlines():
                s = line.strip()
                if s and not s.startswith("||>"):
                    return True
            return False

        # Attempt 1: factory default (no auth) -- just send DMCC directly.
        resp = self.send_command("GET DEVICE.TYPE", read_timeout=1.5)
        if _looks_like_dmcc_reply(resp):
            self._connected = True
            return

        # Attempt 2: a password IS required. Feed User then Password. Some
        # firmwares prompt for both, some skip the username. We send both
        # blindly and let the reader ignore extra lines.
        try:
            self._sock.sendall((self.user + "\r\n").encode("ascii"))
            self._drain(0.4)
            self._sock.sendall((self.password + "\r\n").encode("ascii"))
            self._drain(0.5)
        except Exception:
            pass

        # Retry DMCC.
        resp2 = self.send_command("GET DEVICE.TYPE", read_timeout=2.0)
        if _looks_like_dmcc_reply(resp2):
            self._connected = True
            return


        # Still nothing usable -- give up.
        self.disconnect()
        raise RuntimeError(
            "DataMan did not respond to DMCC. Banner was "
            f"{banner!r}; first attempt got {resp!r}; "
            f"after credentials got {resp2!r}. "
            "If the reader has a password set, enter it in the Password "
            "field. If you forgot it, factory-reset the reader (hold the "
            "trigger button while power-cycling ~20s)."
        )


    def disconnect(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
        self._sock = None
        self._connected = False

    # -------------------------------------------------------- low-level IO
    def _drain(self, secs: float = 0.2) -> bytes:
        """Best-effort read of any pending bytes without blocking for long."""
        if self._sock is None:
            return b""
        old_to = self._sock.gettimeout()
        self._sock.settimeout(secs)
        out = bytearray()
        try:
            while True:
                chunk = self._sock.recv(4096)
                if not chunk:
                    break
                out.extend(chunk)
        except (socket.timeout, OSError):
            pass
        finally:
            try:
                self._sock.settimeout(old_to)
            except Exception:
                pass
        return bytes(out)

    def send_command(self, cmd: str, read_timeout: float = 2.5) -> str:
        """Send a DMCC command and return the full raw response.

        DM262 DMCC WITHOUT checksums (||> prefix, our default) responds in
        a very simple format:
            <command-echo or blank>\r\n
            <data line(s)>\r\n
            <blank>\r\n    <-- end of response

        DMCC WITH checksums responds with "||<cs><status> <data>\r\n".
        We accept either by reading until the socket goes quiet for a short
        idle period OR we see the classic "||S" / "||E" framing.
        """
        with self._lock:
            if self._sock is None:
                raise RuntimeError("Not connected")

            # DMCC prefix is "||>".  Commands are case-insensitive.
            if not cmd.startswith("||>"):
                cmd = "||>" + cmd.lstrip("|> ").strip()
            self._sock.sendall(cmd.encode("ascii") + b"\r\n")

            deadline = time.time() + read_timeout
            out = bytearray()
            last_data_time = time.time()
            IDLE_TERMINATE = 0.30   # treat 300ms of silence as "response done"

            self._sock.settimeout(0.15)
            while time.time() < deadline:
                try:
                    chunk = self._sock.recv(4096)
                    if not chunk:
                        break
                    out.extend(chunk)
                    last_data_time = time.time()
                    text = out.decode("latin-1", errors="ignore")
                    # Classic checksummed response
                    if any(s.strip().startswith(("||S", "||E"))
                           for s in text.splitlines()):
                        return text
                except (socket.timeout, OSError):
                    # If we have data and the reader has been silent for
                    # IDLE_TERMINATE seconds, consider the reply complete.
                    if out and (time.time() - last_data_time) >= IDLE_TERMINATE:
                        break
                    continue
            return out.decode("latin-1", errors="ignore")

    @staticmethod
    def parse_response(raw: str) -> Tuple[bool, str]:
        """Return (ok, value) from a DMCC response.

        Handles TWO formats:
          * Classic checksummed: starts with '||S' (OK) or '||E' (error)
          * No-checksum (our mode): raw data lines separated by CR/LF, with
            no framing prefix.
        """
        # 1) Checksummed format first
        for line in raw.splitlines():
            s = line.strip()
            if s.startswith("||S"):
                parts = s.split(" ", 1)
                return True, (parts[1] if len(parts) > 1 else "")
            if s.startswith("||E"):
                return False, s

        # 2) No-checksum: take the first non-empty line that isn't an
        #    echo of the command (lines starting with "||>").
        lines = [ln.strip() for ln in raw.splitlines()]
        data_lines = [
            ln for ln in lines
            if ln and not ln.startswith("||>") and not ln.startswith("||")
        ]
        if data_lines:
            return True, data_lines[0]

        # 3) Truly empty reply.
        return False, raw.strip() or "(empty reply)"

    # ------------------------------------------------ convenience methods
    def trigger(self) -> Tuple[bool, str]:
        """Fire one trigger (regardless of trigger mode)."""
        raw = self.send_command("TRIGGER ON")
        return self.parse_response(raw)

    def stop_continuous(self) -> Tuple[bool, str]:
        raw = self.send_command("TRIGGER OFF")
        return self.parse_response(raw)

    def get_last_result(self) -> str:
        """Return just the decoded string from the most recent read.
        Empty string if nothing decoded."""
        raw = self.send_command("GET RESULT.LAST-DECODED-DATA")
        ok, val = self.parse_response(raw)
        if ok:
            return val
        return ""

    def get_last_symbology(self) -> str:
        raw = self.send_command("GET RESULT.LAST-SYMBOL-TYPE")
        ok, val = self.parse_response(raw)
        return val if ok else ""

    def get_last_read_time(self) -> str:
        raw = self.send_command("GET RESULT.LAST-DECODE-TIME")
        ok, val = self.parse_response(raw)
        return val if ok else ""

    def set_trigger_mode(self, mode: int) -> Tuple[bool, str]:
        raw = self.send_command(f"SET TRIGGER.TYPE {int(mode)}")
        return self.parse_response(raw)

    def get_trigger_mode(self) -> int:
        raw = self.send_command("GET TRIGGER.TYPE")
        ok, val = self.parse_response(raw)
        try:
            return int(val) if ok else -1
        except Exception:
            return -1

    def set_aimer(self, on: bool) -> Tuple[bool, str]:
        raw = self.send_command(f"SET LIGHT.AIMER-ENABLED {1 if on else 0}")
        return self.parse_response(raw)

    def set_internal_light(self, intensity_0_100: int) -> Tuple[bool, str]:
        v = max(0, min(100, int(intensity_0_100)))
        raw = self.send_command(f"SET LIGHT.INTERNAL-ENABLE {1 if v > 0 else 0}")
        ok, err = self.parse_response(raw)
        if not ok:
            return ok, err
        raw2 = self.send_command(f"SET LIGHT.INTERNAL-POWER {v}")
        return self.parse_response(raw2)

    def set_exposure_us(self, microseconds: int) -> Tuple[bool, str]:
        raw = self.send_command(f"SET CAMERA.EXPOSURE {int(microseconds)}")
        return self.parse_response(raw)

    def set_gain(self, gain: int) -> Tuple[bool, str]:
        raw = self.send_command(f"SET CAMERA.GAIN {int(gain)}")
        return self.parse_response(raw)

    def set_symbology(self, name: str, enabled: bool) -> Tuple[bool, str]:
        raw = self.send_command(
            f"SET SYMBOL.{name}-ENABLE {1 if enabled else 0}"
        )
        return self.parse_response(raw)

    def get_symbology(self, name: str) -> Optional[bool]:
        raw = self.send_command(f"GET SYMBOL.{name}-ENABLE")
        ok, val = self.parse_response(raw)
        if not ok:
            return None
        return val.strip() in ("1", "true", "TRUE", "True")

    # ------------------------------------------------------- device info
    def get_device_info(self) -> Dict[str, str]:
        info: Dict[str, str] = {}
        queries = [
            ("type",     "GET DEVICE.TYPE"),
            ("name",     "GET DEVICE.NAME"),
            ("serial",   "GET DEVICE.ID"),
            ("firmware", "GET DEVICE.FIRMWARE-VER"),
            ("mac",      "GET DEVICE.MAC-ADDRESS"),
            ("ip",       "GET ETHERNET.ADDRESS"),
        ]
        for key, cmd in queries:
            try:
                raw = self.send_command(cmd, read_timeout=1.5)
                ok, val = self.parse_response(raw)
                info[key] = val if ok else f"(n/a)"
            except Exception as e:
                info[key] = f"(err: {e})"
        return info

    # ---------------------------------------------- image pull (optional)
    def pull_last_image(self) -> Optional[bytes]:
        """Pull the most-recent captured image as raw bytes.
        DMCC returns base-64-encoded image data after a GET IMAGE.LAST call.
        Returns None if unsupported or nothing to return."""
        import base64
        raw = self.send_command("GET IMAGE.LAST", read_timeout=5.0)
        ok, val = self.parse_response(raw)
        if not ok:
            return None
        # Some firmwares return the image on the line AFTER ||S...
        # Concatenate any continuation lines and b64-decode.
        payload = val
        lines = raw.splitlines()
        gathering = False
        buf = []
        for line in lines:
            if line.strip().startswith("||S"):
                gathering = True
                # Value on the same line (if any) goes first
                parts = line.split(" ", 1)
                if len(parts) > 1 and parts[1]:
                    buf.append(parts[1])
                continue
            if gathering:
                if line.strip().startswith("||E") or line.strip() == "":
                    break
                buf.append(line.strip())
        payload = "".join(buf) or payload
        if not payload:
            return None
        try:
            return base64.b64decode(payload, validate=False)
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Network discovery helpers
# ---------------------------------------------------------------------------
def _probe_one(ip: str, timeout: float, out: List[str]) -> None:
    """Quick-test: try to open DMCC port. A Cognex DataMan responds with
    either a "DM<model>" banner (factory default, no auth) or a "User:"
    prompt (password set). We also try a DMCC round-trip as a fallback."""
    try:
        with socket.create_connection((ip, DataManReader.DEFAULT_PORT),
                                      timeout=timeout) as s:
            s.settimeout(timeout)
            try:
                banner = s.recv(512)
            except Exception:
                banner = b""
            low = banner.lower().strip()

            # Strongest signals
            if (b"cognex" in low or b"dataman" in low
                    or low.startswith(b"dm")      # "DM262\r\n", "DM470\r\n"
                    or b"user" in low):
                out.append(ip)
                return

            # Fallback: send a harmless DMCC query and see if we get "||S".
            try:
                s.sendall(b"||>GET DEVICE.TYPE\r\n")
                s.settimeout(timeout)
                reply = s.recv(1024)
                if b"||S" in reply or b"||E" in reply:
                    out.append(ip)
            except Exception:
                pass
    except Exception:
        return



# ---------------------------------------------------------------------------
# Data Channel client (port 44444) -- receive-only stream of decoded results
# ---------------------------------------------------------------------------
class DataManDataChannel:
    """Connects to DataMan's "TCP/IP Data Channel" (default port 44444) which
    is enabled out of the factory. Every time the reader decodes a code it
    pushes the text over this socket (plus optional header/footer bytes you
    configure in Setup Tool, usually <STX>data<CR><LF><ETX>).

    We do NOT send anything -- this is purely a listen socket. The reader
    must already be configured to stream on this port (it is by default).

    Call .start(on_result_cb) to begin; callback gets (decoded_string,).
    Call .stop() to close the socket.
    """

    DEFAULT_PORT = 44444

    def __init__(self, host: str, port: int = DEFAULT_PORT,
                 timeout: float = 2.5):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._sock: Optional[socket.socket] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._connected = False
        self._on_result: Optional[Callable[[str], None]] = None

    @property
    def is_connected(self) -> bool:
        return self._connected and self._sock is not None

    def start(self, on_result: Callable[[str], None]) -> None:
        """Open the TCP socket and start a background reader thread that
        invokes on_result(decoded_text) for every complete code received."""
        if self.is_connected:
            return
        self._on_result = on_result
        try:
            self._sock = socket.create_connection(
                (self.host, self.port), timeout=self.timeout
            )
        except ConnectionRefusedError as e:
            raise RuntimeError(
                f"Connection refused by {self.host}:{self.port}. "
                "Confirm that 'TCP/IP Data Channel' or 'Data Channel' is "
                "enabled on the reader (usually is by default) and that the "
                "port matches. This is typically 44444 on a DataMan."
            ) from e
        except (socket.timeout, TimeoutError) as e:
            raise RuntimeError(
                f"Timed out connecting to {self.host}:{self.port}. "
                "Check IP and cabling."
            ) from e
        except OSError as e:
            raise RuntimeError(
                f"Network error connecting to {self.host}:{self.port}: {e}"
            ) from e

        self._sock.settimeout(0.5)
        self._stop.clear()
        self._connected = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._sock is not None:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                self._sock.close()
            except Exception:
                pass
        self._sock = None
        self._connected = False
        if self._thread is not None:
            self._thread.join(timeout=1.5)
        self._thread = None

    def _reader_loop(self) -> None:
        buf = bytearray()
        # Common control bytes the reader may wrap codes with.
        STX, ETX = 0x02, 0x03
        while not self._stop.is_set() and self._sock is not None:
            try:
                chunk = self._sock.recv(4096)
            except socket.timeout:
                continue
            except (OSError, ConnectionResetError):
                break
            if not chunk:
                break  # server closed
            buf.extend(chunk)

            # Split on CR/LF (configurable "Result Suffix" on the reader,
            # typically "\r\n"). Also split on ETX. If we see no terminator
            # we keep buffering.
            while True:
                term_idx = -1
                for b in (b"\r\n", b"\n", b"\r", bytes([ETX])):
                    i = buf.find(b)
                    if i >= 0 and (term_idx < 0 or i < term_idx):
                        term_idx = i
                        term_bytes = b
                if term_idx < 0:
                    break
                line = bytes(buf[:term_idx])
                del buf[: term_idx + len(term_bytes)]
                # Strip a leading STX if present.
                if line and line[0] == STX:
                    line = line[1:]
                text = line.decode("latin-1", errors="replace").strip()
                if text and self._on_result is not None:
                    try:
                        self._on_result(text)
                    except Exception:
                        pass

        self._connected = False


# ---------------------------------------------------------------------------
def discover_on_subnet(local_ip: Optional[str] = None,
                       cidr: int = 24,
                       timeout: float = 0.5,
                       workers: int = 64) -> List[str]:
    """Parallel port-23 probe across the local /24. Returns IPs whose
    Telnet banner mentions Cognex/DataMan.

    Usage:
        ips = discover_on_subnet()           # auto-detect local IP
        ips = discover_on_subnet("192.168.1.10")
    """
    if local_ip is None:
        try:
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
        except Exception:
            return []
    try:
        net = ipaddress.ip_network(f"{local_ip}/{cidr}", strict=False)
    except Exception:
        return []

    hosts = [str(h) for h in net.hosts()]
    hits: List[str] = []

    # Simple thread pool via threading (no stdlib ThreadPoolExecutor to keep
    # the import footprint small).
    from queue import Queue
    q: "Queue[str]" = Queue()
    for h in hosts:
        q.put(h)

    def worker():
        while not q.empty():
            try:
                ip = q.get_nowait()
            except Exception:
                return
            _probe_one(ip, timeout, hits)
            q.task_done()

    ts = [threading.Thread(target=worker, daemon=True) for _ in range(workers)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=timeout * 5)
    return sorted(hits, key=lambda s: list(map(int, s.split("."))))
