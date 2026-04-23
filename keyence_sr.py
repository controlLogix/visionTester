"""
keyence_sr.py
-------------
Pure-Python client for Keyence SR-X100W / SR-X300 / SR-1000 / SR-2000
barcode readers over Ethernet.

Protocol: Keyence "Command & Result" ASCII protocol (sometimes called
          AutoID protocol).

Two TCP ports are used:
  * Command port (default 9004) -- we SEND commands, reader replies OK/ER.
  * Data  port   (default 9005) -- reader PUSHES decoded results to us.

Command framing: commands are terminated with \r (CR). Responses are
terminated with \r\n. The reader's replies are short:
    OK            -- command accepted
    ER,<code>     -- command error (<code> is a Keyence error number)
    <value>       -- for GET-style commands, value on its own line

No authentication is required by default.
Reference: "SR-X Series User's Manual -- Command Communications" chapter
           (Keyence document 96M12920 / 96M13190, freely available from
           www.keyencemanuals.com).

Examples of the short subset we use:
    LON           -- turn ON read mode (start triggering on external signal)
    LOFF          -- turn OFF read mode
    TRG           -- software trigger (fire one read)
    ATRG          -- aimer ON for ~2s
    RS,MODEL      -- GET model name  (replies e.g. "SR-X100W")
    RS,VERSION    -- GET firmware version
    RS,PRM,...    -- GET a numbered parameter
    WS,PRM,...    -- SET a numbered parameter
"""

from __future__ import annotations

import socket
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Command client (port 9004)
# ---------------------------------------------------------------------------
class KeyenceSRReader:
    """Sends command-port commands and reads OK/ER replies."""

    DEFAULT_CMD_PORT = 9004

    def __init__(self, host: str, port: int = DEFAULT_CMD_PORT,
                 timeout: float = 2.5):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._sock: Optional[socket.socket] = None
        self._lock = threading.Lock()
        self._connected = False

    # ------------------------------------------------------------- state
    @property
    def is_connected(self) -> bool:
        return self._connected and self._sock is not None

    # -------------------------------------------------------- connection
    def connect(self) -> None:
        # Clean up any stale socket
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
                "Possible causes:\n"
                " • Ethernet output type is not set to 'TCP' on the reader\n"
                " • Command port number on reader differs from "
                f"{self.port} (check parameter 'Ethernet > Command port')\n"
                " • A previous connection still holds the port -- "
                "power-cycle the reader."
            ) from e
        except (socket.timeout, TimeoutError) as e:
            raise RuntimeError(
                f"Timed out connecting to {self.host}:{self.port}. "
                "Check IP, cabling, and that the reader is on this subnet."
            ) from e
        except OSError as e:
            raise RuntimeError(
                f"Network error connecting to {self.host}:{self.port}: {e}"
            ) from e

        self._sock.settimeout(self.timeout)

        # Sanity check: ask for the model name.
        try:
            model = self.send("RS,MODEL", read_timeout=1.5).strip()
        except Exception:
            model = ""
        if not model:
            # Try a generic echo/ping
            try:
                model = self.send("RS,VERSION", read_timeout=1.5).strip()
            except Exception:
                pass
        if not model:
            self.disconnect()
            raise RuntimeError(
                "Reader did not respond to 'RS,MODEL'. Make sure the reader's "
                "Ethernet output is configured for TCP Command mode and that "
                "you're connecting to the command port (default 9004)."
            )

        self._connected = True

    def disconnect(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
        self._sock = None
        self._connected = False

    # ------------------------------------------------------- raw transport
    def send(self, cmd: str, read_timeout: float = 2.0) -> str:
        """Send a command, return the first non-empty reply line.
        The raw socket reply is logged via the return value so callers can
        log it verbatim."""
        with self._lock:
            if self._sock is None:
                raise RuntimeError("Not connected")
            # Commands are terminated with CR (some firmwares also accept
            # CRLF). Stripping any existing terminator first.
            payload = cmd.rstrip("\r\n") + "\r"
            self._sock.sendall(payload.encode("ascii", errors="ignore"))

            deadline = time.time() + read_timeout
            out = bytearray()
            last = time.time()
            IDLE = 0.25
            self._sock.settimeout(0.15)
            while time.time() < deadline:
                try:
                    chunk = self._sock.recv(4096)
                    if not chunk:
                        break
                    out.extend(chunk)
                    last = time.time()
                    text = out.decode("latin-1", errors="ignore")
                    # OK / ER replies end with \r\n -- stop once we see one
                    if any(line.strip() in ("OK",) or
                           line.strip().startswith("ER")
                           for line in text.splitlines()):
                        return text
                except (socket.timeout, OSError):
                    if out and (time.time() - last) >= IDLE:
                        break
                    continue
            return out.decode("latin-1", errors="ignore")

    # ------------------------------------------------------- helpers
    @staticmethod
    def parse_reply(raw: str) -> Tuple[bool, str]:
        """Return (ok, value). ok=False if an ER line is present.
        value is the first non-empty line that isn't OK / ER / the echo."""
        if not raw:
            return False, "(empty)"
        lines = [ln.strip() for ln in raw.splitlines()]
        # Error?
        for ln in lines:
            if ln.startswith("ER"):
                return False, ln
        # Data line?
        for ln in lines:
            if ln and ln != "OK":
                return True, ln
        # Just "OK" with no data -> success, empty value
        if any(ln == "OK" for ln in lines):
            return True, ""
        return True, raw.strip()

    # ------------------------------------------------- convenience ops
    def trigger(self) -> Tuple[bool, str]:
        return self.parse_reply(self.send("LON"))

    def stop_read_mode(self) -> Tuple[bool, str]:
        return self.parse_reply(self.send("LOFF"))

    def software_trigger(self) -> Tuple[bool, str]:
        return self.parse_reply(self.send("TRG"))

    def aimer_pulse(self) -> Tuple[bool, str]:
        """Flash the aimer beam for ~2 seconds."""
        return self.parse_reply(self.send("ATRG"))

    def get_model(self) -> str:
        ok, val = self.parse_reply(self.send("RS,MODEL"))
        return val if ok else ""

    def get_version(self) -> str:
        ok, val = self.parse_reply(self.send("RS,VERSION"))
        return val if ok else ""

    def get_serial(self) -> str:
        ok, val = self.parse_reply(self.send("RS,SERIAL"))
        return val if ok else ""

    def get_device_info(self) -> Dict[str, str]:
        info: Dict[str, str] = {}
        for key, cmd in [
            ("model", "RS,MODEL"),
            ("version", "RS,VERSION"),
            ("serial", "RS,SERIAL"),
        ]:
            try:
                ok, val = self.parse_reply(self.send(cmd, read_timeout=1.5))
                info[key] = val if ok else "(n/a)"
            except Exception as e:
                info[key] = f"(err: {e})"
        return info


# ---------------------------------------------------------------------------
# Data port client (port 9005) -- receive-only stream
# ---------------------------------------------------------------------------
class KeyenceSRDataStream:
    """Receive-only listener for the Keyence data port (default 9005).
    Each decoded read arrives on this socket as a text line terminated by
    \r\n (or the suffix configured in the reader)."""

    DEFAULT_DATA_PORT = 9005

    def __init__(self, host: str, port: int = DEFAULT_DATA_PORT,
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
                "Set the reader's Ethernet output to 'Data Port' or make "
                "sure the data port is enabled in the reader's config."
            ) from e
        except (socket.timeout, TimeoutError) as e:
            raise RuntimeError(
                f"Timed out connecting to {self.host}:{self.port}."
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
        STX, ETX = 0x02, 0x03
        while not self._stop.is_set() and self._sock is not None:
            try:
                chunk = self._sock.recv(4096)
            except socket.timeout:
                continue
            except (OSError, ConnectionResetError):
                break
            if not chunk:
                break
            buf.extend(chunk)
            while True:
                term_idx = -1
                term_bytes = b""
                for b in (b"\r\n", b"\n", b"\r", bytes([ETX])):
                    i = buf.find(b)
                    if i >= 0 and (term_idx < 0 or i < term_idx):
                        term_idx = i
                        term_bytes = b
                if term_idx < 0:
                    break
                line = bytes(buf[:term_idx])
                del buf[: term_idx + len(term_bytes)]
                if line and line[0] == STX:
                    line = line[1:]
                text = line.decode("latin-1", errors="replace").strip()
                if text and self._on_result is not None:
                    try:
                        self._on_result(text)
                    except Exception:
                        pass
        self._connected = False
