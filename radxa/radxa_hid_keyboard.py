"""Configure a Radxa Zero USB HID keyboard and type a short test string.

Run on the Radxa Zero as root. Connect its USB 2.0 OTG port to a USB host.
The script uses only Python's standard library.
"""

from __future__ import annotations

import argparse
import errno
import os
from pathlib import Path
import secrets
import string
import subprocess
import time
import unicodedata


GADGET = Path("/sys/kernel/config/usb_gadget/radxa_keyboard")
REPORT_DESC = bytes.fromhex(
    "05 01 09 06 a1 01 05 07 19 e0 29 e7 15 00 25 01 "
    "75 01 95 08 81 02 95 01 75 08 81 03 95 05 75 01 "
    "05 08 19 01 29 05 91 02 95 01 75 03 91 03 95 06 "
    "75 08 15 00 25 65 05 07 19 00 29 65 81 00 c0"
)
KEY_RELEASE = bytes(8)
KEY_DELAY_SECONDS = 0.01
USB_RETRY_SECONDS = 2.0
USB_DISCONNECT_ERRNOS = {
    errno.ENOENT, errno.ENODEV, errno.ENOTCONN, errno.ENXIO,
    errno.EPIPE, errno.ETIMEDOUT, errno.ESHUTDOWN,
}

# USB HID key codes for a host using the US keyboard layout.
ASCII_KEYS = {
    char: (0, code) for code, char in enumerate(string.ascii_lowercase, start=0x04)
}
ASCII_KEYS.update({
    char: (0x02, code) for code, char in enumerate(string.ascii_uppercase, start=0x04)
})
ASCII_KEYS.update({
    char: (0, code) for code, char in enumerate("1234567890", start=0x1E)
})
ASCII_KEYS.update({
    shifted: (0x02, code)
    for code, shifted in enumerate("!@#$%^&*()", start=0x1E)
})
ASCII_KEYS.update({"\n": (0, 0x28), "\r": (0, 0x28), "\t": (0, 0x2B), " ": (0, 0x2C)})
for plain, shifted, code in (
    ("-", "_", 0x2D), ("=", "+", 0x2E), ("[", "{", 0x2F),
    ("]", "}", 0x30), ("\\", "|", 0x31), (";", ":", 0x33),
    ("'", '"', 0x34), ("`", "~", 0x35), (",", "<", 0x36),
    (".", ">", 0x37), ("/", "?", 0x38),
):
    ASCII_KEYS[plain] = (0, code)
    ASCII_KEYS[shifted] = (0x02, code)

UNICODE_REPLACEMENTS = {
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-", "\u2026": "...",
}


def keyboard_text(character: str) -> str:
    """Return keys that a US-layout HID keyboard can type for one character."""
    if character in ASCII_KEYS:
        return character
    if character in UNICODE_REPLACEMENTS:
        return UNICODE_REPLACEMENTS[character]
    normalized = unicodedata.normalize("NFKD", character)
    translated = "".join(char for char in normalized if char in ASCII_KEYS)
    return translated or "?"


class HidKeyboardWriter:
    """Send generated characters to the available HID keyboard device."""

    def __init__(self, device: Path | None = None, write_timeout_seconds: float = 5.0) -> None:
        self.device = device
        self.write_timeout_seconds = write_timeout_seconds
        self.fd: int | None = None

    def __enter__(self) -> "HidKeyboardWriter":
        if self.device is None:
            devices = sorted(
                path for path in Path("/dev").glob("hidg*")
                if path.name[4:].isdigit()
            )
            if not devices:
                raise FileNotFoundError("No /dev/hidg* device is available")
            if len(devices) != 1:
                raise RuntimeError(
                    f"Expected one HID device in /dev, found: {[str(path) for path in devices]}"
                )
            self.device = devices[0]
        self.fd = os.open(self.device, os.O_WRONLY | os.O_NONBLOCK)
        return self

    def __exit__(self, *_exc: object) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def _write_report(self, report: bytes) -> None:
        if self.fd is None:
            raise RuntimeError("HID keyboard is not open")
        deadline = time.monotonic() + self.write_timeout_seconds
        while True:
            try:
                if os.write(self.fd, report) != len(report):
                    raise OSError("Incomplete HID keyboard report")
                return
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("HID host is not ready; check the OTG cable and host")
                time.sleep(KEY_DELAY_SECONDS)

    def send_character(self, character: str) -> None:
        for ascii_char in keyboard_text(character):
            modifier, keycode = ASCII_KEYS[ascii_char]
            self._write_report(bytes((modifier, 0, keycode, 0, 0, 0, 0, 0)))
            time.sleep(KEY_DELAY_SECONDS)
            self._write_report(KEY_RELEASE)
            time.sleep(KEY_DELAY_SECONDS)


class OptionalHidKeyboardWriter:
    """Send keys when a USB host is connected; let OLED output continue otherwise."""

    def __init__(self) -> None:
        self.writer: HidKeyboardWriter | None = None
        self.retry_at = 0.0
        self.disconnected = False

    def __enter__(self) -> "OptionalHidKeyboardWriter":
        return self

    def __exit__(self, *_exc: object) -> None:
        self._close()

    def _close(self) -> None:
        if self.writer is not None:
            self.writer.__exit__(None, None, None)
            self.writer = None

    def send_character(self, character: str) -> None:
        if time.monotonic() < self.retry_at:
            return
        try:
            if self.writer is None:
                self.writer = HidKeyboardWriter(write_timeout_seconds=0.5)
                self.writer.__enter__()
                self.writer._write_report(KEY_RELEASE)
            self.writer.send_character(character)
        except OSError as exc:
            if not isinstance(exc, TimeoutError) and exc.errno not in USB_DISCONNECT_ERRNOS:
                raise
            self._close()
            self.retry_at = time.monotonic() + USB_RETRY_SECONDS
            if not self.disconnected:
                print("USB keyboard host unavailable; continuing OLED output", flush=True)
            self.disconnected = True
            return
        if self.disconnected:
            print("USB keyboard host connected; sending new text", flush=True)
            self.disconnected = False


def put(path: Path, value: str | bytes) -> None:
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(value)


def setup() -> None:
    subprocess.run(["modprobe", "libcomposite"], check=True)
    config_root = Path("/sys/kernel/config")
    if not (config_root / "usb_gadget").exists():
        subprocess.run(["mount", "-t", "configfs", "none", str(config_root)], check=True)

    udcs = list(Path("/sys/class/udc").iterdir())
    if not udcs:
        raise RuntimeError("No USB device controller found; check the OTG peripheral mode")

    if GADGET.exists():
        bound = (GADGET / "UDC").read_text().strip()
        if bound:
            print(f"Keyboard gadget already bound to {bound}")
            return
        print(f"Resuming incomplete keyboard gadget at {GADGET}")

    for other in (config_root / "usb_gadget").iterdir():
        if (other / "UDC").read_text().strip():
            raise RuntimeError(f"USB controller is already in use by {other}; stop that gadget first")

    GADGET.mkdir(exist_ok=True)
    put(GADGET / "idVendor", "0x1d6b")  # Linux Foundation test/example ID
    put(GADGET / "idProduct", "0x0104")
    put(GADGET / "bcdUSB", "0x0200")
    put(GADGET / "bcdDevice", "0x0100")

    strings = GADGET / "strings" / "0x409"
    strings.mkdir(parents=True, exist_ok=True)
    put(strings / "serialnumber", "radxa-zero-hid-01")
    put(strings / "manufacturer", "Radxa Zero")
    put(strings / "product", "USB Keyboard Test")

    config = GADGET / "configs" / "c.1"
    config.mkdir(parents=True, exist_ok=True)
    put(config / "MaxPower", "250")  # 500 mA advertised to the USB host
    config_strings = config / "strings" / "0x409"
    config_strings.mkdir(parents=True, exist_ok=True)
    put(config_strings / "configuration", "Keyboard")

    function = GADGET / "functions" / "hid.usb0"
    # Some kernels expose the USB controller but omit the HID configfs function.
    # Loading its module explicitly helps when the function is available as a module.
    module = subprocess.run(["modprobe", "usb_f_hid"], capture_output=True, text=True)
    try:
        function.mkdir(exist_ok=True)
    except OSError as exc:
        detail = module.stderr.strip() if module.returncode else "usb_f_hid loaded"
        raise RuntimeError(
            "Cannot create the HID gadget function. The running kernel may lack "
            "CONFIG_USB_CONFIGFS_F_HID. " + detail
        ) from exc
    put(function / "protocol", "1")
    put(function / "subclass", "1")
    put(function / "report_length", "8")
    put(function / "report_desc", REPORT_DESC)
    link = config / "hid.usb0"
    if not link.exists():
        link.symlink_to(function)

    put(GADGET / "UDC", udcs[0].name)
    print(f"Keyboard gadget bound to {udcs[0].name}; HID device: /dev/hidg0")


def type_random(count: int, device: Path) -> None:
    if not 1 <= count <= 100:
        raise ValueError("count must be between 1 and 100")
    if not device.exists():
        raise RuntimeError(f"{device} does not exist; run setup first")

    text = "".join(secrets.choice(string.ascii_lowercase) for _ in range(count))
    print(f"Typing {text!r} in 3 seconds; focus an iPhone text field now", flush=True)
    time.sleep(3)
    release = bytes(8)
    with device.open("wb", buffering=0) as hid:
        for char in text:
            keycode = ord(char) - ord("a") + 4
            hid.write(bytes((0, 0, keycode, 0, 0, 0, 0, 0)))
            time.sleep(0.06)
            hid.write(release)
            time.sleep(0.06)
        hid.write(release)
    print("Done")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("setup", help="configure and bind the USB keyboard gadget")
    send = sub.add_parser("send-random", help="type random lowercase letters once")
    send.add_argument("--count", type=int, default=12)
    send.add_argument("--device", type=Path, default=Path("/dev/hidg0"))
    args = parser.parse_args()
    if os.geteuid() != 0:
        parser.error("run with sudo")
    if args.command == "setup":
        setup()
    else:
        type_random(args.count, args.device)


if __name__ == "__main__":
    main()
