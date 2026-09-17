# -*- coding: utf-8 -*-
"""
core.logger
-----------
Clean terminal logging, progress bar, time & byte humanizers.
Safe across Windows cmd (cp1252/utf-8), PowerShell, Linux, and Colab.
"""

from __future__ import annotations

import sys
import os
import shutil
import threading
from datetime import datetime

# Configure UTF-8 on Windows consoles to prevent UnicodeEncodeError
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_PROGRESS_LOCK = threading.Lock()


def eprint(msg: str = ""):
    """Print to standard error."""
    try:
        print(msg, file=sys.stderr, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"), file=sys.stderr, flush=True)


def fmt_bytes(size: int | float) -> str:
    """Formats raw bytes into a clean string (KB, MB, GB)."""
    if size < 0:
        return "0 B"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


def fmt_time(seconds: float) -> str:
    """Formats seconds into mm:ss format."""
    s = int(round(seconds))
    return f"{s // 60:02d}:{s % 60:02d}"


def fmt_duration(seconds: float) -> str:
    """Formats seconds into human-readable duration (e.g. 1h 23m 45s)."""
    seconds = max(0.0, float(seconds))
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hrs > 0:
        return f"{hrs}h {mins:02d}m {secs:02d}s"
    if mins > 0:
        return f"{mins}m {secs:02d}s"
    return f"{secs}s"


def fmt_duration_clock(seconds: float) -> str:
    """Formats seconds into hh:mm:ss."""
    seconds = max(0.0, float(seconds))
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


def current_timestamp_str() -> str:
    """Returns formatted current timestamp."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_step(name: str = ""):
    """Prints clean timestamp for step transitions."""
    ts = current_timestamp_str()
    if name:
        print(f"\n[{ts}] {name}")
    else:
        print(f"\n[{ts}]")


def progress(label: str, value: float, extra: str = ""):
    """
    Clean, robust terminal progress bar.
    Dynamically sizes to current terminal width and safely avoids wrapping.
    Compatible with all standard terminals, Windows cmd, PowerShell, and Colab.
    """
    val = max(0.0, min(1.0, float(value)))
    pct = val * 100.0
    bar_len = 20
    filled = int(round(val * bar_len))

    # Detect terminal width dynamically
    try:
        cols = shutil.get_terminal_size((80, 24)).columns
    except Exception:
        cols = 80
    max_w = max(40, cols - 1)

    ext = f" | {extra}" if extra else ""

    with _PROGRESS_LOCK:
        try:
            bar = "█" * filled + " " * (bar_len - filled)
            base_part = f"{label:<20} {bar} {pct:5.1f}%"
            avail = max_w - len(base_part)
            if avail > 4 and ext:
                if len(ext) > avail:
                    ext = ext[: avail - 3] + "..."
            elif avail <= 4:
                ext = ""
            line_str = f"{base_part}{ext}".ljust(max_w)
            if len(line_str) > max_w:
                line_str = line_str[:max_w]
            sys.stdout.write(f"\r{line_str}")
        except UnicodeEncodeError:
            # Fallback for environments lacking UTF-8 support
            bar = "=" * filled + " " * (bar_len - filled)
            base_part = f"{label:<20} [{bar}] {pct:5.1f}%"
            avail = max_w - len(base_part)
            if avail > 4 and ext:
                if len(ext) > avail:
                    ext = ext[: avail - 3] + "..."
            elif avail <= 4:
                ext = ""
            line_str = f"{base_part}{ext}".ljust(max_w)
            if len(line_str) > max_w:
                line_str = line_str[:max_w]
            sys.stdout.write(f"\r{line_str}")

        if val >= 1.0:
            sys.stdout.write("\n")
        sys.stdout.flush()
