# -*- coding: utf-8 -*-
"""
core.media_tools
----------------
Subprocess execution, binary resolution (FFmpeg, FFprobe, yt-dlp),
media probing, downloading, and title sanitization.
"""

from __future__ import annotations

import os
import sys
import json
import shutil
import subprocess
import threading
import time
from pathlib import Path
from urllib.parse import urlparse
import re

from core.logger import progress, fmt_time, eprint


def command_exists(name: str) -> bool:
    """Check if command exists in PATH."""
    return shutil.which(name) is not None


def run(cmd: list[str], capture: bool = False, check: bool = True, cwd: str | Path | None = None) -> subprocess.CompletedProcess:
    """Safe wrapper around subprocess.run with UTF-8 encoding."""
    return subprocess.run(
        cmd,
        text=True,
        capture_output=capture,
        check=check,
        cwd=str(cwd) if cwd else None,
        encoding="utf-8",
        errors="replace",
    )


def _find_media_binary(name: str) -> str | None:
    """Finds FFmpeg/FFprobe binary checking local tools, system PATH, and known locations."""
    exe_suffix = ".exe" if os.name == "nt" else ""
    from core.config import BASE_DIR, BIN_DIR

    local_candidates = [
        BIN_DIR / f"{name}{exe_suffix}",
        BASE_DIR / "tools" / "ffmpeg" / "bin" / f"{name}{exe_suffix}",
        BASE_DIR / "tools" / "ffmpeg" / f"{name}{exe_suffix}",
        Path("tools") / "ffmpeg" / "bin" / f"{name}{exe_suffix}",
    ]
    for c in local_candidates:
        if c.is_file():
            bin_dir = str(c.parent.resolve())
            if bin_dir not in os.environ.get("PATH", ""):
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
            return str(c.resolve())

    # System PATH
    found = shutil.which(name)
    if found:
        return found

    # Windows known directories
    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA", "")
        candidates = []
        if local_app_data:
            winget_pkgs = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
            if winget_pkgs.exists():
                for exe in winget_pkgs.glob(f"**/{name}.exe"):
                    if exe.is_file():
                        candidates.append(exe)
        candidates.extend([
            Path(f"C:/ffmpeg/bin/{name}.exe"),
            Path(f"C:/ProgramData/chocolatey/bin/{name}.exe"),
        ])
        for c in candidates:
            if c.is_file():
                bin_dir = str(c.parent.resolve())
                if bin_dir not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
                return str(c.resolve())

    return None


_FFMPEG_PATH: str | None = None
_FFPROBE_PATH: str | None = None


def ffmpeg_path() -> str:
    """Returns resolved path to ffmpeg executable."""
    global _FFMPEG_PATH
    if _FFMPEG_PATH is None:
        _FFMPEG_PATH = _find_media_binary("ffmpeg")
    if not _FFMPEG_PATH:
        raise RuntimeError("FFmpeg not found. Install FFmpeg or put ffmpeg in PATH or tools/ folder.")
    return _FFMPEG_PATH


_FILTER_COMPLEX_FLAG: str | None = None


def get_filter_complex_script_flag() -> str:
    """
    Returns the appropriate flag to pass a filter_complex script file to FFmpeg.
    - FFmpeg < 7.0 (and standard Ubuntu / Google Colab builds): uses '-filter_complex_script'
    - FFmpeg >= 7.0 / 8.0 / 9.0: uses '-/filter_complex'
    """
    global _FILTER_COMPLEX_FLAG
    if _FILTER_COMPLEX_FLAG is not None:
        return _FILTER_COMPLEX_FLAG

    ff = ffmpeg_path()
    try:
        proc = subprocess.run(
            [ff, "-filter_complex_script"],
            capture_output=True,
            text=True,
            errors="replace"
        )
        err = (proc.stderr or "").lower()
        if "filter_complex_script" in err and ("unrecognized" in err or "not found" in err):
            _FILTER_COMPLEX_FLAG = "-/filter_complex"
        else:
            _FILTER_COMPLEX_FLAG = "-filter_complex_script"
    except Exception:
        _FILTER_COMPLEX_FLAG = "-filter_complex_script"

    return _FILTER_COMPLEX_FLAG


_FPS_MODE_FLAG: list[str] | None = None


def get_fps_mode_flags() -> list[str]:
    """
    Returns the appropriate CFR sync flag for FFmpeg.
    - FFmpeg >= 5.1 / 6 / 7 / 8 / 9: uses ['-fps_mode', 'cfr']
    - FFmpeg < 5.1 (e.g. Ubuntu 20.04/22.04 LTS / Colab default): uses ['-vsync', 'cfr']
    """
    global _FPS_MODE_FLAG
    if _FPS_MODE_FLAG is not None:
        return _FPS_MODE_FLAG

    ff = ffmpeg_path()
    try:
        proc = subprocess.run(
            [ff, "-fps_mode"],
            capture_output=True,
            text=True,
            errors="replace"
        )
        err = (proc.stderr or "").lower()
        if "fps_mode" in err and ("unrecognized" in err or "not found" in err):
            _FPS_MODE_FLAG = ["-vsync", "cfr"]
        else:
            _FPS_MODE_FLAG = ["-fps_mode", "cfr"]
    except Exception:
        _FPS_MODE_FLAG = ["-vsync", "cfr"]

    return _FPS_MODE_FLAG


def ffprobe_path() -> str:
    """Returns resolved path to ffprobe executable."""
    global _FFPROBE_PATH
    if _FFPROBE_PATH is None:
        _FFPROBE_PATH = _find_media_binary("ffprobe")
    if not _FFPROBE_PATH:
        raise RuntimeError("FFprobe not found. Install FFprobe or put ffprobe in PATH or tools/ folder.")
    return _FFPROBE_PATH


def probe(path: Path | str) -> dict:
    """Uses ffprobe to extract media format, streams, and duration."""
    ffp = ffprobe_path()
    r = run([
        ffp, "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path)
    ], capture=True)
    data = json.loads(r.stdout)
    duration = float(data.get("format", {}).get("duration") or 0)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    return {"duration": duration, "video": video, "audio": audio, "raw": data}


def get_stream_dimensions(vstream: dict | None, default: tuple[int, int] = (1280, 720)) -> tuple[int, int]:
    """
    Extracts reliable even dimensions (width, height) from video stream metadata.
    Ensures dimensions are positive even integers for encoder compatibility.
    """
    if not vstream:
        return default
    try:
        w = int(vstream.get("width") or default[0])
        h = int(vstream.get("height") or default[1])
        w = max(2, (w // 2) * 2)
        h = max(2, (h // 2) * 2)
        return (w, h)
    except Exception:
        return default


def get_stream_fps(vstream: dict | None, default: float = 30.0) -> float:
    """
    Calculates reliable playback FPS from a video stream dictionary.
    Prioritizes real frame pacing (avg_frame_rate) over inflated container metadata (like 60000/1001).
    Snaps to standard broadcast rates if very close.
    """
    if not vstream:
        return default
    for key in ["avg_frame_rate", "r_frame_rate"]:
        val = str(vstream.get(key) or "")
        if "/" in val:
            try:
                num, den = val.split("/")
                f = float(num) / float(den)
                if 10.0 <= f <= 120.0:
                    for std in [23.976, 24.0, 25.0, 29.97, 30.0, 50.0, 59.94, 60.0]:
                        if abs(f - std) < 0.15:
                            return std
                    return round(f, 3)
            except Exception:
                pass
    return default


def run_ffmpeg_with_progress(cmd: list[str], total_duration: float, label: str, cwd: str | None = None):
    """Executes FFmpeg with real-time progress parsing."""
    prog_cmd = list(cmd)
    insert_idx = 1
    if "-y" in prog_cmd:
        insert_idx = prog_cmd.index("-y") + 1
    prog_cmd[insert_idx:insert_idx] = ["-progress", "pipe:1", "-nostats"]

    proc = subprocess.Popen(
        prog_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True,
        cwd=cwd,
        encoding="utf-8",
        errors="replace",
    )

    stderr_buffer: list[str] = []
    def _stderr_reader():
        try:
            if proc.stderr:
                for err_l in proc.stderr:
                    stderr_buffer.append(err_l)
                    if len(stderr_buffer) > 200:
                        stderr_buffer.pop(0)
        except Exception:
            pass

    t_err = threading.Thread(target=_stderr_reader, daemon=True)
    t_err.start()

    out_time_us = 0
    speed_str = "1.0x"
    fps_str = "0"
    last_update = 0.0

    try:
        if proc.stdout:
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip()
                    if k == "out_time_us":
                        try:
                            out_time_us = int(v)
                        except ValueError:
                            pass
                    elif k == "speed":
                        speed_str = v
                    elif k == "fps":
                        fps_str = v
                    elif k == "progress" and line == "progress=continue":
                        now = time.time()
                        if now - last_update >= 0.25:
                            cur_sec = out_time_us / 1_000_000.0
                            ratio = min(0.99, cur_sec / total_duration) if total_duration > 0 else 0.5
                            extra = f"{speed_str:>5} | {fmt_time(cur_sec)}/{fmt_time(total_duration)} ({fps_str} fps)"
                            progress(label, ratio, extra)
                            last_update = now

        proc.wait()
        t_err.join(timeout=2)
        if proc.returncode != 0:
            err = "".join(stderr_buffer).strip()
            if not err:
                code_unsigned = proc.returncode & 0xFFFFFFFF
                err = f"Process terminated unexpectedly with exit code {proc.returncode} (0x{code_unsigned:08X})"
                if code_unsigned == 0xC0000006:
                    err += " [STATUS_IN_PAGE_ERROR: Storage drive I/O read failure]"
                elif code_unsigned == 0xC0000005:
                    err += " [STATUS_ACCESS_VIOLATION: Memory crash]"
            raise RuntimeError(f"FFmpeg error: {err}")
        progress(label, 1.0, f"done | {fmt_time(total_duration)}")
    except Exception:
        proc.kill()
        raise


def is_url(s: str) -> bool:
    """Checks if the string is a valid HTTP/HTTPS URL."""
    try:
        p = urlparse(str(s).strip())
        return p.scheme in {"http", "https"}
    except Exception:
        return False


def sanitize_title(title: str, max_chars: int = 55) -> str:
    """Sanitize title for cross-platform safe filesystem naming."""
    cleaned = str(title)
    # Remove social media engagement prefixes
    engagement_pattern = r'^\s*(?:[\d.]+[KMBkmb]?\s*(?:views?|reactions?|shares?|likes?|comments?)\s*[·•|\-—/｜]?\s*)+'
    cleaned = re.sub(engagement_pattern, '', cleaned, flags=re.IGNORECASE)

    # Strip forbidden filesystem characters and unicode pipes
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f\uff5c]', '', cleaned)
    cleaned = re.sub(r'^\s*[·•|\-—_।.,\s]+', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    if not cleaned:
        return "media"
    if len(cleaned) <= max_chars:
        return cleaned

    truncated = cleaned[:max_chars]
    last_space = truncated.rfind(' ')
    if last_space > 12:
        truncated = truncated[:last_space]
    return truncated.strip().rstrip('.,-_।| ')


def download_url(url: str, save_dir: Path, suffix: str = "_Unedited_Raw", temp_dir: Path | None = None) -> Path:
    """Downloads video or audio from URL using yt-dlp with a single unified progress bar."""
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError("yt-dlp is required for URL input. Install it with: pip install yt-dlp") from exc

    save_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_dir or save_dir

    ydl_opts_info = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts_info) as ydl:
        info = ydl.extract_info(url, download=False)
        raw_title = info.get("title", "downloaded_media")

    safe_title = sanitize_title(raw_title)
    out_tmpl = str(save_dir / f"{safe_title}{suffix}.%(ext)s")

    # Unified Single-Bar Download:
    # yt-dlp downloads video + audio as separate streams internally (like IDM multi-part).
    # We accumulate bytes from ALL streams into one shared counter and show ONE clean bar.
    import threading as _th
    _lock = _th.Lock()
    _total = [0]    # total_bytes across all streams
    _done = [0]     # downloaded_bytes across all streams
    _stream_totals: dict[str, int] = {}   # per stream_id totals
    _stream_done: dict[str, int] = {}     # per stream_id downloaded

    def hook(d: dict):
        sid = d.get("tmpfilename") or d.get("filename") or "main"
        status = d.get("status")
        with _lock:
            if status == "downloading":
                this_total = int(d.get("total_bytes") or d.get("total_bytes_estimate") or 0)
                this_done = int(d.get("downloaded_bytes") or 0)
                if sid not in _stream_totals and this_total > 0:
                    _stream_totals[sid] = this_total
                    _total[0] = sum(_stream_totals.values())
                _stream_done[sid] = this_done
                combined_done = sum(_stream_done.values())
                combined_total = _total[0]
                pct = (combined_done / combined_total) if combined_total > 0 else 0.0
                spd = d.get("speed") or 0
                spd_str = f"{spd / (1024 * 1024):.1f} MB/s" if spd else ""
                from core.logger import progress as _prog
                _prog("[DOWNLOADING]", min(pct, 0.99), spd_str)
            elif status == "finished":
                pass  # Don't print 100% until final mux is done

    ydl_opts = {
        "outtmpl": out_tmpl,
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "merge_output_format": "mp4",
        "progress_hooks": [hook],
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    from core.logger import progress as _prog
    _prog("[DOWNLOADING]", 1.0, "Complete")

    for f in save_dir.glob(f"{safe_title}{suffix}.*"):
        if f.is_file() and not f.name.endswith(".part") and not f.name.endswith(".ytdl"):
            return f

    raise RuntimeError(f"Download completed but output file not found in {save_dir}")

