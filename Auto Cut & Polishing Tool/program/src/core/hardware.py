# -*- coding: utf-8 -*-
"""
core.hardware
-------------
GPU telemetry, CUDA, Vulkan, NVDEC probe, and adaptive hardware auto-negotiation.

New in D-0.0.4:
  - HWProfile: singleton dataclass that captures VRAM, CPU cores, NVDEC/NVENC
    availability at startup. All modules read from this single source-of-truth.
  - nvdec_available(): probes FFmpeg for CUDA hardware decode support.
  - CPU thread limits in config.py now auto-scale to real cpu_count().
"""

from __future__ import annotations

import os
import sys
import subprocess
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from core.media_tools import command_exists, run, ffmpeg_path, ffprobe_path


# ---------------------------------------------------------------------------
# HWProfile — singleton hardware capability snapshot
# ---------------------------------------------------------------------------

@dataclass
class HWProfile:
    """
    Immutable hardware capability snapshot captured once at startup.
    All modules (cutter, stabilizer, vad, config) read from this to
    auto-tune their own parameters.
    """
    cpu_cores:       int   = 1      # logical CPU core count
    cpu_threads:     int   = 2      # recommended FFmpeg/numpy thread count
    gpu_available:   bool  = False  # CUDA GPU detected
    gpu_name:        str   = ""     # GPU display name
    vram_total_mb:   int   = 0      # total VRAM in MB
    vram_free_mb:    int   = 0      # estimated free VRAM in MB
    nvenc_ok:        bool  = False  # FFmpeg h264_nvenc works
    nvdec_ok:        bool  = False  # FFmpeg CUDA hwaccel decode works
    cuda_ok:         bool  = False  # PyTorch CUDA available
    platform:        str   = "unknown"


_HW_PROFILE: HWProfile | None = None


def get_hw_profile() -> HWProfile:
    """
    Returns the cached HWProfile singleton.  On the first call it probes the
    system (GPU, CPU, NVDEC, NVENC) and caches the result for the entire
    session — subsequent calls are O(1).
    """
    global _HW_PROFILE
    if _HW_PROFILE is not None:
        return _HW_PROFILE

    import platform as _platform

    # ── CPU
    cores = os.cpu_count() or 1
    # Use up to min(cores, 8) threads for CPU-bound work; always leave 1 core free
    cpu_threads = max(1, min(cores - 1, 8))

    # ── GPU / CUDA
    g = gpu_info()          # already defined below; safe because it has no deps
    cuda_ok = torch_cuda_available()

    # ── VRAM free estimate (total - used - 512 MB OS headroom)
    vram_free = max(0, g["vram_total"] - g["vram_used"] - 512) if g["available"] else 0

    # ── NVENC probe (reuse cached result from get_best_video_encoder_config)
    enc, _, _ = get_best_video_encoder_config()
    nvenc_ok = (enc != "libx264")

    # ── NVDEC probe (lazy — called here so result is cached in HWProfile)
    nvdec_ok = _probe_nvdec()

    _HW_PROFILE = HWProfile(
        cpu_cores=cores,
        cpu_threads=cpu_threads,
        gpu_available=g["available"],
        gpu_name=g["name"],
        vram_total_mb=g["vram_total"],
        vram_free_mb=vram_free,
        nvenc_ok=nvenc_ok,
        nvdec_ok=nvdec_ok,
        cuda_ok=cuda_ok,
        platform=_platform.system(),
    )
    return _HW_PROFILE


def _probe_nvdec() -> bool:
    """
    Probes FFmpeg for functional CUDA hardware video decode (NVDEC).
    Returns True only if a test nullsrc decode succeeds with -hwaccel cuda.
    """
    try:
        ff = ffmpeg_path()
    except Exception:
        return False
    try:
        r = subprocess.run(
            [
                ff, "-y", "-hide_banner", "-loglevel", "error",
                "-hwaccel", "cuda",
                "-f", "lavfi", "-i", "nullsrc=s=320x180:r=30:d=0.5",
                "-frames:v", "1", "-f", "null", "-",
            ],
            capture_output=True, timeout=10,
        )
        return r.returncode == 0
    except Exception:
        return False




def check_system_dependencies() -> bool:
    """
    Silent system readiness check.
    - If everything is ready: prints one short status line and returns True.
    - If anything is missing or warned: prints a detailed issues-only list and returns False (if critical).
    """
    from core.config import BIN_DIR
    import contextlib, io as _io

    issues: list[tuple[bool, str, str]] = []   # (is_critical, label, message)

    # ── Core Binaries ─────────────────────────────────────────────
    if not (command_exists("ffmpeg") or ffmpeg_path()):
        issues.append((True, "FFmpeg", "NOT FOUND — install FFmpeg and add to PATH"))
    if not (command_exists("ffprobe") or ffprobe_path()):
        issues.append((True, "FFprobe", "NOT FOUND — install FFprobe and add to PATH"))

    # ── Python Packages ───────────────────────────────────────────
    packages = [
        ("numpy",      "numpy",          True),
        ("torch",      "torch",          True),
        ("silero_vad", "silero-vad",     True),
        ("yt_dlp",     "yt-dlp",         True),
        ("clearvoice", "clearvoice",     False),
        ("voicefixer", "voicefixer",     False),
        ("cv2",        "opencv-python",  False),
    ]
    # ── Windows N Media Foundation DLL Compatibility Injection ────────
    if os.name == "nt":
        try:
            compat_dir = Path(__file__).resolve().parent / "compat_dlls"
            if compat_dir.exists():
                import sys as _sys
                cv2_target = Path(_sys.prefix) / "Lib" / "site-packages" / "cv2"
                if cv2_target.exists():
                    for stub in compat_dir.glob("*.dll"):
                        dest = cv2_target / stub.name
                        if not dest.exists():
                            import shutil as _shutil
                            _shutil.copy2(stub, dest)
        except Exception:
            pass

    for mod, pkg, critical in packages:
        try:
            __import__(mod)
        except Exception as _err:
            if critical:
                issues.append((True,  f"Package: {pkg}", "MISSING — run setup again"))
            elif mod == "cv2":
                issues.append((False, f"Package: {pkg}", "Optional (PIL engine active for thumbnails)"))
            else:
                issues.append((False, f"Package: {pkg}", "Not installed (optional / DSP fallback active)"))

    # ── AI Model Cache ────────────────────────────────────────────
    silero_ok = False
    try:
        from silero_vad import load_silero_vad
        load_silero_vad()
        silero_ok = True
    except Exception:
        pass
    if not silero_ok:
        issues.append((False, "AI Model: Silero VAD",
                        "Not cached — run setup.bat / setup.sh to pre-download"))

    clearvoice_ok = False
    try:
        from core.config import PROJECT_CHECKPOINTS
        import clearvoice
        if not getattr(clearvoice, "_antigravity_patched", False):
            _orig = clearvoice.network_wrapper.load_args_se
            def _custom(self):
                _orig(self)
                self.args.checkpoint_dir = str(PROJECT_CHECKPOINTS / self.model_name)
            clearvoice.network_wrapper.load_args_se = _custom
            clearvoice._antigravity_patched = True
        with contextlib.redirect_stdout(_io.StringIO()), contextlib.redirect_stderr(_io.StringIO()):
            clearvoice.ClearVoice(task="speech_enhancement", model_names=["MossFormer2_SE_48K"])
        clearvoice_ok = True
    except Exception:
        pass
    if not clearvoice_ok:
        issues.append((False, "AI Model: ClearVoice MossFormer2",
                        "Not cached (optional) — run setup to pre-download"))

    voicefixer_ok = False
    try:
        from voicefixer import VoiceFixer
        VoiceFixer()
        voicefixer_ok = True
    except Exception:
        pass
    if not voicefixer_ok:
        issues.append((False, "AI Model: VoiceFixer",
                        "Not cached (optional) — run setup to pre-download"))


    # ── Output ────────────────────────────────────────────────────
    has_critical = any(crit for crit, _, _ in issues)

    if not issues:
        print("\n[  OK  ]  System ready. Starting program...")
    else:
        print()
        print("=" * 64)
        print("  SYSTEM CHECK — ISSUES FOUND")
        print("=" * 64)
        for crit, label, msg in issues:
            tag = "[ FAIL ]" if crit else "[ WARN ]"
            print(f"  {tag}  {label:<32} {msg}")
        print("=" * 64)
        if has_critical:
            print("  Some critical components are MISSING.")
            print("  Run setup.bat (Windows) or setup.sh (Linux/Mac) to fix.")
        else:
            print("  Optional components missing — core features will still work.")
        print("=" * 64)

    return not has_critical




def gpu_info() -> dict:
    """Queries NVIDIA GPU state via nvidia-smi (or Apple Metal MPS on macOS)."""
    info = {"available": False, "name": "", "vram_total": 0, "vram_used": 0, "util": 0}
    if sys.platform == "darwin":
        try:
            import torch
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                info.update({
                    "available": True,
                    "name": "Apple Silicon GPU (Metal / MPS)",
                    "vram_total": 0,
                    "vram_used": 0,
                    "util": 0,
                })
                return info
        except Exception:
            pass
    if not command_exists("nvidia-smi"):
        return info
    try:
        r = run([
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ], capture=True)
        lines = r.stdout.strip().splitlines()
        if lines:
            line = lines[0]
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 4:
                name, total, used, util = parts[:4]
                info.update({
                    "available": True,
                    "name": name,
                    "vram_total": int(float(total)),
                    "vram_used": int(float(used)),
                    "util": int(float(util)),
                })
    except Exception:
        pass
    return info


def gpu_status_text() -> str:
    """Returns compact string representation of GPU utilization and memory."""
    g = gpu_info()
    if not g["available"]:
        return "GPU unavailable"
    return f"GPU {g['util']}% | VRAM {g['vram_used']}/{g['vram_total']}MB"


def torch_cuda_available() -> bool:
    """Checks if PyTorch with CUDA acceleration is functional."""
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def vulkan_available() -> bool:
    """Checks if Vulkan driver/tools are present for C++ ncnn engines."""
    if command_exists("vulkaninfo"):
        try:
            r = subprocess.run(["vulkaninfo", "--summary"], capture_output=True, timeout=5)
            if r.returncode == 0:
                return True
        except Exception:
            pass
    # In Colab or environments where nvidia driver is loaded, Vulkan is present via ICD
    if os.path.exists("/usr/share/vulkan/icd.d") or os.path.exists("/etc/vulkan/icd.d"):
        return True
    if os.name == "nt":
        # Windows registry or system32 vulkan-1.dll
        sys32 = Path(os.environ.get("WINDIR", "C:\\Windows")) / "System32" / "vulkan-1.dll"
        if sys32.exists():
            return True
    return False


_BEST_ENCODER_CONFIG: tuple[str, list[str], str] | None = None


def get_best_video_encoder_config() -> tuple[str, list[str], str]:
    """
    Universal hardware video encoder auto-negotiation:
      1. NVIDIA NVENC (h264_nvenc) - Cinema Quality P7 VBR
      2. Intel QuickSync (h264_qsv)
      3. AMD AMF (h264_amf)
      4. Apple VideoToolbox (h264_videotoolbox)
      5. Fallback: CPU (libx264)
    Returns: (codec_name, quality_flags, human_readable_display_name)
    """
    global _BEST_ENCODER_CONFIG
    if _BEST_ENCODER_CONFIG is not None:
        return _BEST_ENCODER_CONFIG

    try:
        ff = ffmpeg_path()
    except Exception:
        _BEST_ENCODER_CONFIG = ("libx264", ["-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p"], "CPU (libx264)")
        return _BEST_ENCODER_CONFIG

    # Warm up CUDA before probing NVENC (critical for Colab)
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            _torch.cuda.init()
    except Exception:
        pass

    _probe_env = os.environ.copy()
    if os.name != "nt":
        _nvidia_dirs = [
            "/usr/lib64-nvidia",
            "/usr/local/nvidia/lib64",
            "/usr/local/nvidia/lib",
            "/usr/local/cuda/lib64",
            "/usr/local/cuda/lib",
            "/usr/lib/x86_64-linux-gnu",
            "/usr/lib/x86_64-linux-gnu/nvidia/current",
            "/usr/lib64",
            "/usr/lib",
            "/usr/local/lib",
        ]
        try:
            import glob
            for _p in glob.glob("/usr/**/libnvidia-encode.so*", recursive=True):
                _d = os.path.dirname(_p)
                if _d not in _nvidia_dirs and os.path.isdir(_d):
                    _nvidia_dirs.insert(0, _d)
        except Exception:
            pass

        _existing = [d for d in _nvidia_dirs if os.path.isdir(d)]
        if _existing:
            _cur = _probe_env.get("LD_LIBRARY_PATH", "")
            _new_ld = ":".join(_existing) + (":" + _cur if _cur else "")
            _probe_env["LD_LIBRARY_PATH"] = _new_ld
            os.environ["LD_LIBRARY_PATH"] = _new_ld

    candidates = [
        # 1. NVIDIA NVENC High Quality (P7 VBR, CQ 19, adaptive bitrate)
        # CQ 19 with maxrate 20M/bufsize 40M gives ~10-15 Mbps for 1080p
        # This produces ~2-2.5 GB for a 20-minute 60fps video (vs 5.7 GB uncapped).
        ("h264_nvenc", [
            "-preset", "p7", "-tune", "hq", "-rc:v", "vbr", "-cq", "19", "-b:v", "0",
            "-maxrate", "20M", "-bufsize", "40M", "-spatial-aq", "1", "-temporal-aq", "1",
            "-pix_fmt", "yuv420p"
        ], "NVIDIA NVENC (GPU)", []),
        # 2. NVIDIA NVENC Medium/HQ for Pascal GTX 10-series (GTX 1050 Ti, 1060, etc.)
        ("h264_nvenc", [
            "-preset", "medium", "-tune", "hq", "-rc:v", "vbr", "-cq", "19", "-b:v", "0",
            "-maxrate", "20M", "-bufsize", "40M", "-pix_fmt", "yuv420p"
        ], "NVIDIA NVENC (GPU)", []),
        # 3. NVIDIA NVENC Universal Fallback
        ("h264_nvenc", [
            "-preset", "default", "-pix_fmt", "yuv420p"
        ], "NVIDIA NVENC (GPU)", []),
        ("h264_qsv",  ["-preset", "medium", "-global_quality", "19"], "Intel QuickSync (QSV)", []),
        ("h264_amf",  ["-quality", "quality", "-rc", "cqp", "-qp_i", "19", "-qp_p", "19", "-pix_fmt", "yuv420p"], "AMD AMF (GPU)", []),
        ("h264_videotoolbox", ["-q:v", "75", "-pix_fmt", "yuv420p"], "Apple VideoToolbox (GPU)", []),
    ]

    for enc, flags, name, pre_flags in candidates:
        try:
            cmd = ([ff, "-y", "-hide_banner", "-loglevel", "error"]
                   + pre_flags
                   + ["-f", "lavfi", "-i", "nullsrc=s=256x256:r=30:d=0.5",
                      "-c:v", enc] + flags + ["-f", "null", "-"])
            r = subprocess.run(cmd, capture_output=True, timeout=15, env=_probe_env)
            if r.returncode == 0:
                _BEST_ENCODER_CONFIG = (enc, flags, name)
                return _BEST_ENCODER_CONFIG
        except Exception:
            continue

    _BEST_ENCODER_CONFIG = ("libx264", ["-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p"], "CPU (libx264)")
    return _BEST_ENCODER_CONFIG


def nvenc_available() -> bool:
    """Returns True if any hardware-accelerated video encoder is active."""
    enc, _, _ = get_best_video_encoder_config()
    return enc != "libx264"


def nvdec_available() -> bool:
    """
    Returns True if FFmpeg CUDA hardware decode (NVDEC) is functional.
    Uses HWProfile cache so it's free after the first call.
    """
    return get_hw_profile().nvdec_ok


def print_system():
    """Prints adaptive hardware diagnostic banner."""
    hw = get_hw_profile()
    _, _, enc_name = get_best_video_encoder_config()
    print("=" * 64)
    print("AUTO CUT & POLISHING TOOL — ADAPTIVE HARDWARE ENGINE")
    print("=" * 64)
    if hw.gpu_available:
        print(f"GPU    : {hw.gpu_name}")
        print(f"VRAM   : {hw.vram_free_mb} MB free / {hw.vram_total_mb} MB total")
    else:
        print("GPU    : NOT DETECTED (CPU Fallback Mode)")
    print(f"CPU    : {hw.cpu_cores} cores  →  {hw.cpu_threads} worker threads")
    print(f"CUDA   : {'ON (Tensor Cores Ready)' if hw.cuda_ok else 'OFF'}")
    print(f"NVDEC  : {'ON (GPU decode active)' if hw.nvdec_ok else 'OFF (CPU decode)'}")
    print(f"ENCODER: {enc_name}")
    print("=" * 64)
