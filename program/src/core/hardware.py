# -*- coding: utf-8 -*-
"""
core.hardware
-------------
GPU telemetry, CUDA, Vulkan, and universal hardware video encoder auto-negotiation.
"""

from __future__ import annotations

import os
import subprocess
import shutil
from pathlib import Path

from core.media_tools import command_exists, run, ffmpeg_path, ffprobe_path
from core.config import BIN_DIR, PROJECT_CHECKPOINTS




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
        cv_ckpt = PROJECT_CHECKPOINTS / "MossFormer2_SE_48K" / "last_best_checkpoint.pt"
        if cv_ckpt.exists() and cv_ckpt.stat().st_size > 100_000:
            clearvoice_ok = True
    except Exception:
        pass
    if not clearvoice_ok:
        issues.append((False, "AI Model: ClearVoice MossFormer2",
                        "Not cached (optional) — run setup to pre-download"))

    voicefixer_ok = False
    try:
        from core.config import PROJECT_CHECKPOINTS
        vf_synth = PROJECT_CHECKPOINTS / "voicefixer" / "synthesis_module" / "44100" / "model.ckpt-1490000_trimed.pt"
        vf_analysis = PROJECT_CHECKPOINTS / "voicefixer" / "analysis_module" / "checkpoints" / "vf.ckpt"
        if (vf_synth.exists() and vf_analysis.exists() and
                vf_synth.stat().st_size > 1_000_000 and vf_analysis.stat().st_size > 1_000_000):
            voicefixer_ok = True
    except Exception:
        pass
    if not voicefixer_ok:
        issues.append((False, "AI Model: VoiceFixer",
                        "Not cached (optional) — run setup to pre-download"))

    # ── Real-BasicVSR Weights ─────────────────────────────────────
    realbasicvsr_ok = False
    try:
        from core.config import PROJECT_CHECKPOINTS
        rb_ckpt = PROJECT_CHECKPOINTS / "realbasicvsr" / "realbasicvsr_c64b20_reds.pth"
        if rb_ckpt.exists() and rb_ckpt.stat().st_size > 20_000_000:
            realbasicvsr_ok = True
    except Exception:
        pass
    if not realbasicvsr_ok:
        issues.append((False, "AI Model: Real-BasicVSR",
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
    """Queries NVIDIA GPU state via nvidia-smi with PyTorch CUDA fallback."""
    info = {"available": False, "name": "", "vram_total": 0, "vram_used": 0, "util": 0}
    if command_exists("nvidia-smi"):
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

    # Fallback to PyTorch CUDA device properties if nvidia-smi failed or returned 0
    if not info["available"] or info["vram_total"] == 0:
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                tot_mb = int(props.total_memory / (1024 * 1024))
                used_mb = int(torch.cuda.memory_allocated(0) / (1024 * 1024))
                info.update({
                    "available": True,
                    "name": props.name,
                    "vram_total": tot_mb,
                    "vram_used": used_mb,
                    "util": 0,
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
    # 1. Custom or local ICD profile in program/support/icd
    custom_icd = BIN_DIR.parent / "icd" / "nvidia_icd.json"
    if custom_icd.is_file():
        return True
    vk_env = os.environ.get("VK_ICD_FILENAMES", "")
    if vk_env and any(Path(p).is_file() for p in vk_env.split(":") if p.strip()):
        return True

    # 2. System vulkaninfo tool
    if command_exists("vulkaninfo"):
        try:
            r = subprocess.run(["vulkaninfo", "--summary"], capture_output=True, timeout=5)
            if r.returncode == 0:
                return True
        except Exception:
            pass

    # 3. System Vulkan runtime library check
    if command_exists("vulkaninfo"):
        return True

    # 4. Standard Linux Vulkan ICD directories
    if os.path.exists("/usr/share/vulkan/icd.d") or os.path.exists("/etc/vulkan/icd.d"):
        return True

    # 5. Windows registry or system32 / bundled vulkan-1.dll
    if os.name == "nt":
        sys32 = Path(os.environ.get("WINDIR", "C:\\Windows")) / "System32" / "vulkan-1.dll"
        if sys32.exists() or (BIN_DIR / "vulkan-1.dll").exists():
            return True

    return False


_BEST_ENCODER_CONFIG: tuple[str, list[str], str] | None = None
_BEST_ENCODER_DIAGNOSTIC: str = ""


def get_best_video_encoder_config() -> tuple[str, list[str], str]:
    """
    Universal hardware video encoder auto-negotiation:
      1. NVIDIA NVENC (h264_nvenc) - Cinema Quality P7 VBR
      2. NVIDIA NVENC (h264_nvenc) - Compatible Mode (Pascal/GTX 10xx)
      3. Intel QuickSync (h264_qsv)
      4. AMD AMF (h264_amf)
      5. Apple VideoToolbox (h264_videotoolbox)
      6. Fallback: CPU (libx264)
    Returns: (codec_name, quality_flags, human_readable_display_name)
    """
    global _BEST_ENCODER_CONFIG, _BEST_ENCODER_DIAGNOSTIC
    if _BEST_ENCODER_CONFIG is not None:
        return _BEST_ENCODER_CONFIG

    try:
        ff = ffmpeg_path()
    except Exception as exc:
        _BEST_ENCODER_DIAGNOSTIC = f"FFmpeg not found: {exc}"
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
        # 1. NVIDIA NVENC - High Quality VBR (Target ~8-12 Mbps for 1080p)
        ("h264_nvenc", [
            "-preset", "p6", "-tune", "hq", "-rc:v", "vbr", "-cq", "22", "-b:v", "8M",
            "-maxrate", "14M", "-bufsize", "28M", "-spatial-aq", "1", "-temporal-aq", "1",
            "-pix_fmt", "yuv420p"
        ], "NVIDIA NVENC (GPU)", []),
        # 2. NVIDIA NVENC - Broad Compatible Profile (GTX 10xx / Pascal safe)
        ("h264_nvenc", [
            "-preset", "medium", "-rc:v", "vbr", "-cq", "22", "-b:v", "8M",
            "-maxrate", "14M", "-bufsize", "28M",
            "-pix_fmt", "yuv420p"
        ], "NVIDIA NVENC (GPU)", []),
        # 3. Intel QuickSync
        ("h264_qsv",  ["-preset", "medium", "-global_quality", "22"], "Intel QuickSync (QSV)", []),
        # 4. AMD AMF
        ("h264_amf",  ["-quality", "quality", "-rc", "cqp", "-qp_i", "22", "-qp_p", "22", "-pix_fmt", "yuv420p"], "AMD AMF (GPU)", []),
        # 5. Apple VideoToolbox
        ("h264_videotoolbox", ["-q:v", "62", "-pix_fmt", "yuv420p"], "Apple VideoToolbox (GPU)", []),
    ]

    last_error = ""
    for enc, flags, name, pre_flags in candidates:
        try:
            cmd = ([ff, "-y", "-hide_banner", "-loglevel", "error"]
                   + pre_flags
                   + ["-f", "lavfi", "-i", "nullsrc=s=256x256:r=30:d=0.5",
                      "-c:v", enc] + flags + ["-f", "null", "-"])
            r = subprocess.run(cmd, capture_output=True, timeout=30, env=_probe_env, text=True)
            if r.returncode == 0:
                _BEST_ENCODER_CONFIG = (enc, flags, name)
                _BEST_ENCODER_DIAGNOSTIC = "OK"
                return _BEST_ENCODER_CONFIG
            else:
                err = (r.stderr or "").strip()
                if err and not last_error:
                    # Clean up first line of error
                    first_err_line = err.splitlines()[0] if err.splitlines() else err
                    last_error = f"{enc}: {first_err_line}"
        except Exception as probe_err:
            if not last_error:
                last_error = str(probe_err)
            continue

    _BEST_ENCODER_DIAGNOSTIC = last_error or "Hardware encoder probe returned non-zero exit code"
    _BEST_ENCODER_CONFIG = ("libx264", ["-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p"], "CPU (libx264)")
    return _BEST_ENCODER_CONFIG


def nvenc_available() -> bool:
    """Returns True if any hardware-accelerated video encoder is active."""
    enc, _, _ = get_best_video_encoder_config()
    return enc != "libx264"


def print_system():
    """Dynamically probes and displays active system hardware, AI models, and encoder."""
    g = gpu_info()
    cuda = torch_cuda_available()
    _, _, enc_name = get_best_video_encoder_config()

    print("=" * 64)
    print("AUTO CUT & POLISHING TOOL — HIGH-PERFORMANCE ENGINE")
    print("=" * 64)

    # 1. GPU & VRAM Status
    if g["available"]:
        print(f"GPU Hardware   : {g['name']}")
        print(f"VRAM Telemetry : {g['vram_used']} MB used / {g['vram_total']} MB total")
    else:
        print("GPU Hardware   : None detected (CPU Mode)")

    # 2. CUDA & Compute Architecture
    if cuda:
        try:
            import torch
            cap = torch.cuda.get_device_capability(0)
            print(f"CUDA Compute   : Active (PyTorch {torch.__version__} | CC {cap[0]}.{cap[1]})")
        except Exception:
            print("CUDA Compute   : Active")
    else:
        print("CUDA Compute   : Inactive")

    # 3. Dynamic Video Super-Resolution Engine Probe
    video_ai_models = []
    rb_ckpt = PROJECT_CHECKPOINTS / "realbasicvsr" / "realbasicvsr_c64b20_reds.pth"
    if rb_ckpt.is_file() and rb_ckpt.stat().st_size > 20_000_000:
        mb = rb_ckpt.stat().st_size / (1024 * 1024)
        video_ai_models.append(f"Real-BasicVSR ({mb:.0f} MB, FP16 Ready)")
    # Check for any other VSR model directories dynamically
    for vsr_dir in PROJECT_CHECKPOINTS.glob("*"):
        if vsr_dir.is_dir() and vsr_dir.name not in ("realbasicvsr", "MossFormer2_SE_48K", "voicefixer"):
            pth_files = list(vsr_dir.glob("*.pth")) + list(vsr_dir.glob("*.pt"))
            if pth_files:
                largest = max(pth_files, key=lambda f: f.stat().st_size)
                if largest.stat().st_size > 5_000_000:
                    mb2 = largest.stat().st_size / (1024 * 1024)
                    video_ai_models.append(f"{vsr_dir.name} ({mb2:.0f} MB)")

    if video_ai_models:
        print(f"Video AI Engine: {', '.join(video_ai_models)}")
    else:
        print("Video AI Engine: None cached (auto-fetch on first use)")

    # 4. Dynamic Audio AI Engines Probe
    audio_models = []
    moss_dir = PROJECT_CHECKPOINTS / "MossFormer2_SE_48K"
    if moss_dir.exists():
        try:
            if any(moss_dir.iterdir()):
                audio_models.append("ClearVoice/MossFormer2")
        except Exception:
            pass
    vf_dir = PROJECT_CHECKPOINTS / "voicefixer"
    if vf_dir.exists():
        try:
            if any(vf_dir.iterdir()):
                audio_models.append("VoiceFixer")
        except Exception:
            pass
    # Silero VAD: check project cache first, then torch hub cache
    silero_found = False
    silero_file = PROJECT_CHECKPOINTS / "silero_vad.jit"
    if silero_file.is_file():
        silero_found = True
    if not silero_found:
        try:
            import torch
            hub_dir = Path(torch.hub.get_dir())
            # Silero stores in snakers4_silero-vad_*
            for d in hub_dir.glob("snakers4_silero*"):
                if d.is_dir():
                    silero_found = True
                    break
        except Exception:
            pass
    if not silero_found:
        # Check if it can be imported (already in Python cache)
        try:
            from silero_vad import load_silero_vad  # noqa: F401
            silero_found = True
        except Exception:
            pass
    if silero_found:
        audio_models.append("Silero-VAD")

    if audio_models:
        print(f"Audio AI Engine: {', '.join(audio_models)} (Offline Ready)")
    else:
        print("Audio AI Engine: None cached (auto-fetch on first use)")

    # 5. Dynamic Video Encoder
    if enc_name.startswith("CPU") and g["available"] and _BEST_ENCODER_DIAGNOSTIC:
        print(f"Video Encoder  : {enc_name} [Note: {_BEST_ENCODER_DIAGNOSTIC}]")
    else:
        print(f"Video Encoder  : {enc_name}")

    print("=" * 64)
