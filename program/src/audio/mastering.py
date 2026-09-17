# -*- coding: utf-8 -*-
"""
audio.mastering
---------------
Audio leveling, broadcast mastering, multi-speaker dialogue balancing,
and EBU R128 loudness normalization.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from core.config import SAMPLE_RATE, CHANNELS
from core.media_tools import ffmpeg_path, run, run_ffmpeg_with_progress
from core.logger import eprint


def pre_level_audio(src: Path, dst: Path, duration: float = 0):
    """
    Initial gain staging to standardize input dynamics before VAD & AI filters.
    Prevents low-volume speakers from being misclassified as silence.
    """
    cmd = [
        ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src), "-vn", "-ac", str(CHANNELS), "-ar", str(SAMPLE_RATE),
        "-af", "loudnorm=I=-18:TP=-2.0:LRA=11",
        "-c:a", "pcm_s16le", str(dst)
    ]
    if duration > 0:
        run_ffmpeg_with_progress(cmd, duration, "[1/10] PRE-LEVELING")
    else:
        run(cmd)


def loudness_normalize(src: Path, dst: Path, target: float = -14.0, duration: float = 0):
    """EBU R128 standard loudness normalization."""
    cmd = [
        ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src), "-af", f"loudnorm=I={target}:TP=-1.5:LRA=11",
        "-c:a", "pcm_s16le", str(dst)
    ]
    if duration > 0:
        run_ffmpeg_with_progress(cmd, duration, "[6/10] LOUDNESS")
    else:
        run(cmd)


def voice_fine_tuning(src: Path, dst: Path, duration: float = 0):
    """
    Studio Broadcast Vocal Fine-Tuning & EQ Polish:
    - Warmth & Body: gentle low-mid boost (180Hz)
    - Mud Scoop: subtle cut at 420Hz
    - Articulation & Presence: 3.5kHz clarity boost
    - Studio Air: 9.5kHz smooth highshelf
    - Broadcast Compressor: radio-ready punch
    """
    master_filters = (
        "equalizer=f=180:width_type=o:width=1.0:g=1.5,"
        "equalizer=f=420:width_type=o:width=1.2:g=-1.8,"
        "equalizer=f=3500:width_type=o:width=1.0:g=2.0,"
        "highshelf=f=9500:g=1.2,"
        "acompressor=threshold=-16dB:ratio=2.2:attack=15:release=180"
    )
    cmd = [
        ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error", "-i", str(src),
        "-af", master_filters,
        "-c:a", "pcm_s16le", str(dst)
    ]
    if duration > 0:
        run_ffmpeg_with_progress(cmd, duration, "[7/10] FINE-TUNING")
    else:
        run(cmd)


def balance_multispeaker_volume(audio_in: Path, audio_out: Path, target_lufs: float = -14.0, duration: float = 0.0) -> bool:
    """
    Equalizes speaker volume levels in multi-speaker recordings.
    Uses FFmpeg dynamic audio normalizer (dynaudnorm) + EBU R128 loudnorm.
    """
    if not audio_in.exists():
        return False
    try:
        balance_filter = (
            "dynaudnorm=f=150:g=15:peak=0.95:m=10.0:r=0.9,"
            f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11"
        )
        cmd = [
            ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(audio_in),
            "-af", balance_filter,
            "-c:a", "pcm_s16le", str(audio_out)
        ]
        if duration > 0:
            run_ffmpeg_with_progress(cmd, duration, "[5/10] MULTI-SPEAKER BALANCE")
        else:
            run(cmd)
        return True
    except Exception as exc:
        eprint(f"\n[WARN] Multi-speaker volume balance: {exc}; keeping original.")
        shutil.copy2(audio_in, audio_out)
        return False
