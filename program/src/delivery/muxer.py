# -*- coding: utf-8 -*-
"""
delivery.muxer
--------------
Final audiovisual multiplexing, faststart metadata insertion,
and smart descriptive output filename generation.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from core.media_tools import ffmpeg_path, run, run_ffmpeg_with_progress


def mux_final(video: Path, audio: Path | None, dst: Path, duration: float = 0):
    """Muxes processed video and audio streams together with +faststart."""
    if audio is None:
        shutil.copy2(video, dst)
        return
    cmd = [
        ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(video), "-i", str(audio),
        "-map", "0:v:0?", "-map", "1:a:0?",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest", "-movflags", "+faststart", str(dst)
    ]
    if duration > 0:
        run_ffmpeg_with_progress(cmd, duration, "[10/10] FINAL MUX")
    else:
        run(cmd)


def build_output_suffix(args, meta: dict | None = None) -> str:
    """Builds a descriptive filename suffix based on enabled features."""
    tags = []
    if getattr(args, "remove_silence", False):
        tags.append("AutoCuted")
    if getattr(args, "audio_enhance", False) and getattr(args, "video_enhance", False):
        tags.append("AudVidEnhanced")
    elif getattr(args, "audio_enhance", False):
        tags.append("AudioEnhanced")
    elif getattr(args, "video_enhance", False):
        tags.append("VideoEnhanced")
    if getattr(args, "video_ai", False):
        tags.append("RealBasicVSR")
    if getattr(args, "dereverb", False):
        tags.append("EchoRemoved")
    if getattr(args, "stabilize", False):
        tags.append("Stabilized")
    if getattr(args, "normalize", False) and not getattr(args, "audio_enhance", False) and not getattr(args, "dereverb", False):
        tags.append("Normalized")
    if not tags:
        return "_Processed"
    return "_" + "-".join(tags)
