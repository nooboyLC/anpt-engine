# -*- coding: utf-8 -*-
"""
audio.cleaner
-------------
Deep Learning & DSP speech enhancement, background de-noising, and de-reverberation.
Primary engines: ClearVoice (MossFormer2) and VoiceFixer, with robust DSP fallbacks.
"""

from __future__ import annotations

import os
import sys
import shutil
import contextlib
from pathlib import Path

from core.config import AI_CHUNK_SECONDS
from core.media_tools import ffmpeg_path, run, run_ffmpeg_with_progress
from core.hardware import torch_cuda_available, gpu_status_text
from core.logger import progress
from audio.vad import wav_duration
from audio.cutter import concat_audio_chunks


@contextlib.contextmanager
def suppress_stdout_stderr():
    """Suppress C-level and Python-level stdout/stderr (e.g. from ClearVoice / tqdm)."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def dereverb_with_voicefixer(src: Path, dst: Path):
    """VoiceFixer neural de-reverberation."""
    from voicefixer import VoiceFixer
    vf = VoiceFixer()
    vf.restore(input=str(src), output=str(dst), cuda=torch_cuda_available(), mode=2)
    progress("[3/10] ECHO/REVERB", 1.0, "VoiceFixer de-reverberation complete")


def dereverb_audio(src: Path, dst: Path, engine: str, temp_dir: Path, duration: float = 0) -> str:
    """De-reverbs audio using neural model or acoustic DSP filtering."""
    errors = []
    if engine in {"auto", "voicefixer"}:
        try:
            dereverb_with_voicefixer(src, dst)
            return "VoiceFixer (mode=2)"
        except Exception as exc:
            errors.append(f"VoiceFixer: {exc}")
            if engine == "voicefixer":
                raise RuntimeError("VoiceFixer de-reverb failed: " + str(exc))

    dereverb_filters = (
        "highpass=f=80,"
        "equalizer=f=320:width_type=o:width=1.2:g=-4.0,"
        "equalizer=f=480:width_type=o:width=1.0:g=-2.5,"
        "afftdn=nr=14:nf=-32:tn=1,"
        "agate=threshold=-28dB:ratio=2.5:range=-24dB:attack=10:release=100,"
        "acompressor=threshold=-16dB:ratio=2:attack=20:release=200"
    )
    cmd = [
        ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error", "-i", str(src),
        "-af", dereverb_filters,
        "-c:a", "pcm_s16le", str(dst)
    ]
    if duration > 0:
        run_ffmpeg_with_progress(cmd, duration, "[3/10] ECHO/REVERB")
    else:
        run(cmd)
    return "Acoustic De-reverb DSP"


def enhance_with_clearvoice(src: Path, dst: Path, temp_dir: Path):
    """Speech enhancement with ClearVoice MossFormer2."""
    from core.config import PROJECT_CHECKPOINTS
    import clearvoice
    if not getattr(clearvoice, "_antigravity_patched", False):
        _orig = clearvoice.network_wrapper.load_args_se
        def _custom(self):
            _orig(self)
            self.args.checkpoint_dir = str(PROJECT_CHECKPOINTS / self.model_name)
        clearvoice.network_wrapper.load_args_se = _custom
        clearvoice._antigravity_patched = True
    with suppress_stdout_stderr():
        cv = clearvoice.ClearVoice(task="speech_enhancement", model_names=["MossFormer2_SE_48K"])
    duration = wav_duration(src)
    chunks_dir = temp_dir / "ai_chunks"
    chunks_dir.mkdir(exist_ok=True)
    out_chunks = []
    total_chunks = max(1, int(duration / AI_CHUNK_SECONDS) + (1 if duration % AI_CHUNK_SECONDS else 0))

    for i, start in enumerate(range(0, int(duration + 0.999), AI_CHUNK_SECONDS)):
        dur = min(AI_CHUNK_SECONDS, duration - start)
        if dur <= 0:
            break
        inp = chunks_dir / f"in_{i:05d}.wav"
        out = chunks_dir / f"out_{i:05d}.wav"
        run([
            ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error",
            "-ss", str(start), "-t", str(dur), "-i", str(src),
            "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le", str(inp)
        ])
        with suppress_stdout_stderr():
            result = cv(input_path=str(inp))
            cv.write(result, output_path=str(out))
        inp.unlink(missing_ok=True)
        out_chunks.append(out)
        cur_pct = min(0.99, (i + 1) / total_chunks)
        progress("[4/10] AI AUDIO", cur_pct, f"chunk {i+1}/{total_chunks} | {gpu_status_text()}")

    concat_audio_chunks(out_chunks, dst)
    shutil.rmtree(chunks_dir, ignore_errors=True)
    progress("[4/10] AI AUDIO", 1.0, "ClearVoice complete")


def enhance_with_voicefixer(src: Path, dst: Path):
    """Speech restoration with VoiceFixer mode 0."""
    from voicefixer import VoiceFixer
    vf = VoiceFixer()
    vf.restore(input=str(src), output=str(dst), cuda=torch_cuda_available(), mode=0)
    progress("[4/10] AI AUDIO", 1.0, "VoiceFixer complete")


def enhance_audio(src: Path, dst: Path, engine: str, temp_dir: Path) -> str:
    """Enhances audio clarity via AI models or DSP fallback."""
    errors = []
    if engine in {"auto", "clearvoice"}:
        try:
            enhance_with_clearvoice(src, dst, temp_dir)
            return "ClearVoice"
        except Exception as exc:
            errors.append(f"ClearVoice: {exc}")
            if engine == "clearvoice":
                raise RuntimeError("ClearVoice failed: " + str(exc))

    if engine in {"auto", "voicefixer"}:
        try:
            enhance_with_voicefixer(src, dst)
            return "VoiceFixer"
        except Exception as exc:
            errors.append(f"VoiceFixer: {exc}")
            if engine == "voicefixer":
                raise RuntimeError("VoiceFixer failed: " + str(exc))

    # Safe fallback DSP
    run([
        ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error", "-i", str(src),
        "-af", "highpass=f=60,lowpass=f=11000,acompressor=threshold=-18dB:ratio=2:attack=20:release=250",
        "-c:a", "pcm_s16le", str(dst)
    ])
    progress("[4/10] AI AUDIO", 1.0, "AI unavailable; DSP fallback used")
    return "FFmpeg Audio DSP"
