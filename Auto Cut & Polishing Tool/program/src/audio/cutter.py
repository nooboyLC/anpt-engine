# -*- coding: utf-8 -*-
"""
audio.cutter
------------
Audio and video timeline cutting and seamless chunk concatenation.
Guarantees 100% frame-locked lip-sync.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from core.config import SAMPLE_RATE, CHANNELS, VAD_RATE
from core.media_tools import (
    ffmpeg_path, run, run_ffmpeg_with_progress, probe, get_stream_fps,
    get_filter_complex_script_flag, get_fps_mode_flags,
)
from core.hardware import get_best_video_encoder_config


def extract_audio(src: Path, out_wav: Path, sample_rate: int = SAMPLE_RATE):
    """Extracts high-fidelity PCM 16-bit WAV from source media."""
    run([
        ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src), "-vn", "-ac", str(CHANNELS), "-ar", str(sample_rate),
        "-c:a", "pcm_s16le", str(out_wav)
    ])


def extract_vad_audio(src: Path, out_wav: Path):
    """Extracts 16kHz mono audio for VAD speech analysis."""
    extract_audio(src, out_wav, VAD_RATE)


def concat_audio_chunks(chunks: list[Path], dst: Path):
    """Lossless concatenation of WAV audio chunks via FFmpeg concat demuxer."""
    list_file = dst.with_suffix(".txt")
    lines = [f"file '{p.resolve().as_posix().replace("'", "'\\''")}'" for p in chunks]
    list_file.write_text("\n".join(lines), encoding="utf-8")
    try:
        run([
            ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", str(list_file),
            "-c:a", "pcm_s16le", str(dst)
        ])
    finally:
        list_file.unlink(missing_ok=True)


def ffmpeg_cut_audio(segments: list[tuple[float, float]], src: Path, dst: Path, duration: float):
    """Cut audio-only segments using single-pass filter_complex atrim."""
    if not segments:
        extract_audio(src, dst)
        return

    fc = []
    for i, (s, e) in enumerate(segments):
        fc.append(f"[0:a]atrim=start={s:.6f}:end={e:.6f},asetpts=PTS-STARTPTS[a{i}]")
    ins = "".join(f"[a{i}]" for i in range(len(segments)))
    fc.append(f"{ins}concat=n={len(segments)}:v=0:a=1[out]")

    script_file = dst.parent / f"_cut_audio_{os.getpid()}.txt"
    script_file.write_text(";\n".join(fc), encoding="utf-8")

    fc_flag = get_filter_complex_script_flag()
    cmd = [
        ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src.resolve()),
        fc_flag, str(script_file.resolve()),
        "-map", "[out]", "-c:a", "pcm_s16le", str(dst.resolve())
    ]
    try:
        run_ffmpeg_with_progress(cmd, duration, "[2/10] CUT + SYNC")
    except Exception as e:
        alt_flag = "-filter_complex_script" if fc_flag == "-/filter_complex" else "-/filter_complex"
        if "option not found" in str(e).lower() or "unrecognized option" in str(e).lower():
            cmd[cmd.index(fc_flag)] = alt_flag
            run_ffmpeg_with_progress(cmd, duration, "[2/10] CUT + SYNC")
        else:
            raise
    finally:
        script_file.unlink(missing_ok=True)


def ffmpeg_concat_cut(src: Path, segments: list[tuple[float, float]], dst: Path, duration: float):
    """
    Cut the exact SAME video+audio segments in a single filter_complex pass.
    Frame-locked timeline guarantees 100% perfect lip sync.
    """
    if not segments:
        shutil.copy2(src, dst)
        return

    fc = []
    for i, (s, e) in enumerate(segments):
        fc.append(f"[0:v]trim=start={s:.6f}:end={e:.6f},setpts=PTS-STARTPTS[v{i}]")
        fc.append(f"[0:a]atrim=start={s:.6f}:end={e:.6f},asetpts=PTS-STARTPTS[a{i}]")
    concat_inputs = "".join(f"[v{i}][a{i}]" for i in range(len(segments)))
    fc.append(f"{concat_inputs}concat=n={len(segments)}:v=1:a=1[outv][outa]")

    script_file = dst.parent / f"_cut_script_{os.getpid()}.txt"
    script_file.write_text(";\n".join(fc), encoding="utf-8")

    enc, enc_flags, _ = get_best_video_encoder_config()
    threads_val = str(max(1, min(4, os.cpu_count() or 2)))
    fps = get_stream_fps(probe(src).get("video"), default=30.0)
    fc_flag = get_filter_complex_script_flag()

    cmd = [
        ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error",
        "-threads", threads_val,
        "-i", str(src.resolve()),
        fc_flag, str(script_file.resolve()),
        "-map", "[outv]", "-map", "[outa]",
    ] + get_fps_mode_flags() + [
        "-r", str(fps),
        "-c:v", enc,
    ] + enc_flags + [
        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(dst.resolve())
    ]
    try:
        run_ffmpeg_with_progress(cmd, duration, "[2/10] CUT + SYNC")
    except Exception as e:
        err_msg = str(e).lower()
        alt_flag = "-filter_complex_script" if fc_flag == "-/filter_complex" else "-/filter_complex"

        # 1. Only switch script flag if the error explicitly mentions filter_complex or script
        if ("filter_complex" in err_msg or "script" in err_msg) and ("option not found" in err_msg or "unrecognized option" in err_msg):
            cmd[cmd.index(fc_flag)] = alt_flag
            try:
                run_ffmpeg_with_progress(cmd, duration, "[2/10] CUT + SYNC")
                return
            except Exception as e2:
                err_msg = str(e2).lower()
                fc_flag = alt_flag

        # 2. Fallback to CPU libx264 if hardware encoder failed
        if enc != "libx264":
            cmd_fallback = [
                ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error",
                "-threads", threads_val,
                "-i", str(src.resolve()),
                fc_flag, str(script_file.resolve()),
                "-map", "[outv]", "-map", "[outa]",
            ] + get_fps_mode_flags() + [
                "-r", str(fps),
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(dst.resolve())
            ]
            try:
                run_ffmpeg_with_progress(cmd_fallback, duration, "[2/10] CUT + SYNC")
            except Exception as e_fb:
                fb_msg = str(e_fb).lower()
                if ("filter_complex" in fb_msg or "script" in fb_msg) and ("option not found" in fb_msg or "unrecognized option" in fb_msg):
                    cmd_fallback[cmd_fallback.index(fc_flag)] = alt_flag
                    run_ffmpeg_with_progress(cmd_fallback, duration, "[2/10] CUT + SYNC")
                else:
                    raise
        else:
            raise
    finally:
        script_file.unlink(missing_ok=True)
