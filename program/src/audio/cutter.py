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
from core.media_tools import ffmpeg_path, run, run_ffmpeg_with_progress
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


def _run_single_cut_audio(segments: list[tuple[float, float]], src: Path, dst: Path, duration: float):
    fc = []
    for i, (s, e) in enumerate(segments):
        fc.append(f"[0:a]atrim=start={s:.6f}:end={e:.6f},asetpts=PTS-STARTPTS[a{i}]")
    ins = "".join(f"[a{i}]" for i in range(len(segments)))
    fc.append(f"{ins}concat=n={len(segments)}:v=0:a=1[out]")
    cmd = [
        ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src), "-filter_complex", ";".join(fc),
        "-map", "[out]", "-c:a", "pcm_s16le", str(dst)
    ]
    run_ffmpeg_with_progress(cmd, duration, "[2/10] CUT + SYNC")


def ffmpeg_cut_audio(segments: list[tuple[float, float]], src: Path, dst: Path, duration: float):
    """Cut audio-only segments using filter_complex atrim."""
    if not segments:
        extract_audio(src, dst)
        return

    MAX_BATCH = 100
    if len(segments) <= MAX_BATCH:
        _run_single_cut_audio(segments, src, dst, duration)
        return

    batches = [segments[i:i + MAX_BATCH] for i in range(0, len(segments), MAX_BATCH)]
    part_files = []
    try:
        for idx, batch in enumerate(batches):
            part_dst = dst.parent / f"_audio_part_{idx}.wav"
            part_files.append(part_dst)
            batch_dur = sum(e - s for s, e in batch)
            _run_single_cut_audio(batch, src, part_dst, batch_dur)

        concat_audio_chunks(part_files, dst)
    finally:
        for p in part_files:
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass


def _run_single_concat_cut(src: Path, batch_segments: list[tuple[float, float]], dst: Path, duration: float):
    fc = []
    for i, (s, e) in enumerate(batch_segments):
        fc.append(f"[0:v]trim=start={s:.6f}:end={e:.6f},setpts=PTS-STARTPTS[v{i}]")
        fc.append(f"[0:a]atrim=start={s:.6f}:end={e:.6f},asetpts=PTS-STARTPTS[a{i}]")
    concat_inputs = "".join(f"[v{i}][a{i}]" for i in range(len(batch_segments)))
    fc.append(f"{concat_inputs}concat=n={len(batch_segments)}:v=1:a=1[outv][outa]")
    enc, enc_flags, _ = get_best_video_encoder_config()
    threads_val = str(max(1, min(4, os.cpu_count() or 2)))
    cmd = [
        ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error",
        "-threads", threads_val,
        "-i", str(src), "-filter_complex", ";".join(fc),
        "-map", "[outv]", "-map", "[outa]",
        "-c:v", enc,
    ] + enc_flags + [
        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(dst)
    ]
    try:
        run_ffmpeg_with_progress(cmd, duration, "[2/10] CUT + SYNC")
    except Exception:
        if enc != "libx264":
            cmd_fallback = [
                ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error",
                "-threads", threads_val,
                "-i", str(src), "-filter_complex", ";".join(fc),
                "-map", "[outv]", "-map", "[outa]",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(dst)
            ]
            run_ffmpeg_with_progress(cmd_fallback, duration, "[2/10] CUT + SYNC")
        else:
            raise


def ffmpeg_concat_cut(src: Path, segments: list[tuple[float, float]], dst: Path, duration: float):
    """
    Cut the exact SAME video+audio segments in a single filter_complex pass.
    Frame-locked timeline guarantees 100% perfect lip sync.
    """
    if not segments:
        shutil.copy2(src, dst)
        return

    MAX_BATCH = 80
    if len(segments) <= MAX_BATCH:
        _run_single_concat_cut(src, segments, dst, duration)
        return

    batches = [segments[i:i + MAX_BATCH] for i in range(0, len(segments), MAX_BATCH)]
    part_files = []
    try:
        for idx, batch in enumerate(batches):
            part_dst = dst.parent / f"_cut_part_{idx}.mp4"
            part_files.append(part_dst)
            batch_dur = sum(e - s for s, e in batch)
            _run_single_concat_cut(src, batch, part_dst, batch_dur)

        list_file = dst.parent / "_cut_parts_list.txt"
        lines = [f"file '{p.resolve().as_posix()}'" for p in part_files]
        list_file.write_text("\n".join(lines), encoding="utf-8")
        try:
            run([
                ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error",
                "-f", "concat", "-safe", "0", "-i", str(list_file),
                "-c", "copy", "-movflags", "+faststart", str(dst)
            ])
        finally:
            list_file.unlink(missing_ok=True)
    finally:
        for p in part_files:
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass
