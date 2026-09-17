# -*- coding: utf-8 -*-
"""
audio.vad
---------
Multi-stage Hybrid Voice Activity Detection (Energy VAD + Silero Neural VAD).
Tight silence detection with podcast/aggressive presets to eliminate dead air.
"""

from __future__ import annotations

import sys
import wave
import json
import array
from pathlib import Path

from core.config import (
    VAD_RATE,
    DEFAULT_MIN_SILENCE,
    DEFAULT_KEEP_PAUSE,
    DEFAULT_ENERGY_THRESHOLD,
)
from core.logger import eprint


def wav_duration(path: Path) -> float:
    """Returns duration of WAV file in seconds."""
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


def merge_segments(segments: list[tuple[float, float]], gap: float = 0.05) -> list[tuple[float, float]]:
    """Merges overlapping or closely spaced speech segments."""
    clean = sorted((max(0.0, float(s)), max(0.0, float(e))) for s, e in segments if e > s)
    out: list[list[float]] = []
    for s, e in clean:
        if not out or s - out[-1][1] > gap:
            out.append([s, e])
        else:
            out[-1][1] = max(out[-1][1], e)
    return [(s, e) for s, e in out]


def normalize_segments(segments: list[tuple[float, float]], duration: float, keep_pause: float = DEFAULT_KEEP_PAUSE) -> list[tuple[float, float]]:
    """Clamps segments within [0, duration] and merges any gaps smaller than keep_pause."""
    if not segments:
        return [(0.0, duration)]
    out = []
    for s, e in segments:
        s = max(0.0, s)
        e = min(duration, e)
        if e > s:
            out.append((s, e))
    return merge_segments(out, gap=max(0.15, keep_pause * 0.8))


def write_segments_json(path: Path, segments: list[tuple[float, float]]):
    """Saves segment timeline list to JSON."""
    data = [{"start": round(s, 3), "end": round(e, 3)} for s, e in segments]
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def energy_segments(
    wav_path: Path,
    threshold: float = DEFAULT_ENERGY_THRESHOLD,
    frame_ms: int = 30,
    min_silence: float = DEFAULT_MIN_SILENCE,
    keep_pause: float = DEFAULT_KEEP_PAUSE,
) -> list[tuple[float, float]]:
    """
    Zero-RAM, zero-dependency streaming energy VAD.
    Accurately detects silent gaps without loading the full audio into memory.
    """
    with wave.open(str(wav_path), "rb") as w:
        sr = w.getframerate()
        frame_n = max(1, int(sr * frame_ms / 1000))
        min_sil_frames = max(1, int(min_silence * 1000 / frame_ms))
        speech = False
        start = 0.0
        silent_frames = 0
        pos = 0
        segments = []
        while True:
            raw = w.readframes(frame_n)
            if not raw:
                break
            a = array.array("h")
            a.frombytes(raw)
            if sys.byteorder != "little":
                a.byteswap()
            if a:
                sq = sum(v * v for v in a) / len(a)
                rms = (sq ** 0.5) / 32768.0
            else:
                rms = 0.0
            voiced = rms >= threshold
            t = pos / sr
            if voiced:
                if not speech:
                    start = max(0.0, t - frame_ms / 1000)
                    speech = True
                silent_frames = 0
            elif speech:
                silent_frames += 1
                if silent_frames >= min_sil_frames:
                    end = t - (silent_frames * frame_ms / 1000) + keep_pause
                    end = max(start, end)
                    segments.append((start, end))
                    speech = False
                    silent_frames = 0
            pos += len(a)
        if speech:
            segments.append((start, pos / sr))

    merged = merge_segments(segments, gap=max(0.02, keep_pause * 0.5))
    if not merged:
        return [(0.0, pos / sr)]
    return merged


def silero_segments(
    wav_path: Path,
    min_silence: float = DEFAULT_MIN_SILENCE,
    keep_pause: float = DEFAULT_KEEP_PAUSE,
) -> list[tuple[float, float]]:
    """
    Neural Silero VAD with GPU (CUDA) acceleration when available, falling back to CPU.
    """
    try:
        import torch
        import numpy as np
        from silero_vad import load_silero_vad, VADIterator

        model = load_silero_vad()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
    except Exception as exc:
        raise RuntimeError(f"Silero VAD unavailable: {exc}") from exc

    segments = []
    sr = VAD_RATE
    window = 512  # 32ms at 16kHz
    try:
        vad = VADIterator(
            model,
            sampling_rate=sr,
            min_silence_duration_ms=max(100, int(min_silence * 1000)),
            speech_pad_ms=int(keep_pause * 1000),
        )
        with wave.open(str(wav_path), "rb") as w:
            idx = 0
            active_start = None
            total = w.getnframes()
            while True:
                raw = w.readframes(window)
                if not raw:
                    break
                x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                if len(x) < window:
                    x = np.pad(x, (0, window - len(x)))
                t = idx / sr
                t_tensor = torch.from_numpy(x).to(device)
                with torch.no_grad():
                    out = vad(t_tensor, return_seconds=True)
                if out and "start" in out:
                    active_start = max(0.0, float(out["start"]))
                if out and "end" in out:
                    end = float(out["end"])
                    if active_start is not None:
                        segments.append((active_start, end))
                        active_start = None
                idx += len(x)
            if active_start is not None:
                segments.append((active_start, total / sr))
        if not segments:
            raise RuntimeError("Silero detected no speech segments.")
        return merge_segments(segments, gap=max(0.02, keep_pause * 0.5))
    finally:
        try:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def hybrid_vad_segments(
    wav_path: Path,
    min_silence: float = DEFAULT_MIN_SILENCE,
    keep_pause: float = DEFAULT_KEEP_PAUSE,
    threshold: float = DEFAULT_ENERGY_THRESHOLD,
) -> list[tuple[float, float]]:
    """
    Two-Stage Hybrid VAD (Silero Neural VAD + Energy VAD with Natural Cadence Padding):
    1. Runs Silero Neural VAD (or Energy fallback) to accurately locate human speech.
    2. Applies natural speech boundary padding (front buffer for mouth-opening/consonants,
       tail buffer for natural breath fade-out).
    3. Bridges small intra-sentence pauses (< min_silence) to ensure smooth, natural speech flow
       without abrupt, choppy, or jarring cuts.
    """
    try:
        raw_segs = silero_segments(wav_path, min_silence=min_silence, keep_pause=keep_pause)
    except Exception as ex:
        eprint(f"[VAD] Silero VAD fallback to Energy: {ex}")
        raw_segs = energy_segments(
            wav_path,
            threshold=threshold,
            min_silence=min_silence,
            keep_pause=keep_pause,
        )

    if not raw_segs:
        return [(0.0, wav_duration(wav_path))]

    # Pad boundaries naturally:
    # Front buffer: 50% of keep_pause (allows natural mouth opening & first consonant)
    # Tail buffer:  70% of keep_pause (allows natural voice decay & breath)
    front_pad = max(0.10, keep_pause * 0.5)
    tail_pad  = max(0.15, keep_pause * 0.7)

    padded = []
    for s, e in raw_segs:
        padded.append((max(0.0, s - front_pad), e + tail_pad))

    # Merge any segments whose pause gap is shorter than min_silence (smooth natural breathing)
    merged = merge_segments(padded, gap=max(0.15, min_silence))
    return merged if merged else raw_segs
