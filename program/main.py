#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto Cut & Polishing Tool — Orchestrator
----------------------------------------
Decoupled workflow pipeline:
- Native C++/GPU Acceleration (FFmpeg NVENC/NVDEC, Vulkan Real-ESRGAN, CUDA Tensor Cores)
- Sub-chunk streaming with zero RAM overflow
- Tight VAD silence removal
- Cross-platform support (Windows & Linux / Google Colab)
"""

from __future__ import annotations

import os
import sys
import shutil
import argparse
import tempfile
import time
from pathlib import Path

# Add src/ to Python search path
_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# Core Infrastructure
from core.config import (
    BASE_DIR,
    OUTPUT_DIR,
    PROJECT_TEMP,
    DEFAULT_MIN_SILENCE,
    DEFAULT_KEEP_PAUSE,
    DEFAULT_ENERGY_THRESHOLD,
    is_colab,
    cleanup_memory,
)
from core.logger import (
    eprint,
    progress,
    fmt_time,
    fmt_duration,
    fmt_duration_clock,
    current_timestamp_str,
    log_step,
)
from core.hardware import (
    print_system,
    check_system_dependencies,
    torch_cuda_available,
    vulkan_available,
)
from core.media_tools import (
    ffmpeg_path,
    ffprobe_path,
    probe,
    is_url,
    sanitize_title,
    download_url,
)

# Audio Subsystem
from audio.cutter import (
    extract_audio,
    extract_vad_audio,
    ffmpeg_concat_cut,
    ffmpeg_cut_audio,
)
from audio.vad import (
    hybrid_vad_segments,
    silero_segments,
    energy_segments,
    normalize_segments,
    write_segments_json,
)
from audio.cleaner import enhance_audio, dereverb_audio
from audio.mastering import (
    pre_level_audio,
    loudness_normalize,
    voice_fine_tuning,
    balance_multispeaker_volume,
)

# Video Subsystem
from video.stabilizer import stabilize_video
from video.enhancer_filter import enhance_video
from video.enhancer_ai import enhance_video_ai_vulkan
from video.thumbnail import extract_best_thumbnails

# Delivery Subsystem
from delivery.muxer import mux_final, build_output_suffix
from delivery.cloud_upload import trigger_file_download


def get_input(source: str, temp_dir: Path, output_dir: Path | None = None) -> Path:
    """Resolves local file path or downloads remote URL via yt-dlp.
    When downloading, saves the original raw media into output_dir if available,
    otherwise uses temp_dir (e.g., in cloud-only mode).
    """
    if is_url(source):
        target_dir = output_dir if output_dir is not None else temp_dir
        return download_url(source, target_dir, suffix="_Unedited_Raw", temp_dir=temp_dir)
    p = Path(source).expanduser().resolve()
    if not p.exists() or not p.is_file():
        raise RuntimeError(f"Input file not found: {p}")
    return p


def process(args):
    """Executes the media enhancement and polishing workflow."""
    if not ffmpeg_path() or not ffprobe_path():
        raise RuntimeError("FFmpeg and FFprobe are required and must be installed in PATH.")

    print("\n" + "-" * 70)
    print(f"[{current_timestamp_str()}] START")
    print("-" * 70)

    work_root = Path(getattr(args, "temp", None) or PROJECT_TEMP) / "auto_cut_polishing_tool"
    work_root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="run_", dir=work_root))
    pipeline_t0 = time.time()

    try:
        is_cloud_only = getattr(args, "cloud_only", False) or (getattr(args, "output", None) is None)
        out_dir = None if is_cloud_only else Path(args.output).expanduser().resolve()

        src = get_input(args.input, run_dir, output_dir=out_dir)
        meta = probe(src)

        log_step()
        progress("[MEDIA ANALYSIS]", 1.0, f"Duration: {fmt_duration_clock(meta['duration'])}")

        current_video = src
        current_audio = None
        timeline_duration = meta["duration"]

        # Step 1: Pre-level input audio dynamics
        pre_leveled_audio = None
        if meta["audio"]:
            log_step()
            pre_leveled = run_dir / "pre_leveled.wav"
            pre_level_audio(src, pre_leveled, duration=meta["duration"])
            pre_leveled_audio = pre_leveled

        # Step 2: Silence Removal & Auto-Cut
        if getattr(args, "remove_silence", False):
            log_step()
            vad_wav = run_dir / "vad.wav"
            vad_source = pre_leveled_audio if (pre_leveled_audio and pre_leveled_audio.exists()) else src
            extract_vad_audio(vad_source, vad_wav)

            min_sil = getattr(args, "min_silence", DEFAULT_MIN_SILENCE)
            keep_p = getattr(args, "keep_pause", DEFAULT_KEEP_PAUSE)
            thresh = getattr(args, "energy_threshold", DEFAULT_ENERGY_THRESHOLD)

            segments = hybrid_vad_segments(vad_wav, min_silence=min_sil, keep_pause=keep_p, threshold=thresh)
            vad_wav.unlink(missing_ok=True)
            segments = normalize_segments(segments, meta["duration"], keep_p)
            if not segments:
                segments = [(0.0, meta["duration"])]

            write_segments_json(run_dir / "segments.json", segments)
            kept = sum(e - s for s, e in segments)
            timeline_duration = kept
            print(f"VAD: 2-Stage Hybrid VAD (Energy + Silero) | keep {kept:.1f}s / {meta['duration']:.1f}s")

            cut = run_dir / "cut_media.mp4"
            if meta["video"] and meta["audio"]:
                ffmpeg_concat_cut(src, segments, cut, duration=kept)
            elif meta["audio"]:
                ffmpeg_cut_audio(segments, src, cut, duration=kept)
            else:
                shutil.copy2(src, cut)

            current_video = cut if meta["video"] else src
            current_audio = cut if (meta["audio"] and not meta["video"]) else None
        else:
            current_audio = pre_leveled_audio

        # Extract isolated audio track if subsequent audio steps are requested
        if meta["audio"] and (getattr(args, "dereverb", False) or getattr(args, "audio_enhance", False) or getattr(args, "normalize", False) or getattr(args, "multi_speaker_balance", False)):
            if current_audio is None:
                current_audio = run_dir / "audio.wav"
                extract_audio(current_video, current_audio)

        # Step 3: Echo & Reverb Removal
        if getattr(args, "dereverb", False) and meta["audio"]:
            log_step()
            dereverbed = run_dir / "dereverbed.wav"
            engine_used = dereverb_audio(current_audio, dereverbed, getattr(args, "engine", "auto"), run_dir, duration=timeline_duration)
            current_audio = dereverbed
            print(f"Dereverb engine: {engine_used}")

        # Step 4: Audio AI Enhancement (ClearVoice / VoiceFixer)
        if getattr(args, "audio_enhance", False) and meta["audio"]:
            log_step()
            enhanced = run_dir / "enhanced.wav"
            engine_used = enhance_audio(current_audio, enhanced, getattr(args, "engine", "auto"), run_dir)
            current_audio = enhanced
            print(f"Audio engine: {engine_used}")

        # Step 5: Multi-Speaker Volume Balancing
        if getattr(args, "multi_speaker_balance", False) and meta["audio"]:
            log_step()
            balanced = run_dir / "multispeaker_balanced.wav"
            balance_multispeaker_volume(current_audio, balanced, target_lufs=getattr(args, "target_lufs", -14.0), duration=timeline_duration)
            current_audio = balanced

        # Step 6: Loudness Normalization (-14 LUFS)
        if getattr(args, "normalize", False) and meta["audio"]:
            log_step()
            norm = run_dir / "normalized.wav"
            loudness_normalize(current_audio, norm, getattr(args, "target_lufs", -14.0), duration=timeline_duration)
            current_audio = norm

        # Step 7: Studio Vocal Fine-Tuning & Mastering
        if meta["audio"] and (getattr(args, "audio_enhance", False) or getattr(args, "dereverb", False) or getattr(args, "normalize", False) or getattr(args, "multi_speaker_balance", False)):
            log_step()
            tuned = run_dir / "tuned.wav"
            voice_fine_tuning(current_audio, tuned, duration=timeline_duration)
            current_audio = tuned

        # Step 8: Video Stabilization
        if getattr(args, "stabilize", False) and meta["video"]:
            if current_audio is None and meta["audio"]:
                current_audio = run_dir / "audio_preserved.wav"
                extract_audio(current_video, current_audio)
            log_step()
            stabilized_video = run_dir / "video_stabilized.mp4"
            stab_engine = getattr(args, "stabilize_engine", "gpu")
            stabilize_video(current_video, stabilized_video, timeline_duration, run_dir, engine=stab_engine)
            current_video = stabilized_video

        # Step 9: Video Enhancement (100% GPU Tensor Core Cinema Polish & Natural Sharpening)
        if getattr(args, "video_enhance", False) and meta["video"]:
            if current_audio is None and meta["audio"]:
                current_audio = run_dir / "audio_preserved.wav"
                extract_audio(current_video, current_audio)
            log_step()
            enhanced_video = run_dir / "video_enhanced.mp4"
            v_engine = getattr(args, "video_engine", "auto")
            if v_engine == "vulkan":
                success = enhance_video_ai_vulkan(current_video, enhanced_video, timeline_duration, meta)
                if not success:
                    enhance_video(current_video, enhanced_video, timeline_duration, meta)
            else:
                enhance_video(current_video, enhanced_video, timeline_duration, meta)
            current_video = enhanced_video

        # Step 10: Final Multiplexing & Packaging
        ext = ".mp4" if meta["video"] else ".wav"
        custom_name = getattr(args, "output_name", None)
        if custom_name and str(custom_name).strip():
            clean_name = str(custom_name).strip()
            final_filename = clean_name if clean_name.lower().endswith((".mp4", ".wav", ".mkv")) else f"{clean_name}{ext}"
        else:
            stem = sanitize_title(Path(src).stem, max_chars=55)
            for raw_tag in ["_Unedited_Raw", "_Raw_Downloaded", "_unedited_raw"]:
                if stem.endswith(raw_tag):
                    stem = stem[:-len(raw_tag)]
                    break
            suffix = build_output_suffix(args, meta)
            final_filename = f"{stem}{suffix}{ext}"

        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            dst = out_dir / final_filename
            if not getattr(args, "overwrite", False):
                idx = 1
                dst_stem = dst.stem
                dst_suffix = dst.suffix
                while dst.exists():
                    dst = out_dir / f"{dst_stem}_{idx}{dst_suffix}"
                    idx += 1
        else:
            dst = run_dir / final_filename

        log_step()
        if meta["video"]:
            if not current_video.exists() or current_video.stat().st_size < 1000:
                current_video = src
            mux_final(current_video, current_audio, dst, duration=timeline_duration)
        else:
            if current_audio is not None and current_audio.exists():
                shutil.copy2(current_audio, dst.with_suffix(".wav"))
                dst = dst.with_suffix(".wav")
            else:
                shutil.copy2(current_video, dst)

        # Step 11: AI Expressive Thumbnail Extraction
        if getattr(args, "extract_thumbnails", False) and meta["video"]:
            target_thumb_dir = (out_dir / "thumbnails") if out_dir is not None else (run_dir / "thumbnails")
            target_thumb_dir.mkdir(parents=True, exist_ok=True)
            log_step()
            extract_best_thumbnails(current_video, target_thumb_dir, temp_dir=run_dir, duration=timeline_duration)

        print("\n" + "=" * 64)
        if out_dir is not None:
            print(f"SUCCESS: Processed media saved to:\n{dst}")
        else:
            print("SUCCESS: Processing complete (Cloud-only mode).")
        print("=" * 64)        # Step 12: Cloud Upload / Browser Download trigger
        if getattr(args, "download", False) or is_cloud_only:
            log_step("CLOUD & LOCAL DELIVERY")
            trigger_file_download(dst)

        total_time = time.time() - pipeline_t0
        log_step(f"ALL OPERATIONS COMPLETED | Total Runtime: {fmt_duration(total_time)}")

        return dst
    finally:
        cleanup_memory()
        if getattr(args, "keep_temp", False):
            print(f"[CLEANUP] Scratch workspace kept at: {run_dir}")
        else:
            shutil.rmtree(run_dir, ignore_errors=True)


def choose_input_interactive() -> str:
    """Interactively prompts user for input source."""
    print("\nINPUT SOURCE")
    print("1. Local file")
    print("2. URL (YouTube / Facebook / Direct stream)")
    print("3. Google Drive (/content/drive/MyDrive/...)")
    print("4. Exit")

    drive_root = Path("/content/drive")
    while True:
        choice = input("Select [1-4]: ").strip()
        if choice == "1":
            return input("Enter local media file path: ").strip()
        if choice == "2":
            return input("Enter video URL: ").strip()
        if choice == "3":
            if not drive_root.exists() and is_colab():
                try:
                    from google.colab import drive
                    drive.mount("/content/drive")
                except Exception:
                    pass
            return input("Google Drive path: ").strip()
        if choice == "4":
            sys.exit(0)


def choose_output_interactive() -> tuple[str | None, str | None, bool]:
    """Prompts for output directory, custom name, and automatic download."""
    default_out = str(OUTPUT_DIR)
    print("\nOUTPUT DESTINATION")
    print("1. Default project output folder (.\\output)")
    print("2. Custom directory")
    print("3. Google Drive (/content/drive/MyDrive/...)")

    c = input("Select destination [1-3] (1): ").strip() or "1"
    if c == "1":
        out_folder = default_out
    elif c == "2":
        out_folder = input("Enter output directory path: ").strip() or default_out
    elif c == "3":
        colab_drive = Path("/content/drive/MyDrive")
        if colab_drive.exists():
            sub = input("Subfolder under MyDrive (empty for root): ").strip()
            out_folder = str(colab_drive / sub) if sub else str(colab_drive)
        else:
            out_folder = input("Google Drive path: ").strip() or default_out
    else:
        out_folder = default_out

    out_name = input("\nCustom output filename (Press ENTER for auto-generated name): ").strip()
    dl_prompt = input("\nAuto-download file to your local PC when finished? [y/N]: ").strip().lower()
    want_download = dl_prompt in ["y", "yes"]

    return out_folder, out_name if out_name else None, want_download


def interactive():
    """Interactive guided terminal interface."""
    all_critical_ok = check_system_dependencies()
    print_system()

    if not all_critical_ok:
        print("\n[ERROR] Critical components are missing. Please run setup.bat (Windows) or setup.sh (Linux/Mac) first.")
        input("Press ENTER to exit...")
        sys.exit(1)

    source = choose_input_interactive()
    flags = {
        "remove_silence": False,
        "audio_enhance": False,
        "dereverb": False,
        "normalize": False,
        "stabilize": False,
        "video_enhance": False,
        "extract_thumbnails": False,
    }

    while True:
        print("\nSELECT WORKFLOW FEATURES:")
        items = [
            ("remove_silence", "Remove silence (Audio & Video frame-locked cut)"),
            ("audio_enhance", "AI audio enhancement (ClearVoice / VoiceFixer)"),
            ("dereverb", "Echo & Reverb removal (Dereverberation)"),
            ("normalize", "Loudness normalization (-14 LUFS standard)"),
            ("stabilize", "Video Stabilization (GPU camera deshake)"),
            ("video_enhance", "Video AI Enhancement (Real-ESRGAN GPU)"),
            ("extract_thumbnails", "AI Expressive Thumbnail Generator"),
        ]
        for i, (k, label) in enumerate(items, 1):
            status = "x" if flags[k] else " "
            print(f"[{status}] {i}. {label}")

        print("A=all  N=none  ENTER=start  Q=quit")
        cmd = input("Choice: ").strip().lower()
        if not cmd:
            break
        if cmd == "q":
            return
        if cmd == "a":
            for k in flags:
                flags[k] = True
            print("  -> All features selected.")
            break
        elif cmd == "n":
            for k in flags:
                flags[k] = False
        else:
            import re
            for t in re.findall(r"[1-7]", cmd):
                idx = int(t) - 1
                k_t = items[idx][0]
                flags[k_t] = not flags[k_t]

    print("\n" + "-" * 64)
    multi_prompt = input("Is this a Multi-Speaker video/audio (Podcast, Interview, Press Conf)? [y/N]: ").strip().lower()
    is_multi_speaker = False if multi_prompt in ["n", "no"] else True

    print("-" * 64)
    out_folder, out_name, want_download = choose_output_interactive()

    args = argparse.Namespace(
        input=source,
        output=out_folder,
        output_name=out_name,
        download=want_download,
        cloud_only=(out_folder is None),
        temp=None,
        remove_silence=flags["remove_silence"],
        audio_enhance=flags["audio_enhance"],
        dereverb=flags["dereverb"],
        normalize=flags["normalize"],
        multi_speaker_balance=is_multi_speaker,
        multi_speaker=is_multi_speaker,
        stabilize=flags["stabilize"],
        video_enhance=flags["video_enhance"],
        video_engine="auto",
        extract_thumbnails=flags["extract_thumbnails"],
        min_silence=DEFAULT_MIN_SILENCE,
        keep_pause=DEFAULT_KEEP_PAUSE,
        energy_threshold=DEFAULT_ENERGY_THRESHOLD,
        engine="auto",
        vad_engine="hybrid",
        target_lufs=-14.0,
        stabilize_engine="gpu",
        overwrite=False,
        keep_temp=False,
    )
    process(args)


def build_parser():
    """Builds CLI argument parser."""
    p = argparse.ArgumentParser(description="Auto Cut & Polishing Tool")
    p.add_argument("--input", "-i", help="Path to input media file or URL")
    p.add_argument("--output", "-o", default=str(OUTPUT_DIR), help="Output directory")
    p.add_argument("--output-name", help="Custom output filename")
    p.add_argument("--cloud-only", "--no-local", action="store_true", dest="cloud_only", help="Upload to cloud only; do not save locally")
    p.add_argument("--remove-silence", action="store_true", help="Remove dead pauses and silence")
    p.add_argument("--min-silence", type=float, default=DEFAULT_MIN_SILENCE)
    p.add_argument("--keep-pause", type=float, default=DEFAULT_KEEP_PAUSE)
    p.add_argument("--energy-threshold", type=float, default=DEFAULT_ENERGY_THRESHOLD)
    p.add_argument("--audio-enhance", action="store_true", help="AI audio de-noise and clarity")
    p.add_argument("--dereverb", action="store_true", help="Echo & room reverb removal")
    p.add_argument("--normalize", action="store_true", help="EBU R128 loudness normalization")
    p.add_argument("--multi-speaker-balance", action="store_true", help="Equalize soft and loud speakers")
    p.add_argument("--stabilize", action="store_true", help="GPU video camera stabilization")
    p.add_argument("--video-enhance", action="store_true", help="Video Enhancement (Fast GPU Tensor Cinema Polish)")
    p.add_argument("--video-engine", choices=["auto", "tensor", "vulkan"], default="auto", help="Video engine: auto/tensor (fast natural cinema polish) or vulkan (RealESRGAN AI upscaler)")
    p.add_argument("--extract-thumbnails", action="store_true", help="Extract best AI expressive thumbnails")
    p.add_argument("--all", action="store_true", help="Enable all features: auto-cut, audio enhance, dereverb, normalize, multi-speaker balance, stabilize, video polish, thumbnail extraction")
    p.add_argument("--engine", choices=["auto", "clearvoice", "voicefixer"], default="auto")
    p.add_argument("--vad-engine", choices=["hybrid", "silero", "energy"], default="hybrid")
    p.add_argument("--target-lufs", type=float, default=-14.0)
    p.add_argument("--stabilize-engine", choices=["gpu", "cpu"], default="gpu")
    p.add_argument("--download", action="store_true", help="Upload / trigger download")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    p.add_argument("--keep-temp", action="store_true", help="Keep scratch temp files")
    p.add_argument("--temp", help="Custom temp folder directory")
    return p


def main():
    """Main CLI / Interactive router."""
    parser = build_parser()
    args = parser.parse_args()

    if getattr(args, "all", False):
        args.remove_silence = True
        args.audio_enhance = True
        args.dereverb = True
        args.normalize = True
        args.multi_speaker_balance = True
        args.stabilize = True
        args.video_enhance = True
        args.extract_thumbnails = True

    if not args.input:
        interactive()
    else:
        if getattr(args, "cloud_only", False):
            args.output = None
            args.download = True
        process(args)


if __name__ == "__main__":
    main()
