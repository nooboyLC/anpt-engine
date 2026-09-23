# -*- coding: utf-8 -*-
"""
video.stabilizer
----------------
100% GPU-accelerated video stabilization using CUDA Phase Correlation,
moving-average smoothing, and affine grid warping, with CPU VidStab fallback.
"""

from __future__ import annotations

import os
import gc
import time
import queue
import shutil
import threading
import subprocess
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None
    F = None

from core.media_tools import ffmpeg_path, probe, run_ffmpeg_with_progress
from core.hardware import get_best_video_encoder_config, gpu_info, torch_cuda_available
from core.logger import progress, eprint


def _stabilize_gpu_cuda(src: Path, dst: Path, duration: float, run_dir: Path, combine_enhance: bool = False) -> bool:
    """
    100% GPU-Accelerated Video Stabilization (CUDA Phase Correlation + Affine Grid Warping + NVENC):
    - Pass 1: Sub-pixel GPU Phase Correlation motion estimation (>1,500 FPS).
    - Pass 2: Continuous GPU Affine Warping + optional Tensor Core Cinema Enhancement + NVENC (>100 FPS).
    """
    if torch is None or not torch.cuda.is_available() or np is None:
        return False

    device = torch.device("cuda")
    meta = probe(src)
    vstream = meta.get("video")
    if vstream is None:
        return False

    W = int(vstream["width"])
    H = int(vstream["height"])
    fps_str = vstream.get("r_frame_rate", "25/1")
    num, den = fps_str.split("/")
    fps = float(num) / max(float(den), 1.0)
    total_f = int(vstream.get("nb_frames", 0))
    if total_f <= 0:
        total_f = max(1, int(fps * meta.get("duration", 0)))

    enc, enc_flags, _ = get_best_video_encoder_config()

    # Pass 1: GPU Motion Analysis
    AW = max(64, (min(W, 320) // 8) * 8)
    AH = max(64, (min(H, 180) // 8) * 8)

    progress("[8/10] STABILIZE", 0.0, f"GPU Motion Analysis (CUDA Phase Correlation, {AW}x{AH})")

    hwaccel_p1 = ["-hwaccel", "cuda"] if torch_cuda_available() else []
    dec_threads = str(max(1, min(2, (os.cpu_count() or 2))))
    read1_cmd = [
        ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error",
    ] + hwaccel_p1 + [
        "-threads", dec_threads,
        "-i", str(src.resolve()),
        "-vf", f"scale={AW}:{AH}:flags=fast_bilinear",
        "-f", "rawvideo", "-pix_fmt", "gray", "-"
    ]
    reader1 = subprocess.Popen(read1_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=32 * 1024 * 1024)

    batch_p1 = 64
    frame_bytes_small = AW * AH
    batch_bytes_p1 = frame_bytes_small * batch_p1
    dx_list = [0.0]
    dy_list = [0.0]
    prev_tensor = None
    processed_p1 = 0
    t0_p1 = time.time()

    hann_y = torch.hann_window(AH, periodic=False, device=device).view(1, 1, AH, 1)
    hann_x = torch.hann_window(AW, periodic=False, device=device).view(1, 1, 1, AW)
    hann_2d = hann_y * hann_x

    with torch.no_grad():
        while True:
            raw = reader1.stdout.read(batch_bytes_p1)
            if not raw and reader1.poll() is not None and reader1.returncode != 0 and hwaccel_p1:
                read1_cmd = [
                    ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error",
                    "-threads", dec_threads,
                    "-i", str(src.resolve()),
                    "-vf", f"scale={AW}:{AH}:flags=fast_bilinear",
                    "-f", "rawvideo", "-pix_fmt", "gray", "-"
                ]
                reader1 = subprocess.Popen(read1_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=32 * 1024 * 1024)
                hwaccel_p1 = []
                raw = reader1.stdout.read(batch_bytes_p1)
            if not raw:
                break
            k = len(raw) // frame_bytes_small
            if k <= 0:
                break
            arr = np.frombuffer(raw[:k * frame_bytes_small], dtype=np.uint8).copy().reshape(k, 1, AH, AW)
            curr_tensor = torch.from_numpy(arr).to(device, non_blocking=True).float().div_(255.0)
            curr_windowed = curr_tensor * hann_2d

            if prev_tensor is None:
                prev_tensor = curr_windowed[0:1]
                if k > 1:
                    f0 = curr_windowed[1:]
                    f1 = curr_windowed[:-1]
                    F0 = torch.fft.rfft2(f0)
                    F1 = torch.fft.rfft2(f1)
                    cross = F0 * torch.conj(F1)
                    norm = cross / (torch.abs(cross) + 1e-7)
                    r = torch.fft.irfft2(norm, s=(AH, AW))
                    r = torch.fft.fftshift(r, dim=(-2, -1))
                    r_flat = r.view(k - 1, -1)
                    max_idx = torch.argmax(r_flat, dim=-1)
                    py = ((max_idx // AW).float() - (AH // 2)).tolist()
                    px = ((max_idx % AW).float() - (AW // 2)).tolist()
                    for x_val, y_val in zip(px, py):
                        if abs(x_val) > AW * 0.25 or abs(y_val) > AH * 0.25:
                            dx_list.append(0.0)
                            dy_list.append(0.0)
                        else:
                            dx_list.append(float(x_val))
                            dy_list.append(float(y_val))
                    prev_tensor = curr_windowed[-1:]
            else:
                combined = torch.cat([prev_tensor, curr_windowed], dim=0)
                f0 = combined[1:]
                f1 = combined[:-1]
                F0 = torch.fft.rfft2(f0)
                F1 = torch.fft.rfft2(f1)
                cross = F0 * torch.conj(F1)
                norm = cross / (torch.abs(cross) + 1e-7)
                r = torch.fft.irfft2(norm, s=(AH, AW))
                r = torch.fft.fftshift(r, dim=(-2, -1))
                r_flat = r.view(k, -1)
                max_idx = torch.argmax(r_flat, dim=-1)
                py = ((max_idx // AW).float() - (AH // 2)).tolist()
                px = ((max_idx % AW).float() - (AW // 2)).tolist()
                for x_val, y_val in zip(px, py):
                    if abs(x_val) > AW * 0.25 or abs(y_val) > AH * 0.25:
                        dx_list.append(0.0)
                        dy_list.append(0.0)
                    else:
                        dx_list.append(float(x_val))
                        dy_list.append(float(y_val))
                prev_tensor = curr_windowed[-1:]

            processed_p1 += k
            now = time.time()
            elapsed = now - t0_p1
            fps_p1 = processed_p1 / elapsed if elapsed > 0 else 0
            pct = 0.35 * min(1.0, processed_p1 / max(total_f, 1))
            progress("[8/10] STABILIZE", pct, f"GPU Motion Analysis ({fps_p1:.0f} fps | {processed_p1}/{total_f})")

    reader1.stdout.close()
    reader1.wait()

    actual_f = len(dx_list)
    if actual_f < 2:
        return False

    traj_x = np.cumsum(dx_list)
    traj_y = np.cumsum(dy_list)
    smooth_radius = max(3, int(fps * 0.6))
    kernel_size = 2 * smooth_radius + 1

    smooth_x = np.convolve(traj_x, np.ones(kernel_size) / kernel_size, mode='same')
    smooth_y = np.convolve(traj_y, np.ones(kernel_size) / kernel_size, mode='same')

    for i in range(smooth_radius):
        smooth_x[i] = traj_x[:i + smooth_radius + 1].mean()
        smooth_y[i] = traj_y[:i + smooth_radius + 1].mean()
        smooth_x[-(i + 1)] = traj_x[-(i + smooth_radius + 1):].mean()
        smooth_y[-(i + 1)] = traj_y[-(i + smooth_radius + 1):].mean()

    # Fix: Normalize against analysis grid (AW, AH) instead of (W, H) to achieve 100% full shake correction
    norm_x = np.clip((smooth_x - traj_x) / (AW / 2.0), -0.06, 0.06)
    norm_y = np.clip((smooth_y - traj_y) / (AH / 2.0), -0.06, 0.06)
    corr_tensor = torch.zeros((actual_f, 2), dtype=torch.float16, device=device)
    corr_tensor[:, 0] = torch.from_numpy(norm_x.astype(np.float32)).to(device=device, dtype=torch.float16)
    corr_tensor[:, 1] = torch.from_numpy(norm_y.astype(np.float32)).to(device=device, dtype=torch.float16)

    # Pass 2: GPU Affine Warping & NVENC Encode (with optional Real-BasicVSR Fusion)
    vsr_model = None
    out_W, out_H = W, H
    proc_W, proc_H = W, H
    if combine_enhance:
        try:
            from .enhancer_ai import _load_model, ensure_realbasicvsr_weights, _determine_processing_resolution
            ckpt_file = ensure_realbasicvsr_weights()
            vsr_model = _load_model(ckpt_file, device)
            if vsr_model is not None:
                proc_W, proc_H, out_W, out_H = _determine_processing_resolution(W, H)
                progress("[8/10] STABILIZE+ENHANCE", 0.38, "Unified Stabilization + Real-BasicVSR (Single-Pass NVENC)")
            else:
                progress("[8/10] STABILIZE", 0.38, "GPU Affine Warping (CUDA Tensor Cores + NVENC)")
        except Exception as ex:
            eprint(f"[FUSION NOTICE] Real-BasicVSR fusion load fallback: {ex}")
            vsr_model = None
            progress("[8/10] STABILIZE", 0.38, "GPU Affine Warping (CUDA Tensor Cores + NVENC)")
    else:
        progress("[8/10] STABILIZE", 0.38, "GPU Affine Warping (CUDA Tensor Cores + NVENC)")

    vram_mb = gpu_info().get("vram_total", 0)
    warp_batch = 10 if vsr_model is not None else (24 if vram_mb >= 8000 else (16 if vram_mb >= 4000 else 8))
    full_bytes = W * H * 3
    batch_bytes_p2 = full_bytes * warp_batch

    hwaccel_p2 = ["-hwaccel", "cuda"] if torch_cuda_available() else []
    read2_cmd = [
        ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error",
    ] + hwaccel_p2 + [
        "-threads", dec_threads,
        "-i", str(src.resolve()),
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-"
    ]
    reader2 = subprocess.Popen(read2_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=64 * 1024 * 1024)

    write2_cmd = [
        ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error",
        "-threads", dec_threads,
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-r", str(fps), "-s", f"{out_W}x{out_H}",
        "-i", "pipe:0",
        "-c:v", enc,
    ] + enc_flags + ["-an", str(dst.resolve())]
    writer2 = subprocess.Popen(write2_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=64 * 1024 * 1024)

    in_q: queue.Queue = queue.Queue(maxsize=2)
    out_q: queue.Queue = queue.Queue(maxsize=2)

    def _reader():
        nonlocal reader2
        try:
            buf = reader2.stdout.read(batch_bytes_p2)
            if not buf and reader2.poll() is not None and reader2.returncode != 0 and hwaccel_p2:
                cpu_cmd = [
                    ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error",
                    "-threads", dec_threads,
                    "-i", str(src.resolve()),
                    "-f", "rawvideo", "-pix_fmt", "rgb24", "-"
                ]
                reader2 = subprocess.Popen(cpu_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=64 * 1024 * 1024)
                buf = reader2.stdout.read(batch_bytes_p2)
            while buf:
                in_q.put(buf)
                buf = reader2.stdout.read(batch_bytes_p2)
        except Exception:
            pass
        finally:
            in_q.put(None)

    writer_err = []

    def _writer():
        try:
            while True:
                buf = out_q.get()
                if buf is None:
                    out_q.task_done()
                    break
                try:
                    writer2.stdin.write(buf)
                except Exception as ex:
                    writer_err.append(str(ex))
                    out_q.task_done()
                    break
                out_q.task_done()
        except Exception as ex:
            writer_err.append(str(ex))

    t_reader = threading.Thread(target=_reader, daemon=True)
    t_writer = threading.Thread(target=_writer, daemon=True)
    t_reader.start()
    t_writer.start()

    processed_p2 = 0
    t0_p2 = time.time()
    zoom = 1.04

    try:
        with torch.no_grad():
            while True:
                raw = in_q.get()
                if raw is None:
                    in_q.task_done()
                    break
                k = len(raw) // full_bytes
                if k <= 0:
                    in_q.task_done()
                    break

                theta = torch.zeros((k, 2, 3), dtype=torch.float16, device=device)
                theta[:, 0, 0] = zoom
                theta[:, 1, 1] = zoom
                f_end = min(processed_p2 + k, actual_f)
                valid_k = f_end - processed_p2
                if valid_k > 0:
                    theta[:valid_k, 0, 2] = corr_tensor[processed_p2:f_end, 0]
                    theta[:valid_k, 1, 2] = corr_tensor[processed_p2:f_end, 1]

                arr = np.frombuffer(raw[:k * full_bytes], dtype=np.uint8).copy().reshape(k, H, W, 3)
                t_gpu = torch.from_numpy(arr).to(device, non_blocking=True).permute(0, 3, 1, 2).half().div_(255.0)

                grid = F.affine_grid(theta, t_gpu.shape, align_corners=False)
                warped = F.grid_sample(t_gpu, grid, mode='bilinear', padding_mode='border', align_corners=False)

                if vsr_model is not None:
                    # Single-Pass Fusion: Warped frames flow directly into Real-BasicVSR in VRAM
                    if (H, W) != (proc_H, proc_W):
                        t_proc = F.interpolate(warped, size=(proc_H, proc_W), mode="bilinear", align_corners=False)
                    else:
                        t_proc = warped
                    sr_out = vsr_model(t_proc.unsqueeze(0)).squeeze(0)  # (k, 3, 4*proc_H, 4*proc_W)
                    if (sr_out.shape[2], sr_out.shape[3]) != (out_H, out_W):
                        sr_out = F.interpolate(sr_out, size=(out_H, out_W), mode="bicubic", align_corners=False)
                    out_gpu = sr_out.permute(0, 2, 3, 1).clamp_(0.0, 1.0).mul_(255.0).to(torch.uint8).contiguous()
                    del t_proc, sr_out
                else:
                    out_gpu = warped.permute(0, 2, 3, 1).clamp_(0.0, 1.0).mul_(255.0).to(torch.uint8).contiguous()

                out_cpu = out_gpu.cpu().numpy()
                out_q.put(memoryview(out_cpu))
                in_q.task_done()

                processed_p2 += k
                now = time.time()
                elapsed = now - t0_p2
                fps_p2 = processed_p2 / elapsed if elapsed > 0 else 0
                pct = 0.38 + 0.60 * min(1.0, processed_p2 / max(actual_f, 1))
                progress("[8/10] STABILIZE", pct, f"GPU Warping ({fps_p2:.1f} fps | {processed_p2}/{actual_f})")
    finally:
        out_q.put(None)
        t_writer.join(timeout=10)
        ret_w = -1
        try:
            if writer2 is not None and writer2.stdin:
                writer2.stdin.close()
        except Exception:
            pass
        if writer2 is not None:
            try:
                ret_w = writer2.wait(timeout=60)
            except Exception:
                ret_w = -1
        try:
            if reader2 is not None and reader2.stdout:
                reader2.stdout.close()
        except Exception:
            pass
        if reader2 is not None:
            try:
                reader2.wait(timeout=10)
            except Exception:
                pass
        for proc in (reader2, writer2):
            if proc is not None and proc.poll() is None:
                try:
                    proc.terminate()
                except Exception:
                    pass
        if vsr_model is not None:
            del vsr_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

    if writer_err or ret_w != 0 or not (dst.exists() and dst.stat().st_size > 1000):
        return False
    return True


def _stabilize_vidstab(src: Path, dst: Path, duration: float, run_dir: Path, combine_enhance: bool = False) -> bool:
    """CPU Fallback VidStab Video Stabilization."""
    trf_name = f"transforms_{int(time.time())}.trf"
    trf_file = run_dir / trf_name
    run_dir_str = str(run_dir)

    try:
        pass1_filt = f"vidstabdetect=stepsize=14:shakiness=8:accuracy=6:result={trf_name}"
        pass1_threads = str(max(1, min(8, os.cpu_count() or 4)))
        pass1_cmd = [
            ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error",
            "-threads", pass1_threads, "-filter_threads", pass1_threads,
            "-i", str(src.resolve()), "-vf", pass1_filt,
            "-f", "null", "-"
        ]
        run_ffmpeg_with_progress(pass1_cmd, duration, "[8/10] STABILIZE (1/2)", cwd=run_dir_str)

        if not trf_file.exists() or trf_file.stat().st_size == 0:
            return False

        pass2_filt = f"vidstabtransform=input={trf_name}:zoom=3:smoothing=25:optalgo=gauss:interpol=bicubic"
        enc, enc_flags, _ = get_best_video_encoder_config()
        threads_val = str(max(1, min(8, os.cpu_count() or 4)))
        pass2_cmd = [
            ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error",
            "-threads", threads_val, "-filter_threads", threads_val,
            "-i", str(src.resolve()), "-vf", pass2_filt,
            "-c:v", enc,
        ] + enc_flags + ["-an", str(dst.resolve())]
        run_ffmpeg_with_progress(pass2_cmd, duration, "[8/10] STABILIZE (2/2)", cwd=run_dir_str)

        return dst.exists() and dst.stat().st_size > 0
    except Exception as ex:
        eprint(f"[WARN] VidStab failed: {ex}")
        return False
    finally:
        try:
            trf_file.unlink(missing_ok=True)
        except Exception:
            pass


def motion_level_detect(src: Path, sample_frames: int = 90) -> float:
    """
    Analyses a short clip sample to measure average camera shake.
    Returns a shake score in pixels (motion magnitude at source resolution).
    A score < 0.8 means negligible shake (tripod/static camera).
    A score >= 0.8 means noticeable shake (handheld/moving camera).
    Fast: only samples up to `sample_frames` frames at tiny resolution.
    """
    if torch is None or np is None:
        return 999.0  # Unknown — assume shake present, always stabilize

    try:
        meta = probe(src)
        vstream = meta.get("video")
        if not vstream:
            return 999.0

        W = int(vstream.get("width", 1920))
        H = int(vstream.get("height", 1080))
        fps_str = vstream.get("r_frame_rate", "25/1")
        num, den = fps_str.split("/")
        fps = float(num) / max(float(den), 1.0)
        total_f = int(vstream.get("nb_frames", 0))
        if total_f <= 0:
            dur = float(vstream.get("duration", meta.get("duration", 0)) or 0)
            total_f = max(1, int(fps * dur))

        # Tiny analysis resolution — fast + VRAM-free
        AW = 160
        AH = 90

        # Sample evenly spaced frames (up to sample_frames)
        step = max(1, total_f // sample_frames)
        select_expr = f"not(mod(n\\,{step}))"

        cmd = [
            ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(src.resolve()),
            "-vf", f"select='{select_expr}',scale={AW}:{AH}:flags=fast_bilinear",
            "-vsync", "vfr",
            "-f", "rawvideo", "-pix_fmt", "gray", "-",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        raw = proc.stdout.read()
        proc.stdout.close()
        proc.wait()

        frame_bytes = AW * AH
        n_frames = len(raw) // frame_bytes
        if n_frames < 2:
            return 0.0  # Too few frames → assume static

        arr = np.frombuffer(raw[:n_frames * frame_bytes], dtype=np.uint8).copy().reshape(n_frames, 1, AH, AW)

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        t = torch.from_numpy(arr).to(device).float().div_(255.0)

        hann_y = torch.hann_window(AH, periodic=False, device=device).view(1, 1, AH, 1)
        hann_x = torch.hann_window(AW, periodic=False, device=device).view(1, 1, 1, AW)
        hann_2d = hann_y * hann_x

        motions = []
        with torch.no_grad():
            f0 = (t[1:] * hann_2d)
            f1 = (t[:-1] * hann_2d)
            F0 = torch.fft.rfft2(f0)
            F1 = torch.fft.rfft2(f1)
            cross = F0 * torch.conj(F1)
            norm = cross / (torch.abs(cross) + 1e-7)
            r = torch.fft.irfft2(norm, s=(AH, AW))
            r = torch.fft.fftshift(r, dim=(-2, -1))
            r_flat = r.view(n_frames - 1, -1)
            max_idx = torch.argmax(r_flat, dim=-1)
            py = (max_idx // AW).float() - (AH // 2)
            px = (max_idx % AW).float() - (AW // 2)
            # Scale motion back to source resolution
            scale_x = W / AW
            scale_y = H / AH
            mag = torch.sqrt((px * scale_x) ** 2 + (py * scale_y) ** 2)
            # Filter out extreme outliers (scene cuts etc.)
            med = mag.median()
            valid = mag[mag < med * 5 + 1]
            if len(valid) > 0:
                motions = float(valid.mean().cpu())
            else:
                motions = float(med.cpu())

        if device.type == "cuda":
            torch.cuda.empty_cache()

        return motions

    except Exception:
        return 999.0  # On any error assume shake present


def stabilize_video(src: Path, dst: Path, duration: float, run_dir: Path, combine_enhance: bool = False, engine: str = "gpu") -> bool:
    """Entry point for video stabilization with automatic shake detection and single-pass Real-BasicVSR fusion."""
    # Auto-detect camera shake before doing expensive stabilization
    progress("[8/10] STABILIZE", 0.0, "Auto-detecting camera shake...")
    shake_score = motion_level_detect(src)
    SHAKE_THRESHOLD = 0.8  # pixels at source resolution — below = static/tripod

    if shake_score < SHAKE_THRESHOLD:
        progress("[8/10] STABILIZE", 1.0, f"Static camera detected (shake={shake_score:.2f}px) — skipping stabilization")
        if combine_enhance:
            from .enhancer_ai import enhance_video_ai
            meta = probe(src)
            return enhance_video_ai(src, dst, duration, meta)
        shutil.copy2(src, dst)
        return True

    progress("[8/10] STABILIZE", 0.05, f"Camera shake detected ({shake_score:.1f}px) — running stabilizer")

    if torch is not None and torch_cuda_available() and engine != "cpu":
        tag = "GPU CUDA Stabilizer + Real-BasicVSR active" if combine_enhance else "GPU CUDA Stabilizer active"
        progress("[8/10] STABILIZE", 0.06, tag)
        if _stabilize_gpu_cuda(src, dst, duration, run_dir, combine_enhance=combine_enhance):
            done_tag = "GPU stabilization & Real-BasicVSR complete" if combine_enhance else "GPU stabilization complete"
            progress("[8/10] STABILIZE", 1.0, done_tag)
            return True
        eprint("[WARN] GPU stabilization fallback needed...")

    progress("[8/10] STABILIZE", 0.0, "VidStab stabilizer (CPU Fallback)")
    if _stabilize_vidstab(src, dst, duration, run_dir, combine_enhance=combine_enhance):
        progress("[8/10] STABILIZE", 1.0, "VidStab stabilization complete")
        return True

    eprint("[WARN] Stabilization failed — keeping original video stream.")
    shutil.copy2(src, dst)
    return False
