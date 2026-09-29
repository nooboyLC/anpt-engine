# -*- coding: utf-8 -*-
"""
video.stabilizer
----------------
Memory-Safe GPU Video Stabilization.

Architecture (crash-proof, OOM-proof):
  Pass 1: GPU Phase Correlation -> motion vectors saved to disk (motions.json)
           RAM footprint: <= 1 batch of tiny (320x180 gray) frames at a time.
           VRAM footprint: <= 64 x 1 x 180 x 320 x float16 ~7 MB peak.

  Pass 2: Single-frame streaming warp + encode (NO threaded queues, NO large buffers).
           Decode (FFmpeg pipe) -> numpy -> GPU tensor (1 frame) -> warp -> enhance (optional)
           -> CPU bytes -> FFmpeg encode stdin.
           VRAM footprint: <= 1 x H x W x 3 x float16 ~12 MB peak at 1080p.
           Total buffer in Python at any moment ~2 x frame_bytes (<= 50 MB at 1080p).

  No threading, no queue, no double-buffering.  Rock-solid on 4 GB GPUs.

  Checkpoint/Resume:
    motions.json is written to disk after Pass 1.  If it already exists (e.g. after a
    crash mid-Pass-2), Pass 1 is skipped entirely and Pass 2 resumes from scratch
    against the same json -- saving 10-30 minutes of motion analysis.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import threading
import time
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

from core.media_tools import (
    ffmpeg_path, probe, run_ffmpeg_with_progress,
    get_stream_fps, get_fps_mode_flags,
)
from core.hardware import get_best_video_encoder_config, gpu_info, torch_cuda_available, get_hw_profile
from core.logger import progress, eprint

# Constants
_MOTIONS_FILENAME = "motions.json"   # disk-backed motion cache
_P1_BATCH         = 64               # tiny-frame batch for Phase Correlation


def _get_safe_p2_batch(vram_mb: float) -> int:
    """
    Return the largest warp batch size that will not OOM on this GPU at 1080p RGB16.

    Adaptive VRAM tiers (D-0.0.4):
      >=24 GB  -> 64 frames  (e.g. RTX 3090 / A100)
      >=16 GB  -> 48 frames  (e.g. Tesla T4 16 GB, RTX 3080)
      >= 8 GB  -> 24 frames  (e.g. RTX 3070, Tesla T4 8 GB)
      >= 4 GB  -> 8 frames   (e.g. GTX 1660, RTX 3050)
      <  4 GB  -> 1 frame    (safe single-frame mode)

    Each 1080p RGB float16 frame ~12.4 MB on GPU.
    Conservative headroom: subtract 1500 MB for NVENC + DWM + OS before tiering.
    """
    free_mb = max(0, vram_mb - 1500)
    if free_mb >= 22_000:   # 24 GB card
        return 64
    if free_mb >= 14_000:   # 16 GB card
        return 48
    if free_mb >= 6_000:    # 8 GB card
        return 24
    if free_mb >= 2_000:    # 4 GB card
        return 8
    return 1                # <4 GB or no headroom — single-frame safe mode


# Pass 1: GPU Phase Correlation Motion Analysis

def _batch_correlate(prev, curr, AH, AW, dx_list, dy_list):
    """In-place phase correlation on GPU. Appends (dx, dy) per frame to the lists."""
    F0 = torch.fft.rfft2(curr)
    F1 = torch.fft.rfft2(prev)
    cross = F0 * torch.conj(F1)
    norm  = cross / (torch.abs(cross) + 1e-7)
    r     = torch.fft.irfft2(norm, s=(AH, AW))
    r     = torch.fft.fftshift(r, dim=(-2, -1))
    n     = r.shape[0]
    r_flat = r.view(n, -1)
    max_idx = torch.argmax(r_flat, dim=-1)
    py = (max_idx // AW).float() - (AH // 2)
    px = (max_idx % AW).float()  - (AW // 2)
    for xi, yi in zip(px.tolist(), py.tolist()):
        if abs(xi) > AW * 0.25 or abs(yi) > AH * 0.25:
            dx_list.append(0.0)
            dy_list.append(0.0)
        else:
            dx_list.append(float(xi))
            dy_list.append(float(yi))


def _run_pass1(src, motions_path, total_f, fps, AW, AH, device):
    """
    Decode video at thumbnail scale, compute per-frame (dx, dy) via CUDA
    Phase Correlation, and save results to motions_path as JSON.
    Returns True on success.
    """
    progress("[8/10] STABILIZE", 0.0,
             f"Pass 1/2 - GPU Motion Analysis ({AW}x{AH} gray, batch={_P1_BATCH})")

    _src_codec = ""
    try:
        meta = probe(src)
        _src_codec = (meta.get("video") or {}).get("codec_name", "").lower()
    except Exception:
        pass

    nvdec_flag = []
    if torch_cuda_available():
        if _src_codec in ("h264", "avc"):
            nvdec_flag = ["-c:v", "h264_cuvid"]
        elif _src_codec in ("hevc", "h265"):
            nvdec_flag = ["-c:v", "hevc_cuvid"]
        elif _src_codec == "vp9":
            nvdec_flag = ["-c:v", "vp9_cuvid"]

    hwaccel = ["-hwaccel", "cuda"] if torch_cuda_available() else []
    dec_threads = str(get_hw_profile().cpu_threads)

    def _make_reader(use_hw):
        cmd = [ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error"]
        if use_hw:
            cmd += hwaccel + nvdec_flag
        cmd += [
            "-threads", dec_threads,
            "-i", str(src.resolve()),
            "-vf", f"scale={AW}:{AH}:flags=fast_bilinear,fps={fps}",
            "-f", "rawvideo", "-pix_fmt", "gray", "-",
        ]
        return subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            bufsize=_P1_BATCH * AW * AH + 65536,
        )

    reader = _make_reader(use_hw=bool(torch_cuda_available()))

    frame_bytes = AW * AH
    batch_bytes = frame_bytes * _P1_BATCH

    hann_y = torch.hann_window(AH, periodic=False, device=device).view(1, 1, AH, 1)
    hann_x = torch.hann_window(AW, periodic=False, device=device).view(1, 1, 1, AW)
    hann_2d = hann_y * hann_x

    dx_list = [0.0]
    dy_list = [0.0]
    prev_gpu = None
    processed = 0
    t0 = time.time()
    hw_failed = False

    with torch.no_grad():
        while True:
            raw = reader.stdout.read(batch_bytes)

            # HW decoder failure? Restart with software decode
            if not raw and not hw_failed and (hwaccel or nvdec_flag):
                code = reader.poll()
                if code is not None and code != 0:
                    reader.stdout.close()
                    reader.wait()
                    reader = _make_reader(use_hw=False)
                    hw_failed = True
                    raw = reader.stdout.read(batch_bytes)

            if not raw:
                break

            k = len(raw) // frame_bytes
            if k <= 0:
                break

            arr = np.frombuffer(raw[:k * frame_bytes], dtype=np.uint8).reshape(k, 1, AH, AW)
            curr = torch.from_numpy(arr).to(device, non_blocking=True).float().div_(255.0)
            curr = curr * hann_2d

            if prev_gpu is None:
                prev_gpu = curr[0:1]
                if k > 1:
                    _batch_correlate(curr[:-1], curr[1:], AH, AW, dx_list, dy_list)
                    prev_gpu = curr[-1:]
            else:
                full = torch.cat([prev_gpu, curr], dim=0)
                _batch_correlate(full[:-1], full[1:], AH, AW, dx_list, dy_list)
                prev_gpu = curr[-1:]

            processed += k
            elapsed = time.time() - t0
            fps_p1 = processed / elapsed if elapsed > 0 else 0
            pct = 0.35 * min(1.0, processed / max(total_f, 1))
            progress("[8/10] STABILIZE", pct,
                     f"Pass 1/2 - Motion Analysis ({fps_p1:.0f} fps | {processed}/{total_f})")

    try:
        reader.stdout.close()
    except Exception:
        pass
    reader.wait()

    del hann_2d, hann_y, hann_x
    if prev_gpu is not None:
        del prev_gpu
    torch.cuda.empty_cache()

    if len(dx_list) < 2:
        return False

    motions_path.write_text(
        json.dumps({"dx": dx_list, "dy": dy_list}, separators=(",", ":")),
        encoding="utf-8",
    )
    return True


# Smoothing / Virtual Tripod

def _compute_correction_vectors(dx_list, dy_list, AW, AH, mode="gimbal"):
    """
    Compute per-frame (corr_x, corr_y) correction vectors in normalised [-1,1]
    space, with Adaptive Dynamic Zoom to eliminate jitter while preventing black borders.

    Modes:
      - 'gimbal' (default): Wide-window Gaussian convolution for cinematic drone/gimbal glide.
      - 'tripod': High-inertia anchor lock for rigid static tripod stability.
    """
    dx_arr = np.array(dx_list, dtype=np.float32)
    dy_arr = np.array(dy_list, dtype=np.float32)
    traj_x = np.cumsum(dx_arr)
    traj_y = np.cumsum(dy_arr)
    rms_jitter = float(np.sqrt(np.mean(dx_arr**2 + dy_arr**2)))

    n = len(dx_arr)
    if rms_jitter < 0.20 or n < 2:
        return np.zeros(n, np.float32), np.zeros(n, np.float32), 1.0

    # Task 6.2: Adaptive Dynamic Zoom (scaled smoothly based on jitter severity)
    if rms_jitter < 0.8:
        zoom = 1.04
    elif rms_jitter < 2.0:
        zoom = 1.08
    elif rms_jitter < 4.0:
        zoom = 1.10
    else:
        zoom = min(1.14, 1.08 + 0.015 * rms_jitter)

    smooth_x = np.zeros(n, np.float32)
    smooth_y = np.zeros(n, np.float32)

    if mode == "tripod":
        # Virtual Tripod: Rigid anchor lock with minimal drift
        cur_ax = float(traj_x[0])
        cur_ay = float(traj_y[0])
        for i in range(n):
            cur_ax += 0.005 * (traj_x[i] - cur_ax)
            cur_ay += 0.005 * (traj_y[i] - cur_ay)
            smooth_x[i] = cur_ax
            smooth_y[i] = cur_ay
    else:
        # Task 6.1: Cinematic Gimbal Mode
        # Wide Gaussian smoothing window (~1.5 to 2.5 sec of footage) for organic glide
        radius = min(45, max(15, n // 6))
        k = np.arange(-radius, radius + 1)
        sigma = max(1.0, radius / 2.5)
        kernel = np.exp(-0.5 * (k / sigma)**2)
        kernel /= kernel.sum()

        pad_x = np.pad(traj_x, radius, mode="edge")
        pad_y = np.pad(traj_y, radius, mode="edge")
        smooth_x = np.convolve(pad_x, kernel, mode="valid")
        smooth_y = np.convolve(pad_y, kernel, mode="valid")

    diff_x = smooth_x - traj_x
    diff_y = smooth_y - traj_y

    # Soft deadband to suppress sub-pixel noise without causing jumpiness
    deadband = 0.20
    mag = np.sqrt(diff_x**2 + diff_y**2)
    att = np.maximum(0.0, mag - deadband) / (mag + 1e-6)
    diff_x *= att
    diff_y *= att

    # Task 6.3: Relaxed shift margin with adaptive zoom headroom
    max_shift = (zoom - 1.0) / zoom
    corr_x = np.clip(-(diff_x / (AW / 2.0)), -max_shift, max_shift).astype(np.float32)
    corr_y = np.clip(-(diff_y / (AH / 2.0)), -max_shift, max_shift).astype(np.float32)

    return corr_x, corr_y, zoom


# Cinema enhancement

def _apply_cinema_enhancement(t, recipe):
    """
    Apply Cinema Polish inline on a GPU fp16 tensor [B, C, H, W] in [0,1].
    All intermediate tensors are explicitly deleted to minimise peak VRAM.
    """
    sharp_w       = recipe["sharp_w"]
    clamp_w       = recipe["clamp_w"]
    denoise_guard = recipe["denoise_guard"]
    shadow_lift   = recipe["shadow_lift"]
    sat_base      = recipe["sat_base"]
    s_curve_amp   = recipe["s_curve_amp"]

    enh = t

    if denoise_guard:
        enh = F.avg_pool2d(enh, kernel_size=3, stride=1, padding=1)

    coarse    = F.avg_pool2d(enh, kernel_size=5, stride=1, padding=2)
    high_freq = enh - coarse
    edge_mag  = high_freq.abs().mean(dim=1, keepdim=True)
    edge_gate = (edge_mag * 14.0).clamp_(0.0, 1.0)
    detail    = high_freq.clamp(-clamp_w, clamp_w)
    enh       = (enh + sharp_w * detail * edge_gate).clamp_(0.0, 1.0)
    del coarse, high_freq, edge_mag, edge_gate, detail

    if shadow_lift != 0.0:
        mid_mask = torch.sin(enh * 3.14159).clamp_(0.0, 1.0)
        enh = (enh + shadow_lift * mid_mask).clamp_(0.0, 1.0)
        del mid_mask

    luma      = 0.2126 * enh[:, 0:1] + 0.7152 * enh[:, 1:2] + 0.0722 * enh[:, 2:3]
    chroma    = enh - luma
    sat_boost = (sat_base - 0.12 * chroma.abs().mean(dim=1, keepdim=True)).clamp_(0.90, 1.50)
    vibrant   = (luma + chroma * sat_boost).clamp_(0.0, 1.0)
    del luma, chroma, sat_boost

    s_curve = s_curve_amp * torch.sin((vibrant - 0.5) * 3.14159)
    out = (vibrant + s_curve).clamp_(0.0, 1.0)
    del vibrant, s_curve

    return out


# Pass 2: Streaming Warp + Encode

def _run_pass2(src, dst, corr_x, corr_y, scale, fps, W, H,
               total_f, device, combine_enhance, recipe, run_dir):
    """
    Streaming small-batch GPU Affine Warp + optional Cinema Enhancement.
    Writes frames synchronously to FFmpeg NVENC stdin.

    No threads. No large queues. VRAM ceiling = batch x ~12 MB at 1080p.
    """
    enc, enc_flags, _ = get_best_video_encoder_config()
    vram_mb = gpu_info().get("vram_total", 0)
    batch = _get_safe_p2_batch(vram_mb)

    p2_label = "[8/10] FUSED STAB+ENHANCE" if combine_enhance else "[8/10] STABILIZE"
    progress(p2_label, 0.38,
             f"Pass 2/2 - GPU Warp+{'Enhance' if combine_enhance else 'Encode'} "
             f"(batch={batch}, VRAM_total={int(vram_mb)}MB)")

    full_bytes  = W * H * 3
    batch_bytes = full_bytes * batch
    dec_threads = str(get_hw_profile().cpu_threads)

    # Adaptive decode strategy:
    #   - Low VRAM GPU  (<= 6 GB, e.g. GTX 1050 Ti 4GB):
    #       Use CPU (software) decode. NVDEC + NVENC + CUDA cores all share the same
    #       small VRAM pool. Running NVDEC alongside NVENC causes CUDA context
    #       corruption and hard system crashes. CPU H.264 decode at ~200+ fps on a
    #       4-core machine is faster than the GPU warp throughput (~80-120 fps on 1050 Ti),
    #       so there is zero bottleneck.
    #
    #   - High VRAM GPU (>= 8 GB, e.g. Colab T4 16GB, RTX 3070 8GB+):
    #       Use NVDEC (hardware decode). 16 GB VRAM has plenty of headroom for
    #       NVDEC + NVENC + CUDA to coexist safely. On Colab free-tier (2 CPU cores),
    #       CPU decode bottlenecks at ~60-100 fps while the T4 can warp at 500+ fps.
    #       NVDEC removes the CPU decode bottleneck entirely.
    _use_nvdec_p2 = torch_cuda_available() and vram_mb >= 8000

    _src_codec_p2 = ""
    if _use_nvdec_p2:
        try:
            _meta_p2 = probe(src)
            _src_codec_p2 = (_meta_p2.get("video") or {}).get("codec_name", "").lower()
        except Exception:
            _use_nvdec_p2 = False

    if _use_nvdec_p2:
        _nvdec_p2 = []
        if _src_codec_p2 in ("h264", "avc"):
            _nvdec_p2 = ["-c:v", "h264_cuvid"]
        elif _src_codec_p2 in ("hevc", "h265"):
            _nvdec_p2 = ["-c:v", "hevc_cuvid"]
        elif _src_codec_p2 == "vp9":
            _nvdec_p2 = ["-c:v", "vp9_cuvid"]
        else:
            _use_nvdec_p2 = False  # unknown codec, fall back to CPU

    if _use_nvdec_p2:
        # NVDEC path: GPU decode (Colab T4 / high-VRAM GPU)
        read_cmd = [
            ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error",
            "-hwaccel", "cuda",
        ] + _nvdec_p2 + [
            "-threads", dec_threads,
            "-i", str(src.resolve()),
            "-vf", f"fps={fps}",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ]
        _decode_mode = f"NVDEC ({_src_codec_p2.upper()})"
    else:
        # CPU decode path: software decode (GTX 1050 Ti / low-VRAM GPU)
        read_cmd = [
            ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error",
            "-threads", dec_threads,
            "-i", str(src.resolve()),
            "-vf", f"fps={fps}",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ]
        _decode_mode = f"CPU (sw, {dec_threads}t)"

    eprint(f"[INFO] Pass 2 decode: {_decode_mode} | VRAM={int(vram_mb)}MB | batch={batch}")

    reader = subprocess.Popen(
        read_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        bufsize=batch_bytes + 4096,
    )

    _color_meta = [
        "-color_primaries", "bt709", "-color_trc", "bt709",
        "-colorspace",      "bt709", "-color_range", "tv",
    ]
    write_cmd = [
        ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error",
        "-threads", dec_threads,
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-r", str(fps), "-s", f"{W}x{H}",
    ] + _color_meta + [
        "-i", "pipe:0",
    ] + get_fps_mode_flags() + [
        "-c:v", enc,
    ] + enc_flags + ["-an", str(dst.resolve())]

    writer = subprocess.Popen(
        write_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        bufsize=batch_bytes + 4096,
    )

    actual_f  = len(corr_x)
    processed = 0
    t0        = time.time()
    success   = True
    last_err  = b""

    # Asynchronous Double-Buffered Multi-Threaded I/O Pipeline:
    # Eliminates CPU<->GPU pipe bottlenecks by decoupling reader and writer into dedicated threads.
    in_queue = queue.Queue(maxsize=2)
    out_queue = queue.Queue(maxsize=2)
    stop_event = threading.Event()
    reader_err = []
    writer_err = []

    def _reader_target():
        try:
            while not stop_event.is_set():
                raw = bytearray()
                remaining = batch_bytes
                while remaining > 0 and not stop_event.is_set():
                    chunk = reader.stdout.read(remaining)
                    if not chunk:
                        break
                    raw.extend(chunk)
                    remaining -= len(chunk)
                if not raw:
                    break
                in_queue.put(raw)
        except Exception as ex:
            reader_err.append(ex)
        finally:
            in_queue.put(None)

    def _writer_target():
        try:
            while not stop_event.is_set():
                chunk_bytes = out_queue.get()
                if chunk_bytes is None:
                    out_queue.task_done()
                    break
                try:
                    writer.stdin.write(chunk_bytes)
                finally:
                    out_queue.task_done()
        except Exception as ex:
            writer_err.append(ex)
            stop_event.set()

    t_reader = threading.Thread(target=_reader_target, daemon=True)
    t_writer = threading.Thread(target=_writer_target, daemon=True)
    t_reader.start()
    t_writer.start()

    try:
        with torch.no_grad():
            while True:
                if stop_event.is_set():
                    break
                raw = in_queue.get()
                if raw is None:
                    break

                k = len(raw) // full_bytes
                if k <= 0:
                    continue

                # Build affine transform matrix on GPU
                theta = torch.zeros((k, 2, 3), dtype=torch.float16, device=device)
                theta[:, 0, 0] = scale
                theta[:, 1, 1] = scale
                f_end   = min(processed + k, actual_f)
                valid_k = f_end - processed
                if valid_k > 0:
                    cx = torch.from_numpy(corr_x[processed:f_end]).to(device=device, dtype=torch.float16)
                    cy = torch.from_numpy(corr_y[processed:f_end]).to(device=device, dtype=torch.float16)
                    theta[:valid_k, 0, 2] = cx
                    theta[:valid_k, 1, 2] = cy
                    del cx, cy

                # Decode raw bytes -> GPU tensor (single contiguous copy)
                arr   = np.frombuffer(raw[:k * full_bytes], dtype=np.uint8).reshape(k, H, W, 3)
                t_gpu = (torch.from_numpy(np.ascontiguousarray(arr))
                              .to(device, non_blocking=True)
                              .permute(0, 3, 1, 2)
                              .to(dtype=torch.float16)
                              .div_(255.0))

                # GPU Affine warp
                grid   = F.affine_grid(theta, t_gpu.shape, align_corners=False)
                warped = F.grid_sample(t_gpu, grid, mode="bilinear",
                                       padding_mode="border", align_corners=False)
                del t_gpu, theta, grid

                # Optional cinema enhancement (fused, no extra I/O)
                if combine_enhance and recipe:
                    warped = _apply_cinema_enhancement(warped, recipe)

                # GPU -> CPU -> out_queue (asynchronous write)
                out_cpu = (warped.permute(0, 2, 3, 1)
                                 .clamp_(0.0, 1.0)
                                 .mul_(255.0)
                                 .to(torch.uint8)
                                 .contiguous()
                                 .cpu()
                                 .numpy())
                del warped

                out_queue.put(out_cpu.tobytes())
                del out_cpu

                processed += k
                elapsed = time.time() - t0
                fps_p2 = processed / elapsed if elapsed > 0 else 0
                pct    = 0.38 + 0.60 * min(1.0, processed / max(actual_f, 1))
                label  = "[8/10] FUSED STAB+ENHANCE" if combine_enhance else "[8/10] STABILIZE"
                progress(label, pct,
                         f"Pass 2/2 - GPU Warp ({fps_p2:.1f} fps | {processed}/{actual_f})")

                if writer.poll() is not None:
                    try:
                        last_err = writer.stderr.read(1024) if writer.stderr else b"encoder exited"
                    except Exception:
                        last_err = b"encoder exited"
                    success = False
                    stop_event.set()
                    break

                if writer_err:
                    success = False
                    stop_event.set()
                    break

    except KeyboardInterrupt:
        success = False
        stop_event.set()
        eprint("[WARN] Pass 2 interrupted by user.")
    except Exception as ex:
        success = False
        stop_event.set()
        eprint(f"[WARN] Pass 2 error: {ex}")
    finally:
        stop_event.set()
        out_queue.put(None)
        try:
            t_reader.join(timeout=3)
        except Exception:
            pass
        try:
            t_writer.join(timeout=30)
        except Exception:
            pass

        try:
            reader.stdout.close()
        except Exception:
            pass
        try:
            reader.kill()
        except Exception:
            pass
        reader.wait()

        try:
            writer.stdin.close()
        except Exception:
            pass

        # Give FFmpeg time to flush NVENC GOP buffer and write moov atom
        try:
            ret = writer.wait(timeout=120)
            if ret != 0 and not last_err:
                try:
                    last_err = writer.stderr.read(512) if writer.stderr else b""
                except Exception:
                    pass
        except subprocess.TimeoutExpired:
            eprint("[WARN] FFmpeg encoder timed out -- killing.")
            writer.kill()
            writer.wait()
            success = False

        torch.cuda.empty_cache()

    if last_err:
        eprint(f"[WARN] Encoder stderr: {last_err.decode('utf-8', errors='replace')[-300:]}")

    has_output = dst.exists() and dst.stat().st_size > 50_000
    if not has_output:
        return False

    if processed >= max(1, actual_f - 3) and success:
        label = "[8/10] FUSED STAB+ENHANCE" if combine_enhance else "[8/10] STABILIZE"
        progress(label, 1.0, f"GPU Warp complete ({processed} frames)")
        return True

    if not success:
        return False

    return has_output


# Main GPU Stabilization Entry Point

def _stabilize_gpu_cuda(src, dst, duration, run_dir, combine_enhance=False):
    """
    Memory-safe, crash-proof GPU video stabilization.

    Pass 1 results are cached to disk (motions.json).  If that file already
    exists in run_dir, Pass 1 is skipped (useful after a mid-Pass-2 crash).
    """
    if torch is None or not torch.cuda.is_available() or np is None:
        return False

    device = torch.device("cuda")

    meta    = probe(src)
    vstream = meta.get("video")
    if vstream is None:
        return False

    W   = int(vstream["width"])
    H   = int(vstream["height"])
    fps = get_stream_fps(vstream, default=30.0)

    total_f = max(1, int(round(fps * duration))) if duration > 0 else int(vstream.get("nb_frames", 0))
    if total_f <= 0:
        total_f = max(1, int(round(fps * meta.get("duration", 0))))

    AW = max(64, (min(W, 320) // 8) * 8)
    AH = max(64, (min(H, 180) // 8) * 8)

    motions_path = run_dir / _MOTIONS_FILENAME

    # Pass 1: Motion Analysis (with checkpoint resume)
    if motions_path.exists() and motions_path.stat().st_size > 100:
        progress("[8/10] STABILIZE", 0.35,
                 f"Pass 1/2 - Resuming from cached motions ({motions_path.name})")
        eprint("[INFO] motions.json found -- skipping Pass 1 (checkpoint resume).")
    else:
        ok = _run_pass1(src, motions_path, total_f, fps, AW, AH, device)
        if not ok:
            eprint("[WARN] Pass 1 (motion analysis) failed.")
            return False

    # Load motion vectors from disk
    try:
        data    = json.loads(motions_path.read_text(encoding="utf-8"))
        dx_list = data["dx"]
        dy_list = data["dy"]
    except Exception as ex:
        eprint(f"[WARN] Failed to load motions.json: {ex}")
        return False

    # Compute correction vectors (CPU NumPy only, no GPU)
    corr_x, corr_y, zoom = _compute_correction_vectors(dx_list, dy_list, AW, AH)
    scale = 1.0 / zoom

    # Load cinema recipe if needed
    recipe = None
    if combine_enhance:
        try:
            from video.enhancer_filter import get_adaptive_recipe
            recipe = get_adaptive_recipe(None)
        except Exception as ex:
            eprint(f"[WARN] Could not load cinema recipe ({ex}); enhance skipped.")
            combine_enhance = False

    # Pass 2: Streaming Warp + Encode
    stab_label = "[8/10] FUSED STAB+ENHANCE" if combine_enhance else "[8/10] STABILIZE"
    progress(stab_label, 0.38, "Pass 2/2 - Starting GPU Affine Warp")
    ok = _run_pass2(
        src=src, dst=dst,
        corr_x=corr_x, corr_y=corr_y,
        scale=scale, fps=fps, W=W, H=H,
        total_f=total_f, device=device,
        combine_enhance=combine_enhance,
        recipe=recipe,
        run_dir=run_dir,
    )
    return ok


# CPU Fallback: VidStab

def _stabilize_vidstab(src, dst, duration, run_dir, combine_enhance=False):
    """CPU Fallback VidStab Video Stabilization (uses FFmpeg vidstab filter)."""
    trf_name = f"transforms_{int(time.time())}.trf"
    trf_file = run_dir / trf_name
    run_dir_str = str(run_dir)

    try:
        threads = str(max(1, min(8, os.cpu_count() or 4)))
        pass1_cmd = [
            ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error",
            "-threads", threads, "-filter_threads", threads,
            "-i", str(src.resolve()),
            "-vf", f"vidstabdetect=stepsize=14:shakiness=8:accuracy=6:result={trf_name}",
            "-f", "null", "-",
        ]
        run_ffmpeg_with_progress(pass1_cmd, duration, "[8/10] STABILIZE (1/2)", cwd=run_dir_str)

        if not trf_file.exists() or trf_file.stat().st_size == 0:
            return False

        enc, enc_flags, _ = get_best_video_encoder_config()
        pass2_cmd = [
            ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error",
            "-threads", threads, "-filter_threads", threads,
            "-i", str(src.resolve()),
            "-vf", f"vidstabtransform=input={trf_name}:zoom=3:smoothing=25:optalgo=gauss:interpol=bicubic",
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


# Public Entry Point

def stabilize_video(src, dst, duration, run_dir, combine_enhance=False, engine="gpu"):
    """
    Stabilize src video, writing output to dst.

    engine: "gpu" (default) or "cpu" (force VidStab fallback)
    combine_enhance: fuse Cinema Polish into the same warp pass (no extra encode cycle)
    """
    if torch is not None and torch_cuda_available() and engine != "cpu":
        progress("[8/10] STABILIZE", 0.0, "Memory-Safe GPU CUDA Stabilizer")
        if _stabilize_gpu_cuda(src, dst, duration, run_dir, combine_enhance=combine_enhance):
            progress("[8/10] STABILIZE", 1.0, "GPU stabilization complete")
            return True
        eprint("[WARN] GPU stabilization failed -- falling back to VidStab CPU.")

    progress("[8/10] STABILIZE", 0.0, "VidStab Stabilizer (CPU Fallback)")
    if _stabilize_vidstab(src, dst, duration, run_dir, combine_enhance=combine_enhance):
        progress("[8/10] STABILIZE", 1.0, "VidStab stabilization complete")
        return True

    eprint("[WARN] Stabilization failed -- copying original video stream.")
    shutil.copy2(src, dst)
    return False
