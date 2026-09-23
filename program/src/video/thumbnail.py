# -*- coding: utf-8 -*-
"""
video.thumbnail
---------------
AI Expressive Frame Selection & Thumbnail Extraction:
- Analyzes candidate frames using OpenCV facial feature detection & Laplacian variance.
- Selects top 3 expressive, sharp frames.
- Multi-speaker aware: isolates separate speakers by horizontal frame location.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

try:
    import cv2
except Exception:
    cv2 = None

from core.media_tools import ffmpeg_path, run
from core.hardware import torch_cuda_available


def extract_best_thumbnails(
    video_path: Path,
    output_dir: Path,
    temp_dir: Path | None = None,
    multi_speaker: bool = False,
    duration: float = 0.0
) -> list[Path]:
    """Analyzes full video content to extract sharpest, clearest expressive thumbnail frames."""
    if not video_path.exists() or not ffmpeg_path():
        return []

    print("\n" + "=" * 64)
    print("🎨 AI THUMBNAIL GENERATOR (Best Expressive Frame Extraction)")
    print("=" * 64)

    base_tmp = temp_dir if temp_dir is not None else Path(tempfile.gettempdir())
    temp_thumb_dir = base_tmp / "_temp_thumb_frames"
    temp_thumb_dir.mkdir(parents=True, exist_ok=True)

    try:
        interval = max(1.0, min(15.0, duration / 36.0)) if duration >= 1.0 else 0.5
        dec_threads = str(max(1, min(2, (os.cpu_count() or 2))))
        cmd = [
            ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error",
            "-threads", dec_threads,
            "-i", str(video_path),
            "-vf", f"fps=1/{interval:.2f}",
            "-pix_fmt", "yuvj420p",
            "-strict", "unofficial",
            "-q:v", "2",
            str(temp_thumb_dir / "frame_%04d.jpg")
        ]
        run(cmd, check=False)

        frame_files = sorted(list(temp_thumb_dir.glob("frame_*.jpg")))
        if not frame_files:
            cmd_fallback = [
                ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error",
                "-threads", dec_threads,
                "-i", str(video_path),
                "-vframes", "1",
                "-pix_fmt", "yuvj420p",
                "-strict", "unofficial",
                "-q:v", "2",
                str(temp_thumb_dir / "frame_0001.jpg")
            ]
            run(cmd_fallback, check=False)
            frame_files = sorted(list(temp_thumb_dir.glob("frame_*.jpg")))

        if not frame_files:
            print("[THUMBNAIL] No candidate frames extracted.")
            return []

        scored_frames = []
        face_cascade = None
        if cv2 is not None:
            try:
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                if os.path.exists(cascade_path):
                    face_cascade = cv2.CascadeClassifier(cascade_path)
            except Exception:
                face_cascade = None

        for idx_f, fpath in enumerate(frame_files):
            sharpness = 0.0
            mean_brightness = 128.0
            faces = []
            img_h, img_w = 720, 1280

            if cv2 is not None:
                img = cv2.imread(str(fpath))
                if img is not None:
                    img_h, img_w = img.shape[:2]
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    mean_brightness = float(cv2.mean(gray)[0])
                    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

                    if face_cascade is not None:
                        detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
                        faces = list(detected)
            else:
                try:
                    from PIL import Image, ImageFilter, ImageStat
                    with Image.open(fpath) as pil_img:
                        img_w, img_h = pil_img.size
                        gray = pil_img.convert("L")
                        stat_gray = ImageStat.Stat(gray)
                        mean_brightness = float(stat_gray.mean[0]) if stat_gray.mean else 128.0
                        edges = gray.filter(ImageFilter.FIND_EDGES)
                        stat = ImageStat.Stat(edges)
                        sharpness = float(stat.var[0]) if stat.var else 0.0
                except Exception:
                    fsize = fpath.stat().st_size
                    sharpness = float(fsize) / 1000.0

            face_score = 0.0
            face_centers = []
            for (x, y, w, h) in faces:
                area_ratio = (w * h) / (img_w * img_h)
                center_x = (x + w / 2.0) / img_w
                face_centers.append(center_x)
                face_score += 1.0 + (area_ratio * 5.0)

            total_score = sharpness * (1.0 + face_score)
            scored_frames.append({
                "path": fpath,
                "frame_idx": idx_f,
                "mean_brightness": mean_brightness,
                "score": total_score,
                "sharpness": sharpness,
                "faces": faces,
                "face_centers": face_centers
            })

        output_paths = []
        if not scored_frames:
            print("[THUMBNAIL] [WARN] No valid candidate frames could be scored.")
            return []

        # Filter out extreme dark (black frames) or extreme bright (white flashes)
        valid_candidates = [f for f in scored_frames if f.get("mean_brightness", 128.0) > 15.0 and f.get("mean_brightness", 128.0) < 242.0]
        if not valid_candidates:
            # Fallback to all scored frames if filter was too strict
            valid_candidates = scored_frames

        # Sort candidate frames by total quality score (sharpness + face composition) descending
        valid_candidates.sort(key=lambda x: x["score"], reverse=True)

        # Select top distinct frames (enforce minimum temporal spacing so thumbnails aren't identical)
        selected_frames: list[dict] = []
        min_frame_dist = max(1, len(frame_files) // 12)  # spacing in candidate indices

        for cand in valid_candidates:
            cand_idx = cand.get("frame_idx", 0)
            # Check if this frame is sufficiently spaced from already selected frames
            if all(abs(cand_idx - s.get("frame_idx", 0)) >= min_frame_dist for s in selected_frames):
                selected_frames.append(cand)
            if len(selected_frames) >= 3:
                break

        # If diversity constraint was too tight, fill up from remaining top candidates
        if len(selected_frames) < 3:
            for cand in valid_candidates:
                if cand not in selected_frames:
                    selected_frames.append(cand)
                if len(selected_frames) >= 3:
                    break

        # Always save sequentially: Thumbnail_01.jpg, Thumbnail_02.jpg, Thumbnail_03.jpg
        for rank, frm in enumerate(selected_frames, 1):
            out_thumb = output_dir / f"Thumbnail_{rank:02d}.jpg"
            try:
                shutil.copy2(frm["path"], out_thumb)
                output_paths.append(out_thumb)
                has_face = " (Face detected)" if frm.get("faces") else ""
                print(f"  [OK] Saved thumbnail: {out_thumb.name} (Sharpness: {frm['sharpness']:.1f}{has_face})")
            except Exception as exc:
                print(f"  [WARN] Failed to write thumbnail {out_thumb.name}: {exc}")

        # If multi-speaker mode is requested, also create speaker-tagged copies
        if multi_speaker and any(f.get("face_centers") for f in scored_frames):
            speaker_bins: dict[str, list[dict]] = {"Speaker1": [], "Speaker2": [], "Speaker3": []}
            for item in scored_frames:
                centers = item.get("face_centers", [])
                if not centers:
                    continue
                avg_x = sum(centers) / len(centers)
                if avg_x < 0.40:
                    speaker_bins["Speaker1"].append(item)
                elif avg_x > 0.60:
                    speaker_bins["Speaker2"].append(item)
                else:
                    speaker_bins["Speaker3"].append(item)

            for spk_name, group in speaker_bins.items():
                if group:
                    group.sort(key=lambda x: x["score"], reverse=True)
                    best_spk = group[0]
                    spk_out = output_dir / f"Thumbnail_{spk_name}.jpg"
                    try:
                        shutil.copy2(best_spk["path"], spk_out)
                        output_paths.append(spk_out)
                        print(f"  [OK] Saved {spk_name} thumbnail: {spk_out.name} (Sharpness: {best_spk['sharpness']:.1f})")
                    except Exception as exc:
                        print(f"  [WARN] Failed to write {spk_name} thumbnail: {exc}")

        print("=" * 64 + "\n")
        return output_paths
    finally:
        shutil.rmtree(temp_thumb_dir, ignore_errors=True)
