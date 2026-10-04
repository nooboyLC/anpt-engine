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
        interval = max(4.0, min(15.0, duration / 36.0)) if duration > 0 else 5.0
        dec_threads = str(max(1, min(2, (os.cpu_count() or 2))))
        hwaccel_args = ["-hwaccel", "cuda"] if torch_cuda_available() else []
        cmd = [
            ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error",
        ] + hwaccel_args + [
            "-threads", dec_threads,
            "-i", str(video_path),
            "-vf", f"scale=trunc(iw/2)*2:trunc(ih/2)*2,fps=1/{interval:.2f}",
            "-q:v", "2",
            str(temp_thumb_dir / "frame_%04d.jpg")
        ]
        run(cmd, check=False)

        frame_files = sorted(list(temp_thumb_dir.glob("frame_*.jpg")))
        if not frame_files:
            cmd_cpu = [
                ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error",
                "-threads", dec_threads,
                "-i", str(video_path),
                "-vf", f"scale=trunc(iw/2)*2:trunc(ih/2)*2,fps=1/{interval:.2f}",
                "-q:v", "2",
                str(temp_thumb_dir / "frame_%04d.jpg")
            ]
            run(cmd_cpu, check=False)
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

        for fpath in frame_files:
            sharpness = 0.0
            faces = []
            img_h, img_w = 720, 1280

            if cv2 is not None:
                img = cv2.imread(str(fpath))
                if img is not None:
                    img_h, img_w = img.shape[:2]
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

                    if face_cascade is not None:
                        detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
                        faces = list(detected)
            else:
                try:
                    from PIL import Image, ImageFilter, ImageStat
                    with Image.open(fpath) as pil_img:
                        img_w, img_h = pil_img.size
                        gray = pil_img.convert("L")
                        edges = gray.filter(ImageFilter.FIND_EDGES)
                        stat = ImageStat.Stat(edges)
                        sharpness = stat.var[0] if stat.var else 0.0
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
                "score": total_score,
                "sharpness": sharpness,
                "faces": faces,
                "face_centers": face_centers
            })

        output_paths = []

        if multi_speaker and any(f["face_centers"] for f in scored_frames):
            left_speaker_frames = []
            right_speaker_frames = []
            center_speaker_frames = []

            for item in scored_frames:
                centers = item["face_centers"]
                if not centers:
                    continue
                avg_x = sum(centers) / len(centers)
                if avg_x < 0.45:
                    left_speaker_frames.append(item)
                elif avg_x > 0.55:
                    right_speaker_frames.append(item)
                else:
                    center_speaker_frames.append(item)

            speaker_groups = [
                ("Speaker1", left_speaker_frames),
                ("Speaker2", right_speaker_frames),
                ("Speaker3", center_speaker_frames),
            ]

            count = 0
            for spk_name, group in speaker_groups:
                if not group:
                    continue
                group.sort(key=lambda x: x["score"], reverse=True)
                top_frames = group[:2]
                for idx, frm in enumerate(top_frames, 1):
                    out_thumb = output_dir / f"Thumbnail_{spk_name}_{idx:02d}.jpg"
                    shutil.copy2(frm["path"], out_thumb)
                    output_paths.append(out_thumb)
                    print(f"  [OK] Saved {spk_name} thumbnail: {out_thumb.name} (Sharpness: {frm['sharpness']:.1f})")
                    count += 1

            if count == 0:
                scored_frames.sort(key=lambda x: x["score"], reverse=True)
                for idx, frm in enumerate(scored_frames[:3], 1):
                    out_thumb = output_dir / f"Thumbnail_{idx:02d}.jpg"
                    shutil.copy2(frm["path"], out_thumb)
                    output_paths.append(out_thumb)
                    print(f"  [OK] Saved thumbnail: {out_thumb.name} (Sharpness: {frm['sharpness']:.1f})")
        else:
            scored_frames.sort(key=lambda x: x["score"], reverse=True)
            top_3 = scored_frames[:3]
            for idx, frm in enumerate(top_3, 1):
                out_thumb = output_dir / f"Thumbnail_{idx:02d}.jpg"
                shutil.copy2(frm["path"], out_thumb)
                output_paths.append(out_thumb)
                print(f"  [OK] Saved thumbnail: {out_thumb.name} (Sharpness: {frm['sharpness']:.1f})")

        print("=" * 64 + "\n")
        return output_paths
    finally:
        shutil.rmtree(temp_thumb_dir, ignore_errors=True)
