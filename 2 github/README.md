<div align="center">

<h1>&#x2702;&#xFE0F; Auto Cut &amp; Polishing Tool</h1>

<p><strong>AI-powered automatic video &amp; audio post-production pipeline.</strong><br>
Remove silence, clean audio, stabilize footage, and deliver polished media &mdash; in a single automated run.</p>

<p>
  <img src="https://img.shields.io/badge/Version-a001-brightgreen?style=for-the-badge" alt="Version a001">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS%20%7C%20Colab-lightgrey?style=for-the-badge" alt="Platform">
  <img src="https://img.shields.io/badge/GPU-CUDA%20%7C%20NVENC-76b900?style=for-the-badge&logo=nvidia&logoColor=white" alt="GPU CUDA">
  <img src="https://img.shields.io/badge/License-MIT-orange?style=for-the-badge" alt="MIT License">
</p>

<p>
  <a href="#-features">Features</a> &bull;
  <a href="#-quick-start">Quick Start</a> &bull;
  <a href="#%EF%B8%8F-how-it-works">How It Works</a> &bull;
  <a href="#-performance">Performance</a> &bull;
  <a href="#-cli-reference">CLI Reference</a> &bull;
  <a href="#-requirements">Requirements</a>
</p>

</div>

---

## &#x1F680; What Is This?

**Auto Cut &amp; Polishing Tool** is a fully automated, cross-platform media post-production engine.  
It transforms raw, unedited video or audio files into studio-quality, broadcast-ready masters — with **zero manual editing**.

Point it at a local file **or** any public URL (YouTube, Facebook, direct stream), select the features you want, and it handles everything automatically.

```
Raw Recording  →  [Auto Cut & Polishing Tool]  →  Polished Master

• Dead silence & pauses     → 2-Stage Hybrid VAD Cut (frame-locked, zero drift)
• Background fan & noise    → ClearVoice MossFormer2 AI Denoise (48 kHz)
• Room echo & reverb        → VoiceFixer Neural De-reverberation
• Uneven speaker volumes    → Multi-Speaker Dynamic Volume Balancing
• Shaky handheld footage    → 2-Pass GPU Phase Correlation Stabilization
• Dull, soft video quality  → 100% GPU Tensor Core Cinema Polish
• No thumbnail              → AI Expressive Frame Selector (Top 3 JPEGs)
• Slow file delivery        → Universal Dual Delivery (CDN + Localtunnel)
```

---

## &#x2728; Features

| Feature | Technology | Speed |
|---|---|---|
| **2-Stage Hybrid Silence Cut** | Silero Neural VAD + Energy RMS VAD | ~50&times; realtime |
| **Frame-Locked Audio/Video Cut** | FFmpeg filter_complex (single-pass) | Zero lip-sync drift |
| **AI Speech Denoising** | ClearVoice MossFormer2 (48 kHz) | 2&ndash;5&times; realtime (GPU) |
| **Neural De-reverberation** | VoiceFixer Vocoder | Sub-chunk streaming |
| **Broadcast Loudness Mastering** | EBU R128 (&minus;14 LUFS) | Real-time FFmpeg |
| **Multi-Speaker Volume Leveling** | Dynamic windowed normalization | All speakers balanced |
| **GPU Video Stabilization** | CUDA Phase Correlation (2-Pass FFT) | 60&ndash;90 FPS |
| **Tensor Cinema Enhancement** | DoG Edge Sharpening + Contrast Lift | 100&ndash;150+ FPS |
| **AI Thumbnail Extraction** | OpenCV Facial Landmarks + Laplacian | Top 3 best frames |
| **Universal Dual Delivery** | Gofile CDN + Localtunnel HTTPS Server | 2000&ndash;3000+ Mbps |

---

## &#x26A1; Quick Start

### Windows (Double-Click Setup)

```
1. Double-click:   start\windows\setup.bat    (first time only — installs venv & AI models)
2. Double-click:   start\windows\run.bat      (every run)
3. Done! Your output is in the  output\  folder.
```

### Linux / macOS / Google Colab (Terminal)

```bash
# First-time setup (installs venv and downloads AI models)
bash start/linux_mac/setup.sh

# Run the tool
bash start/linux_mac/run.sh
```

### Google Colab — Save to Google Drive

When prompted for output destination, type `1,3` to save to both your local `output/` folder **and** your Google Drive simultaneously.

---

## &#x1F5A5;&#xFE0F; How It Works

When you launch the tool, an interactive terminal menu guides you through 3 steps:

**Step 1 &mdash; Choose your input:**
```
INPUT SOURCE
  1. Local file
  2. URL  (YouTube / Facebook / Direct stream)
  3. Google Drive
  4. Exit

Select [1-4]: 2
Enter video URL: https://www.youtube.com/watch?v=...
```

**Step 2 &mdash; Select features:**
```
SELECT WORKFLOW FEATURES:
  [x] 1. Remove silence  (frame-locked auto-cut)
  [x] 2. AI audio enhancement  (ClearVoice / VoiceFixer)
  [x] 3. Echo & Reverb removal
  [x] 4. Loudness normalization  (−14 LUFS)
  [x] 5. Video Stabilization  (GPU deshake)
  [x] 6. GPU Video Enhancement  (Cinema Polish)
  [x] 7. AI Expressive Thumbnail Generator

  A=all  N=none  ENTER=start  Q=quit
```

**Step 3 &mdash; Live progress tracking:**
```
[3/10] AI AUDIO ENHANCEMENT
[████████████████████░░░░░░░] 72.4%   Elapsed: 02:14   ETA: 00:51
GPU: 61%  |  VRAM: 2.1 / 8.0 GB  |  Speed: 3.4× realtime
```

---

## &#x1F4CA; Performance

Benchmarked on a **1-hour 1080p podcast recording** with all 7 features enabled:

| Metric | Result |
|---|---|
| **Total Processing Time** | ~33 minutes |
| **Peak RAM Usage** | ≤ 51.5% (no crash) |
| **GPU VRAM Peak** | ~7 MB (Pass 1) / ~50 MB (Pass 2) |
| **Network Delivery Speed** | 2,251 Mbps avg / 3,124 Mbps peak |
| **Output Quality** | Broadcast −14 LUFS, studio-clean speech |

### Zero-RAM-Crash Architecture

This tool **never** loads an entire video into memory. It uses **sub-chunk streaming** to keep RAM and VRAM consumption flat, regardless of input duration or resolution.

- **Audio VAD:** 30 ms sliding windows (&lt;1 MB RAM)
- **AI Enhancement:** 30&ndash;60 second overlapping chunks (flushed immediately after each)
- **Video Processing:** Frames streamed one-by-one: decoder &rarr; GPU CUDA &rarr; NVENC encoder (never buffered)
- **Result:** A 3-hour 4K video processes safely on an 8 GB RAM laptop without crashing.

---

## &#x2328;&#xFE0F; CLI Reference

For batch automation and scripting, run `program/main.py` directly:

```bash
python program/main.py \
  --input "https://www.youtube.com/watch?v=EXAMPLE" \
  --output "./output" \
  --all \
  --download
```

| Flag | Default | Description |
|---|---|---|
| `--input`, `-i` | *required* | Local file path or public web URL |
| `--output`, `-o` | `./output` | Output directory |
| `--output-name` | *auto* | Custom output filename |
| `--all` | `False` | Enable all 7 processing features |
| `--remove-silence` | `False` | 2-stage Hybrid VAD auto-cut |
| `--min-silence` | `0.50` | Min silence to cut (seconds) |
| `--keep-pause` | `0.30` | Natural breath pause to keep (seconds) |
| `--audio-enhance` | `False` | ClearVoice / VoiceFixer AI denoising |
| `--dereverb` | `False` | Room echo &amp; reverb removal |
| `--normalize` | `False` | EBU R128 loudness normalization |
| `--target-lufs` | `-14.0` | Target loudness in LUFS |
| `--multi-speaker-balance` | `False` | Equalize multiple speaker volumes |
| `--stabilize` | `False` | 2-pass GPU Phase Correlation deshake |
| `--stabilize-engine` | `gpu` | `gpu` (CUDA FFT) or `cpu` (vidstab) |
| `--video-enhance` | `False` | GPU Tensor Core cinema polish |
| `--extract-thumbnails` | `False` | Extract top 3 AI-scored thumbnails |
| `--engine` | `auto` | Audio engine: `auto`, `clearvoice`, `voicefixer` |
| `--vad-engine` | `hybrid` | VAD mode: `hybrid`, `silero`, `energy` |
| `--download` | `False` | Trigger Universal Dual Delivery |
| `--cloud-only` | `False` | Skip local save; CDN delivery only |
| `--overwrite` | `False` | Overwrite existing output files |
| `--keep-temp` | `False` | Keep temp files for debugging |

---

## &#x1F4E6; Requirements

### System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| **OS** | Windows 10, Ubuntu 20.04, macOS 11 | Windows 11 / Ubuntu 22.04 |
| **Python** | 3.10+ | 3.11 or 3.12 |
| **CPU** | 4 cores | 8+ cores |
| **RAM** | 8 GB | 16&ndash;32 GB |
| **GPU** | Optional (CPU fallback available) | NVIDIA RTX 2060+ (CUDA 12.x) |
| **Disk** | 10 GB free | NVMe SSD |

### Python Dependencies (auto-installed by setup scripts)

```
numpy>=1.26.0       # Audio waveform processing
yt-dlp>=2026.7.7    # YouTube / public URL downloader
silero-vad>=6.0     # Neural Voice Activity Detection
torch>=2.0.0        # GPU-accelerated AI inference (CUDA or CPU)
torchaudio>=2.0.0   # Audio tensor operations
clearvoice>=0.1.0   # ClearVoice MossFormer2 speech enhancement
voicefixer>=0.1.3   # Neural de-reverberation & voice restoration
opencv-python       # Face detection & thumbnail sharpness scoring
Pillow>=9.0.0       # Fallback image processing
tqdm>=4.0.0         # Upload progress bars
requests>=2.28.0    # CDN upload HTTP client
```

> **Note:** All dependencies install **only** into an isolated `support/venv/` virtual environment. Nothing is installed globally on your system.

---

## &#x1F4C1; Project Structure

```
Auto Cut & Polishing Tool/
├── start/
│   ├── windows/              # setup.bat  run.bat  clean.bat
│   └── linux_mac/            # setup.sh   run.sh   clean.sh
├── program/
│   ├── main.py               # Master pipeline orchestrator & CLI entry point
│   ├── requirements.txt      # All Python dependencies
│   ├── patch_packages.py     # Post-install model patch & compatibility fixes
│   ├── prefetch_models.py    # Pre-downloads all AI model weights at setup time
│   └── src/
│       ├── core/             # config · hardware_monitor · logger · media_tools
│       ├── audio/            # vad · silence_cutter · enhancer · mastering
│       ├── video/            # stabilizer · cinema_polish · thumbnail_extractor
│       └── delivery/         # muxer · cloud_upload · local_server
├── support/                  # [auto-generated] venv · AI models · cache · temp
└── output/                   # Your finished videos & thumbnails land here
```

For a complete deep-dive into every module, class, and data-flow, see  
[**CODEBASE_ARCHITECTURE_GUIDE.md**](Auto%20Cut%20%26%20Polishing%20Tool/CODEBASE_ARCHITECTURE_GUIDE.md).

---

## &#x1F504; Version History

| Tag | Date | Summary |
|---|---|---|
| **a001** | 2026-10-03 | First public release. Full 8-stage pipeline: Hybrid VAD Cut · ClearVoice AI Denoise · VoiceFixer De-reverb · EBU R128 Mastering · Multi-Speaker Balance · GPU Stabilization · Tensor Cinema Polish · AI Thumbnails · Universal Dual Delivery. |

---

## &#x1F4C4; License

This project is licensed under the **MIT License** &mdash; see [LICENSE](LICENSE) for details.

---

<div align="center">
<sub>Built with precision &nbsp;&bull;&nbsp; Zero-RAM crash architecture &nbsp;&bull;&nbsp; Studio-grade output on any hardware.</sub>
</div>
