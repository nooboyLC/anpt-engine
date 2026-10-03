# 📘 Auto Cut & Polishing Tool — The Architecture & Codebase Guide
>
> **Version:** a001

---

## 📑 Table of Contents

1. [🌟 Executive Summary & Mission](#1--executive-summary--mission)
2. [🗺️ High-Level System Architecture (Text-Paint Diagram)](#2--high-level-system-architecture-text-paint-diagram)
3. [📂 Directory Tree & Path Anatomy (Text-Paint Map)](#3--directory-tree--path-anatomy-text-paint-map)
4. [🖥️ Complete User Interface Experience (Terminal Text-Paint Mockups)](#4--complete-user-interface-experience-terminal-text-paint-mockups)
5. [🎮 User Guide: How to Run the Tool (Windows, Linux, Colab & CLI)](#5--user-guide-how-to-run-the-tool-windows-linux-colab--cli)
6. [🔄 End-to-End Program Workflow & Decision Logic (Master Flowchart)](#6--end-to-end-program-workflow--decision-logic-master-flowchart)
7. [🔬 Deep Dive: Python File-by-File & Function-by-Function Reference](#7--deep-dive-python-file-by-file--function-by-function-reference)
   - 7.1. Root Orchestration & Lifecycle (`program/*.py`)
   - 7.2. Core & Infrastructure Subsystem (`program/src/core/`)
   - 7.3. Audio Processing Subsystem (`program/src/audio/`)
   - 7.4. Video Processing Subsystem (`program/src/video/`)
   - 7.5. Packaging & Delivery Subsystem (`program/src/delivery/`)
8. [💾 Data Flow & Scratch File Lifecycle](#8--data-flow--scratch-file-lifecycle)
9. [🛡️ Zero-RAM Crash Architecture & Hardware Acceleration](#9--zero-ram-crash-architecture--hardware-acceleration)
10. [⚠️ Edge Cases, Fallbacks & Self-Healing Mechanisms](#10--edge-cases-fallbacks--self-healing-mechanisms)
11. [👨‍💻 Developer Extension Handbook](#11--developer-extension-handbook)

---

## 🌟 1. Executive Summary & Mission

### The Core Problem

Recording raw video or audio content (podcasts, tutorials, lectures, vlogs) introduces severe production defects:

- Up to **30%–50% of the duration is dead silence**, hesitation pauses, or awkward breaths.
- Microphones pick up **acoustic room echo**, ventilation noise, computer fans, and traffic hiss.
- Multiple speakers have **drastically mismatched volume levels** (one whispers, another shouts).
- Handheld cameras suffer from **distracting micro-jitters and camera shakes**.
- Video colors look **washed-out, soft, or blurry**.
- Sharing the final 2–5 GB master file from cloud environments (like Google Colab) is often slow, painful, and prone to broken downloads.

Manual editing requires hours of cutting on a timeline, tweaking noise gates, applying compressors, running color grades, stabilizing shaky clips, and rendering thumbnails.

### The Solution: Auto Cut & Polishing Tool

The **Auto Cut & Polishing Tool** is a fully automated, cross-platform, modular post-production engine. It takes an unedited raw video file or a public web URL (YouTube, Facebook, direct link) and transforms it into a polished, broadcast-ready master file in a single automated run.

```
+---------------------------------------------------------------------------------------------------+
|                                      THE CORE TRANSFORMATION                                      |
+---------------------------------------------------------------------------------------------------+
|   RAW UNEDITED MEDIA                     AUTO CUT & POLISHING TOOL                 POLISHED MASTER|
|                                                                                                   |
|   ┌──────────────────────────┐           ┌─────────────────────────────┐           ┌──────────────┐
|   │ • Awkward Silences       │           │ 1. 2-Stage Hybrid VAD Cut   │           │ • Tight Flow │
|   │ • Background Fan Noise   │   ─────▶  │ 2. ClearVoice AI Denoise    │   ─────▶  │ • Studio Mic │
|   │ • Room Echo / Reverb     │           │ 3. VoiceFixer De-reverb     │           │ • Dry Vocal  │
|   │ • Uneven Speaker Volumes │           │ 4. Multi-Speaker Balancing  │           │ • Level Dial.│
|   │ • Shaky Handheld Footage │           │ 5. GPU Video Stabilization  │           │ • Steadicam  │
|   │ • Dull Color & Soft Blur │           │ 6. GPU Tensor Cinema Polish │           │ • Vibrant 4K │
|   │ • No Thumbnail Image     │           │ 7. Expressive AI Thumbnail  │           │ • 3 Top JPGs │
|   │ • Painful File Sharing   │           │ 8. Universal Dual Delivery  │           │ • CDN+Tunnel │
|   └──────────────────────────┘           └─────────────────────────────┘           └──────────────┘
+---------------------------------------------------------------------------------------------------+
```

---

## 🗺️ 2. High-Level System Architecture (Text-Paint Diagram)

The codebase is engineered around **strict modularity** and **total environment isolation**. It is divided into 4 sovereign zones:

```
+===================================================================================================+
|                                    SYSTEM ARCHITECTURE TOPOLOGY                                   |
+===================================================================================================+
|                                                                                                   |
|  [USER ENTRYPOINTS]                                                                               |
|  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐  |
|  │  start/windows/ (setup.bat, run.bat, clean.bat)                                             │  |
|  │  start/linux_mac/ (setup.sh, run.sh, clean.sh)                                              │  |
|  └──────────────────────────────────────────────┬──────────────────────────────────────────────┘  |
|                                                 │ Invokes Python venv                             |
|                                                 ▼                                                 |
|  [PROGRAM ENGINE] (program/)                                                                      |
|  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐  |
|  │  main.py (Master Orchestrator / Pipeline Conductor)                                         │  |
|  │  ┌───────────────────────────────────────────────────────────────────────────────────────┐  │  |
|  │  │                                  src/ (Subsystems)                                    │  │  |
|  │  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐           │  │  |
|  │  │  │   src/core/   │  │  src/audio/   │  │  src/video/   │  │ src/delivery/ │           │  │  |
|  │  │  │ • config      │  │ • vad         │  │ • stabilizer  │  │ • muxer       │           │  │  |
|  │  │  │ • hardware    │  │ • cutter      │  │ • enhancer    │  │ • cloud_upload│           │  │  |
|  │  │  │ • logger      │  │ • cleaner     │  │ • thumbnail   │  │ • server      │           │  │  |
|  │  │  │ • media_tools │  │ • mastering   │  │               │  │               │           │  │  |
|  │  │  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘           │  │  |
|  │  └──────────┼──────────────────┼──────────────────┼──────────────────┼───────────────────┘  │  |
|  └─────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────────┘  |
|                │                  │                  │                  │                         |
|                ▼                  ▼                  ▼                  ▼                         |
|  [SUPPORT ISOLATION ZONE] (support/)                                                              |
|  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐  |
|  │ • venv/          (Isolated Python environment; never pollutes user's OS)                    │  |
|  │ • bin/           (FFmpeg, FFprobe, and native accelerator executables)                      │  |
|  │ • checkpoints/   (Neural weights: ClearVoice MossFormer2, Silero VAD, VoiceFixer)           │  │
|  │ • temp/          (Scratch processing space; per-run temporary directories)                  │  |
|  │ • cache/         (Centralized pycache, Torch extensions, Matplotlib configs)                │  |
|  └─────────────────────────────────────────────────────────────────────────────────────────────┘  |
|                                                 │                                                 |
|                                                 ▼ Produces Final Output                           |
|  [OUTPUT DELIVERY BOX] (output/)                                                                  |
|  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐  |
|  │ • Polished Master Media File (e.g. MyVideo_AutoCut_Enhanced.mp4)                            │  |
|  │ • output/thumbnails/ (Top 3 AI-selected expressive, high-resolution JPEG frames)            │  |
|  │ * CRITICAL LAW: The output/ directory is NEVER modified or wiped by any clean scripts.      │  |
|  └─────────────────────────────────────────────────────────────────────────────────────────────┘  |
+===================================================================================================+
```

---

## 📂 3. Directory Tree & Path Anatomy (Text-Paint Map)

Below is the complete filesystem map of the project, documenting the location, purpose, and classification of every single file:

```
Auto Cut & Polishing Tool/
│
├── CODEBASE_ARCHITECTURE_GUIDE.md  <-- Complete architectural blueprint (You are reading this)
│
├── start/                          <-- User Launcher Scripts (Double-click or shell execution)
│   │
│   ├── windows/                    <-- Windows 10/11 Batch scripts
│   │   ├── setup.bat               <-- Creates supportenv, installs wheels, triggers prefetch
│   │   ├── run.bat                 <-- Injects isolated environment variables & launches main.py
│   │   └── clean.bat               <-- Wipes support emp and support\cache without touching output│   │
│   └── linux_mac/                  <-- POSIX Shell scripts (Linux, macOS, Google Colab)
│       ├── setup.sh                <-- Linux/Mac/Colab environment installer
│       ├── run.sh                  <-- Linux/Mac/Colab interactive launcher
│       └── clean.sh                <-- Linux/Mac/Colab cache cleanup script
│
├── program/                        <-- Core Application Source Code
│   │
│   ├── main.py                     <-- Entry point, CLI argument parser & 12-step pipeline director
│   ├── patch_packages.py           <-- Runtime patcher (redirects VoiceFixer/HuggingFace paths)
│   ├── prefetch_models.py          <-- Pre-downloads AI weights so the tool can run offline
│   ├── requirements.txt            <-- Pinned Python dependencies (PyTorch, Silero, OpenCV, etc.)
│   │
│   └── src/                        <-- Python Modular Subsystems
│       ├── __init__.py             <-- Package initialization
│       │
│       ├── core/                   <-- Foundations, Hardware & IO
│       │   ├── __init__.py         <-- Sub-package marker
│       │   ├── config.py           <-- Paths, environment variables, thread limits & audio presets
│       │   ├── hardware.py         <-- HWProfile singleton, CUDA, NVDEC, NVENC, Vulkan, CPU detection
│       │   ├── logger.py           <-- Color terminal printer, progress bars, time formatters
│       │   ├── media_tools.py      <-- FFmpeg/FFprobe wrappers, probe(), yt-dlp downloader
│       │   └── compat_dlls/        <-- Windows N media foundation fallback DLL stubs (MF.dll, etc.)
│       │
│       ├── audio/                  <-- Audio Processing Pipeline
│       │   ├── __init__.py         <-- Sub-package marker
│       │   ├── vad.py              <-- Energy VAD + Silero Neural VAD + Natural Cadence Padding
│       │   ├── cutter.py           <-- Lossless extraction, concat demuxer & frame-locked cutting
│       │   ├── cleaner.py          <-- ClearVoice MossFormer2 (48kHz) & VoiceFixer neural restoration
│       │   └── mastering.py        <-- Pre-leveling, multi-speaker balance, studio EQ & -14 LUFS
│       │
│       ├── video/                  <-- Video Processing Pipeline
│       │   ├── __init__.py         <-- Sub-package marker
│       │   ├── stabilizer.py       <-- 2-Pass crash-proof GPU Phase Correlation video deshaker
│       │   ├── enhancer_filter.py  <-- 100% GPU Tensor Core Cinema Polish & DoG edge sharpening
│       │   └── thumbnail.py        <-- AI facial expression detection & top-3 thumbnail picker
│       │
│       └── delivery/               <-- Muxing & Distribution
│           ├── __init__.py         <-- Sub-package marker
│           ├── muxer.py            <-- Faststart MP4 packaging with AAC 192k audio
│           ├── cloud_upload.py     <-- Universal Dual Delivery (Gofile / TmpFiles CDN upload)
│           └── server.py           <-- Multi-threaded HTTP Range server + Localtunnel reverse proxy
│
├── support/                        <-- Auto-Generated Isolated Runtime (Excluded from Git)
│   ├── venv/                       <-- Virtual environment (Python binaries & site-packages)
│   ├── bin/                        <-- Native tools (ffmpeg.exe, ffprobe.exe)
│   ├── checkpoints/                <-- Downloaded neural weights (ClearVoice, VoiceFixer, Silero)
│   ├── cache/                      <-- Bytecode (__pycache__), Matplotlib and Torch extensions
│   ├── temp/                       <-- Scratch folder where temporary audio/video chunks live
│   └── py_userbase/                <-- Isolated pip user directory
│
└── output/                         <-- Final User Deliverables (Always preserved)
    ├── [Master_Output].mp4         <-- Your final polished video
    └── thumbnails/                 <-- Best thumbnail candidate images (frame_001.jpg, etc.)
```

---

## 🖥️ 4. Complete User Interface Experience (Terminal Text-Paint Mockups)

When a user runs the tool via `run.bat` or `run.sh`, they are presented with an intuitive, guided terminal interface. Below are exact visual mockups of every single screen:

### Screen 1: System Verification & Hardware Audit

```
================================================================================
  AUTO CUT & POLISHING TOOL — SYSTEM INITIALIZATION
================================================================================
  OS Platform       : Windows 11 (64-bit) / Linux Ubuntu 22.04
  CPU Logical Cores : 8 Cores (Allocated Worker Threads: 7)
  GPU Accelerator   : NVIDIA GeForce RTX 3070 (8192 MB VRAM)
  CUDA Capability   : Available (PyTorch CUDA Active)
  Hardware Decode   : NVDEC (H.264 / HEVC Hardware Acceleration: YES)
  Hardware Encode   : NVENC (h264_nvenc: YES)
  Vulkan Compute    : Available
  Dependency Check  : [OK] FFmpeg, FFprobe, Silero VAD, ClearVoice, VoiceFixer
================================================================================
```

### Screen 2: Input Media Selection Menu

```
INPUT SOURCE
1. Local file
2. URL (YouTube / Facebook / Direct stream)
3. Google Drive (/content/drive/MyDrive/...)
4. Exit

Select [1-4]: 2
Enter video URL: https://www.youtube.com/watch?v=dQw4w9WgXcQ
[DOWNLOAD] Fetching best quality stream via yt-dlp...
[DOWNLOAD] Complete: "My_Raw_Interview_Unedited_Raw.mp4" (1080p, 285.4 MB)
```

### Screen 3: Interactive Feature Selection Matrix

```
SELECT WORKFLOW FEATURES:
[ ] 1. Remove silence (Audio & Video frame-locked cut)
[ ] 2. AI audio enhancement (ClearVoice / VoiceFixer)
[ ] 3. Echo & Reverb removal (Dereverberation)
[ ] 4. Loudness normalization (-14 LUFS standard)
[ ] 5. Video Stabilization (GPU camera deshake)
[ ] 6. GPU Video Enhancement (Fast Tensor Cinema Polish)
[ ] 7. AI Expressive Thumbnail Generator

A=all  N=none  ENTER=start  Q=quit
Choice: A
  -> All features selected.

[x] 1. Remove silence (Audio & Video frame-locked cut)
[x] 2. AI audio enhancement (ClearVoice / VoiceFixer)
[x] 3. Echo & Reverb removal (Dereverberation)
[x] 4. Loudness normalization (-14 LUFS standard)
[x] 5. Video Stabilization (GPU camera deshake)
[x] 6. GPU Video Enhancement (Fast Tensor Cinema Polish)
[x] 7. AI Expressive Thumbnail Generator

Choice: [ENTER]
----------------------------------------------------------------
Is this a Multi-Speaker video/audio (Podcast, Interview)? [y/N]: y
----------------------------------------------------------------
```

### Screen 4: Output Destination & Delivery Preferences

```
OUTPUT DESTINATION
1. Default project output folder (.\output)
2. Custom directory
3. Google Drive (/content/drive/MyDrive/...)
Note: Multiple destinations can be combined (e.g. '1,3' to save both locally and to Google Drive)

Select destination [1-3] (1): 1,3
Subfolder under MyDrive (empty for root, e.g. AutoCut): AutoCut
Custom output filename (Press ENTER for auto-generated name): My_Podcast_Episode_01
Auto-download file to your local PC when finished? [y/N]: y
```

### Screen 5: Active Processing Live Telemetry

```
================================================================================
[02/10/2026 15:58:20] [2/10] CUT + SYNC
--------------------------------------------------------------------------------
Processing:  [████████████████████░░░░░░░░░░] 64.2%  Elapsed: 00:01:14  ETA: 00:00:41
FPS: 84.5  |  Speed: 2.81x  |  GPU: 48%  |  VRAM: 1.84 / 8.00 GB
================================================================================
```

### Screen 6: Universal Dual Delivery & Download Ready Screen

```
================================================================================
DOWNLOAD READY: My_Podcast_Episode_01.mp4  (Duration: 18m 42s | Size: 412 MB)
================================================================================

1. 🌐 HIGH-SPEED CLOUD CDN LINK (Uploaded via Gofile CDN):
   https://gofile.io/d/AbCdEf123
   • [COPIED] Link copied to your clipboard! Press Ctrl+V in browser.
   • Full-speed download with pause & resume support.

2. 🚀 GLOBAL PUBLIC INTERNET TUNNEL (Localtunnel):
   https://autocut-fast-delivery.loca.lt/My_Podcast_Episode_01.mp4
   • Password / Tunnel IP: 34.125.88.14 (Enter this if prompted by Localtunnel)

3. 📶 LOCAL NETWORK (LAN / WI-FI) LINK:
   http://192.168.1.105:8888/My_Podcast_Episode_01.mp4
   • Download directly to your phone/tablet connected to the same Wi-Fi!
   • Uses ZERO internet data; transfers at maximum router speed (~300–800 Mbps).

4. 💾 ONE-CLICK TERMINAL DOWNLOAD COMMANDS:
   Windows PowerShell:
     curl.exe -L "https://gofile.io/d/AbCdEf123" -o "$HOME\Downloads\My_Podcast_Episode_01.mp4"
   Mac / Linux:
     curl -L "https://gofile.io/d/AbCdEf123" -o ~/Downloads/My_Podcast_Episode_01.mp4
================================================================================
```

---

---

## 🎮 5. User Guide: How to Run the Tool (Windows, Linux, Colab & CLI)

Running the Auto Cut & Polishing Tool requires zero technical setup. Everything is handled via automated 1-click launchers.

---

### 💻 5.1. Running on Windows 10 / 11

1. **First-Time Installation:**
   - Open the folder: `start\windows\`
   - Double-click `setup.bat`.
   - *What happens:* It creates an isolated Python virtual environment inside `support\venv\`, installs required wheels, and pre-downloads AI models. You only need to run this once.

2. **Starting a Post-Production Job:**
   - Open: `start\windows\`
   - Double-click `run.bat`.
   - Follow the interactive terminal prompts: enter your media file path or paste a YouTube/web URL, select features with `A` (All) or individual numbers, and hit Enter.

3. **Cleaning Cache (Housekeeping):**
   - Double-click `clean.bat` whenever you want to free up disk space. It wipes temporary files inside `support\temp\` and `support\cache\` while keeping your files in `output\` 100% safe.

---

### 🍏 5.2. Running on Linux or macOS

1. Open your terminal in the repository root:

   ```bash
   cd "start/linux_mac"
   ```

2. **First-time setup:**

   ```bash
   bash setup.sh
   ```

3. **Launch the tool:**

   ```bash
   bash run.sh
   ```

4. **Clean scratch cache:**

   ```bash
   bash clean.sh
   ```

---

### ☁️ 5.3. Running on Google Colab (Free or Pro Cloud GPU)

You can run this entire tool in the cloud for free using Google Colab GPUs (e.g. Tesla T4):

1. Open a terminal in Google Colab (or run inside a notebook cell):

   ```bash
   cd "start/linux_mac"
   bash setup.sh
   bash run.sh
   ```

2. **Mounting Google Drive:**
   - When prompted for output destination, choose option `3` to save directly to `/content/drive/MyDrive/AutoCut`.
   - Or enter `1,3` to save to both the local directory and your Google Drive simultaneously!
3. **Downloading the Output:**
   - At the end of the run, the program displays both a **Gofile CDN download link** and a **Localtunnel global HTTPS link**. Click either link to download your finished video to your personal PC at ultra-high network speeds (often exceeding 2000–3000 Mbps!).

---

### ⌨️ 5.4. Complete Command-Line (CLI) Flags Reference

For power users, scripts, or automated batch processing, `program/main.py` can be invoked directly with CLI flags:

```bash
# Example: Fully automated YouTube podcast polish
python program/main.py \
  --input "https://www.youtube.com/watch?v=EXAMPLE" \
  --output "./output" \
  --all \
  --download
```

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--input`, `-i` | String | *Required* | Path to local media file OR public web URL (YouTube, FB, stream) |
| `--output`, `-o` | String | `./output` | Primary destination folder for finished master files |
| `--output-name` | String | *Auto* | Custom name for the final output media file |
| `--cloud-only` | Flag | `False` | Disables local disk saving; delivers file strictly via Cloud CDN & Tunnel |
| `--all` | Flag | `False` | Enables all 7 processing features in a single automated pass |
| `--remove-silence` | Flag | `False` | Enables 2-stage Hybrid VAD silence removal and frame-locked cut |
| `--min-silence` | Float | `0.50` | Minimum duration (seconds) of silence required to trigger a cut |
| `--keep-pause` | Float | `0.30` | Natural breath room (seconds) preserved before and after speech |
| `--energy-threshold` | Float | `0.018` | Energy VAD volume threshold for detecting quiet speech |
| `--audio-enhance` | Flag | `False` | Enables neural speech enhancement (ClearVoice / VoiceFixer) |
| `--dereverb` | Flag | `False` | Removes room acoustic reverberation and echo |
| `--normalize` | Flag | `False` | Normalizes final audio to EBU R128 (-14 LUFS) broadcast standard |
| `--target-lufs` | Float | `-14.0` | Target loudness level in LUFS (-14.0 for YouTube/Spotify, -23.0 for TV) |
| `--multi-speaker-balance` | Flag | `False` | Equalizes volume differences between multiple dialogue participants |
| `--stabilize` | Flag | `False` | Enables 2-Pass GPU Phase Correlation video camera deshaker |
| `--stabilize-engine` | Choice | `gpu` | Engine for video stabilization (`gpu` for CUDA FFT, `cpu` for vidstab) |
| `--video-enhance` | Flag | `False` | Enables 100% GPU Tensor Core Cinema Polish & DoG edge sharpening |
| `--extract-thumbnails` | Flag | `False` | Extracts top 3 AI-scored expressive, high-resolution JPEG thumbnails |
| `--engine` | Choice | `auto` | Preferred neural audio engine (`auto`, `clearvoice`, `voicefixer`) |
| `--vad-engine` | Choice | `hybrid` | Voice Activity Detection mode (`hybrid`, `silero`, `energy`) |
| `--download` | Flag | `False` | Triggers Universal Dual Delivery (Gofile CDN + Localtunnel server) |
| `--overwrite` | Flag | `False` | Overwrites existing output files instead of appending `_1`, `_2` |
| `--keep-temp` | Flag | `False` | Keeps scratch temporary files in `support/temp/` for debugging |
| `--temp` | String | *Auto* | Overrides default scratch temp directory path |

## 🔄 6. End-to-End Program Workflow & Decision Logic (Master Flowchart)

Below is the definitive visual flowchart showing every conditional branch, step sequence, and fallback pathway in the application lifecycle:

```
[START: User invokes run.bat or run.sh]
   │
   ▼
[Step 0: System Verification (core/hardware.py)]
   ├── Probes CPU cores, VRAM, NVDEC, NVENC, Vulkan
   ├── Checks FFmpeg, FFprobe, and Python packages
   └── Missing critical dependencies? ──YES──▶ [Print Help & Terminate]
         │ NO
         ▼
[Input Ingestion (program/main.py -> get_input())]
   ├── Is Input a URL? (YouTube/Facebook/Stream)
   │     ├── YES ──▶ Download raw media via yt-dlp to output/ or scratch/
   │     └── NO  ──▶ Verify local path exists on disk
   ▼
[Media Probing (core/media_tools.py -> probe())]
   ├── Detects streams: Video stream present? Audio stream present?
   └── Extracts duration, fps, resolution, sample rate, bit rate
   │
   ▼
[Step 1: Audio Pre-Leveling (audio/mastering.py -> pre_level_audio())]
   ├── Does source contain Audio?
   │     ├── NO  ──▶ Skip to Step 8
   │     └── YES ──▶ Run EBU R128 pre-leveling (-18 LUFS)
   │                 (Ensures low-volume speech is not misclassified as silence)
   ▼
[Step 2: Silence Removal & Auto-Cut (audio/vad.py & audio/cutter.py)]
   ├── Was --remove-silence requested?
   │     ├── NO  ──▶ Keep full original timeline; pass raw streams to Step 3
   │     └── YES ──▶ 
   │           ├── 1. Extract 16kHz mono audio (extract_vad_audio)
   │           ├── 2. Run 2-Stage Hybrid VAD (hybrid_vad_segments)
   │           │      ├── Energy VAD (Streaming RMS in 30ms frames)
   │           │      └── Silero Neural VAD (Batch GPU Tensor inference)
   │           ├── 3. Inject Natural Cadence Padding (front & tail breath buffers)
   │           ├── 4. Bridge sub-pause gaps (< min_silence)
   │           └── 5. Execute Frame-Locked Concatenation:
   │                  ├── Video + Audio: ffmpeg_concat_cut() (trim + atrim filter_complex)
   │                  └── Audio Only   : ffmpeg_cut_audio() (atrim filter_complex)
   ▼
[Step 3: Echo & Reverb Removal (audio/cleaner.py -> dereverb_audio())]
   ├── Was --dereverb requested?
   │     ├── NO  ──▶ Pass audio to Step 4
   │     └── YES ──▶ 
   │           ├── Engine "voicefixer": Neural VoiceFixer mode=0 (30s sub-chunks)
   │           └── Fallback: Acoustic DSP Filter (afftdn + agate + acompressor)
   ▼
[Step 4: AI Audio Enhancement (audio/cleaner.py -> enhance_audio())]
   ├── Was --audio-enhance requested?
   │     ├── NO  ──▶ Pass audio to Step 5
   │     └── YES ──▶
   │           ├── Primary: ClearVoice MossFormer2_SE_48K (48kHz sample rate)
   │           │   ├── Slices into 60s chunks with 1.0s overlap
   │           │   └── Stitches chunks with FFmpeg acrossfade (NO boundary clicks!)
   │           └── Fallback: FFmpeg anlmdn / afftdn high-order spectral filters
   ▼
[Step 5: Multi-Speaker Volume Balancing (audio/mastering.py)]
   ├── Was --multi-speaker requested?
   │     ├── NO  ──▶ Pass audio to Step 6
   │     └── YES ──▶ Two-pass dynamic window leveling (equalizes quiet & loud speakers)
   ▼
[Step 6: Studio Vocal Fine-Tuning (audio/mastering.py -> voice_fine_tuning())]
   └── Applied if ANY audio enhancement was selected:
         ├── 180 Hz (+1.5 dB) Body Warmth
         ├── 420 Hz (-1.8 dB) Mud Resonance Scoop
         ├── 3.5 kHz (+2.0 dB) Consonant Clarity & Presence
         ├── 9.5 kHz (+1.2 dB) Studio Vocal Air
         └── Broadcast Dynamic Range Compressor
   ▼
[Step 7: Final Loudness Normalization (audio/mastering.py -> loudness_normalize())]
   └── Sets final audio track to exact EBU R128 -14 LUFS broadcast standard
   │
   ▼
[Step 8: Video Stabilization (video/stabilizer.py -> stabilize_video())]
   ├── Was --stabilize requested AND does media have Video?
   │     ├── NO  ──▶ Pass video to Step 9
   │     └── YES ──▶
   │           ├── Check for existing motions.json (Resume checkpoint)
   │           │     ├── Found? ──▶ Skip Pass 1!
   │           │     └── Missing? ──▶ Pass 1: 320x180 Grayscale Phase Correlation (VRAM ~7MB)
   │           ├── Pass 2: Adaptive VRAM Batch Warp (Streams frames to NVENC)
   │           └── Fused Enhancement Check:
   │                 Is --video-enhance ALSO selected with CUDA available?
   │                 ├── YES ──▶ [FUSED PASS] Deshake + Cinema Polish in 1 pass! (Saves ~25 min)
   │                 └── NO  ──▶ Standard stabilization warp
   ▼
[Step 9: GPU Video Cinema Enhancement (video/enhancer_filter.py)]
   ├── Was --video-enhance requested AND NOT already handled by Fused Mode?
   │     ├── NO  ──▶ Pass video to Step 10
   │     └── YES ──▶
   │           ├── Profile video quality in <1s (contrast, sharpness, compression artifacts)
   │           ├── DoG (Difference of Gaussians) edge-aware sharpening (zero halos)
   │           ├── Dynamic sharpness gating (disables on already-sharp or noisy frames)
   │           └── 100% GPU Tensor Core execution (100–150+ FPS)
   ▼
[Step 10: Final Muxing & Packaging (delivery/muxer.py -> mux_final())]
   ├── Merges final video stream + final mastered audio stream
   ├── Audio encoded to high-fidelity AAC 192k stereo/mono
   └── Injects -movflags +faststart (optimizes MP4 for instant web streaming)
   │
   ▼
[Step 11: AI Expressive Thumbnail Generation (video/thumbnail.py)]
   ├── Was --extract-thumbnails requested?
   │     ├── NO  ──▶ Proceed to Step 12
   │     └── YES ──▶
   │           ├── Samples candidate frames across video duration
   │           ├── Evaluates facial landmarks, eye openness & smile expressions
   │           ├── Scores focus quality via Laplacian variance
   │           └── Exports Top 3 best thumbnails to output/thumbnails/
   ▼
[Step 12: Universal Dual Delivery (delivery/cloud_upload.py & delivery/server.py)]
   ├── Save copies to secondary destinations (e.g. Google Drive if requested)
   ├── Launch Channel 1: High-Speed Cloud CDN Upload (Gofile / TmpFiles)
   ├── Launch Channel 2: Multi-Threaded HTTP Range Server on Port 8888
   ├── Establish Localtunnel reverse proxy (global HTTPS link)
   ├── Copy preferred download link to clipboard via OSC 52
   └── Display interactive links and curl commands
   │
   ▼
[CLEANUP & SUCCESS EXIT]
   ├── Wipes run-specific temporary scratch directory (run_dir)
   ├── Frees GPU and system memory caches (cleanup_memory())
   └── Keeps output/ completely intact
```

---

## 🔬 7. Deep Dive: Python File-by-File & Function-by-Function Reference

This section provides an exhaustive technical analysis of every Python file, class, and function across the entire project.

---

### 7.1. Root Orchestration & Lifecycle (`program/*.py`)

#### 📄 `program/main.py`

- **Purpose:** The conductor of the entire orchestra. Coordinates all 12 steps, handles the user interface, parses arguments, and manages run-specific scratch workspaces.
- **Key Functions:**
  - `get_input(source: str, temp_dir: Path, output_dir: Path | None) -> Path`:  
    Determines if `source` is a web URL or local file. If it is a URL, calls `download_url()` to pull the media.
  - `process(args: argparse.Namespace) -> Path`:  
    The master workflow function. Creates an isolated temporary directory `run_XXXXXX` inside `support/temp/`, steps through all audio/video/delivery transformations, handles error catching, copies results to primary and secondary outputs, and cleans up the scratch space.
  - `choose_input_interactive() -> str`:  
    Interactive terminal menu prompting user for Local File, Web URL, or Google Drive path.
  - `choose_output_interactive() -> tuple[str, list[str], str, bool]`:  
    Handles multiple output paths (e.g. `1,3` saves to both `output/` and `/content/drive/MyDrive/AutoCut`).
  - `interactive()`:  
    The main menu loop presenting feature toggles `[x]` and multi-speaker questions.

#### 📄 `program/patch_packages.py`

- **Purpose:** Fixes hardcoded paths in third-party libraries without needing to fork them.
- **Why it is needed:** Third-party packages (specifically `voicefixer`) contain hardcoded calls to `os.path.expanduser("~/.cache/voicefixer")`. On Windows, this litters the user's `C:\Users\username\` folder. `patch_packages.py` surgically replaces these strings inside `site-packages` with `support/checkpoints/voicefixer/`.
- **Key Function:**
  - `patch_voicefixer(site_packages_dir: Path)`: Finds `voicefixer/restorer/model.py` and rewrites cache directories to respect project isolation.

#### 📄 `program/prefetch_models.py`

- **Purpose:** Downloads all required neural network checkpoints during initial setup.
- **Why it is needed:** Prevents the program from freezing or failing during a live video editing session due to an interrupted internet connection.
- **Key Function:**
  - `prefetch_all()`: Sequentially downloads Silero VAD weights, ClearVoice `MossFormer2_SE_48K` models, and VoiceFixer vocoder checkpoints into `support/checkpoints/`.

---

### 7.2. Core & Infrastructure Subsystem (`program/src/core/`)

#### 📄 `src/core/config.py`

- **Purpose:** Defines global constants, paths, presets, and environmental containment rules.
- **Key Highlights:**
  - **Path Constants:** `BASE_DIR`, `ROOT_DIR`, `SUPPORT_DIR`, `OUTPUT_DIR`, `PROJECT_TEMP`, `PROJECT_CHECKPOINTS`, `BIN_DIR`.
  - **Environment Redirections:** Overwrites `TEMP`, `TMPDIR`, `TORCH_HOME`, `HF_HOME`, `XDG_CACHE_HOME` to point strictly inside `support/`.
  - **Thread Scaling:** Automatically inspects CPU core count (`os.cpu_count()`) and sets `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and `OPENBLAS_NUM_THREADS` to `max(1, min(cores - 1, 8))`. This prevents 100% thread locking while maximizing performance.
  - **Cadence Presets:** Dictionary `SILENCE_PRESETS` containing fine-tuned `min_silence`, `keep_pause`, and `energy_threshold` parameters.

#### 📄 `src/core/hardware.py`

- **Purpose:** Comprehensive hardware telemetry and auto-negotiation.
- **Key Class & Functions:**
  - `HWProfile (Dataclass)`: Immutable snapshot of CPU cores, thread allocations, GPU name, total/free VRAM, CUDA capability, NVDEC decoding, and NVENC encoding.
  - `get_hw_profile() -> HWProfile`: Singleton factory function that probes hardware once and caches the result for O(1) future access.
  - `_probe_nvdec() -> bool`: Tests FFmpeg with a 0.5-second test decode using `-hwaccel cuda`. Returns `True` only if hardware decoding is verified functional.
  - `get_best_video_encoder_config() -> tuple[str, list[str], str]`: Determines whether to use `h264_nvenc` (NVIDIA GPU), `h264_vaapi` (Linux Intel/AMD), or `libx264` (CPU fallback).
  - `check_system_dependencies() -> bool`: Inspects FFmpeg, FFprobe, and required Python packages. If any are missing, prints actionable guidance.

#### 📄 `src/core/logger.py`

- **Purpose:** Terminal presentation, progress reporting, and timestamping.
- **Key Functions:**
  - `progress(step_label: str, percent: float, extra: str)`: Renders smooth ANSI progress bars with ETA and GPU telemetry.
  - `log_step(msg: str)`: Prints standardized step dividers with current local time.
  - `fmt_duration(seconds: float) -> str`: Converts raw seconds into human-readable strings (e.g. `1h 14m 22s`).

#### 📄 `src/core/media_tools.py`

- **Purpose:** Low-level media analysis and external CLI invocations.
- **Key Functions:**
  - `probe(path: Path) -> dict`: Executes FFprobe with `-show_format -show_streams -print_format json` to extract stream configurations, codec types, frame rates, and durations.
  - `download_url(url: str, output_dir: Path, suffix: str, temp_dir: Path) -> Path`: Executes `yt-dlp` to download media at maximum available quality.
  - `run_ffmpeg_with_progress(cmd: list, duration: float, step_label: str)`: Spawns an FFmpeg subprocess with `-progress pipe:1`, reading frame/time updates in real-time to render the interactive terminal progress bar.

---

### 7.3. Audio Processing Subsystem (`program/src/audio/`)

#### 📄 `src/audio/vad.py`

- **Purpose:** Multi-stage Hybrid Voice Activity Detection.
- **Key Functions:**
  - `energy_segments(wav_path: Path, ...) -> list[tuple[float, float]]`:  
    Streaming energy-based VAD. Reads audio in 30ms frames, calculates Root Mean Square (RMS) energy, and detects silence gaps with **zero RAM usage**.
  - `silero_segments(wav_path: Path, ...) -> list[tuple[float, float]]`:  
    Loads audio onto GPU as a tensor and runs the deep Silero Neural model in batch mode. Detects human speech patterns with near-perfect accuracy.
  - `hybrid_vad_segments(...) -> list[tuple[float, float]]`:  
    Combines both algorithms. Adds safety margins (front padding for consonants like P, T, K, and tail padding for breath fade-outs) and merges micro-pauses within sentences so the speech flows naturally.

#### 📄 `src/audio/cutter.py`

- **Purpose:** Frame-locked lossless timeline cutting and concatenation.
- **Key Functions:**
  - `extract_audio(src: Path, out_wav: Path, sample_rate: int)`: Rips high-fidelity 16-bit PCM WAV audio from any container.
  - `ffmpeg_concat_cut(src: Path, segments: list, dst: Path, duration: float)`:  
    **The Lip-Sync Anchor.** Writes a dynamic FFmpeg `filter_complex` script where every speech segment is extracted using synchronized video `trim` and audio `atrim` filters:

    ```
    [0:v]trim=start=1.2:end=5.4,setpts=PTS-STARTPTS[v0];
    [0:a]atrim=start=1.2:end=5.4,asetpts=PTS-STARTPTS[a0];
    ...
    [v0][a0][v1][a1]concat=n=N:v=1:a=1[outv][outa]
    ```

    This mathematical alignment guarantees **100% zero audio/video drift**.

#### 📄 `src/audio/cleaner.py`

- **Purpose:** Deep-learning neural speech restoration and echo removal.
- **Key Functions:**
  - `enhance_with_clearvoice(src: Path, dst: Path, temp_dir: Path)`:  
    Uses `MossFormer2_SE_48K` to strip background noise.
    - *Anti-Click Technique:* Processes audio in 60-second chunks with a 1.0-second overlap. Recombines chunks using FFmpeg's `acrossfade` filter, completely preventing boundary clicks.
  - `dereverb_with_voicefixer(src: Path, dst: Path)`:  
    Uses `VoiceFixer` neural vocoder in 30-second sub-chunks to eliminate room reverb.
  - `dereverb_audio(...)`: Tries neural de-reverberation first; if unavailable, seamlessly falls back to high-order acoustic DSP filters (`afftdn`, `agate`, `acompressor`).

#### 📄 `src/audio/mastering.py`

- **Purpose:** Dialogue leveling, acoustic EQ polish, and broadcast loudness normalization.
- **Key Functions:**
  - `pre_level_audio(src: Path, dst: Path)`: Normalizes dynamic range *before* VAD so quiet whispering isn't accidentally cut out as silence.
  - `balance_multispeaker_volume(src: Path, dst: Path)`: Uses dynamic windowing to boost quiet speakers while restraining loud speakers, ensuring comfortable listening.
  - `voice_fine_tuning(src: Path, dst: Path)`: Studio mastering chain injecting warmth (180 Hz), clarity (3.5 kHz), studio air (9.5 kHz), and scooping out boxy mud (420 Hz).
  - `loudness_normalize(src: Path, dst: Path, target: float = -14.0)`: Formats audio to the international **EBU R128 (-14 LUFS)** standard.

---

### 7.4. Video Processing Subsystem (`program/src/video/`)

#### 📄 `src/video/stabilizer.py`

- **Purpose:** Crash-proof, 2-Pass GPU Phase Correlation video deshaker.
- **Key Functions:**
  - `_phase_correlation_shift(f1, f2) -> tuple[dx, dy]`: Computes sub-pixel translation between frames using Fast Fourier Transforms (FFT).
  - `_get_safe_p2_batch(vram_mb: float) -> int`: Dynamically calculates how many full-resolution frames can be buffered on the GPU without risking Out-Of-Memory errors.
  - `stabilize_video(...)`:
    - **Pass 1:** Downscales frames to 320x180 grayscale and calculates motion vectors. Peak VRAM is only **~7 MB**. Writes vectors to `motions.json`.
    - **Pass 2:** Reads original frames, applies smoothed inverse transform matrices, and streams warped frames directly to the NVENC encoder.
    - **Checkpoint Safety:** If interrupted, Pass 1 is skipped on restart by loading `motions.json`.
    - **Fused Mode:** If video enhancement is also requested, it combines deshaking and color polishing inside the same GPU pass, saving ~25 minutes of encoding time!

#### 📄 `src/video/enhancer_filter.py`

- **Purpose:** 100% GPU Tensor Core Cinema Polish & Natural Sharpening.
- **Key Functions:**
  - `profile_video_quality(src: Path, duration: float, meta: dict) -> dict`: Analyzes contrast, exposure, saturation, and compression artifact risk in under 1 second.
  - `_enhance_gpu_tensor(...)`: Runs Difference-of-Gaussians (DoG) spatial convolution on GPU Tensor Cores. Sharpens edges naturally without halos, lifts mid-tone contrast, and polishes skin tones at **100–150+ FPS**.

#### 📄 `src/video/thumbnail.py`

- **Purpose:** AI Expressive Frame Selection & Thumbnail Generation.
- **Key Functions:**
  - `extract_best_thumbnails(...) -> list[Path]`: Extracts candidate frames across the video, evaluates facial landmark clarity, open eyes, positive expressions, and Laplacian sharpness variance. Saves the **Top 3 sharpest, most engaging frames** to `output/thumbnails/`.

---

### 7.5. Packaging & Delivery Subsystem (`program/src/delivery/`)

#### 📄 `src/delivery/muxer.py`

- **Purpose:** Final stream multiplexing and web optimization.
- **Key Functions:**
  - `build_output_suffix(args, meta) -> str`: Generates descriptive file suffixes (e.g. `_AutoCut_Enhanced_Stabilized.mp4`).
  - `mux_final(video_path: Path, audio_path: Path, dst: Path)`: Merges the processed video and audio streams using AAC 192k encoding and applies `-movflags +faststart` for instant web streaming.

#### 📄 `src/delivery/cloud_upload.py` & `src/delivery/server.py`

- **Purpose:** Universal Dual Delivery.
- **Key Functions:**
  - `upload_to_gofile(file_path: Path) -> str | None`: Uploads the master file to Gofile CDN with live progress updates.
  - `upload_to_tmpfiles(file_path: Path) -> str | None`: High-speed backup CDN uploader.
  - `start_http_file_server(file_path: Path, port: int)`: Spins up a multi-threaded Python HTTP server supporting HTTP Range requests (allowing multi-connection acceleration in IDM/Aria2).
  - `start_public_tunnel(port: int) -> tuple[proc, url]`: Exposes the local server to the global internet via `npx localtunnel`, providing an instant HTTPS download link.
  - `trigger_file_download(dst: Path)`: The master delivery coordinator that launches both CDN uploads and local HTTP server + tunnel endpoints.

---

## 💾 8. Data Flow & Scratch File Lifecycle

To keep disk usage clean and predictable, all intermediate processing steps write to a temporary scratch directory created inside `support/temp/run_XXXXXX/`.

Below is the lifecycle map of temporary files during a full run:

```
[support/temp/run_1a2b3c/]
│
├── pre_leveled.wav         <-- Created by Step 1: Pre-leveled audio dynamics
├── vad.wav                 <-- Created by Step 2: 16kHz mono audio for VAD analysis
├── segments.json           <-- Created by Step 2: JSON list of speech time ranges
├── cut_media.mp4           <-- Created by Step 2: Frame-locked cut video/audio
├── audio.wav               <-- Extracted from cut_media.mp4 for audio processing
├── dereverbed.wav          <-- Created by Step 3: Echo-removed audio
├── enhanced.wav            <-- Created by Step 4: ClearVoice AI denoised audio
│   └── ai_chunks/          <-- Sub-chunks (60s with 1s overlap) for ClearVoice
├── multispeaker_balanced.wav <-- Created by Step 5: Leveled multi-speaker audio
├── tuned.wav               <-- Created by Step 6: 5-band studio mastered audio
├── normalized.wav          <-- Created by Step 7: Final -14 LUFS audio track
├── motions.json            <-- Created by Step 8: Pass 1 stabilization motion vectors
├── video_stabilized.mp4    <-- Created by Step 8: Deshaked video stream
├── video_enhanced.mp4      <-- Created by Step 9: Tensor Core polished video
├── _temp_thumb_frames/     <-- Created by Step 11: Candidate thumbnail frames
│
└── [PROCESS COMPLETE]      ==> Final master file moved to output/Master.mp4
                            ==> run_1a2b3c/ is completely deleted via shutil.rmtree()!
```

---

## 🛡️ 9. Zero-RAM Crash Architecture & Hardware Acceleration

### How the Program Eliminates RAM / VRAM Crashes

Traditional media processing tools load entire uncompressed video files into system RAM. A 10-minute 4K video consumes over **35 GB of RAM** when uncompressed into raw pixel arrays, causing immediate out-of-memory crashes on consumer PCs.

This tool solves memory exhaustion through **Sub-Chunk Streaming**:

1. **Audio Streaming:** VAD analyzes audio in 30ms sliding windows (<1 MB RAM). ClearVoice and VoiceFixer operate on 30–60 second sub-chunks, flushing tensors immediately after processing.
2. **Video Streaming via Pipes:** Full-resolution frames are never loaded all at once. FFmpeg decodes frames on-demand into an anonymous OS pipe; Python reads one batch at a time, processes it on the GPU, and pipes the output directly to the hardware encoder.
3. **Phase Correlation Downsampling:** Video stabilization tracks camera motions using tiny 320x180 grayscale representations, keeping Pass 1 VRAM footprint at an ultra-low **~7 MB**.

```
+-----------------------------------------------------------------------------------+
|                        SUB-CHUNK STREAMING MEMORY MODEL                           |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  [Disk: 4K Raw Video] ──▶ [FFmpeg C++ Decoder]                                    |
|                                  │ (Streams only 1-8 frames at a time via pipe)   |
|                                  ▼                                                |
|                           [GPU VRAM Tensor]  <-- Constant Footprint (~15-50 MB!)  |
|                                  │ (Warp / Color Polish / DoG Sharpening)         |
|                                  ▼                                                |
|                           [NVENC C++ Encoder] ──▶ [Disk: Polished Video]          |
|                                                                                   |
|  * RAM usage stays flat whether the video is 1 minute or 5 hours long!            |
+-----------------------------------------------------------------------------------+
```

---

## ⚠️ 10. Edge Cases, Fallbacks & Self-Healing Mechanisms

The tool is engineered to **never crash ungracefully**. Every advanced feature has an automated self-healing fallback:

| Scenario / Edge Case | Primary Engine | Automated Fallback Behavior |
| --- | --- | --- |
| **No NVIDIA GPU detected** | CUDA / NVENC / Tensor | Automatically falls back to multi-core CPU mode (`libx264`, CPU Silero VAD, and FFmpeg vidstab). |
| **GPU VRAM exhaustion risk** | Multi-frame batching | `_get_safe_p2_batch()` detects VRAM limits and falls back to safe single-frame streaming mode. |
| **ClearVoice AI fails or crashes** | Neural MossFormer2 | Automatically catches exception and switches to FFmpeg `afftdn` spectral subtraction denoiser. |
| **VoiceFixer fails or crashes** | Neural Vocoder | Automatically falls back to high-order acoustic DSP dereverberation filter chain. |
| **Variable Frame Rate (VFR) Video** | Native timestamps | Enforces Constant Frame Rate (CFR) flags and explicit PTS rebuilding to avoid audio desync. |
| **Zero speech detected by VAD** | Silero Neural VAD | Falls back to Energy VAD. If still empty, preserves the entire original timeline without cutting. |
| **Localtunnel blocked or offline** | Localtunnel reverse proxy | Gofile CDN direct upload and Local LAN Wi-Fi server continue operating independently. |

---

## 👨‍💻 11. Developer Extension Handbook

### Adding a New Custom Audio Filter

1. Open `program/src/audio/mastering.py`.
2. Define your filter function:

   ```python
   def apply_my_custom_filter(src: Path, dst: Path, duration: float = 0):
       threads = str(get_hw_profile().cpu_threads)
       cmd = [
           ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error",
           "-threads", threads, "-i", str(src),
           "-af", "my_ffmpeg_filter_string",
           "-c:a", "pcm_s16le", str(dst)
       ]
       if duration > 0:
           run_ffmpeg_with_progress(cmd, duration, "[STEP] CUSTOM FILTER")
       else:
           run(cmd)
   ```

3. Call your function inside `program/main.py` at the desired pipeline stage.

### Adding a New Video Transformation

1. Open `program/src/video/enhancer_filter.py`.
2. Insert your PyTorch tensor operation inside `_enhance_gpu_tensor()` or add an FFmpeg video filter flag inside `_build_cinema_filter_string()`.

### Codebase Hygiene Rules

- **Rule 1:** Always use `Path` from `pathlib` for file operations (ensures 100% cross-platform path compatibility between Windows backslashes `\` and Linux forward slashes `/`).
- **Rule 2:** Never write temporary files to `program/` or the project root. Always use `run_dir` inside `support/temp/`.
- **Rule 3:** Never commit files inside `support/` or `output/` to Git.

---
*End of Master Architecture & Codebase Encyclopedia. Built with precision for the Auto Cut & Polishing Tool.* 🚀
