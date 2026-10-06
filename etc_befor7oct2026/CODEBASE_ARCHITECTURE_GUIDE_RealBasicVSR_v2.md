# Auto Cut & Polishing Tool
## Codebase Architecture Guide
### Version: Real-BasicVSR v2 | Date: 2026-09-21

এই গাইডটি কখন পড়বেন:
- নতুন কোনো ফিচার যোগ করতে চাইলে
- কোনো বাগ ট্র্যাক করতে চাইলে
- প্রোগ্রামের ভেতর কী হচ্ছে বুঝতে চাইলে
- AI এজেন্টকে context দিতে চাইলে

---

## ১. সামগ্রিক চিত্র (Big Picture)

```
ব্যবহারকারী
    |
    v
start\windows\run.bat          <- প্রোগ্রাম চালু করার দরজা
    |
    v
program\main.py                <- সব কিছুর কেন্দ্র (Orchestrator)
    |
    |-- core\               <- ভিত্তি (কনফিগ, হার্ডওয়্যার, টুলস)
    |-- audio\              <- অডিও প্রসেসিং পাইপলাইন
    |-- video\              <- ভিডিও প্রসেসিং পাইপলাইন
    `-- delivery\           <- আউটপুট ডেলিভারি (সার্ভার, ক্লাউড)
```

মূল ধারণা: প্রতিটি মিডিয়া ফাইল একটি 10-ধাপের পাইপলাইনে পাস হয়।
প্রতিটি ধাপ স্বাধীন — কোনো ধাপ বন্ধ রাখলে বাকিগুলো নিজেদের মতো চলে।

---

## ২. ফোল্ডার স্ট্রাকচার

```
Auto Cut & Polishing Tool\
|
|-- AGENTS.md                          <- AI এজেন্টের নিয়মকানুন
|-- CODEBASE_ARCHITECTURE_GUIDE.md     <- পুরনো গাইড (D-0.0.1 ভার্সনের)
|-- CODEBASE_ARCHITECTURE_GUIDE_RealBasicVSR_v2.md  <- এই ফাইল
|-- HARDWARE_AND_QUALITY_ANALYSIS_REPORT.md
|-- HARDWARE_AND_QUALITY_ANALYSIS_REPORT_SHISHU_KALYAN_50MIN.md
|-- version_note.txt
|-- .gitignore
|
|-- output\                            <- প্রসেস করা ভিডিও
|
|-- start\
|   |-- windows\
|   |   |-- run.bat      <- প্রোগ্রাম চালু
|   |   |-- setup.bat    <- প্রথমবার সেটআপ
|   |   `-- clean.bat    <- ক্যাশ/টেম্প পরিষ্কার
|   `-- linux_mac\
|       |-- run.sh
|       |-- setup.sh
|       `-- clean.sh
|
`-- program\
    |-- main.py              <- মেইন অর্কেস্ট্রেটর (539 লাইন)
    |-- patch_packages.py    <- VoiceFixer source-level patch
    |-- prefetch_models.py   <- Setup-এর সময় AI মডেল আগে থেকে ডাউনলোড
    |-- requirements.txt
    |
    |-- venv\                <- Python Virtual Environment
    |
    |-- support\             <- সব রানটাইম ফাইল (C: drive isolated)
    |   |-- bin\             <- FFmpeg, FFprobe binary
    |   |-- checkpoints\     <- AI মডেলের weights
    |   |   |-- realbasicvsr\    <- Real-BasicVSR (141 MB)
    |   |   |-- MossFormer2_SE_48K\  <- ClearVoice audio model
    |   |   |-- voicefixer\      <- VoiceFixer audio model
    |   |   `-- hub\             <- Silero-VAD (torch hub cache)
    |   |-- cache\           <- Python/matplotlib cache
    |   `-- temp\            <- রানটাইম temporary ফাইল
    |
    `-- src\
        |-- __init__.py
        |-- core\
        |-- audio\
        |-- video\
        `-- delivery\
```

---

## ৩. Source Files বিস্তারিত

### core\ — ভিত্তি স্তর

#### core/config.py
দায়িত্ব:
  - সব Path নির্ধারণ (BASE_DIR, OUTPUT_DIR, SUPPORT_DIR)
  - Environment isolation: সব cache program\support\ এ রাখা
  - অডিও constants (SAMPLE_RATE=48000, VAD_RATE=16000)
  - Silence Presets (natural / standard / aggressive)
  - cleanup_memory() — VRAM ও RAM release

গুরুত্বপূর্ণ Variables:
  BASE_DIR            = program\
  OUTPUT_DIR          = Auto Cut & Polishing Tool\output\
  SUPPORT_DIR         = program\support\
  PROJECT_CHECKPOINTS = program\support\checkpoints\
  BIN_DIR             = program\support\bin\


#### core/hardware.py
দায়িত্ব:
  - GPU ও VRAM তথ্য (nvidia-smi -> PyTorch fallback)
  - CUDA availability চেক
  - Vulkan availability চেক
  - Video encoder auto-negotiation (NVENC > QSV > AMF > libx264)
  - print_system() — Dynamic startup banner (কোনো hardcode নেই)
  - check_system_dependencies() — Silent system check

Key Functions:
  gpu_info()                      -> dict (name, vram_total, vram_used, util)
  gpu_status_text()               -> "GPU 45% | VRAM 1024/4096MB"
  torch_cuda_available()          -> bool
  vulkan_available()              -> bool
  get_best_video_encoder_config() -> (codec, flags, display_name)
  nvenc_available()               -> bool
  print_system()                  -> startup banner
  check_system_dependencies()     -> bool

[Real-BasicVSR v2 পরিবর্তন]:
  - print_system() এখন সম্পূর্ণ dynamic
  - Video AI: disk probe করে — installed model দেখায়, না থাকলে "None cached"
  - Audio AI: ClearVoice, VoiceFixer, Silero-VAD আলাদাভাবে চেক করে
  - Silero: project cache -> torch hub -> Python import — তিন স্তরে চেক
  - যেকোনো নতুন VSR model checkpoints\<name>\*.pth ফাইল রাখলে auto-detect


#### core/media_tools.py
দায়িত্ব:
  - FFmpeg/FFprobe path খোঁজা (PATH -> support\bin\ fallback)
  - মিডিয়া probe (duration, width, height, fps, audio/video track)
  - URL detection (YouTube, Facebook, Direct stream)
  - yt-dlp দিয়ে URL থেকে ডাউনলোড
  - Title sanitization

Key Functions:
  ffmpeg_path()    -> FFmpeg executable path
  ffprobe_path()   -> FFprobe executable path
  probe(path)      -> dict (duration, width, height, fps, audio, video)
  is_url(s)        -> bool
  download_url()   -> downloaded file Path
  sanitize_title() -> safe filename string


#### core/logger.py
Key Functions:
  eprint()              -> stderr error print
  progress()            -> [STEP] message
  fmt_time()            -> "1h 23m 45s"
  fmt_duration()        -> seconds to human-readable
  fmt_duration_clock()  -> "01:23:45" clock format
  log_step()            -> step separator line
  current_timestamp_str() -> "2026-09-21 20:09:52"

---

### audio\ — অডিও পাইপলাইন

#### audio/vad.py — Voice Activity Detection
Engine:
  1. Energy-based VAD (দ্রুত, offline)
  2. Silero-VAD (AI, সর্বোচ্চ নির্ভুলতা)
  3. Hybrid (Energy + Silero) <- Default

Key Functions:
  hybrid_vad_segments()  -> [(start, end), ...] segment list
  silero_segments()      -> Silero AI segments
  energy_segments()      -> Energy-based segments
  normalize_segments()   -> Overlap ও edge smooth করা
  write_segments_json()  -> JSON ফাইলে সেভ

Parameters:
  min_silence      = 0.50s (default)
  keep_pause       = 0.30s (default)
  energy_threshold = 0.018 (default)


#### audio/cutter.py — অডিও ও ভিডিও কাটা
Key Functions:
  extract_audio()      -> ভিডিও থেকে WAV বের করা
  extract_vad_audio()  -> VAD-এর জন্য 16kHz mono WAV
  ffmpeg_concat_cut()  -> Segment অনুযায়ী ভিডিও+অডিও কাটা (frame-locked)
  ffmpeg_cut_audio()   -> শুধু অডিও কাটা


#### audio/cleaner.py — AI Audio Enhancement
Engines (auto-detection order):
  1. ClearVoice MossFormer2_SE_48K — Primary (48kHz, 60s chunks)
  2. VoiceFixer — Fallback (Mode 2 for dereverberation)

Key Functions:
  enhance_audio()   -> Noise removal ও voice clarity
  dereverb_audio()  -> Echo ও room reverb removal


#### audio/mastering.py — Audio Mastering
Key Functions:
  pre_level_audio()           -> Input dynamics normalize (FFmpeg loudnorm)
  loudness_normalize()        -> EBU R128 -14 LUFS
  voice_fine_tuning()         -> Studio vocal EQ (highpass 80Hz, presence boost 3kHz)
  balance_multispeaker_volume() -> একাধিক বক্তার volume সমান করা

---

### video\ — ভিডিও পাইপলাইন

#### video/stabilizer.py — Video Stabilization
Engine:
  GPU mode: FFmpeg vidstabdetect + vidstabtransform (CUDA)
  CPU mode: fallback

Fusion Mode (combine_enhance=True):
  Stabilization + Real-BasicVSR একসাথে -> Single pass

Key Function:
  stabilize_video(src, dst, duration, run_dir, combine_enhance=False, engine="gpu")


#### video/enhancer_ai.py — AI Video Super-Resolution (PRIMARY)
[Real-BasicVSR v2 — Main Engine]

Architecture:
  SmartFrameJudge:
    - প্রতিটি frame analyze করে
    - Inter-frame motion diff (L1 norm)
    - Laplacian blur variance
    -> "Anchor frame" হলে: Real-BasicVSR দিয়ে process
    -> অন্যথায়: Optical flow warp (150+ FPS)

Processing:
  - 16-frame streaming chunks (4GB VRAM safe)
  - FFmpeg pipe (RAM overflow নেই)
  - FP16 inference
  - torch.compile() guard: CC >= 7.0 only (GTX 1050 Ti safe)

Checkpoint:
  program\support\checkpoints\realbasicvsr\realbasicvsr_c64b20_reds.pth (141 MB)

Key Functions:
  enhance_video_ai()              -> Main entry point
  SmartFrameJudge                 -> Anchor frame detection class
  propagate_intermediate_frames() -> Optical flow warping


#### video/enhancer_filter.py — GPU Tensor Filter (Fallback)
Real-BasicVSR না থাকলে চলে:
  - Upscale (1920x1080, lanczos)
  - Sharpen (unsharp mask)
  - Denoise (hqdn3d)
  - Color grade (curves, eq)
  - NVENC encoding

Key Function:
  enhance_video()


#### video/thumbnail.py — AI Thumbnail Generator
Algorithm:
  1. Candidate frames নেওয়া (সমান ব্যবধানে)
  2. Expressiveness score: contrast + sharpness + brightness
  3. Top-3 JPEG export

Key Function:
  extract_best_thumbnails()

---

### delivery\ — Output Delivery

#### delivery/muxer.py
Key Functions:
  mux_final()          -> Video + Audio -> MP4 (NVENC encoding)
  build_output_suffix() -> "_AutoCuted-AudVidEnhanced-..." filename suffix

#### delivery/server.py
  - Port 8888
  - HTTP Range Request support (IDM/browser resume)
  - Local: http://localhost:8888/
  - LAN: http://192.168.x.x:8888/

#### delivery/cloud_upload.py
Methods (priority order):
  1. Colab files.download() (Colab environment)
  2. Gofile CDN upload (public download link)
  3. Local HTTP server (Windows/Linux)

Key Function:
  trigger_file_download(path)

---

## ৪. 10-ধাপ পাইপলাইন (main.py::process())

```
INPUT (local file / URL / Google Drive)
    |
    v
[Step 1]  Pre-level Audio          mastering.py::pre_level_audio()
    |
    v
[Step 2]  Silence Removal          vad.py + cutter.py
          Hybrid VAD -> frame-locked video cut
    |
    v
[Step 3]  Echo/Reverb Removal      cleaner.py::dereverb_audio()
          VoiceFixer Mode 2
    |
    v
[Step 4]  AI Audio Enhancement     cleaner.py::enhance_audio()
          ClearVoice MossFormer2 (48kHz, 60s chunks)
    |
    v
[Step 5]  Multi-Speaker Balance    mastering.py::balance_multispeaker_volume()
    |
    v
[Step 6]  Vocal Fine-Tuning        mastering.py::voice_fine_tuning()
    |
    v
[Step 7]  Loudness Normalization   mastering.py::loudness_normalize()
          EBU R128 -14 LUFS
    |
    v
[Step 8]  Video Stabilization      stabilizer.py::stabilize_video()
          GPU deshake [Fusion: Step 8+9 single pass]
    |
    v
[Step 9]  AI Video Enhancement     enhancer_ai.py::enhance_video_ai()
          Real-BasicVSR (141MB, FP16, 16-frame chunks)
          Fallback: enhancer_filter.py::enhance_video()
    |
    v
[Step 10] Final Mux                muxer.py::mux_final()
          Video + Audio -> MP4 (NVENC)
    |
    v
[Step 11] Thumbnail Extraction     thumbnail.py (optional)
    |
    v
[Step 12] Delivery                 server.py + cloud_upload.py
    |
    v
OUTPUT\
```

---

## ৫. CLI Flag Reference

```bash
# সব ধাপ একসাথে
python main.py --input video.mp4 --all

# নির্দিষ্ট ধাপ
python main.py --input video.mp4 --remove-silence --dereverb --audio-enhance --normalize --stabilize --video-enhance

# Video engine
--video-engine auto          # Default (realbasicvsr -> tensor fallback)
--video-engine realbasicvsr  # Only Real-BasicVSR
--video-engine tensor        # Only GPU filter (fast, lower quality)

# VAD engine
--vad-engine hybrid          # Default (Energy + Silero)
--vad-engine silero          # Silero AI only
--vad-engine energy          # Energy-based only (fast, offline)

# Silence tuning
--min-silence 0.5            # Default: 0.5s
--keep-pause 0.30            # Default: 0.30s
--energy-threshold 0.018     # Default: 0.018

# Audio
--engine auto                # ClearVoice -> VoiceFixer -> DSP fallback
--target-lufs -14.0          # Default: -14 LUFS

# Other
--output ./my_output         # Output folder
--output-name "my_file"      # Custom filename
--download                   # Serve via HTTP server
--overwrite                  # Overwrite existing output
--keep-temp                  # Keep temp files (for debug)
--multi-speaker-balance      # Multi-speaker mode
```

---

## ৬. Environment Isolation

সমস্যা: PyTorch, HuggingFace, pip সবাই C:\Users\...\AppData তে ফাইল রাখে।

সমাধান (core/config.py):
  TORCH_HOME             -> program\support\checkpoints\
  HF_HOME                -> program\support\checkpoints\
  HUGGINGFACE_HUB_CACHE  -> program\support\checkpoints\hub\
  VOICEFIXER_CACHE       -> program\support\checkpoints\voicefixer\
  PIP_CACHE_DIR          -> program\support\temp\pip_cache\
  PYTHONUSERBASE         -> program\support\py_userbase\
  TMPDIR/TEMP/TMP        -> program\support\temp\
  XDG_CACHE_HOME         -> program\support\cache\

ফলে: পুরো প্রোগ্রাম self-contained।
     clean.bat চালালে support\ মুছলেই সব পরিষ্কার।

---

## ৭. AI Model তালিকা

| Model | Location | Size | Purpose |
|-------|----------|------|---------|
| Real-BasicVSR | checkpoints\realbasicvsr\realbasicvsr_c64b20_reds.pth | 141 MB | Video Super-Resolution |
| ClearVoice MossFormer2 | checkpoints\MossFormer2_SE_48K\ | ~300 MB | Audio noise removal |
| VoiceFixer | checkpoints\voicefixer\ | ~500 MB | Audio dereverberation |
| Silero-VAD | checkpoints\hub\snakers4_silero*\ | ~5 MB | Voice Activity Detection |

---

## ৮. GTX 1050 Ti বিশেষ বিবেচনা (CUDA CC 6.1)

সীমাবদ্ধতা:
  - torch.compile() কাজ করে না (Triton only supports CC >= 7.0)
  - Tensor Cores নেই (FP16 software emulation)
  - 4GB VRAM — বড় model একসাথে লোড করা যাবে না

সমাধান (enhancer_ai.py):
  - torch.compile() guard:
      if cap[0] >= 7: model = torch.compile(model)  # Ampere+
      else: model.eval()                              # Pascal fallback
  - FP16 inference (VRAM বাঁচায়)
  - 16-frame streaming chunks
  - cleanup_memory() প্রতিটি ধাপের পরে

---

## ৯. Startup Flow

```
run.bat
  |
  v
  venv activate
  |
  v
  python main.py (no args -> interactive mode)
  |
  v
  check_system_dependencies()  -> [OK] বা issues list
  |
  v
  print_system()               -> Dynamic hardware banner:
    GPU Hardware   : NVIDIA GeForce GTX 1050 Ti    (nvidia-smi)
    VRAM Telemetry : 575 MB / 4096 MB              (live)
    CUDA Compute   : Active (PyTorch 2.6.0 | CC 6.1)
    Video AI Engine: Real-BasicVSR (141 MB, FP16)  (disk probe)
    Audio AI Engine: ClearVoice/MossFormer2, VoiceFixer, Silero-VAD
    Video Encoder  : NVIDIA NVENC (GPU)             (FFmpeg probe)
  |
  v
  choose_input_interactive()   -> 1=Local / 2=URL / 3=Drive
  |
  v
  Feature selection menu       -> [x] 1-7 বা A=all
  |
  v
  process(args)                -> 10-step pipeline
```

---

## ১০. Output Filename Convention

```
{source_title}{suffix}.mp4

Suffix (active steps based):
  -AutoCuted       -> remove_silence=True
  -AudEnhanced     -> audio_enhance=True
  -EchoRemoved     -> dereverb=True
  -Normalized      -> normalize=True
  -Stabilized      -> stabilize=True
  -VidEnhanced     -> video_enhance=True

Example:
  শিশু কল্যান পরিষদ_AutoCuted-AudVidEnhanced-EchoRemoved-Stabilized.mp4
```

---

## ১১. Known Issues (Real-BasicVSR v2)

| Issue | Cause | Status |
|-------|-------|--------|
| torch.compile() freeze on GTX 1050 Ti | Triton CC<7.0 | Fixed: CC>=7 guard |
| MJPEG thumbnail error | pixel format mismatch | Fixed: -pix_fmt yuvj420p |
| ClearVoice CPU-only (slow) | No GPU acceleration | Open issue |
| Video quality loss with Real-BasicVSR | Model/param tuning needed | Under evaluation |
| Processing time 7-8h for 50min video | Heavy AI pipeline | Optimization pending |

---

## ১২. নতুন ফিচার যোগ করার গাইড

### নতুন অডিও ধাপ:
1. audio\ তে নতুন .py ফাইল তৈরি
2. audio\__init__.py তে import যোগ
3. main.py::process() তে নতুন Step যোগ
4. build_parser() তে CLI flag যোগ
5. Interactive menu তে items list এ যোগ

### নতুন Video AI Model:
1. video\enhancer_ai.py তে engine logic লিখুন
2. PROJECT_CHECKPOINTS / "model_name" / "weights.pth" এ রাখুন
3. hardware.py::print_system() এ আপনাআপনি detect হবে (dynamic glob)
4. prefetch_models.py তে download logic যোগ করুন

---

*Guide Created: 2026-09-21 | Version: Real-BasicVSR v2*
*পরবর্তী গাইড: ভিডিও কোয়ালিটি মূল্যায়নের পর নতুন VSR engine নির্বাচনের উপর ভিত্তি করে*
