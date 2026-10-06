# 📋 Auto Cut & Polishing Tool — সম্পূর্ণ চ্যাট সারাংশ

> **প্রজেক্ট ফাইল:** `main.py` (single-file AI audio & video processing tool)
> **চ্যাট সময়কাল:** ২১ আগস্ট ২০২৬ থেকে ২৯ আগস্ট ২০২৬
> **মূল ফোকাস:** `main.py`-এর pipeline কারেকশন, সিঙ্ক বাগ ফিক্স, এবং আর্কিটেকচার অপ্টিমাইজেশন

---

## 🗓️ দিনওয়ারি উন্নয়নের ধারা

---

### 📅 ২১ আগস্ট ২০২৬ — প্রথম পরিকল্পনা ও ভিত্তি স্থাপন

**পরিস্থিতি:** ইউজার একটি নতুন AI মিডিয়া প্রসেসিং টুল তৈরির পরিকল্পনা নিয়ে আসেন।

**আলোচনা ও সিদ্ধান্ত:**
- টুলটির ফিচার লিস্ট নির্ধারণ: Silence Removal, AI Voice Enhancement, Echo Removal, Loudness Normalization, Video Stabilization, Video Enhancement
- সিঙ্গেল-ফাইল আর্কিটেকচার ঠিক করা (`main.py`)
- GPU-first (CUDA/NVENC), CPU fallback
- ইনপুট: Local file, URL (yt-dlp), Google Drive

---

### 📅 ২২–২৩ আগস্ট ২০২৬ — Silence Removal ও Audio/Video Sync সমস্যা চিহ্নিত

**সমস্যা (ইউজারের রিপোর্ট):**
> *"অডিও আর ভিডিওকে কোনোভাবেই আলাদা করা যাবে না যখন সাইলেন্ট অংশ রিমুভ করতেছে... টাইমলাইনে অডিও ও ভিডিও একসাথে রেখে টাইমলাইন ধরে কাটতে হবে।"*

**রুট কজ (চিহ্নিত):**
পুরনো কোড ভিডিও চ্যাঙ্কগুলো **আলাদা আলাদাভাবে** FFmpeg দিয়ে কাটত এবং পরে জোড়া দিত। এর ফলে:
- প্রতিটি চ্যাঙ্কে AAC encoder-এর ~20ms priming delay জমত
- 50–100টা কাট মিলিয়ে **1–3 সেকেন্ডের lip-sync drift** তৈরি হত

**সমাধান:**
পুরনো chunk-based slicing বাদ দিয়ে **Single-Pass FFmpeg `filter_complex`** সিস্টেমে রূপান্তর:

```
[0:v]trim=start:end,setpts=PTS-STARTPTS[v0];
[0:a]atrim=start:end,asetpts=PTS-STARTPTS[a0];
...concat=n=N:v=1:a=1[outv][outa]
```

এতে FFmpeg একটিমাত্র অভ্যন্তরীণ frame clock-এ অডিও ও ভিডিও একসাথে কাটে — কোনো container timestamp shift হয় না।

---

### 📅 ২৪ আগস্ট ২০২৬ — Audio Treatment Pipeline ক্রম সংশোধন

**ইউজারের নির্দেশ:**
> *"অডিও ট্রিটমেন্টের ক্রমটা হবে নিচের মত — প্রথম স্টেপ প্রি-লেভেলিং, দ্বিতীয় স্টেপ সাইলেন্ট রিমুভাল, তারপর ইকো রিমুভাল, লাউডনেস নরমালাইজেশন, AI ভয়েস এনহ্যান্সমেন্ট..."*

**আগের ভুল ক্রম:**
```
❌ Silence Removal → Pre-Leveling → AI Enhancement → Dereverb → Normalize
```

**সঠিক ক্রম (implement করা হয়েছে):**
```
✅ 1. Pre-Leveling (Gain Staging)
✅ 2. Silence Removal & Auto-Cut (frame-locked single-pass)
✅ 3. Echo & Reverb Removal (Dereverberation)
✅ 4. Loudness Normalization (-14 LUFS)
✅ 5. AI Voice Enhancement (ClearVoice)
✅ 6. Vocal Studio Fine-Tuning & Mastering
✅ 7. Video Stabilization
✅ 8. Video Enhancement
✅ 9. Final Mux
```

**কারণ:** Pre-Leveling না করে AI মডেলে পাঠালে অডিও হয় অনেক বেশি শান্ত বা অনেক বেশি উচ্চস্বরে যায়, ফলে AI-এর কাজের মান কমে। Pre-Leveling করলে AI সবসময় একটি স্বাভাবিক ভলিউম পায়।

---

### 📅 ২৫ আগস্ট ২০২৬ — Video Treatment ক্রম যাচাই

**আলোচনা:**
ইউজার ভিডিও ট্রিটমেন্টের সঠিক ক্রম জানতে চান। নিশ্চিত করা হয়:
1. Video Stabilization (2-pass VidStab) — প্রথমে ঝাঁকুনি ঠিক করা
2. Video Enhancement (শার্পনিং + কালার) — তারপর ভিজুয়াল কোয়ালিটি উন্নত করা
3. Final Mux — শেষে অডিও-ভিডিও একত্রিত করা

---

### 📅 ২৫–২৬ আগস্ট ২০২৬ — Video Enhancement: Destructive Bug ফিক্স

**সমস্যা (ইউজারের রিপোর্ট):**
> *"যে জায়গায় ফ্রেম ভালো... সেই জায়গায় এনহ্যান্সমেন্ট করে সেখানেও ঘোলা করে দিচ্ছে... যে ফ্রেম ভালো আছে সেটার ওপর এক্সট্রিম এনহ্যান্সমেন্ট দেওয়ার দরকার নেই।"*

**রুট কজ:**
পুরনো কোডে `bilateral` blur ফিল্টার ব্যবহার হত যা সব ফ্রেমে একই মাত্রায় প্রয়োগ হত এবং ইতোমধ্যে শার্প ফ্রেমকে ঘোলা করে দিত।

**সমাধান — Non-Destructive Enhancement:**

```python
# আগে (ভুল):
bilateral=d=1:sigmaColor=0.04:sigmaSpace=2.0

# পরে (সঠিক):
cas=0.30  # Contrast-Adaptive Sharpening
eq=contrast=1.04:brightness=0.01:saturation=1.05
```

CAS স্বয়ংক্রিয়ভাবে ফ্রেমের কনটেন্ট বিশ্লেষণ করে — যেখানে দরকার সেখানে শার্পনেস দেয়, যেখানে দরকার নেই সেখানে দেয় না।

---

### 📅 ২৬ আগস্ট ২০২৬ — AI মডেল আর্কিটেকচার ব্যাখ্যা

**প্রোগ্রামে ব্যবহৃত AI মডেলগুলো:**

| স্টেপ | মডেল | কাজ | কেন এটা? |
|---|---|---|---|
| Silence Detection | **Silero VAD** + Energy VAD (Hybrid) | কথা ও নীরবতা শনাক্ত | দুটো পদ্ধতির intersection — সর্বোচ্চ নির্ভুলতা |
| Echo/Reverb Removal | **VoiceFixer** (mode=2) | Room acoustic শুষে নেয় | শুধু dereverberation, noise removal নয় |
| Voice Enhancement | **ClearVoice** (MossFormer2_SE_48K) | Background noise সম্পূর্ণ মুছে | Deep learning speech separation |

**কেন VoiceFixer আগে, ClearVoice পরে:**
VoiceFixer যদি ঘরের ইকো সরিয়ে না দেয়, তাহলে ClearVoice সেই ইকোকে "speech feature" ভেবে রেখে দেবে। তাই ক্রমটা critical।

---

### 📅 ২৭ আগস্ট ২০২৬ — Hybrid VAD আর্কিটেকচার (২-ধাপ সিস্টেম)

**ইউজারের প্রশ্ন:** Silero এবং Energy VAD-এর হাইব্রিড কম্বিনেশন কীভাবে কাজ করে?

**Pass 1 — Energy VAD (দ্রুত স্ক্যান):**
- অডিওর RMS energy পরিমাপ করে (threshold = `0.015`)
- দ্রুত কিন্তু fan noise-কে মাঝে মাঝে "speech" ভাবে

**Pass 2 — Silero Neural AI (গভীর বিশ্লেষণ):**
- Deep neural network — মানুষের গলার vocal tract pattern চেনে (threshold = `0.5`)
- ফ্যানের বাতাস বা keyboard click-কে speech হিসেবে গণ্য করে না

**Intersection (চূড়ান্ত সিদ্ধান্ত):**
```
final_segments = Energy VAD ∩ Silero VAD
```
শুধুমাত্র যখন **দুটোই একমত** — সেই অংশই রাখা হয়।
Fail-safe: Silero unavailable হলে Energy VAD একা কাজ চালায়।

**Merge Parameters:**
- `keep_pause = 0.35s` — কথার শুরু ও শেষে 350ms padding
- `gap = 0.08s` — 80ms-এর কম ফাঁক থাকলে দুটো segment জোড়া দেওয়া হয়

---

### 📅 ২৮ আগস্ট ২০২৬ — clean.bat রিরাইট

**সমস্যা:** পুরনো `clean.bat` `venv/` ডিলিট করতে ব্যর্থ হত কারণ Python process-এর locked ফাইল থাকত।

**সমাধান:**
```batch
taskkill /F /IM python.exe   ← locked process সরানো
robocopy empty_dir venv /MIR ← Windows-safe force purge
rmdir /s /q venv
```
`checkpoints/` ডিলিটের আগে ইউজারকে জিজ্ঞেস করা হয় (AI model weights মূল্যবান)।

---

### 📅 ২৮ আগস্ট ২০২৬ — Whisper Experiment (বাদ দেওয়া হয়েছে)

**পরীক্ষা:** `main copy.py` নামে একটি experimental ফাইল তৈরি হয়েছিল যেখানে Whisper ASR দিয়ে ৩-ধাপের VAD pipeline পরীক্ষা করা হয়েছিল।

**ফলাফল:**
> ইউজার: *"না আসলে আমাদের এক্সপেরিমেন্টটা ফেইল করছে... হুইস্পার দিয়ে কাজ করাটা পসিবল না।"*

**সিদ্ধান্ত:** Whisper সম্পূর্ণ বাদ। `requirements.txt` থেকে `faster-whisper` সরানো হয়েছে। `main.py`-তে Whisper কোড কখনও যোগ করা হয়নি।

---

### 📅 ২৯ আগস্ট ২০২৬ — ডকুমেন্টেশন অডিট ও পারফেক্ট করা

| ফাইল | সমস্যা | সংশোধন |
|---|---|---|
| `requirements.txt` | `faster-whisper` ছিল | সম্পূর্ণ সরানো হয়েছে |
| `README.md` | Pipeline ক্রম ভুল ছিল | সঠিক ক্রমে লেখা হয়েছে |
| `README.md` | `setup.sh`/`run.sh` মেনশন (ফাইল নেই) | সরিয়ে Linux manual command দেওয়া |
| `README.md` | `--vad-engine` CLI arg ছিল না | CLI টেবিলে যোগ করা হয়েছে |
| `README.md` | Video Enhancement "blur" বলছিল | Non-destructive CAS সঠিকভাবে লেখা |
| `run.bat` | কোনো startup banner ছিল না | Python পাথ ও নাম দেখানো হয় |

---

## 🏗️ চূড়ান্ত `main.py` আর্কিটেকচার (বর্তমান অবস্থা)

```
main.py (১৭২২ লাইন, ~68KB)
│
├── Constants & Configuration
│   ├── SAMPLE_RATE = 48,000 Hz
│   ├── DEFAULT_MIN_SILENCE = 1.0s
│   ├── DEFAULT_KEEP_PAUSE = 0.35s
│   └── energy_threshold = 0.015
│
├── System Detection
│   ├── is_colab()         — Google Colab শনাক্ত
│   ├── gpu_info()         — NVIDIA GPU/VRAM মনিটর
│   └── nvenc_available()  — Hardware video encoder চেক
│
├── Audio Pipeline
│   ├── pre_level_audio()       — [Step 1] Gain staging (I=-18, TP=-2dB)
│   ├── hybrid_vad_segments()   — [Step 2] 2-pass Energy+Silero VAD
│   ├── ffmpeg_concat_cut()     — [Step 2] Single-pass filter_complex cut
│   ├── dereverb_audio()        — [Step 3] VoiceFixer mode=2
│   ├── loudness_normalize()    — [Step 4] EBU R128 -14 LUFS
│   ├── enhance_audio()         — [Step 5] ClearVoice MossFormer2_SE_48K
│   └── voice_fine_tuning()     — [Step 6] EQ + Compressor + True Peak Limiter
│
├── Video Pipeline
│   ├── stabilize_video()       — [Step 7] 2-pass VidStab (shakiness=8, smooth=16)
│   └── enhance_video_fast()    — [Step 8] Non-destructive CAS=0.30 + eq
│
├── I/O System
│   ├── download_url()          — yt-dlp + Raw file retention
│   ├── serve_download()        — HTTP browser-based download
│   ├── choose_input_interactive()
│   └── choose_output_interactive()
│
└── Entry Points
    ├── interactive()    — Terminal guided mode (double-click run.bat)
    ├── build_parser()   — CLI argument parser
    └── main()           — Auto-detect: interactive vs CLI
```

---

## 📊 প্রজেক্ট ফাইল স্ট্রাকচার (বর্তমান)

```
AI_Media_Processor_V3--chatgpt/
├── main.py                      ✅ মূল প্রোগ্রাম (১৭২২ লাইন)
├── requirements.txt             ✅ সংশোধিত (faster-whisper সরানো)
├── setup.bat                    ✅ Windows one-click installer
├── run.bat                      ✅ সংশোধিত (startup banner যোগ)
├── clean.bat                    ✅ সম্পূর্ণ রিরাইট (force-kill + robocopy)
├── README.md                    ✅ সম্পূর্ণ অডিট ও সংশোধন
├── lkl2aiEav__main - Copy.py   📌 গোল্ডেন রেফারেন্স (ডিলিট করবেন না)
├── venv/                        🔒 Virtual environment
├── checkpoints/                 🤖 AI model weights cache
├── temp/                        🗑️ Processing temp files
└── output/                      📁 সব প্রসেসড ফাইল এখানে
```

---

## 🔑 মূল টেকনিক্যাল সিদ্ধান্তসমূহ

| সিদ্ধান্ত | কারণ |
|---|---|
| **Single-pass filter_complex** | Lip-sync drift সম্পূর্ণ দূর করতে |
| **Pre-Leveling আগে, AI পরে** | AI model-এ consistent ভলিউম যেন যায় |
| **VoiceFixer আগে, ClearVoice পরে** | ইকো সরিয়ে তারপর noise remove |
| **Hybrid VAD (Energy ∩ Silero)** | Fan noise-এ ভুল কাট ও slow processing দুটোই এড়ানো |
| **Non-destructive CAS** | ভালো ফ্রেম নষ্ট না করে শুধু দরকারমতো শার্পনেস |
| **Whisper বাদ** | RAM ও speed-এ অগ্রহণযোগ্য overhead |

---

> 📌 **নোট:** `main copy.py` (Whisper experimental) এবং `lkl2aiEav__main - Copy.py` (গোল্ডেন রেফারেন্স) — এই দুটি ফাইল production pipeline-এর অংশ নয়। শুধুমাত্র reference হিসেবে রাখা আছে।
