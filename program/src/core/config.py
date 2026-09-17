# -*- coding: utf-8 -*-
"""
core.config
-----------
Central configuration, paths, environment isolation, and audio/video presets.
"""

from __future__ import annotations

import os
import sys
import gc
import tempfile
from pathlib import Path

# Base workspace directory (program/ folder — one level above src/)
BASE_DIR    = Path(__file__).resolve().parent.parent.parent
SRC_DIR     = BASE_DIR / "src"
# Support folder: all runtime/downloaded files stay here (keeps program/ clean)
SUPPORT_DIR = BASE_DIR / "support"
# Output goes to the project root (one level above program/)
OUTPUT_DIR  = BASE_DIR.parent / "output"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Isolated runtime directories inside program/support/
PROJECT_TEMP        = SUPPORT_DIR / "temp"
PROJECT_CHECKPOINTS = SUPPORT_DIR / "checkpoints"
PROJECT_CACHE       = SUPPORT_DIR / "cache"
PROJECT_USERBASE    = SUPPORT_DIR / "py_userbase"
BIN_DIR             = SUPPORT_DIR / "bin"

# Set environment isolation
os.environ["TMPDIR"]   = str(PROJECT_TEMP)
os.environ["TEMP"]     = str(PROJECT_TEMP)
os.environ["TMP"]      = str(PROJECT_TEMP)
tempfile.tempdir       = str(PROJECT_TEMP)

os.environ["TORCH_HOME"]        = str(PROJECT_CHECKPOINTS)
os.environ["TORCH_EXTENSIONS_DIR"] = str(PROJECT_CACHE / "torch_extensions")

os.environ["HF_HOME"]                = str(PROJECT_CHECKPOINTS)
os.environ["TRANSFORMERS_CACHE"]     = str(PROJECT_CHECKPOINTS / "transformers")
os.environ["HF_DATASETS_CACHE"]      = str(PROJECT_CHECKPOINTS / "datasets")
os.environ["HUGGINGFACE_HUB_CACHE"]  = str(PROJECT_CHECKPOINTS / "hub")

os.environ["ASTEROID_CACHE"]              = str(PROJECT_CHECKPOINTS / "asteroid")
os.environ["SPEECHBRAIN_CACHE"]           = str(PROJECT_CHECKPOINTS / "speechbrain")
os.environ["SENTENCE_TRANSFORMERS_HOME"]  = str(PROJECT_CHECKPOINTS / "sentence_transformers")
os.environ["VOICEFIXER_CACHE"]            = str(PROJECT_CHECKPOINTS / "voicefixer")
os.environ["VOICEFIXER_HOME"]             = str(PROJECT_CHECKPOINTS / "voicefixer")

os.environ["PIP_CACHE_DIR"] = str(PROJECT_TEMP / "pip_cache")
os.environ["PYTHONUSERBASE"] = str(PROJECT_USERBASE)
os.environ["XDG_CACHE_HOME"] = str(PROJECT_CACHE)
os.environ["MPLCONFIGDIR"] = str(PROJECT_CACHE / "matplotlib")

# Thread limit to prevent CPU throttling
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")

# Audio Constants
SAMPLE_RATE = 48_000
CHANNELS = 1
SAMPLE_WIDTH = 2
VAD_RATE = 16_000
VAD_CHUNK_SECONDS = 30
AI_CHUNK_SECONDS = 10

# Audio Silence Presets
SILENCE_PRESETS = {
    "natural": {
        "name": "Natural Cadence (Smooth & Breathing)",
        "min_silence": 0.50,
        "keep_pause": 0.30,
        "energy_threshold": 0.018,
    },
    "standard": {
        "name": "Lecture / Standard YouTube",
        "min_silence": 0.50,
        "keep_pause": 0.25,
        "energy_threshold": 0.018,
    },
    "aggressive": {
        "name": "Podcast / Rapid-Fire (Tight Cut)",
        "min_silence": 0.40,
        "keep_pause": 0.18,
        "energy_threshold": 0.022,
    },
}

DEFAULT_MIN_SILENCE = 0.50
DEFAULT_KEEP_PAUSE = 0.30
DEFAULT_ENERGY_THRESHOLD = 0.018


def is_colab() -> bool:
    """Returns True if running inside a Google Colab notebook environment."""
    return "google.colab" in sys.modules or os.path.exists("/content")


def cleanup_memory():
    """Aggressive garbage collection and VRAM release."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        pass
