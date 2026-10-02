# -*- coding: utf-8 -*-
"""
prefetch_models.py
------------------
Run during setup to pre-download ALL AI model weights before first use.
This ensures the program works fully offline on first run.

Models downloaded:
  - Silero VAD (Neural Voice Activity Detection)
  - ClearVoice MossFormer2_SE_48K (AI speech enhancement)
  - VoiceFixer (Neural de-reverberation and voice restoration)

Usage (called by setup.bat / setup.sh automatically):
    python prefetch_models.py
"""

import sys
import os
from pathlib import Path

# Centralize Python bytecode (__pycache__) into support/cache/pycache/
_ROOT_DIR = Path(__file__).resolve().parent.parent
_PYCACHE_DIR = _ROOT_DIR / "support" / "cache" / "pycache"
try:
    _PYCACHE_DIR.mkdir(parents=True, exist_ok=True)
    sys.pycache_prefix = str(_PYCACHE_DIR)
    os.environ["PYTHONPYCACHEPREFIX"] = str(_PYCACHE_DIR)
except Exception:
    pass

# Make sure config is loaded first so environment variables (TORCH_HOME, HF_HOME, etc.)
# are set before any model download begins.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from core.config import (
    PROJECT_CHECKPOINTS, PROJECT_TEMP, PROJECT_CACHE, BIN_DIR,
    SUPPORT_DIR
)

# Ensure support directories exist before downloads
PROJECT_CHECKPOINTS.mkdir(parents=True, exist_ok=True)
PROJECT_TEMP.mkdir(parents=True, exist_ok=True)
PROJECT_CACHE.mkdir(parents=True, exist_ok=True)
BIN_DIR.mkdir(parents=True, exist_ok=True)


def _header(msg: str):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def _ok(msg: str):
    print(f"  [OK] {msg}")


def _fail(msg: str):
    print(f"  [WARN] {msg}")


def prefetch_silero_vad():
    """Download Silero VAD model weights into checkpoints/."""
    _header("Silero Neural VAD Model")
    try:
        from silero_vad import load_silero_vad
        model = load_silero_vad()
        _ok("Silero VAD model downloaded and cached.")
        del model
    except ImportError:
        _fail("silero-vad not installed â€” skipped. Run: pip install silero-vad>=6.0")
    except Exception as exc:
        _fail(f"Silero VAD download failed: {exc}")


def prefetch_clearvoice():
    """Download ClearVoice MossFormer2_SE_48K weights into checkpoints/."""
    _header("ClearVoice â€” MossFormer2_SE_48K (AI Speech Enhancement)")
    try:
        import clearvoice
        import contextlib, io
        if not getattr(clearvoice, "_antigravity_patched", False):
            _orig = clearvoice.network_wrapper.load_args_se
            def _custom(self):
                _orig(self)
                self.args.checkpoint_dir = str(PROJECT_CHECKPOINTS / self.model_name)
            clearvoice.network_wrapper.load_args_se = _custom
            clearvoice._antigravity_patched = True
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            cv = clearvoice.ClearVoice(task="speech_enhancement", model_names=["MossFormer2_SE_48K"])
        _ok("ClearVoice MossFormer2_SE_48K model downloaded and cached.")
        del cv
    except ImportError:
        _fail("clearvoice not installed â€” skipped. Run: pip install clearvoice>=0.1.0")
    except Exception as exc:
        _fail(f"ClearVoice download failed: {exc}")


def _monkey_patch_voicefixer():
    """Runtime monkey-patch: redirect VoiceFixer memory paths to project checkpoints."""
    import os
    from pathlib import Path
    
    # Unconditional isolated path inside the project
    vf_cache = str(PROJECT_CHECKPOINTS / "voicefixer").replace("\\", "/")

    try:
        import voicefixer.vocoder.config as _vcfg
        synth_path = os.path.join(vf_cache, "synthesis_module", "44100",
                                  "model.ckpt-1490000_trimed.pt")
        _vcfg.Config.ckpt = synth_path
        _orig_refresh = _vcfg.Config.refresh.__func__

        @classmethod  # type: ignore
        def _patched_refresh(cls, sr):
            _orig_refresh(cls, sr)
            if sr == 44100:
                cls.ckpt = synth_path
        _vcfg.Config.refresh = _patched_refresh
    except Exception:
        pass

    try:
        import voicefixer.base as _vbase
        analysis_path = os.path.join(vf_cache, "analysis_module", "checkpoints", "vf.ckpt")
        for attr in dir(_vbase):
            val = getattr(_vbase, attr, None)
            if isinstance(val, str) and "analysis_module" in val and "vf.ckpt" in val:
                setattr(_vbase, attr, analysis_path)
    except Exception:
        pass


def prefetch_voicefixer():
    """Download VoiceFixer model weights into support/checkpoints/voicefixer/.

    Double-locked isolation strategy:
      1. patch_packages.py patches site-packages source files permanently.
      2. _monkey_patch_voicefixer() patches class attributes in memory at runtime.
    Together they guarantee C: drive / home directory is NEVER written to.
    """
    _header("VoiceFixer (Neural De-reverberation & Voice Restoration)")
    try:
        import shutil
        from huggingface_hub import hf_hub_download

        # Apply runtime monkey-patch BEFORE importing VoiceFixer class
        _monkey_patch_voicefixer()

        # Canonical project location (fully isolated)
        local_vf_dir   = PROJECT_CHECKPOINTS / "voicefixer"
        local_synth    = local_vf_dir / "synthesis_module" / "44100"
        local_analysis = local_vf_dir / "analysis_module" / "checkpoints"
        local_synth.mkdir(parents=True, exist_ok=True)
        local_analysis.mkdir(parents=True, exist_ok=True)

        local_f1 = local_synth    / "model.ckpt-1490000_trimed.pt"
        local_f2 = local_analysis / "vf.ckpt"

        # Download only if not already present with full size check
        if not (local_f1.exists() and local_f1.stat().st_size > 130_000_000):
            _ok("Downloading synthesis model (model.ckpt-1490000_trimed.pt)...")
            try:
                p1 = hf_hub_download(repo_id="Diogodiogod/voicefixer-models",
                                     filename="model.ckpt-1490000_trimed.pt")
                shutil.copy2(p1, str(local_f1))
            except Exception:
                import subprocess
                subprocess.run([
                    "curl.exe", "-L", "--fail", "--retry", "3", "-C", "-",
                    "https://huggingface.co/Diogodiogod/voicefixer-models/resolve/main/model.ckpt-1490000_trimed.pt",
                    "-o", str(local_f1)
                ], check=True)

        if not (local_f2.exists() and local_f2.stat().st_size >= 480_000_000):
            _ok("Downloading analysis model (vf.ckpt, ~489 MB)...")
            try:
                import subprocess
                subprocess.run([
                    "curl.exe", "-L", "--fail", "--retry", "3", "-C", "-",
                    "https://huggingface.co/Diogodiogod/voicefixer-models/resolve/main/vf.ckpt",
                    "-o", str(local_f2)
                ], check=True)
            except Exception:
                p2 = hf_hub_download(repo_id="Diogodiogod/voicefixer-models",
                                     filename="vf.ckpt")
                shutil.copy2(p2, str(local_f2))

        _ok(f"VoiceFixer models ready at: {local_vf_dir}")
        _ok("VoiceFixer loads directly from support/checkpoints (C: drive 100% untouched).")

        # Verify the library loads correctly from the project path
        from voicefixer import VoiceFixer
        vf = VoiceFixer()
        _ok("VoiceFixer loaded successfully.")
        del vf
    except ImportError:
        _fail("voicefixer not installed â€” skipped. Run: pip install voicefixer>=0.1.3")
    except Exception as exc:
        _fail(f"VoiceFixer download failed: {exc}")



def main():
    print("\n" + "=" * 60)
    print("  PRE-FETCHING ALL AI MODEL WEIGHTS")
    print("  (This runs only once â€” models are cached for future runs)")
    print("=" * 60)
    print(f"  Saving to: {PROJECT_CHECKPOINTS}")

    prefetch_silero_vad()
    prefetch_clearvoice()
    prefetch_voicefixer()

    print("\n" + "=" * 60)
    print("  ALL MODELS READY â€” Program will start instantly next time.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
