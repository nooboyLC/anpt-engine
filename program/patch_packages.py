# -*- coding: utf-8 -*-
"""
patch_packages.py
-----------------
Called automatically by setup.bat / setup.sh after pip install.

Patches the installed voicefixer library in site-packages so that
every hardcoded  os.path.expanduser("~/.cache/voicefixer/...")
is replaced with the project-local path inside program/support/checkpoints/voicefixer.

This makes the library 100% isolated — no files ever land on C: (Windows)
or ~/  (Linux/macOS).

Safe to re-run multiple times; idempotent.
"""

import sys
import os
import re
from pathlib import Path


def find_site_packages(venv_dir: Path):
    """Return the site-packages directory inside the venv."""
    for candidate in venv_dir.glob("Lib/site-packages"):    # Windows
        if candidate.is_dir():
            return candidate
    for candidate in venv_dir.glob("lib/python*/site-packages"):  # Unix
        if candidate.is_dir():
            return candidate
    return None


def patch_file(path: Path, replacements: list) -> bool:
    """Apply regex replacements to a file. Returns True if file was changed."""
    original = path.read_text(encoding="utf-8")
    patched  = original
    for pattern, repl in replacements:
        patched = re.sub(pattern, repl, patched)
    if patched == original:
        return False   # already patched or pattern not found
    path.write_text(patched, encoding="utf-8")
    return True


def _replacements_for(filename: str, cache_path: str) -> list:
    """Return (pattern, replacement) pairs tailored to each voicefixer file."""

    if filename == "config.py":
        p1 = (
            r'os\.path\.join\(\s*os\.path\.expanduser\(["\']~["\']\),\s*'
            r'["\']\.cache/voicefixer/synthesis_module/44100/model\.ckpt-1490000_trimed\.pt["\']\s*,?\s*\)',
            f'os.path.join(r"{cache_path}", "synthesis_module", "44100", "model.ckpt-1490000_trimed.pt")'
        )
        return [p1]

    if filename == "base.py":
        p1 = (
            r'os\.path\.join\(\s*os\.path\.expanduser\(["\']~["\']\),\s*'
            r'["\']\.cache/voicefixer/analysis_module/checkpoints/vf\.ckpt["\']\s*,?\s*\)',
            f'os.path.join(r"{cache_path}", "analysis_module", "checkpoints", "vf.ckpt")'
        )
        return [p1]

    if filename == "__init__.py":
        p1 = (
            r'os\.path\.join\(\s*os\.path\.expanduser\(["\']~["\']\),\s*'
            r'["\']\.cache/voicefixer/analysis_module/checkpoints/vf\.ckpt["\']\s*,?\s*\)',
            f'os.path.join(r"{cache_path}", "analysis_module", "checkpoints", "vf.ckpt")'
        )
        return [p1]

    return []


TARGETS = [
    ("voicefixer/vocoder/config.py",   "config.py"),
    ("voicefixer/base.py",              "base.py"),
    ("voicefixer/restorer/__init__.py", "__init__.py"),
]


def main():
    venv_dir = Path(sys.executable).parent.parent  # .../venv/Scripts/python.exe -> .../venv
    program_dir = Path(__file__).resolve().parent
    cache_path = (program_dir / "support" / "checkpoints" / "voicefixer").as_posix()
    
    sp = find_site_packages(venv_dir)
    if sp is None:
        print("[PATCH] Could not locate site-packages — skipping.")
        return

    print(f"[PATCH] site-packages -> {sp}")
    any_changed = False
    for rel_path, fname in TARGETS:
        target = sp / rel_path
        if not target.exists():
            print(f"[PATCH] Not found (skip): {rel_path}")
            continue
        repls = _replacements_for(fname, cache_path)
        if not repls:
            continue
        changed = patch_file(target, repls)
        status  = "patched OK" if changed else "already patched (skip)"
        print(f"[PATCH] {rel_path}: {status}")
        any_changed = any_changed or changed

    if any_changed:
        print("[PATCH] VoiceFixer library patched — C: drive will NOT be used.")
    else:
        print("[PATCH] VoiceFixer already fully patched.")


if __name__ == "__main__":
    main()
