# -*- coding: utf-8 -*-
"""
delivery.cloud_upload
---------------------
Gofile CDN upload, TmpFiles CDN direct download, Colab browser download, and LAN server dispatchers.
"""

from __future__ import annotations

import os
import sys
import json
import socket
import threading
import time
import subprocess
import urllib.parse
from pathlib import Path

from core.config import is_colab
from core.logger import fmt_duration
from core.media_tools import command_exists
from delivery.server import start_http_file_server


class TqdmFileReader:
    """Wrapper that updates tqdm progress bar during requests.post stream."""
    def __init__(self, filepath: Path, desc: str = "[UPLOAD]"):
        self.filepath = Path(filepath)
        self.total_size = self.filepath.stat().st_size
        self.file = open(self.filepath, "rb")
        try:
            from tqdm import tqdm
            self.pbar = tqdm(
                total=self.total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=desc,
                ncols=80,
                leave=True
            )
        except Exception:
            self.pbar = None

    def read(self, size: int = -1) -> bytes:
        chunk = self.file.read(size)
        if chunk and self.pbar:
            self.pbar.update(len(chunk))
        return chunk

    def __len__(self) -> int:
        return self.total_size

    def close(self):
        if self.pbar:
            self.pbar.close()
        self.file.close()


def upload_to_gofile(dst: Path) -> str | None:
    """Uploads file to high-speed Gofile CDN with live progress bar."""
    try:
        print("[UPLOAD] Uploading to high-speed Gofile CDN...")

        # Method 1: Python requests with live streaming progress
        try:
            import requests
            r = requests.get("https://api.gofile.io/servers", timeout=15)
            servers = r.json().get("data", {}).get("servers", [])
            if servers:
                server = servers[0]["name"]
                reader = TqdmFileReader(dst, desc="[UPLOAD]")
                try:
                    up = requests.post(
                        f"https://{server}.gofile.io/contents/uploadfile",
                        files={"file": (dst.name, reader)},
                        timeout=3600
                    )
                    res = up.json()
                    if res.get("status") == "ok":
                        return res["data"]["downloadPage"]
                finally:
                    reader.close()
        except KeyboardInterrupt:
            return None
        except Exception:
            pass

        # Method 2: curl streaming fallback
        try:
            import urllib.request
            req = urllib.request.Request("https://api.gofile.io/servers", headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as r:
                server = json.loads(r.read().decode())["data"]["servers"][0]["name"]

            cmd = ["curl", "-#", "-F", f"file=@{dst.resolve()}", f"https://{server}.gofile.io/contents/uploadfile"]
            out = subprocess.check_output(cmd, timeout=3600).decode("utf-8", errors="replace")
            res = json.loads(out)
            if res.get("status") == "ok":
                return res["data"]["downloadPage"]
        except KeyboardInterrupt:
            return None
        except Exception:
            pass

    except KeyboardInterrupt:
        return None
    except Exception:
        pass
    return None


def upload_to_tmpfiles(dst: Path) -> str | None:
    """Uploads file to tmpfiles.org for instant 1-click direct download with live progress."""
    try:
        print("[UPLOAD] Uploading to high-speed direct CDN (tmpfiles.org)...")
        import requests
        reader = TqdmFileReader(dst, desc="[UPLOAD]")
        try:
            r = requests.post("https://tmpfiles.org/api/v1/upload", files={"file": (dst.name, reader)}, timeout=3600)
            if r.status_code == 200:
                res = r.json()
                if res.get("status") == "success":
                    raw_url = res["data"]["url"]
                    if "tmpfiles.org/" in raw_url:
                        return raw_url.replace("tmpfiles.org/", "tmpfiles.org/dl/")
                    return raw_url
        finally:
            reader.close()
    except KeyboardInterrupt:
        return None
    except Exception:
        pass
    return None


def trigger_file_download(dst: Path):
    """Dispatches file download via GDrive, Colab browser, Gofile CDN, TmpFiles CDN, or local explorer."""
    from core.logger import log_step, current_timestamp_str, fmt_bytes
    dst_str = str(dst.resolve())
    is_gdrive = "/content/drive/" in dst_str or "MyDrive" in dst_str

    # 1. Google Drive output
    if is_gdrive:
        print("\n" + "=" * 64)
        print("SAVED TO GOOGLE DRIVE:")
        print(f"Path: {dst}")
        print("Google Drive Web: https://drive.google.com/drive/my-drive")
        print("=" * 64)
        return

    # 2. Local PC: Open folder directly if on Windows
    if not is_colab() and sys.platform == "win32":
        try:
            print("\n[LOCAL PC] Opening output folder...")
            subprocess.run(["explorer", "/select,", str(dst.resolve())], check=False)
        except Exception:
            pass

    # 3. Colab browser download
    if is_colab():
        def _trigger_colab():
            try:
                from google.colab import files as _colab_files
                print(f"\n[{current_timestamp_str()}] [DOWNLOAD] Triggering browser download for '{dst.name}'...")
                _colab_files.download(str(dst))
                print(f"[{current_timestamp_str()}] [DOWNLOAD] Browser download initiated.")
            except Exception:
                pass
        threading.Thread(target=_trigger_colab, daemon=True).start()

    # 4. High-Speed Cloud CDNs (Gofile & TmpFiles) for shareable link
    t_upload_start = time.time()
    try:
        cloud_url = upload_to_gofile(dst)
        if not cloud_url:
            cloud_url = upload_to_tmpfiles(dst)
        upload_elapsed = time.time() - t_upload_start

        if cloud_url:
            print(f"\n[{current_timestamp_str()}] " + "=" * 60)
            print(f"SHAREABLE CLOUD DOWNLOAD LINK:  [Uploaded in {fmt_duration(upload_elapsed)}]")
            print(f"\n  👉  {cloud_url}\n")
            print("• You can use this link to download the file from any other computer.")
            print("• Full-speed download with pause & resume support.")
            print("=" * 64)
        else:
            print(f"\n[{current_timestamp_str()}] [INFO] Cloud upload finished/skipped.")
            print(f"[INFO] Your processed file is safely stored at:\n  {dst}")
    except KeyboardInterrupt:
        print(f"\n[{current_timestamp_str()}] [UPLOAD] Upload skipped. Your file is safely stored locally at:")
        print(f"  {dst}")

    # 5. LAN fallback server
    port = 8888
    encoded_filename = urllib.parse.quote(dst.name)
    try:
        httpd = start_http_file_server(dst, port)
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
        except Exception:
            ip = "127.0.0.1"
        finally:
            s.close()

        dl_url = f"http://{ip}:{port}/{encoded_filename}"
        print(f"\n[{current_timestamp_str()}] " + "=" * 60)
        print("LOCAL NETWORK DOWNLOAD LINK:")
        print(f"\n  👉  {dl_url}")
        # Smart filename display: truncate middle of long names but always show extension
        fname = dst.name
        if len(fname) > 72:
            fname = fname[:60] + "..." + fname[-12:]
        print(f"  📁  File: {fname}\n")
        print("(Press ENTER or CTRL+C when finished)")
        print("=" * 64)

        try:
            if sys.stdin and hasattr(sys.stdin, "isatty") and sys.stdin.isatty():
                input("\n[Press ENTER or CTRL+C to close download server]\n")
        except (KeyboardInterrupt, EOFError):
            pass
        httpd.shutdown()
        print(f"[{current_timestamp_str()}] [SERVER] Download server stopped.")
    except Exception:
        print(f"\n[{current_timestamp_str()}] [DOWNLOAD] File saved at: {dst}")
