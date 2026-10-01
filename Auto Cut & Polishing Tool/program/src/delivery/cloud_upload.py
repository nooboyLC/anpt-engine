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
    _last_error = ""
    try:
        print("[UPLOAD] Uploading to high-speed Gofile CDN...")

        # Method 1: Python requests with live streaming progress
        try:
            import requests
            r = requests.get("https://api.gofile.io/servers", timeout=15)
            data = r.json().get("data", {})
            # Gofile API v2: returns "server" as a string (not a list)
            server = data.get("server") or (data.get("servers", [{}])[0].get("name") if data.get("servers") else None)
            if server:
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
                    _last_error = f"Gofile API error: {res.get('status')} – {res}"
                finally:
                    reader.close()
            else:
                _last_error = f"Gofile: no server returned – response: {r.text[:200]}"
        except KeyboardInterrupt:
            return None
        except Exception as exc:
            _last_error = f"Gofile requests method: {exc}"

        # Method 2: curl streaming fallback
        try:
            import urllib.request
            req = urllib.request.Request("https://api.gofile.io/servers", headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as r:
                api_data = json.loads(r.read().decode()).get("data", {})
                server = api_data.get("server") or api_data.get("servers", [{}])[0].get("name")

            cmd = ["curl", "-#", "-F", f"file=@{dst.resolve()}", f"https://{server}.gofile.io/contents/uploadfile"]
            out = subprocess.check_output(cmd, timeout=3600).decode()
            res = json.loads(out)
            if res.get("status") == "ok":
                return res["data"]["downloadPage"]
            _last_error = f"Gofile curl fallback API error: {res}"
        except KeyboardInterrupt:
            return None
        except Exception as exc:
            _last_error = f"Gofile curl method: {exc}"

    except KeyboardInterrupt:
        return None
    except Exception as exc:
        _last_error = str(exc)

    if _last_error:
        print(f"[UPLOAD] Gofile failed: {_last_error}")
    return None


def upload_to_tmpfiles(dst: Path) -> str | None:
    """Uploads file to tmpfiles.org for instant 1-click direct download with live progress.
    Skipped for files > 3 GB (tmpfiles.org has unreliable large-file handling).
    Hard timeout of 120s prevents hanging when server is slow or unresponsive.
    """
    # Safety caps: skip large files and use a strict timeout
    _SIZE_LIMIT_GB = 3
    _UPLOAD_TIMEOUT_S = 120
    try:
        file_gb = dst.stat().st_size / (1024 ** 3)
        if file_gb > _SIZE_LIMIT_GB:
            print(f"[UPLOAD] tmpfiles.org skipped — file size {file_gb:.1f} GB exceeds {_SIZE_LIMIT_GB} GB cap.")
            return None
    except Exception:
        pass

    _last_error = ""
    try:
        print("[UPLOAD] Uploading to high-speed direct CDN (tmpfiles.org)...")
        import requests
        reader = TqdmFileReader(dst, desc="[UPLOAD]")
        try:
            r = requests.post("https://tmpfiles.org/api/v1/upload", files={"file": (dst.name, reader)}, timeout=_UPLOAD_TIMEOUT_S)
            if r.status_code == 200:
                res = r.json()
                if res.get("status") == "success":
                    raw_url = res["data"]["url"]
                    if "tmpfiles.org/" in raw_url:
                        return raw_url.replace("tmpfiles.org/", "tmpfiles.org/dl/")
                    return raw_url
                _last_error = f"tmpfiles API returned: {res}"
            else:
                _last_error = f"tmpfiles HTTP {r.status_code}: {r.text[:200]}"
        finally:
            reader.close()
    except KeyboardInterrupt:
        return None
    except Exception as exc:
        _last_error = str(exc)

    if _last_error:
        print(f"[UPLOAD] tmpfiles failed: {_last_error}")
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
            except ImportError:
                print(f"\n[{current_timestamp_str()}] [DOWNLOAD] Notice: Direct browser popup download requires execution inside a Google Colab notebook cell.")
                print(f"[{current_timestamp_str()}] [DOWNLOAD] When running in Colab terminal, please use the Cloud CDN link (Gofile) or your mounted Google Drive destination below.")
            except Exception as ex:
                print(f"\n[{current_timestamp_str()}] [DOWNLOAD] Colab browser download notice: {ex}")
                print(f"[{current_timestamp_str()}] [DOWNLOAD] Please use the high-speed Cloud CDN link below or your synced Google Drive destination.")
        t_colab = threading.Thread(target=_trigger_colab, daemon=True)
        t_colab.start()
        t_colab.join(timeout=2.0)

    # 4. High-Speed Cloud CDNs (Gofile & TmpFiles) for shareable link
    # Smart CDN: Only try tmpfiles.org if Gofile failed — never upload the same big file twice.
    t_upload_start = time.time()
    try:
        cloud_url = upload_to_gofile(dst)
        if not cloud_url:
            # Gofile failed — try tmpfiles.org as fallback (with size cap + timeout)
            cloud_url = upload_to_tmpfiles(dst)
        else:
            print("[UPLOAD] Gofile succeeded — skipping backup CDN upload.")
        upload_elapsed = time.time() - t_upload_start

        if cloud_url:
            # Copy download link directly to user's local PC clipboard via OSC 52
            try:
                import base64
                b64 = base64.b64encode(cloud_url.encode("utf-8")).decode("ascii")
                sys.stdout.write(f"\033]52;c;{b64}\x07")
                sys.stdout.flush()
            except Exception:
                pass

            print(f"\n[{current_timestamp_str()}] " + "=" * 60)
            print(f"SHAREABLE CLOUD DOWNLOAD LINK:  [Uploaded in {fmt_duration(upload_elapsed)}]")
            print(f"\n  👉  {cloud_url}\n")
            print("• [COPIED] Link copied to your local PC clipboard! Press Ctrl+V in browser.")
            print("• Full-speed download with pause & resume support.")
            print("=" * 64)
            print(f"\n📥 ONE-CLICK DOWNLOAD COMMAND FOR YOUR LOCAL PC:")
            print(f"  Windows (PowerShell):")
            print(f"    curl.exe -L \"{cloud_url}\" -o \"$HOME\\Downloads\\{dst.name}\"")
            print(f"  Mac / Linux:")
            print(f"    curl -L \"{cloud_url}\" -o ~/Downloads/\"{dst.name}\"")
            print("=" * 64)
        else:
            print(f"\n[{current_timestamp_str()}] [INFO] Cloud upload finished/skipped.")
            print(f"[INFO] Your processed file is safely stored at:\n  {dst}")
    except KeyboardInterrupt:
        print(f"\n[{current_timestamp_str()}] [UPLOAD] Upload skipped. Your file is safely stored locally at:")
        print(f"  {dst}")

    # 5. Local LAN HTTP Server (For downloading across home/office network from another PC)
    # Fully active on all local PCs (Windows / Linux / Mac). Only skipped on Google Colab cloud datacenter
    # where 172.30.x.x is an internal Google VM IP unreachable from home routers.
    if not is_colab():
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
            print("🌐 LOCAL NETWORK (LAN) DOWNLOAD LINK:")
            print(f"\n  👉  {dl_url}")
            fname = dst.name
            if len(fname) > 72:
                fname = fname[:60] + "..." + fname[-12:]
            print(f"  📁  File: {fname}")
            print("  💡  Use this link from any other computer or phone on your home/office Wi-Fi.")
            print("=" * 64)

            try:
                if sys.stdin and hasattr(sys.stdin, "isatty") and sys.stdin.isatty():
                    input("\n[Press ENTER or CTRL+C when you have finished downloading to close server]\n")
            except (KeyboardInterrupt, EOFError):
                pass
            httpd.shutdown()
            print(f"[{current_timestamp_str()}] [SERVER] Local LAN server stopped.")
        except Exception:
            print(f"\n[{current_timestamp_str()}] [DOWNLOAD] File saved at: {dst}")
    else:
        print(f"\n[{current_timestamp_str()}] [CLOUD NOTE] Google Colab cloud container detected.")
        print(f"  Internal container IP (172.30.x.x) is inside Google's datacenter and unreachable over home Wi-Fi.")
        print(f"  Please use the Shareable Cloud Link (Gofile) or Google Drive destination to download to your local PC.")
