# -*- coding: utf-8 -*-
"""
delivery.cloud_upload
---------------------
Gofile CDN upload, TmpFiles CDN direct download, Colab browser download,
and Local LAN + Global Public Shareable Tunnel (Localtunnel) dispatchers.
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
from delivery.server import (
    start_http_file_server,
    start_public_tunnel,
    stop_public_tunnel,
    get_public_ip,
)


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


def get_lan_ip() -> str:
    """Fetch the local LAN IPv4 address on the local Wi-Fi or Ethernet network."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


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


def _upload_to_cdns(dst: Path) -> str | None:
    """Uploads to Gofile or TmpFiles CDN and prints formatted download commands."""
    from core.logger import current_timestamp_str
    t_upload_start = time.time()
    try:
        cloud_url = upload_to_gofile(dst)
        if not cloud_url:
            cloud_url = upload_to_tmpfiles(dst)
        else:
            print("[UPLOAD] Gofile succeeded — skipping backup CDN upload.")
        upload_elapsed = time.time() - t_upload_start

        if cloud_url:
            try:
                import base64
                b64 = base64.b64encode(cloud_url.encode("utf-8")).decode("ascii")
                sys.stdout.write(f"\033]52;c;{b64}\x07")
                sys.stdout.flush()
            except Exception:
                pass

            print(f"\n[{current_timestamp_str()}] " + "=" * 64)
            print(f"SHAREABLE CLOUD DOWNLOAD LINK:  [Uploaded in {fmt_duration(upload_elapsed)}]")
            print(f"\n  👉  {cloud_url}\n")
            print("• [COPIED] Link copied to your clipboard! Press Ctrl+V in browser.")
            print("• Full-speed download with pause & resume support.")
            print("=" * 64)
            print(f"\n📥 ONE-CLICK DOWNLOAD COMMAND FOR YOUR LOCAL PC:")
            print(f"  Windows (PowerShell):")
            print(f'    curl.exe -L "{cloud_url}" -o "$HOME\\Downloads\\{dst.name}"')
            print(f"  Mac / Linux:")
            print(f'    curl -L "{cloud_url}" -o ~/Downloads/"{dst.name}"')
            print("=" * 64)
            return cloud_url
        else:
            print(f"\n[{current_timestamp_str()}] [INFO] Cloud upload finished/skipped.")
            print(f"[INFO] Your processed file is safely stored at:\n  {dst}")
    except KeyboardInterrupt:
        print(f"\n[{current_timestamp_str()}] [UPLOAD] Upload skipped. Your file is safely stored locally at:")
        print(f"  {dst}")
    return None


def trigger_file_download(dst: Path):
    """
    Dispatches file download:
    - On Google Colab / GDrive: triggers Google Drive or Cloud CDN upload.
    - On Local Environment: starts high-speed HTTP Range Server, providing:
        1. Local Network (LAN/Wi-Fi) Link for phones/PCs on the same router (zero internet data, max router speed).
        2. Global Public Internet Link (Localtunnel) for remote devices outside the local network.
    """
    from core.logger import log_step, current_timestamp_str

    if not dst.exists():
        print(f"[ERROR] Output file not found for delivery: {dst}")
        return

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
        # files.download() MUST run on main thread — IPython kernel is inaccessible from threads
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

        # In Colab: Upload to Gofile / TmpFiles as resilient cloud backup
        _upload_to_cdns(dst)
        # Continue to launch HTTP Server & Localtunnel so Colab ALSO provides live global tunnel download!

    # 4. Local Environment: High-Speed Multi-Threaded HTTP Server + Public Tunnel
    port = 8888
    httpd = None
    tunnel_proc = None

    try:
        print(f"\n[{current_timestamp_str()}] [SERVER] Starting high-speed HTTP Range download server on port {port}...")
        httpd = start_http_file_server(dst, port)

        lan_ip = get_lan_ip()
        encoded_filename = urllib.parse.quote(dst.name)
        lan_url = f"http://{lan_ip}:{port}/{encoded_filename}" if lan_ip != "127.0.0.1" else f"http://localhost:{port}/{encoded_filename}"

        print(f"[{current_timestamp_str()}] [TUNNEL] Establishing global public tunnel...")
        tunnel_proc, tunnel_url = start_public_tunnel(port=port, timeout=20)
        public_ip = get_public_ip(timeout=4)

        fname = dst.name
        if len(fname) > 72:
            fname = fname[:60] + "..." + fname[-12:]

        # Copy preferred link to clipboard via OSC 52
        copy_url = tunnel_url or lan_url
        if copy_url:
            try:
                import base64
                b64 = base64.b64encode(copy_url.encode("utf-8")).decode("ascii")
                sys.stdout.write(f"\033]52;c;{b64}\x07")
                sys.stdout.flush()
            except Exception:
                pass

        print(f"\n[{current_timestamp_str()}] " + "=" * 68)
        print(f"📥 DOWNLOAD READY: {fname}")
        print("=" * 68)

        # Section 1: Local Network (LAN / Wi-Fi) Link (Only for Local PC on same Wi-Fi router)
        if not is_colab():
            print("\n1️⃣  📶 LOCAL (WI-FI / LAN):")
            print(f"    👉  {lan_url}")

        # Section 2: Global Public Internet Link (Localtunnel)
        sec_num = "1️⃣" if is_colab() else "2️⃣"
        if tunnel_url:
            print(f"\n{sec_num}  🌐 GLOBAL (INTERNET):")
            print(f"    👉  {tunnel_url}")
            if public_ip:
                print(f"    🔑  Tunnel Password (IP): {public_ip}")
            print(f"    💻  One-Click Command (for remote client PC):")
            print(f'        Windows (PowerShell):')
            print(f'          curl.exe -L -H "Bypass-Tunnel-Reminder: true" "{tunnel_url}" -o "$HOME\\Downloads\\{dst.name}"')
            print(f'        Mac / Linux:')
            print(f'          curl -L -H "Bypass-Tunnel-Reminder: true" "{tunnel_url}" -o ~/Downloads/"{dst.name}"')
        else:
            print(f"\n{sec_num}  🌐 GLOBAL (INTERNET):")
            print("    [Notice: Public tunnel unavailable; use the Cloud CDN link above]")

        print("\n" + "=" * 68)
        print("• [COPIED] Link copied to clipboard! Press Ctrl+V in browser.")
        print("• Supports pause & resume and multi-threaded parallel downloads (IDM / Aria2).")
        print("=" * 68)

        try:
            if sys.stdin and hasattr(sys.stdin, "isatty") and sys.stdin.isatty():
                input("\n[Press ENTER or CTRL+C when you have finished downloading to close server]\n")
            else:
                # Non-interactive grace period
                time.sleep(30)
        except (KeyboardInterrupt, EOFError):
            pass
    except (KeyboardInterrupt, EOFError):
        pass
    except Exception as exc:
        print(f"\n[{current_timestamp_str()}] [SERVER] Notice: {exc}")
    finally:
        if tunnel_proc:
            stop_public_tunnel(tunnel_proc)
        if httpd:
            try:
                httpd.shutdown()
                httpd.server_close()
            except Exception:
                pass
            print(f"[{current_timestamp_str()}] [SERVER] Download server and tunnel cleanly stopped.")
