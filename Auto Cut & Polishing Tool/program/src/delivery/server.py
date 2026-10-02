# -*- coding: utf-8 -*-
"""
delivery.server
---------------
Multi-threaded HTTP Range server supporting parallel segment downloads (IDM / Aria2)
and public reverse tunnel integration (Localtunnel) for global remote downloads.
"""

from __future__ import annotations

import http.server
import re
import shutil
import socketserver
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path


class ThreadedTCPServer(socketserver.ThreadingTCPServer):
    """Threaded TCP Server with SO_REUSEADDR enabled before bind."""
    allow_reuse_address = True


def get_public_ip(timeout: int = 5) -> str | None:
    """Fetch the host's public IP address for tunnel authentication/verification."""
    services = [
        "https://api.ipify.org",
        "https://ifconfig.me/ip",
        "https://icanhazip.com",
    ]
    for url in services:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "curl/7.68.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                ip = resp.read().decode("utf-8").strip()
                if ip and re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", ip):
                    return ip
        except Exception:
            continue
    return None


def start_public_tunnel(port: int = 8888, timeout: int = 25) -> tuple[subprocess.Popen | None, str | None]:
    """
    Launch a public reverse tunnel to expose the local HTTP server to the internet.
    Uses localtunnel via npx for zero-config global HTTPS endpoints.
    """
    npx_exe = shutil.which("npx") or "npx"
    try:
        proc = subprocess.Popen(
            [npx_exe, "-y", "localtunnel", "--port", str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        url = None
        start_t = time.time()
        while time.time() - start_t < timeout:
            line = proc.stdout.readline()
            if not line and proc.poll() is not None:
                break
            m = re.search(r"https://[a-zA-Z0-9-]+\.loca\.lt", line)
            if m:
                url = m.group(0)
                break

        if url:
            return proc, url

        # If timeout or no URL found, stop process
        stop_public_tunnel(proc)
    except Exception as exc:
        print(f"[TUNNEL] Failed to launch localtunnel: {exc}")

    return None, None


def stop_public_tunnel(proc: subprocess.Popen | None) -> None:
    """Safely terminate a public tunnel process and all its child processes."""
    if not proc:
        return
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except Exception:
                proc.kill()
    except Exception:
        pass


def start_http_file_server(dst: Path, port: int = 8888) -> socketserver.ThreadingTCPServer:
    """Start a background HTTP server serving the output file with byte-range acceleration."""
    class _OneFileHandler(http.server.BaseHTTPRequestHandler):
        def do_HEAD(self):
            req_path = urllib.parse.unquote(self.path).lstrip("/")
            if req_path in ("", dst.name):
                file_size = dst.stat().st_size
                self.send_response(200)
                self.send_header("Content-Type", "video/mp4" if dst.suffix.lower() == ".mp4" else "application/octet-stream")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(file_size))
                self.end_headers()
            else:
                self.send_error(404, "Not found")

        def do_GET(self):
            req_path = urllib.parse.unquote(self.path).lstrip("/")
            if req_path in ("", dst.name):
                try:
                    file_size = dst.stat().st_size
                    range_header = self.headers.get("Range")

                    start = 0
                    end = file_size - 1
                    status_code = 200

                    # IDM / Aria2 multi-threaded range support
                    if range_header and range_header.startswith("bytes="):
                        parts = range_header[6:].split("-")
                        try:
                            if parts[0].strip():
                                start = int(parts[0].strip())
                            if len(parts) > 1 and parts[1].strip():
                                end = int(parts[1].strip())
                            end = min(end, file_size - 1)
                            if 0 <= start <= end < file_size:
                                status_code = 206
                            else:
                                self.send_error(416, "Requested Range Not Satisfiable")
                                return
                        except ValueError:
                            pass

                    content_length = (end - start) + 1
                    self.send_response(status_code)
                    self.send_header("Content-Type", "video/mp4" if dst.suffix.lower() == ".mp4" else "application/octet-stream")
                    # RFC 5987 / 6266 standard encoding for Unicode/non-ASCII filenames
                    ascii_name = dst.name.encode("ascii", "ignore").decode("ascii").strip()
                    if not ascii_name or ascii_name == dst.suffix:
                        ascii_name = f"video{dst.suffix}"
                    quoted_name = urllib.parse.quote(dst.name, encoding="utf-8")
                    self.send_header("Content-Disposition", f'attachment; filename="{ascii_name}"; filename*=UTF-8\'\'{quoted_name}')
                    self.send_header("Accept-Ranges", "bytes")
                    if status_code == 206:
                        self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                    self.send_header("Content-Length", str(content_length))
                    self.end_headers()

                    with open(dst, "rb") as f:
                        f.seek(start)
                        remaining = content_length
                        while remaining > 0:
                            chunk_size = min(remaining, 1 << 20)  # 1 MB chunk
                            chunk = f.read(chunk_size)
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            remaining -= len(chunk)
                except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError, TimeoutError, OSError):
                    return
                except Exception:
                    try:
                        self.send_error(500, "Internal Server Error")
                    except Exception:
                        pass
            else:
                try:
                    self.send_error(404, "Not found")
                except Exception:
                    pass

        def log_message(self, fmt, *args):
            pass

    httpd = ThreadedTCPServer(("", port), _OneFileHandler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd
