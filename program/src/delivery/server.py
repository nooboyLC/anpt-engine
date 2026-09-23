# -*- coding: utf-8 -*-
"""
delivery.server
---------------
Multi-threaded HTTP Range server supporting parallel segment downloads (IDM / Aria2).
"""

from __future__ import annotations

import http.server
import socketserver
import urllib.parse
import threading
import unicodedata
from pathlib import Path


def start_http_file_server(dst: Path, port: int = 8888) -> socketserver.ThreadingTCPServer:
    """Start a background HTTP server serving the output file with byte-range acceleration."""
    class _OneFileHandler(http.server.BaseHTTPRequestHandler):
        def send_response(self, code, message=None):
            if message is not None:
                try:
                    message = str(message).encode("latin-1").decode("latin-1")
                except UnicodeEncodeError:
                    message = str(message).encode("utf-8", "replace").decode("latin-1", "replace")
            try:
                super().send_response(code, message)
            except UnicodeEncodeError:
                super().send_response(code, "OK" if 200 <= code < 300 else "Error")

        def send_error(self, code, message=None, explain=None):
            try:
                super().send_error(code, message, explain)
            except (UnicodeEncodeError, UnicodeDecodeError):
                try:
                    self.send_response(code, message)
                    self.send_header("Content-Type", "text/plain; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(f"Error {code}: {message or 'Internal Error'}".encode("utf-8"))
                except Exception:
                    pass

        def do_HEAD(self):
            req_path = urllib.parse.unquote(urllib.parse.urlparse(self.path).path).lstrip("/")
            req_path_norm = unicodedata.normalize("NFC", req_path)
            dst_name_norm = unicodedata.normalize("NFC", dst.name)
            if req_path_norm in ("", dst_name_norm) or req_path_norm.lower() == dst_name_norm.lower():
                file_size = dst.stat().st_size
                self.send_response(200)
                self.send_header("Content-Type", "video/mp4" if dst.suffix.lower() == ".mp4" else "application/octet-stream")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(file_size))
                self.end_headers()
            else:
                self.send_error(404, "Not found")

        def do_GET(self):
            req_path = urllib.parse.unquote(urllib.parse.urlparse(self.path).path).lstrip("/")
            req_path_norm = unicodedata.normalize("NFC", req_path)
            dst_name_norm = unicodedata.normalize("NFC", dst.name)
            if req_path_norm in ("", dst_name_norm) or req_path_norm.lower() == dst_name_norm.lower():
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

    httpd = socketserver.ThreadingTCPServer(("", port), _OneFileHandler)
    httpd.allow_reuse_address = True
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd
