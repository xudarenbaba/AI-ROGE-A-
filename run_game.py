"""Serve the game client as static files (development use)."""
from __future__ import annotations

import http.server
import os
import sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8082

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "game"))


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        # 开发时禁止缓存 JS/HTML，避免 generator/coop 旧文件导致「没有分房」
        path = (self.path or "").split("?", 1)[0]
        if path.endswith((".js", ".html", ".css")) or path in ("/", ""):
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
        super().end_headers()

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"[game] {self.address_string()} - {fmt % args}")


print(f"Game client : http://127.0.0.1:{PORT}")
print(f"NPC API     : http://127.0.0.1:5100  (run `python run.py` separately)")
print("Press Ctrl+C to stop.\n")

with http.server.HTTPServer(("", PORT), Handler) as httpd:
    httpd.serve_forever()
