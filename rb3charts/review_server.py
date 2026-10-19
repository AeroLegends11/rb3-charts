"""Serve a chart review to devices on the same trusted local network."""
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import socket
from urllib.parse import unquote, urlsplit


_AUDIO_FILES = {
    "/audio/song.wav",
    *(f"/audio/timing-audit-{instrument}.wav" for instrument in ("guitar", "bass", "drums", "keys")),
}


def _handler_for(root):
    root = root.resolve()

    class ReviewHandler(SimpleHTTPRequestHandler):
        def _asset(self, request_path=None):
            path = unquote(urlsplit(request_path or self.path).path)
            if path != "/review.html" and path not in _AUDIO_FILES:
                return None
            asset = (root / path.lstrip("/")).resolve()
            if not asset.is_relative_to(root) or not asset.is_file():
                return None
            return asset

        def do_GET(self):
            if self._asset() is None:
                self.send_error(404)
                return
            super().do_GET()

        def do_HEAD(self):
            if self._asset() is None:
                self.send_error(404)
                return
            super().do_HEAD()

        def translate_path(self, path):
            asset = self._asset(path)
            return str(asset) if asset is not None else str(root / ".review-not-found")

        def list_directory(self, path):
            self.send_error(404)
            return None

    return ReviewHandler


def create_review_server(folder, host="0.0.0.0", port=8000):
    folder = Path(folder).resolve()
    if not (folder / "review.html").is_file() or not (folder / "audio" / "song.wav").is_file():
        raise ValueError("Folder must contain review.html and audio/song.wav")
    return ThreadingHTTPServer((host, port), _handler_for(folder))


def _lan_address():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as connection:
            connection.connect(("192.0.2.1", 80))
            return connection.getsockname()[0]
    except OSError:
        return "<computer-ip>"


def serve_review(folder, host="0.0.0.0", port=8000):
    server = create_review_server(folder, host, port)
    address, actual_port = server.server_address[:2]
    print("Review server is read-only; connect your phone to the same trusted Wi-Fi.")
    if address in ("0.0.0.0", "::"):
        address = _lan_address()
    print(f"Open http://{address}:{actual_port}/review.html on your phone. Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
