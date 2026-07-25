"""Run the local Phase 1A continuous-behavior audit service."""

from __future__ import annotations

import argparse
import gzip
import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from market_cycle.audit.replay import load_phase1a_continuous_replay
from market_cycle.data import DEFAULT_SNAPSHOT_ID

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_STATIC_DIR = _PROJECT_ROOT / "web" / "dist"
_REPLAY_PATH = "/api/replay.json"


def _content_type(path: Path) -> str:
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def _make_handler(*, replay_payload: bytes, static_dir: Path) -> type[BaseHTTPRequestHandler]:
    class AuditRequestHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
            path = urlparse(self.path).path
            if path == _REPLAY_PATH:
                self._send_replay()
                return
            self._send_static(path)

        def _send_replay(self) -> None:
            body = gzip.compress(replay_payload, compresslevel=6)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_static(self, request_path: str) -> None:
            relative = "index.html" if request_path in ("", "/") else request_path.lstrip("/")
            candidate = (static_dir / relative).resolve()
            if static_dir not in candidate.parents and candidate != static_dir:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            if not candidate.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            body = candidate.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", _content_type(candidate))
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            print(f"{self.address_string()} - {format % args}")

    return AuditRequestHandler


def serve(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    snapshot_id: str = DEFAULT_SNAPSHOT_ID,
    static_dir: Path = _DEFAULT_STATIC_DIR,
) -> None:
    """Compute the continuous replay bundle once, then serve the local audit page."""
    static_dir = static_dir.resolve()
    if not static_dir.is_dir():
        raise FileNotFoundError(
            f"Chart assets not found: {static_dir}. Run `npm run build` in web first."
        )

    payload = json.dumps(load_phase1a_continuous_replay(snapshot_id), separators=(",", ":")).encode("utf-8")
    handler = _make_handler(replay_payload=payload, static_dir=static_dir)
    server = ThreadingHTTPServer((host, port), handler)
    print(f"Phase 1A continuous audit: http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local Phase 1A continuous audit page.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    parser.add_argument("--snapshot-id", default=DEFAULT_SNAPSHOT_ID)
    parser.add_argument("--static-dir", default=_DEFAULT_STATIC_DIR, type=Path)
    args = parser.parse_args()
    serve(host=args.host, port=args.port, snapshot_id=args.snapshot_id, static_dir=args.static_dir)


if __name__ == "__main__":
    main()