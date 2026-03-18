from __future__ import annotations

import argparse
import base64
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class AuthHandler(SimpleHTTPRequestHandler):
    username = ""
    password = ""
    directory = ""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=self.directory, **kwargs)

    def _expected_header(self) -> str:
        token = base64.b64encode(f"{self.username}:{self.password}".encode("utf-8")).decode("ascii")
        return f"Basic {token}"

    def _unauthorized(self) -> None:
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="gupiao-preview"')
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write("Authentication required.".encode("utf-8"))

    def do_AUTHHEAD(self) -> None:
        self._unauthorized()

    def _authorized(self) -> bool:
        return self.headers.get("Authorization", "") == self._expected_header()

    def do_GET(self) -> None:
        if not self._authorized():
            self._unauthorized()
            return
        super().do_GET()

    def do_HEAD(self) -> None:
        if not self._authorized():
            self._unauthorized()
            return
        super().do_HEAD()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)
    args = parser.parse_args()

    directory = Path(args.directory).resolve()
    AuthHandler.directory = str(directory)
    AuthHandler.username = str(args.username)
    AuthHandler.password = str(args.password)
    server = ThreadingHTTPServer((str(args.host), int(args.port)), AuthHandler)
    print(f"Serving {directory} on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
