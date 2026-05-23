import json
import asyncio
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import miniadk.models._http as http_module
from miniadk.models._http import JsonHttpClient


def test_json_http_client_validates_configuration():
    with pytest.raises(ValueError, match="timeout_seconds must be > 0"):
        JsonHttpClient(timeout_seconds=0)

    with pytest.raises(ValueError, match="retries must be >= 0"):
        JsonHttpClient(retries=-1)

    with pytest.raises(ValueError, match="retry_delay must be >= 0"):
        JsonHttpClient(retry_delay=-0.1)


async def test_json_http_client_posts_json_to_local_server():
    seen = {}

    def handle(handler, body):
        seen["path"] = handler.path
        seen["payload"] = json.loads(body)
        seen["content_type"] = handler.headers["content-type"]
        seen["auth"] = handler.headers["authorization"]
        _send_json(handler, {"ok": True})

    with _server(handle) as url:
        result = await JsonHttpClient(timeout_seconds=2).post_json(
            f"{url}/chat",
            {"message": "hello"},
            {"authorization": "Bearer key"},
        )

    assert result == {"ok": True}
    assert seen == {
        "path": "/chat",
        "payload": {"message": "hello"},
        "content_type": "application/json",
        "auth": "Bearer key",
    }


async def test_json_http_client_streams_sse_from_local_server():
    def handle(handler, body):
        handler.send_response(200)
        handler.send_header("content-type", "text/event-stream")
        handler.end_headers()
        handler.wfile.write(b"data: {\"delta\": \"hel\"}\n\n")
        handler.wfile.write(b"data: {\"delta\": \"lo\"}\n\n")
        handler.wfile.write(b"data: [DONE]\n\n")

    with _server(handle) as url:
        events = [
            event
            async for event in JsonHttpClient(timeout_seconds=2).post_sse(
                f"{url}/stream",
                {"stream": True},
                {},
            )
        ]

    assert events == [{"delta": "hel"}, {"delta": "lo"}]


async def test_json_http_client_reports_malformed_sse_json():
    def handle(handler, body):
        handler.send_response(200)
        handler.send_header("content-type", "text/event-stream")
        handler.end_headers()
        handler.wfile.write(b"data: {bad json}\n\n")

    with _server(handle) as url:
        with pytest.raises(RuntimeError, match="SSE event was not valid JSON"):
            [
                event
                async for event in JsonHttpClient(timeout_seconds=2).post_sse(
                    f"{url}/stream",
                    {"stream": True},
                    {},
                )
            ]


async def test_json_http_client_retries_transient_json_http_errors():
    seen = {"requests": 0}

    def handle(handler, body):
        seen["requests"] += 1
        if seen["requests"] == 1:
            handler.send_response(503)
            handler.end_headers()
            return
        _send_json(handler, {"ok": True})

    with _server(handle) as url:
        result = await JsonHttpClient(
            timeout_seconds=2,
            retries=1,
            retry_delay=0,
        ).post_json(
            f"{url}/chat",
            {"message": "hello"},
            {},
        )

    assert result == {"ok": True}
    assert seen["requests"] == 2


async def test_json_http_client_retries_transient_sse_http_errors():
    seen = {"requests": 0}

    def handle(handler, body):
        seen["requests"] += 1
        if seen["requests"] == 1:
            handler.send_response(503)
            handler.end_headers()
            return
        handler.send_response(200)
        handler.send_header("content-type", "text/event-stream")
        handler.end_headers()
        handler.wfile.write(b"data: {\"delta\": \"ok\"}\n\n")
        handler.wfile.write(b"data: [DONE]\n\n")

    with _server(handle) as url:
        events = [
            event
            async for event in JsonHttpClient(
                timeout_seconds=2,
                retries=1,
                retry_delay=0,
            ).post_sse(
                f"{url}/stream",
                {"stream": True},
                {},
            )
        ]

    assert events == [{"delta": "ok"}]
    assert seen["requests"] == 2


async def test_json_http_client_can_stop_sse_without_waiting_for_server_finish():
    def handle(handler, body):
        handler.send_response(200)
        handler.send_header("content-type", "text/event-stream")
        handler.end_headers()
        handler.wfile.write(b"data: {\"delta\": \"first\"}\n\n")
        handler.wfile.flush()
        time.sleep(2)
        handler.wfile.write(b"data: {\"delta\": \"late\"}\n\n")

    with _server(handle) as url:
        stream = JsonHttpClient(timeout_seconds=5).post_sse(
            f"{url}/stream",
            {"stream": True},
            {},
        )
        started = time.monotonic()
        first = await stream.__anext__()
        await stream.aclose()
        elapsed = time.monotonic() - started

    assert first == {"delta": "first"}
    assert elapsed < 1


async def test_json_http_client_closes_sse_response_when_stream_closes(monkeypatch):
    closed = threading.Event()

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.close()

        def __iter__(self):
            yield b"data: {\"delta\": \"first\"}\n"
            yield b"\n"
            closed.wait(timeout=2)

        def close(self):
            closed.set()

    def fake_urlopen(request, timeout):
        return FakeResponse()

    monkeypatch.setattr(http_module, "urlopen", fake_urlopen)

    stream = JsonHttpClient(timeout_seconds=5).post_sse(
        "https://provider.example.test/stream",
        {"stream": True},
        {},
    )

    first = await stream.__anext__()
    await stream.aclose()

    assert first == {"delta": "first"}
    assert closed.wait(timeout=0.2)


async def test_json_http_client_closes_json_response_when_request_is_cancelled(monkeypatch):
    opened = threading.Event()
    closed = threading.Event()

    class FakeResponse:
        def __enter__(self):
            opened.set()
            return self

        def __exit__(self, exc_type, exc, tb):
            self.close()

        def read(self):
            closed.wait(timeout=2)
            return b'{"ok": true}'

        def close(self):
            closed.set()

    def fake_urlopen(request, timeout):
        return FakeResponse()

    monkeypatch.setattr(http_module, "urlopen", fake_urlopen)

    task = asyncio.create_task(
        JsonHttpClient(timeout_seconds=5).post_json(
            "https://provider.example.test/chat",
            {"message": "hello"},
            {},
        )
    )
    await asyncio.sleep(0)
    assert await asyncio.to_thread(opened.wait, timeout=0.2)

    started = time.monotonic()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert time.monotonic() - started < 1
    assert await asyncio.to_thread(closed.wait, timeout=0.2)


class _Handler(BaseHTTPRequestHandler):
    route = None

    def do_POST(self):
        length = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(length)
        self.route(self, body)

    def log_message(self, format, *args):
        return


class _server:
    def __init__(self, route):
        self.route = route
        self.httpd = None
        self.thread = None

    def __enter__(self):
        route = self.route

        class Handler(_Handler):
            pass

        class Server(ThreadingHTTPServer):
            daemon_threads = True

        Handler.route = staticmethod(route)
        self.httpd = Server(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.httpd.server_address
        return f"http://{host}:{port}"

    def __exit__(self, exc_type, exc, tb):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=2)


def _send_json(handler, payload):
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(200)
    handler.send_header("content-type", "application/json")
    handler.send_header("content-length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)
