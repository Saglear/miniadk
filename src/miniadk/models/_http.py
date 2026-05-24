import asyncio
import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(slots=True)
class JsonHttpClient:
    # Default 10 minutes. Reasoning models (o-series, Claude with
    # extended thinking, etc.) routinely think for several minutes;
    # large requests with long context add another minute or two.
    # 60s — the old default — was a Web-API-from-2015 number, not an
    # agent number. Per-call override is still available, and the
    # ``MINIADK_HTTP_TIMEOUT`` env var trumps both.
    timeout_seconds: float = 600
    retries: int = 0
    retry_delay: float = 0.25

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        if self.retries < 0:
            raise ValueError("retries must be >= 0")
        if self.retry_delay < 0:
            raise ValueError("retry_delay must be >= 0")

    async def post_json(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        stopped = threading.Event()
        response_handle = _ResponseHandle()

        def finish(item) -> None:
            if future.done():
                return
            if isinstance(item, Exception):
                future.set_exception(item)
            else:
                future.set_result(item)

        def complete(item) -> None:
            if stopped.is_set() or loop.is_closed():
                return
            loop.call_soon_threadsafe(finish, item)

        def worker() -> None:
            try:
                complete(
                    self._post_json_sync(
                        url,
                        payload,
                        headers,
                        stopped=stopped,
                        set_response=response_handle.set,
                    )
                )
            except Exception as error:  # noqa: BLE001 - forwarded to async caller
                complete(error)

        threading.Thread(target=worker, daemon=True).start()
        try:
            return await future
        except BaseException:
            stopped.set()
            response_handle.close_later()
            raise

    def _post_json_sync(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        *,
        stopped: threading.Event | None = None,
        set_response: Callable[[Any], None] | None = None,
    ) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            url=url,
            data=body,
            headers={
                "content-type": "application/json",
                **headers,
            },
            method="POST",
        )

        response_body = self._request_with_retries(
            request,
            stopped=stopped,
            set_response=set_response,
        ).decode("utf-8")

        try:
            return json.loads(response_body)
        except json.JSONDecodeError as error:
            raise RuntimeError("Model response was not valid JSON") from error

    async def post_sse(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
    ):
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        done = object()
        stopped = threading.Event()
        response_handle = _ResponseHandle()

        def put(item) -> None:
            if stopped.is_set() or loop.is_closed():
                return
            loop.call_soon_threadsafe(queue.put_nowait, item)

        def worker() -> None:
            try:
                for item in self._post_sse_sync(
                    url,
                    payload,
                    headers,
                    stopped=stopped,
                    set_response=response_handle.set,
                ):
                    if stopped.is_set():
                        return
                    put(item)
            except Exception as error:  # noqa: BLE001 - forwarded to async caller
                put(error)
            finally:
                put(done)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        try:
            while True:
                item = await queue.get()
                if item is done:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            stopped.set()
            response_handle.close_later()

    def _post_sse_sync(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        *,
        stopped: threading.Event | None = None,
        set_response: Callable[[Any], None] | None = None,
    ):
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            url=url,
            data=body,
            headers={
                "content-type": "application/json",
                "accept": "text/event-stream",
                **headers,
            },
            method="POST",
        )

        for attempt in range(self.retries + 1):
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    if set_response is not None:
                        set_response(response)
                    try:
                        data_lines: list[str] = []
                        for raw_line in response:
                            if stopped is not None and stopped.is_set():
                                return
                            line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                            if line == "":
                                if data_lines:
                                    text = "\n".join(data_lines)
                                    data_lines = []
                                    if text == "[DONE]":
                                        break
                                    yield _loads_sse_json(text)
                                continue
                            if line.startswith("data:"):
                                data_lines.append(line[5:].lstrip())
                        if data_lines:
                            text = "\n".join(data_lines)
                            if text != "[DONE]":
                                yield _loads_sse_json(text)
                    finally:
                        if set_response is not None:
                            set_response(None)
                return
            except HTTPError as error:
                if not self._should_retry_http(error, attempt):
                    raise
                self._sleep_before_retry(attempt)
            except URLError as error:
                if not self._should_retry(attempt):
                    raise RuntimeError(f"Model request failed: {error.reason}") from error
                self._sleep_before_retry(attempt)

    def _request_with_retries(
        self,
        request: Request,
        *,
        stopped: threading.Event | None = None,
        set_response: Callable[[Any], None] | None = None,
    ) -> bytes:
        for attempt in range(self.retries + 1):
            try:
                if stopped is not None and stopped.is_set():
                    raise RuntimeError("Model request was cancelled")
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    if set_response is not None:
                        set_response(response)
                    try:
                        return response.read()
                    finally:
                        if set_response is not None:
                            set_response(None)
            except HTTPError as error:
                if not self._should_retry_http(error, attempt):
                    raise
                self._sleep_before_retry(attempt)
            except URLError as error:
                if not self._should_retry(attempt):
                    raise RuntimeError(f"Model request failed: {error.reason}") from error
                self._sleep_before_retry(attempt)
        raise RuntimeError("Model request failed after retries")

    def _should_retry_http(self, error: HTTPError, attempt: int) -> bool:
        return error.code in {408, 429, 500, 502, 503, 504} and self._should_retry(attempt)

    def _should_retry(self, attempt: int) -> bool:
        return attempt < self.retries

    def _sleep_before_retry(self, attempt: int) -> None:
        delay = self.retry_delay * (2 ** attempt)
        if delay > 0:
            time.sleep(delay)


class _ResponseHandle:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._response = None

    def set(self, response) -> None:
        with self._lock:
            self._response = response

    def close_later(self) -> None:
        threading.Thread(target=self.close, daemon=True).start()

    def close(self) -> None:
        with self._lock:
            response = self._response
        if response is None:
            return
        try:
            response.close()
        except Exception:
            pass


def _loads_sse_json(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise RuntimeError("SSE event was not valid JSON") from error
