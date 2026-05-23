from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from ..core.tools import tool


@dataclass(frozen=True, slots=True)
class FetchResult:
    url: str
    status: int
    content_type: str
    text: str
    truncated: bool = False

    def __str__(self) -> str:
        prefix = ""
        if self.status and (self.status < 200 or self.status >= 300):
            prefix = f"HTTP {self.status}\n"
        elif self.truncated:
            prefix = f"HTTP {self.status}\n"
        suffix = "\n...[truncated]" if self.truncated else ""
        return prefix + self.text + suffix


def make_fetch_url(
    *,
    timeout: float = 10,
    max_bytes: int | None = 200_000,
    allow: Callable[[str], bool | str | None] | None = None,
    user_agent: str = "MiniADK/0",
):
    def validate_fetch_url(url: str) -> bool | str:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            return "fetch_url failed: url must start with http:// or https://"
        if not parsed.netloc:
            return "fetch_url failed: url must include a host"
        if timeout <= 0:
            return "fetch_url failed: timeout must be > 0"
        if max_bytes is not None and max_bytes < 1:
            return "fetch_url failed: max_bytes must be >= 1"
        if allow is not None:
            decision = allow(url)
            if decision is False:
                return "fetch_url failed: url is not allowed"
            if isinstance(decision, str):
                return decision
        return True

    @tool(
        read_only=True,
        concurrency_safe=True,
        validate=validate_fetch_url,
        schema={
            "url": {
                "type": "string",
                "minLength": 1,
                "pattern": r"^https?://",
            }
        },
    )
    async def fetch_url(url: str) -> FetchResult:
        """Fetch a HTTP(S) URL as UTF-8-ish text."""
        try:
            return await asyncio.to_thread(
                _fetch,
                url,
                timeout=timeout,
                max_bytes=max_bytes,
                user_agent=user_agent,
            )
        except HTTPError as error:
            return FetchResult(
                url=url,
                status=error.code,
                content_type="",
                text=f"fetch_url failed: HTTP {error.code} {error.reason}",
            )
        except URLError as error:
            return FetchResult(
                url=url,
                status=0,
                content_type="",
                text=f"fetch_url failed: {error.reason}",
            )
        except Exception as error:  # noqa: BLE001 - tools report readable failures
            return FetchResult(
                url=url,
                status=0,
                content_type="",
                text=f"fetch_url failed: {error}",
            )

    return fetch_url


def _fetch(
    url: str,
    *,
    timeout: float,
    max_bytes: int | None,
    user_agent: str,
) -> FetchResult:
    request = Request(url, headers={"User-Agent": user_agent})
    with urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get("Content-Type", "")
        limit = None if max_bytes is None else max_bytes + 1
        body = response.read(limit)
        truncated = max_bytes is not None and len(body) > max_bytes
        if truncated:
            body = body[:max_bytes]
        charset = response.headers.get_content_charset() or "utf-8"
        text = body.decode(charset, errors="replace")
        return FetchResult(
            url=url,
            status=getattr(response, "status", 200),
            content_type=content_type,
            text=text,
            truncated=truncated,
        )
