from urllib.error import HTTPError


def http_error(error: HTTPError) -> RuntimeError:
    discard_body(error)
    return RuntimeError(
        f"Model request failed with HTTP {error.code}: {error.reason}"
    )


def discard_body(error: HTTPError) -> None:
    if error.fp is None:
        return
    try:
        error.read()
    except Exception:
        pass
