"""Download the right ``miniadk-tui`` binary for this platform.

Two entry points:

* ``miniadk-tui-fetch`` (CLI) — explicit pre-fetch, useful for CI or
  air-gapped staging.
* :func:`ensure_binary` — called by ``find_tui_command`` on first
  miss so a plain ``pip install miniadk`` + ``run_cli`` sequence
  works end-to-end without a separate fetch step.

The binary is cached at ``~/.cache/miniadk/tui/<version>/<asset>``. Set
``MINIADK_TUI_BIN`` to a manual path to override; set
``MINIADK_TUI_NO_FETCH=1`` to opt out of the auto-fetch (e.g. in
sandboxed environments where the Textual fallback is preferred).

Why fetch instead of bundle:

- Each binary is ~50–90 MB (Bun runtime is embedded). Shipping 5
  per-platform wheels makes PyPI uploads noisy and Pip's resolver gets
  unhappy when users mix and match.
- Fetching on demand keeps the wheel itself tiny and lets us ship TUI
  updates independently of the Python release cycle.
"""

from __future__ import annotations

import os
import platform
import shutil
import stat
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Pinned to the matching upstream release. Bumped together with
# miniadk's PyPI version when the protocol or UI changes.
DEFAULT_RELEASE = "v0.2.0"
DEFAULT_REPO = "Saglear/miniadk"

ARCH_MAP = {
    ("Linux", "x86_64"): "miniadk-tui-linux-x64",
    ("Linux", "aarch64"): "miniadk-tui-linux-arm64",
    ("Linux", "arm64"): "miniadk-tui-linux-arm64",
    ("Darwin", "x86_64"): "miniadk-tui-darwin-x64",
    ("Darwin", "arm64"): "miniadk-tui-darwin-arm64",
    ("Windows", "AMD64"): "miniadk-tui-windows-x64.exe",
    ("Windows", "x86_64"): "miniadk-tui-windows-x64.exe",
}


def cache_dir() -> Path:
    base = os.environ.get("MINIADK_CACHE_DIR")
    if base:
        return Path(base) / "tui"
    return Path.home() / ".cache" / "miniadk" / "tui"


def detect_asset() -> str:
    key = (platform.system(), platform.machine())
    asset = ARCH_MAP.get(key)
    if asset is None:
        raise SystemExit(
            f"miniadk-tui: no prebuilt binary for {key}. "
            "Build from source: clone the repo and run "
            "`bun build --compile src/index.tsx -o dist/miniadk-tui` "
            "in tui-ts/, then point MINIADK_TUI_BIN at the result."
        )
    return asset


def cached_path(release: str, asset: str) -> Path:
    return cache_dir() / release / asset


def _emit(msg: str) -> None:
    sys.stderr.write(msg)
    sys.stderr.flush()


def download(
    release: str,
    asset: str,
    repo: str = DEFAULT_REPO,
    *,
    progress: bool = True,
) -> Path:
    """Fetch ``asset`` for ``release`` from GitHub Releases.

    A simple percentage indicator is written to stderr so first-run
    downloads don't appear to hang. Pass ``progress=False`` for a
    silent fetch (useful when capturing stderr in tests).
    """
    url = f"https://github.com/{repo}/releases/download/{release}/{asset}"
    dest = cached_path(release, asset)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    if progress:
        _emit(f"miniadk-tui: fetching {asset} ({release})\n")
    with urllib.request.urlopen(url, timeout=120) as response:
        total = int(response.headers.get("Content-Length") or 0)
        downloaded = 0
        last_pct = -1
        with open(tmp, "wb") as out:
            chunk = response.read(65_536)
            while chunk:
                out.write(chunk)
                downloaded += len(chunk)
                if progress and total > 0:
                    pct = int(downloaded * 100 / total)
                    if pct != last_pct and pct % 5 == 0:
                        _emit(f"\r  {pct:3d}%  {downloaded // 1024} KiB / {total // 1024} KiB")
                        last_pct = pct
                chunk = response.read(65_536)
    if progress:
        _emit("\n")
    tmp.replace(dest)
    if os.name != "nt":
        dest.chmod(dest.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return dest


def ensure_binary(*, quiet: bool = False) -> Path | None:
    """Return a path to the cached binary, downloading it if missing.

    Returns ``None`` when the binary can't be obtained — caller is
    expected to fall back (typically to the Textual TUI). Reasons for
    a ``None`` return:

    * ``MINIADK_TUI_NO_FETCH=1`` opt-out.
    * No prebuilt binary for the current platform.
    * Network failure (the Python parent prints a one-liner and
      degrades; we don't want a missing TUI to take down the whole
      app).
    """
    if os.environ.get("MINIADK_TUI_NO_FETCH"):
        return None
    try:
        asset = detect_asset()
    except SystemExit:
        return None
    release = os.environ.get("MINIADK_TUI_RELEASE", DEFAULT_RELEASE)
    path = cached_path(release, asset)
    if path.exists():
        return path
    if not quiet:
        _emit(
            "miniadk-tui: first run — downloading the terminal UI binary\n"
            f"  release: {release}\n"
            f"  asset:   {asset}\n"
            f"  cache:   {path}\n"
            "  (set MINIADK_TUI_NO_FETCH=1 to skip and use the Textual fallback)\n"
        )
    try:
        return download(release, asset, progress=not quiet)
    except (urllib.error.URLError, OSError) as exc:
        if not quiet:
            _emit(f"miniadk-tui: download failed ({exc}). Falling back.\n")
        return None


def main() -> None:
    release = os.environ.get("MINIADK_TUI_RELEASE", DEFAULT_RELEASE)
    asset = detect_asset()
    path = cached_path(release, asset)
    if path.exists() and "--force" not in sys.argv[1:]:
        print(f"miniadk-tui: already present at {path}", file=sys.stderr)
        return
    path = download(release, asset)
    print(str(path))


if __name__ == "__main__":
    main()
