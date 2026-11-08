#!/usr/bin/env python3
"""Download a reproducible 500-file Openclipart SVG subset.

The source is the public GitHub mirror:
    https://github.com/cyanidecupcake/openclipart-svg

This script downloads SVG source files only. It does not rasterize them and
does not create JPEG degraded/clean pairs yet. The downloaded SVG files are
the lossless source material for the next pipeline stage.

Usage:
    python download_openclipart_500.py

Optional arguments:
    python download_openclipart_500.py --count 500 --seed 20261006

An optional GitHub token can be supplied through GITHUB_TOKEN to avoid the
unauthenticated API rate limit:
    set GITHUB_TOKEN=...       # Windows PowerShell/cmd
    export GITHUB_TOKEN=...    # Linux/macOS
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


REPO_OWNER = "cyanidecupcake"
REPO_NAME = "openclipart-svg"
REPO_REF = "master"
REPO_URL = f"https://github.com/{REPO_OWNER}/{REPO_NAME}"
API_TREE_URL = (
    f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}"
    f"/git/trees/{REPO_REF}?recursive=1"
)
RAW_BASE_URL = (
    f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}"
    f"/{REPO_REF}/"
)


def request_bytes(url: str, *, headers: Dict[str, str], retries: int = 4) -> bytes:
    """Download one URL with exponential backoff for transient failures."""
    last_error: Optional[Exception] = None
    for attempt in range(retries):
        try:
            request = Request(url, headers=headers)
            with urlopen(request, timeout=60) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc
            if isinstance(exc, HTTPError) and exc.code in {401, 403, 404}:
                raise
            if attempt + 1 < retries:
                time.sleep(2 ** attempt)
    assert last_error is not None
    raise last_error


def safe_filename(path: str) -> str:
    """Turn a repository path into a stable local filename."""
    name = Path(path).name
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    if not name.lower().endswith(".svg"):
        name += ".svg"
    return name


def unique_local_name(path: str, used: set[str]) -> str:
    """Avoid collisions when different source folders share a filename."""
    base = safe_filename(path)
    if base not in used:
        used.add(base)
        return base

    stem = Path(base).stem
    suffix = Path(base).suffix
    digest = hashlib.sha1(path.encode("utf-8")).hexdigest()[:10]
    candidate = f"{stem}__{digest}{suffix}"
    used.add(candidate)
    return candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "openclipart_500",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="redownload files that already exist in the output directory",
    )
    return parser.parse_args()


def get_svg_paths(headers: Dict[str, str]) -> List[str]:
    raw = request_bytes(API_TREE_URL, headers=headers)
    payload = json.loads(raw.decode("utf-8"))
    if payload.get("truncated"):
        raise RuntimeError(
            "GitHub returned a truncated repository tree; cannot select a "
            "reproducible subset safely."
        )

    paths = [
        entry["path"]
        for entry in payload.get("tree", [])
        if entry.get("type") == "blob"
        and entry.get("path", "").lower().endswith(".svg")
    ]
    if not paths:
        raise RuntimeError("No SVG files were found in the repository tree.")
    return sorted(paths)


def write_text(path: Path, text: str) -> None:
    # Path.write_text has no newline= argument on supported Python versions.
    # The generated text already uses explicit \n line endings.
    path.write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.count <= 0:
        raise SystemExit("--count must be positive")

    args.output.mkdir(parents=True, exist_ok=True)
    svg_dir = args.output / "svg_raw"
    svg_dir.mkdir(parents=True, exist_ok=True)

    token = __import__("os").environ.get("GITHUB_TOKEN")
    headers = {
        "User-Agent": "openclipart-500-downloader/1.0",
        "Accept": "application/vnd.github+json",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    print(f"Reading SVG file list from {REPO_URL} ...")
    paths = get_svg_paths(headers)
    print(f"Found {len(paths):,} SVG files in the mirror.")

    if args.count > len(paths):
        raise SystemExit(
            f"Requested {args.count} files, but only {len(paths)} SVG files exist."
        )

    rng = random.Random(args.seed)
    selected = sorted(rng.sample(paths, args.count))

    manifest: List[Dict[str, object]] = []
    used_names: set[str] = set()
    failures: List[Dict[str, str]] = []

    for index, source_path in enumerate(selected, start=1):
        local_name = unique_local_name(source_path, used_names)
        destination = svg_dir / local_name
        source_url = RAW_BASE_URL + source_path

        if destination.exists() and not args.overwrite:
            data = destination.read_bytes()
            status = "existing"
        else:
            try:
                data = request_bytes(source_url, headers=headers)
                destination.write_bytes(data)
                status = "downloaded"
            except Exception as exc:  # keep the batch going
                failures.append(
                    {"source_path": source_path, "url": source_url, "error": repr(exc)}
                )
                print(f"[{index:04d}/{args.count}] FAILED {source_path}: {exc}")
                continue

        sha256 = hashlib.sha256(data).hexdigest()
        manifest.append(
            {
                "index": len(manifest),
                "source_path": source_path,
                "source_url": source_url,
                "local_file": str(Path("svg_raw") / local_name),
                "sha256": sha256,
                "bytes": len(data),
                "status": status,
            }
        )
        print(f"[{index:04d}/{args.count}] {status:10s} {source_path}")

    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                "source_repository": REPO_URL,
                "source_ref": REPO_REF,
                "seed": args.seed,
                "requested_count": args.count,
                "downloaded_count": len(manifest),
                "failed_count": len(failures),
                "files": manifest,
                "failures": failures,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    write_text(
        args.output / "README.txt",
        (
            "Openclipart SVG subset\n\n"
            f"Source: {REPO_URL}\n"
            f"Reference: {REPO_REF}\n"
            f"Random seed: {args.seed}\n"
            f"Requested files: {args.count}\n"
            f"Downloaded files: {len(manifest)}\n"
            f"Failed files: {len(failures)}\n\n"
            "The files in svg_raw are source SVGs. They have not been rasterized "
            "or JPEG-compressed. Use them to render clean PNG targets, then create "
            "JPEG degraded inputs. See manifest.json for source paths and hashes.\n\n"
            "The GitHub mirror is marked CC0-1.0 by its repository metadata. Verify "
            "the licensing information of individual source artwork before any "
            "redistribution or publication.\n"
        ),
    )

    print(
        f"\nFinished: {len(manifest)}/{args.count} files in {svg_dir}\n"
        f"Manifest: {args.output / 'manifest.json'}"
    )
    return 0 if not failures else 2


if __name__ == "__main__":
    sys.exit(main())
