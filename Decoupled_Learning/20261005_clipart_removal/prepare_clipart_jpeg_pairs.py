#!/usr/bin/env python3
"""Render Openclipart SVGs to clean PNGs and generate JPEG inputs.

Pipeline:
    SVG source -> high-resolution clean PNG -> low-quality JPEG

The PNG is the clean reference for the synthetic experiment. The JPEG files
are the degraded inputs for WLS/V8 evaluation. The default output keeps the
same base name for every quality factor and writes a manifest describing all
generated files.

Install the renderer once if needed:
    python -m pip install cairosvg

Example:
    python prepare_clipart_jpeg_pairs.py
    python prepare_clipart_jpeg_pairs.py --quality 10 20 30 50
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from PIL import Image


DEFAULT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = DEFAULT_ROOT / "openclipart_500" / "svg_raw"
DEFAULT_OUTPUT = DEFAULT_ROOT / "openclipart_500" / "jpeg_pairs_512"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="maximum clean image side in pixels; default: 512",
    )
    parser.add_argument(
        "--quality",
        type=int,
        nargs="+",
        default=[10, 20, 30, 50],
        help="JPEG quality factors; default: 10 20 30 50",
    )
    parser.add_argument(
        "--subsampling",
        type=int,
        choices=[0, 1, 2],
        default=2,
        help="JPEG chroma subsampling: 0=4:4:4, 1=4:2:2, 2=4:2:0",
    )
    parser.add_argument(
        "--background",
        default="#ffffff",
        help="background used to flatten SVG transparency; default: white",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing clean PNG and JPEG files",
    )
    return parser.parse_args()


def load_renderer():
    try:
        import cairosvg  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "CairoSVG is required to render SVG files. Install it with:\n"
            "  python -m pip install cairosvg\n"
            f"Original error: {exc}"
        ) from exc
    return cairosvg


def flatten_rgba(path: Path, background: str, size: int) -> Image.Image:
    """Render one SVG and center it on a square RGB canvas without distortion."""
    cairosvg = load_renderer()
    # output_width scales the SVG while preserving its intrinsic aspect ratio.
    # A temporary PNG avoids assumptions about SVG width/height declarations.
    png_bytes = cairosvg.svg2png(url=str(path), output_width=size)
    from io import BytesIO

    with Image.open(BytesIO(png_bytes)) as rendered:
        rgba = rendered.convert("RGBA")

    # Fit the long side to `size`, preserving aspect ratio.
    scale = min(size / rgba.width, size / rgba.height)
    new_size = (
        max(1, round(rgba.width * scale)),
        max(1, round(rgba.height * scale)),
    )
    if new_size != rgba.size:
        rgba = rgba.resize(new_size, Image.Resampling.LANCZOS)

    canvas = Image.new("RGBA", (size, size), background)
    offset = ((size - rgba.width) // 2, (size - rgba.height) // 2)
    canvas.alpha_composite(rgba, offset)
    return canvas.convert("RGB")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_svgs(input_dir: Path) -> Iterable[Path]:
    return sorted(input_dir.rglob("*.svg"))


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input directory does not exist: {args.input}")
    if args.size < 64:
        raise SystemExit("--size must be at least 64")
    if any(q < 1 or q > 95 for q in args.quality):
        raise SystemExit("JPEG quality must be between 1 and 95")

    svg_paths = list(iter_svgs(args.input))
    if not svg_paths:
        raise SystemExit(f"No SVG files found under: {args.input}")

    # Import once before creating output files so a missing renderer fails early.
    load_renderer()
    clean_dir = args.output / "clean_png"
    jpeg_dirs = {q: args.output / f"jpeg_q{q:02d}" for q in args.quality}
    clean_dir.mkdir(parents=True, exist_ok=True)
    for directory in jpeg_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, object]] = []
    failures: List[Dict[str, str]] = []

    for index, svg_path in enumerate(svg_paths, start=1):
        relative = svg_path.relative_to(args.input)
        # Include a short path hash to avoid collisions between folders sharing a name.
        source_key = relative.as_posix()
        digest = hashlib.sha1(source_key.encode("utf-8")).hexdigest()[:10]
        stem = f"{svg_path.stem}__{digest}"
        clean_path = clean_dir / f"{stem}.png"

        try:
            if clean_path.exists() and not args.overwrite:
                clean = Image.open(clean_path).convert("RGB")
            else:
                clean = flatten_rgba(svg_path, args.background, args.size)
                clean.save(clean_path, format="PNG", optimize=False)

            clean_sha = sha256_file(clean_path)
            for quality, jpeg_dir in jpeg_dirs.items():
                jpeg_path = jpeg_dir / f"{stem}.jpg"
                if not jpeg_path.exists() or args.overwrite:
                    clean.save(
                        jpeg_path,
                        format="JPEG",
                        quality=quality,
                        subsampling=args.subsampling,
                        optimize=False,
                        progressive=False,
                    )

                records.append(
                    {
                        "source_svg": str(relative),
                        "clean_png": str(clean_path.relative_to(args.output)),
                        "jpeg": str(jpeg_path.relative_to(args.output)),
                        "quality": quality,
                        "subsampling": args.subsampling,
                        "width": clean.width,
                        "height": clean.height,
                        "clean_sha256": clean_sha,
                        "jpeg_sha256": sha256_file(jpeg_path),
                    }
                )
            clean.close()
            print(f"[{index:04d}/{len(svg_paths)}] {relative}")
        except Exception as exc:
            failures.append({"source_svg": str(relative), "error": repr(exc)})
            print(f"[{index:04d}/{len(svg_paths)}] FAILED {relative}: {exc}")

    manifest = {
        "input": str(args.input),
        "output": str(args.output),
        "size": args.size,
        "background": args.background,
        "quality": args.quality,
        "subsampling": args.subsampling,
        "source_count": len(svg_paths),
        "record_count": len(records),
        "failure_count": len(failures),
        "records": records,
        "failures": failures,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with (args.output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        if records:
            writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)

    readme = (
        "Synthetic clip-art JPEG pairs\n\n"
        f"SVG sources: {len(svg_paths)}\n"
        f"Successful SVGs: {len({r['source_svg'] for r in records})}\n"
        f"JPEG qualities: {args.quality}\n"
        f"Canvas: {args.size}x{args.size}, background={args.background}\n"
        f"Subsampling: {args.subsampling}\n\n"
        "clean_png contains the rasterized SVG references. jpeg_qXX contains "
        "the corresponding degraded inputs. manifest.csv/json record alignment.\n"
    )
    (args.output / "README.txt").write_text(readme, encoding="utf-8")

    print(
        f"\nFinished. Generated {len(records)} pairs in {args.output}. "
        f"Failures: {len(failures)}."
    )
    return 0 if not failures else 2


if __name__ == "__main__":
    sys.exit(main())
