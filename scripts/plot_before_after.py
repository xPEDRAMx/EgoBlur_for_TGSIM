#!/usr/bin/env python3
"""Stack before/after images vertically (original on top, blurred below). Same basename pairing."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def _read_bgr(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path))
    if img is None or img.size == 0:
        return None
    return img


def _label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.5, min(w, h) / 900.0)
    thickness = max(1, int(round(2 * scale)))
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x, y = 12, min(h - 12, th + 16)
    cv2.putText(out, text, (x, y), font, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(out, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out


def _resize_match(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ha, wa = a.shape[:2]
    hb, wb = b.shape[:2]
    if (ha, wa) == (hb, wb):
        return a, b
    b_resized = cv2.resize(b, (wa, ha), interpolation=cv2.INTER_AREA)
    return a, b_resized


def _maybe_downscale_max_height(combined: np.ndarray, max_height: int | None) -> np.ndarray:
    if max_height is None or combined.shape[0] <= max_height:
        return combined
    scale = max_height / float(combined.shape[0])
    new_w = int(combined.shape[1] * scale)
    new_h = int(combined.shape[0] * scale)
    return cv2.resize(combined, (new_w, new_h), interpolation=cv2.INTER_AREA)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Save stacked before/after plots (original on top, blurred below)."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("test"),
        help="Directory with original images (default: test)",
    )
    parser.add_argument(
        "--blurred_dir",
        type=Path,
        default=Path("test/output"),
        help="Directory with blurred outputs, same filenames (default: test/output)",
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        default=Path("test/comparison_plots"),
        help="Where to write comparison PNGs (default: test/comparison_plots)",
    )
    parser.add_argument(
        "--gap",
        type=int,
        default=12,
        help="White gap height in pixels between top and bottom panels (default: 12)",
    )
    parser.add_argument(
        "--max_height",
        type=int,
        default=4000,
        help="If combined image is taller than this, shrink (default: 4000). Use 0 to disable.",
    )
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        raise SystemExit(f"input_dir not found: {args.input_dir.resolve()}")
    if not args.blurred_dir.is_dir():
        raise SystemExit(f"blurred_dir not found: {args.blurred_dir.resolve()}")

    args.save_dir.mkdir(parents=True, exist_ok=True)

    max_h = args.max_height if args.max_height and args.max_height > 0 else None

    paths = sorted(
        p
        for p in args.input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise SystemExit(f"No images found in {args.input_dir.resolve()}")

    n_ok = 0
    n_skip = 0
    for inp in paths:
        blurred = args.blurred_dir / inp.name
        if not blurred.is_file():
            print(f"skip (no pair): {inp.name}")
            n_skip += 1
            continue
        before = _read_bgr(inp)
        after = _read_bgr(blurred)
        if before is None or after is None:
            print(f"skip (read failed): {inp.name}")
            n_skip += 1
            continue

        before, after = _resize_match(before, after)
        before = _label(before, "Before")
        after = _label(after, "After")

        h, w = before.shape[:2]
        gh = max(0, args.gap)
        if gh > 0:
            gap = np.full((gh, w, 3), 255, dtype=np.uint8)
            combined = np.vstack([before, gap, after])
        else:
            combined = np.vstack([before, after])
        combined = _maybe_downscale_max_height(combined, max_h)

        out_path = args.save_dir / f"compare_{inp.stem}.png"
        if not cv2.imwrite(str(out_path), combined):
            print(f"write failed: {out_path}")
            n_skip += 1
            continue
        print(f"wrote {out_path}")
        n_ok += 1

    print(f"done: {n_ok} comparisons, {n_skip} skipped")
    return 0 if n_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
