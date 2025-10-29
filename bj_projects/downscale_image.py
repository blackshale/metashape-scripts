#!/usr/bin/env python3
"""
Batch downscale images to a target size while preserving metadata.

Usage:
  python batch_downscale_preserve_meta.py <input_folder> <output_folder> --size 1600x1200

- Reads all image files from <input_folder> (non-recursive) with common extensions.
- Saves into <output_folder> using filename pattern: {originalname}_W_{WIDTH}x{HEIGHT}{ext}
- Preserves EXIF & ICC via Pillow. If `exiftool` is available, copies ALL metadata
  (EXIF/XMP/GPS/MakerNotes) from the source to the output.
"""

import argparse, os, sys, shutil, subprocess
from pathlib import Path
from PIL import Image, ImageOps

SUPPORTED_EXTS = {".jpg", ".jpeg", ".tif", ".tiff", ".png", ".webp"}

def parse_size(size_str: str):
    try:
        w_str, h_str = size_str.lower().split("x")
        w, h = int(w_str), int(h_str)
        assert w > 0 and h > 0
        return (w, h)
    except Exception:
        raise argparse.ArgumentTypeError("Size must be like 1600x1200")

def collect_images(input_folder: Path):
    files = []
    for p in sorted(input_folder.iterdir()):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            files.append(p)
    return files

def save_with_metadata(src: Path, dst: Path, target_size):
    dst.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im)
        im = im.resize(target_size, Image.Resampling.LANCZOS)

        # Pull-through EXIF & ICC when saving
        exif_bytes = im.info.get("exif")
        if not exif_bytes:
            try:
                with Image.open(src) as src_im:
                    exif_bytes = src_im.info.get("exif")
            except Exception:
                exif_bytes = None

        icc_profile = im.info.get("icc_profile")
        if not icc_profile:
            try:
                with Image.open(src) as src_im:
                    icc_profile = src_im.info.get("icc_profile")
            except Exception:
                icc_profile = None

        # Choose save format by extension
        ext = dst.suffix.lower()
        fmt = None
        if ext in (".jpg", ".jpeg"):
            fmt = "JPEG"
            save_kwargs = {"quality": 95, "optimize": True, "subsampling": 1}
            if exif_bytes: save_kwargs["exif"] = exif_bytes
            if icc_profile: save_kwargs["icc_profile"] = icc_profile
        elif ext in (".tif", ".tiff"):
            fmt = "TIFF"
            save_kwargs = {"compression": "tjpeg"}  # good balance
            if exif_bytes: save_kwargs["exif"] = exif_bytes
            if icc_profile: save_kwargs["icc_profile"] = icc_profile
        else:
            # PNG/WEBP etc.
            fmt = None  # let Pillow infer from extension
            save_kwargs = {}
            if icc_profile: save_kwargs["icc_profile"] = icc_profile

        im.save(dst, format=fmt, **save_kwargs)

    exiftool = shutil.which("exiftool")
    if exiftool:
        try:
            subprocess.run(
                [exiftool, "-overwrite_original", "-TagsFromFile", str(src), "-all:all", "-unsafe", str(dst)],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
        except subprocess.CalledProcessError as e:
            print(f"[WARN] exiftool failed for {dst.name}: {e}")

def main():
    ap = argparse.ArgumentParser(description="Batch downscale with metadata preservation")
    ap.add_argument("input_folder", type=Path, help="Folder containing images")
    ap.add_argument("output_folder", type=Path, help="Folder to write resized images")
    ap.add_argument("--size", default="1600x1200", type=parse_size, help="Target size WxH, e.g., 1600x1200")
    args = ap.parse_args()

    input_folder: Path = args.input_folder
    output_folder: Path = args.output_folder
    target_size = args.size

    if not input_folder.exists():
        print(f"[ERR] Input folder not found: {input_folder}")
        sys.exit(1)

    images = collect_images(input_folder)
    if not images:
        print("[INFO] No images found.")
        return

    w, h = target_size
    for src in images:
        stem = src.stem
        ext = src.suffix  # keep original extension
        dst_name = f"{stem}_{w}x{h}{ext}"
        dst = output_folder / dst_name
        try:
            save_with_metadata(src, dst, target_size)
            print(f"[OK] {src.name} -> {dst.name}")
        except Exception as e:
            print(f"[FAIL] {src.name}: {e}")

if __name__ == "__main__":
    main()
