# translate_image.py
from __future__ import annotations
import sys
import os
from typing import Tuple

try:
    import rasterio
    from rasterio.transform import Affine
except Exception as e:
    raise RuntimeError("translate_image.py requires rasterio. Please install rasterio.") from e


def translate_image(input_path: str, pt1_map: Tuple[float, float], pt2_map: Tuple[float, float]) -> str:
    dx = float(pt2_map[0]) - float(pt1_map[0])
    dy = float(pt2_map[1]) - float(pt1_map[1])

    base, ext = os.path.splitext(input_path)
    if ext.lower() not in (".tif", ".tiff"):
        ext = ".tif"
    out_path = f"{base}_translated{ext}"

    with rasterio.open(input_path) as src:
        T = src.transform
        new_T = Affine(T.a, T.b, T.c + dx, T.d, T.e, T.f + dy)

        profile = src.profile.copy()
        profile.update(transform=new_T)

        with rasterio.open(out_path, "w", **profile) as dst:
            data = src.read()
            dst.write(data)
            dst.update_tags(**src.tags())
            for i in range(1, src.count + 1):
                dst.update_tags(i, **src.tags(i))

    return out_path


def _main(argv=None):
    argv = list(sys.argv if argv is None else argv)
    if len(argv) != 6:
        print("Usage: python translate_image.py input.tif x1 y1 x2 y2")
        return 2
    input_path = argv[1]
    x1 = float(argv[2]); y1 = float(argv[3])
    x2 = float(argv[4]); y2 = float(argv[5])
    out = translate_image(input_path, (x1, y1), (x2, y2))
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
