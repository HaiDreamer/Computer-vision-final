'''
LAB+CLAHE preprocessing is a photometric-only transform (it changes contrast/brightness locally but does not move pixels geometrically), 
    so IC15 quad labels stay valid
    IC15 uses quadrilaterals (clockwise label)
IC15 “incidental” scene text images often have:
    uneven illumination (shadow, glare),
    low contrast text vs background,
    locally washed-out or locally too-dark regions
Usage: boosts contrast/visibility while reducing unwanted hue shifts
For: better CNN model
    easier feature extraction in early CNN layers (cleaner edges)
    faster / more stable convergence by reducing illuminance variance
TODO: check multiply variant 
'''
import argparse
import shutil
import re
from pathlib import Path

import cv2

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
ID_RE = re.compile(r"img_(\d+)$", re.IGNORECASE)  # stem like img_123

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def find_images_dir(split_dir: Path) -> Path:
    # split/images
    cand = split_dir / "images"
    return cand if cand.exists() else split_dir

def find_labels_dir(split_dir: Path) -> Path:
    # split/labels
    cand = split_dir / "labels"
    return cand if cand.exists() else split_dir

def iter_images(img_dir: Path):
    for p in img_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p

def derive_label_path(img_path: Path, labels_dir: Path) -> Path:
    """
    IC15 convention: img_123.jpg -> gt_img_123.txt
    Fallback: same-stem .txt (img_123.txt) if present.
    """
    stem = img_path.stem  # e.g., img_123
    m = ID_RE.match(stem)
    if m:
        i = m.group(1)
        cand = labels_dir / f"gt_img_{i}.txt"
        if cand.exists():
            return cand

    # fallback: same stem
    cand2 = labels_dir / f"{stem}.txt"
    if cand2.exists():
        return cand2

    # try to find any gt file that contains the numeric id
    if m:
        i = m.group(1)
        cand3 = labels_dir / f"gt_{stem}.txt"
        if cand3.exists():
            return cand3
        # Could add more heuristics if needed.

    raise FileNotFoundError(f"Label not found for image: {img_path.name} (looked in {labels_dir})")

def lab_clahe_bgr(bgr, clip_limit: float, tile_grid: int):
    """
    Convert BGR->LAB (CIELAB color space), apply CLAHE on L channel, then LAB->BGR.
    Why?
        L* represents perceptual lightness
        a* and b* represent chromatic opponent axes (green-red, blue-yellow)
        => enhance contrast on lightness without directly distorting colors
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)

    # clipLimit = 2, tile grid=(8, 8) as default. split image to 64 part w/8, h/8 then local histogram equalization(adapt constrast per region)
    #   tile grid
    #   bigger: help with strong with illumination changes, but can make local artifacts / texture amplification
    #   smaller: smooth, less artifact but may not fix local shadow, glare good
    #   clip limit
    #   higher: improve ability to reveal faint text stroke(ink like to make a char) but can amp noises, over-crisp edges
    #   smaller: less noisy but may not save very low constract text
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit),    
                            tileGridSize=(int(tile_grid), int(tile_grid)))
    L2 = clahe.apply(L)

    lab2 = cv2.merge((L2, a, b))
    out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    return out

def process_split(split_in: Path, split_out: Path, clip_limit: float, tile_grid: int,
                  jpg_quality: int, overwrite: bool):
    img_in = find_images_dir(split_in)
    lab_in = find_labels_dir(split_in)

    img_out = split_out / "images"
    lab_out = split_out / "labels"
    ensure_dir(img_out)
    ensure_dir(lab_out)

    count = 0
    for img_path in iter_images(img_in):
        # read img
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[WARN] Could not read image: {img_path}")
            continue

        # enhance
        out_bgr = lab_clahe_bgr(bgr, clip_limit=clip_limit, tile_grid=tile_grid)

        # write image (preserve filename)
        out_img_path = img_out / img_path.name
        if out_img_path.exists() and not overwrite:
            pass
        else:
            if out_img_path.suffix.lower() in {".jpg", ".jpeg"}:
                cv2.imwrite(str(out_img_path), out_bgr, [cv2.IMWRITE_JPEG_QUALITY, int(jpg_quality)])
            else:
                cv2.imwrite(str(out_img_path), out_bgr)

        # copy label
        try:
            gt_path = derive_label_path(img_path, lab_in)
        except FileNotFoundError as e:
            raise FileNotFoundError(str(e) + "\nTip: ensure labels are in split/labels or split root "
                                            "and follow IC15 naming gt_img_#.txt") from e

        out_gt_path = lab_out / gt_path.name
        if out_gt_path.exists() and not overwrite:
            pass
        else:
            shutil.copy2(gt_path, out_gt_path)

        count += 1

    print(f"[OK] {split_in.name}: processed {count} images -> {split_out}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=r"D:\ic15_raw\train")
    ap.add_argument("--valid", default=r"D:\ic15_raw\valid")
    ap.add_argument("--test",  default=r"D:\ic15_raw\test")
    ap.add_argument("--out",   default=r"D:\ic15_raw\p4")

    ap.add_argument("--clip_limit", type=float, default=2.0,
                    help="CLAHE clipLimit (start mild: 2.0)")
    ap.add_argument("--tile_grid", type=int, default=8,
                    help="CLAHE tileGridSize as NxN (common: 8)")
    ap.add_argument("--jpg_quality", type=int, default=95)
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing outputs")
    args = ap.parse_args()

    out_root = Path(args.out)
    ensure_dir(out_root)

    for name, in_dir in [("train", Path(args.train)),
                         ("valid", Path(args.valid)),
                         ("test",  Path(args.test))]:
        if not in_dir.exists():
            raise FileNotFoundError(f"Input split not found: {in_dir}")

        process_split(
            split_in=in_dir,
            split_out=out_root / name,
            clip_limit=args.clip_limit,
            tile_grid=args.tile_grid,
            jpg_quality=args.jpg_quality,
            overwrite=args.overwrite
        )

    print("\nDone.")
    print(f"Output dataset: {out_root}")

if __name__ == "__main__":
    main()