import os
import shutil
import cv2
import numpy as np
from tqdm import tqdm
from skimage.filters import threshold_sauvola

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def preprocess_p2(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.bilateralFilter(g, 7, 50, 50)
    blur = cv2.GaussianBlur(g, (0, 0), 1.2)
    return cv2.addWeighted(g, 1.6, blur, -0.6, 0)

def preprocess_p3(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    t = threshold_sauvola(g, window_size=51, k=0.10)
    b = ((g > t).astype(np.uint8) * 255)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN, k, iterations=1)
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, k, iterations=1)
    return b

def copy_labels(src_labels, dst_labels):
    if not os.path.isdir(src_labels):
        return
    ensure_dir(dst_labels)
    for f in os.listdir(src_labels):
        if f.lower().endswith(".txt"):
            shutil.copy2(os.path.join(src_labels, f), os.path.join(dst_labels, f))

def process_split(split_dir, out_dir, variant):
    fn = preprocess_p2 if variant == "p2" else preprocess_p3

    src_images = os.path.join(split_dir, "images")
    src_labels = os.path.join(split_dir, "labels")
    dst_images = os.path.join(out_dir, "images")
    dst_labels = os.path.join(out_dir, "labels")

    ensure_dir(dst_images)
    copy_labels(src_labels, dst_labels)

    imgs = []
    for f in os.listdir(src_images):
        if os.path.splitext(f.lower())[1] in IMG_EXTS:
            imgs.append(f)

    for f in tqdm(imgs, desc=f"{variant}:{os.path.basename(split_dir)}"):
        src = os.path.join(src_images, f)
        dst = os.path.join(dst_images, f)
        img = cv2.imread(src, cv2.IMREAD_COLOR)
        out = fn(img)
        cv2.imwrite(dst, out)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--valid", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--variant", required=True, choices=["p2", "p3"])
    args = ap.parse_args()

    out_root = os.path.join(args.output, args.variant)
    process_split(args.train, os.path.join(out_root, "train"), args.variant)
    process_split(args.valid, os.path.join(out_root, "valid"), args.variant)
    process_split(args.test,  os.path.join(out_root, "test"),  args.variant)

if __name__ == "__main__":
    main()