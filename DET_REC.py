import os
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from paddleocr import PaddleOCR

# ============================================
# CẤU HÌNH ĐƯỜNG DẪN
# ============================================

BASE_DIR = Path(r"C:\Users\Daonq\OneDrive\Documents\USTH\computer vision\Final")

# QUAN TRỌNG:
# Nếu cấu trúc của bạn là:
# P0_No preprocessing (baseline)\p0\train\images
# P1_...\p1\train\images
# ...
# thì hãy sửa thành:
# "P0": BASE_DIR / "P0_No preprocessing (baseline)" / "p0"
# "P1": BASE_DIR / "P1_..." / "p1"
# ...

VARIANT_DIRS = {
    "P0": BASE_DIR / "P0_No preprocessing (baseline)" / "p0",
    "P1": BASE_DIR / "P1_Grayscale_HE" / "p1",
    "P2": BASE_DIR / "P2_Grayscale + Bilateral Filter + Sharpen" / "p2",
    "P3": BASE_DIR / "P3_Grayscale + Sauvola Adaptive Threshold + Morphology" / "p3",
    "P4": BASE_DIR / "P4_LAB enhancement + CLAHE (color-based contrast boosting)" / "p4"
}

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"✓ Base directory: {BASE_DIR}")
print(f"✓ Output directory: {OUTPUT_DIR}")

# ============================================
# KHỞI TẠO MODEL
# ============================================

print("\n⏳ Initializing PaddleOCR model...")
ocr = PaddleOCR(
    lang="en",
    det=True,
    rec=True,
    use_angle_cls=False,
    show_log=False
)
print("✓ PaddleOCR initialized successfully\n")

# ============================================
# CÁC HÀM XỬ LÝ
# ============================================

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images_recursive(folder: Path):
    """
    Tìm tất cả ảnh trong folder theo kiểu đệ quy.
    Nếu có thư mục con tên 'images' thì ưu tiên lấy ảnh trong đó.
    """
    all_imgs = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]

    imgs_in_images_dir = [
        p for p in all_imgs
        if "images" in [part.lower() for part in p.parts]
    ]

    if len(imgs_in_images_dir) > 0:
        return sorted(imgs_in_images_dir)

    return sorted(all_imgs)


def poly_to_bbox(poly):
    """Chuyển polygon thành bounding box"""
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def filter_boxes(polys, scores, min_area=100, min_conf=0.3, min_w=6, min_h=6):
    """Lọc các boxes không hợp lệ"""
    keep_polys = []
    keep_scores = []

    for i, poly in enumerate(polys):
        x1, y1, x2, y2 = poly_to_bbox(poly)

        w = x2 - x1
        h = y2 - y1
        area = w * h

        if w < min_w or h < min_h:
            continue

        if area < min_area:
            continue

        if scores[i] < min_conf:
            continue

        keep_polys.append(poly)
        keep_scores.append(float(scores[i]))

    return keep_polys, keep_scores


def crop_with_padding(img, poly, pad=4):
    """Crop ảnh với padding"""
    h, w = img.shape[:2]

    x1, y1, x2, y2 = poly_to_bbox(poly)

    x1 = max(x1 - pad, 0)
    y1 = max(y1 - pad, 0)
    x2 = min(x2 + pad, w - 1)
    y2 = min(y2 + pad, h - 1)

    crop = img[int(y1):int(y2), int(x1):int(x2)]

    return crop, (x1, y1, x2, y2)


def normalize_crop(img, target_h=48):
    """Resize crop về chiều cao cố định"""
    if img is None or img.size == 0:
        return None

    h, w = img.shape[:2]
    if h == 0:
        return None

    new_w = max(1, int(w * (target_h / h)))
    img = cv2.resize(img, (new_w, target_h))

    return img


def detect_text(img):
    """Detection text boxes bằng detector trực tiếp"""
    dt_boxes, _ = ocr.text_detector(img)

    polys = []
    scores = []

    if dt_boxes is None:
        return polys, scores

    # dt_boxes thường là numpy array shape [N,4,2]
    for box in dt_boxes:
        poly = [(int(p[0]), int(p[1])) for p in box]
        polys.append(poly)
        scores.append(1.0)   # detector direct không luôn trả score rõ, tạm fixed = 1.0

    return polys, scores

def recognize_text(crops):
    """Recognition text từ crops bằng recognizer trực tiếp"""
    if len(crops) == 0:
        return [], []

    rec_res, _ = ocr.text_recognizer(crops)

    texts = []
    confs = []

    if rec_res is None:
        return texts, confs

    for r in rec_res:
        # mỗi r thường là [text, conf]
        text = r[0]
        conf = r[1]

        texts.append(text)
        confs.append(float(conf))

    return texts, confs


def draw_boxes(img, polys, texts):
    """Vẽ boxes và text lên ảnh"""
    out = img.copy()

    for i, poly in enumerate(polys):
        pts = np.array(poly).reshape((-1, 1, 2))
        cv2.polylines(out, [pts], True, (0, 255, 0), 2)

        if i < len(texts):
            x, y = poly[0]

            cv2.putText(
                out,
                str(texts[i])[:30],
                (int(x), max(int(y) - 5, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    return out


def make_safe_stem(img_path: Path, root_folder: Path):
    """
    Tạo tên file output không bị trùng nếu ảnh nằm ở train/test/valid...
    Ví dụ:
    train/images/img1.jpg -> train__images__img1
    """
    rel = img_path.relative_to(root_folder)
    rel_no_suffix = rel.with_suffix("")
    safe_name = "__".join(rel_no_suffix.parts)
    return safe_name


# ============================================
# MAIN PIPELINE
# ============================================

def main():
    summary = []

    for variant, folder in VARIANT_DIRS.items():
        print(f"\n{'=' * 60}")
        print(f"Processing {variant}...")
        print(f"Folder: {folder}")
        print(f"{'=' * 60}")

        if not folder.exists():
            print(f"⚠️ Folder not found: {folder}")
            continue

        # Tạo thư mục output
        out_variant = OUTPUT_DIR / variant
        (out_variant / "json").mkdir(parents=True, exist_ok=True)
        (out_variant / "vis").mkdir(parents=True, exist_ok=True)

        # Lấy danh sách ảnh
        imgs = list_images_recursive(folder)

        if len(imgs) == 0:
            print(f"⚠️ No images found in {folder}")
            continue

        print(f"✓ Found {len(imgs)} images")

        times = []
        processed_count = 0

        # Xử lý từng ảnh
        for img_path in tqdm(imgs, desc=f"{variant}"):
            img = cv2.imread(str(img_path))

            if img is None:
                print(f"⚠️ Cannot read image: {img_path}")
                continue

            t0 = time.time()

            # 1. Detection
            polys, scores = detect_text(img)

            # 2. Filtering
            polys, scores = filter_boxes(polys, scores)

            # 3. Crop
            crops = []
            boxes = []

            for poly in polys:
                crop, box = crop_with_padding(img, poly)
                crop = normalize_crop(crop)

                if crop is None:
                    continue

                crops.append(crop)
                boxes.append(box)

            # 4. Recognition
            texts, confs = recognize_text(crops)

            t1 = time.time()
            times.append(t1 - t0)
            processed_count += 1

            # 5. Save results
            predictions = []

            for i in range(len(polys)):
                predictions.append({
                    "poly": polys[i],
                    "bbox": boxes[i] if i < len(boxes) else None,
                    "text": texts[i] if i < len(texts) else "",
                    "rec_conf": float(confs[i]) if i < len(confs) else 0.0,
                    "det_conf": float(scores[i]) if i < len(scores) else 0.0
                })

            json_out = {
                "image": str(img_path),
                "variant": variant,
                "predictions": predictions,
                "timing_sec": float(t1 - t0)
            }

            safe_stem = make_safe_stem(img_path, folder)

            # Save JSON
            with open(out_variant / "json" / f"{safe_stem}.json", "w", encoding="utf-8") as f:
                json.dump(json_out, f, indent=2, ensure_ascii=False)

            # Save visualization
            vis = draw_boxes(img, polys, texts)
            cv2.imwrite(str(out_variant / "vis" / f"{safe_stem}.jpg"), vis)

        # Thống kê
        summary.append({
            "variant": variant,
            "num_images_found": len(imgs),
            "num_images_processed": processed_count,
            "time_per_image_mean": float(np.mean(times)) if times else 0.0,
            "time_per_image_std": float(np.std(times)) if times else 0.0
        })

    # In báo cáo tổng kết
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    df_summary = pd.DataFrame(summary)
    print(df_summary.to_string(index=False))

    # Lưu báo cáo
    df_summary.to_csv(OUTPUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    print(f"\n✅ Summary saved to: {OUTPUT_DIR / 'summary.csv'}")
    print(f"✅ All results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()