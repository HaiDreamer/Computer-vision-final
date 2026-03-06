import json
import numpy as np
import pandas as pd
from pathlib import Path
from shapely.geometry import Polygon
from rapidfuzz.distance import Levenshtein
from jiwer import wer

# =====================================================
# CONFIG
# =====================================================

BASE_DIR = Path(r"C:\Users\Daonq\OneDrive\Documents\USTH\computer vision\Final")

OUTPUT_DIR = BASE_DIR / "outputs"

VARIANT_DIRS = {
    "P0": BASE_DIR / "P0_No preprocessing (baseline)" / "p0",
    "P1": BASE_DIR / "P1_Grayscale_HE" / "p1",
    "P2": BASE_DIR / "P2_Grayscale + Bilateral Filter + Sharpen" / "p2",
    "P3": BASE_DIR / "P3_Grayscale + Sauvola Adaptive Threshold + Morphology" / "p3",
    "P4": BASE_DIR / "P4_LAB enhancement + CLAHE (color-based contrast boosting)" / "p4",
}

SPLITS = ["train", "valid", "test"]

IOU_THR = 0.5
E2E_CER_THR = 0.1

# =====================================================
# GT PARSER
# =====================================================

def parse_ic15_gt(gt_path: Path):
    """
    IC15 format:
    x1,y1,x2,y2,x3,y3,x4,y4,text
    """
    items = []

    if not gt_path.exists():
        return items

    lines = gt_path.read_text(encoding="utf-8-sig", errors="ignore").splitlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split(",")
        if len(parts) < 9:
            continue

        nums = list(map(int, parts[:8]))
        text = ",".join(parts[8:]).strip()

        poly = [
            (nums[0], nums[1]),
            (nums[2], nums[3]),
            (nums[4], nums[5]),
            (nums[6], nums[7]),
        ]

        items.append({
            "poly": poly,
            "text": text
        })

    return items

# =====================================================
# METRICS
# =====================================================

def normalize_text(s: str):
    return (s or "").strip().lower()

def cer(gt: str, pred: str):
    gt = gt or ""
    pred = pred or ""
    if len(gt) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return Levenshtein.distance(gt, pred) / len(gt)

def poly_iou(poly1, poly2):
    p1 = Polygon(poly1)
    p2 = Polygon(poly2)

    if (not p1.is_valid) or (not p2.is_valid):
        return 0.0

    inter = p1.intersection(p2).area
    union = p1.union(p2).area

    return inter / union if union > 0 else 0.0

def greedy_match(gt_polys, pred_polys, iou_thr=0.5):
    pairs = []
    for gi, g in enumerate(gt_polys):
        for pi, p in enumerate(pred_polys):
            iou = poly_iou(g, p)
            pairs.append((iou, gi, pi))

    pairs.sort(reverse=True, key=lambda x: x[0])

    used_gt = set()
    used_pred = set()
    matches = []

    for iou, gi, pi in pairs:
        if iou < iou_thr:
            break
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        matches.append((gi, pi, iou))

    return matches, used_gt, used_pred

# =====================================================
# MAP JSON -> GT FILE
# =====================================================

def prediction_json_to_gt_file(pred_json_path: Path, variant_root: Path):
    """
    JSON name được save kiểu:
    train__images__img_1.json
    valid__images__img_99.json
    test__images__img_20.json

    -> map thành:
    variant_root/train/labels/gt_img_1.txt
    """
    stem = pred_json_path.stem
    parts = stem.split("__")

    if len(parts) < 3:
        return None, None

    split_name = parts[0].lower()

    img_name = None
    for p in reversed(parts):
        if p.startswith("img_"):
            img_name = p
            break

    if img_name is None:
        return None, None

    gt_file = variant_root / split_name / "labels" / f"gt_{img_name}.txt"
    return split_name, gt_file

# =====================================================
# CORE EVAL
# =====================================================

def evaluate_json_files(variant_name, variant_root: Path, json_files):
    det_tp = 0
    det_fp = 0
    det_fn = 0

    cer_scores = []
    wer_scores = []
    exact_scores = []

    e2e_strict_scores = []
    e2e_soft_scores = []

    per_image_rows = []
    used_images = 0

    for jf in json_files:
        split_name, gt_file = prediction_json_to_gt_file(jf, variant_root)
        if gt_file is None or (not gt_file.exists()):
            continue

        pred = json.loads(jf.read_text(encoding="utf-8"))

        gt_items = parse_ic15_gt(gt_file)
        if len(gt_items) == 0:
            continue

        gt_polys = [x["poly"] for x in gt_items]
        gt_texts = [x["text"] for x in gt_items]

        ignore_mask = [normalize_text(t) == "###" for t in gt_texts]

        pred_items = pred.get("predictions", [])
        pred_polys = [x["poly"] for x in pred_items]
        pred_texts = [x.get("text", "") for x in pred_items]

        matches, used_gt, used_pred = greedy_match(gt_polys, pred_polys, IOU_THR)

        # Detection
        tp = len(matches)
        fp = len(pred_polys) - len(used_pred)
        fn = len(gt_polys) - len(used_gt)

        det_tp += tp
        det_fp += fp
        det_fn += fn

        img_cers = []
        img_wers = []
        img_exact = []
        img_e2e_strict = []
        img_e2e_soft = []

        for gi, pi, iou in matches:
            if ignore_mask[gi]:
                continue

            gt_text = normalize_text(gt_texts[gi])
            pred_text = normalize_text(pred_texts[pi])

            c = cer(gt_text, pred_text)
            w = wer(gt_text, pred_text) if len(gt_text.split()) > 0 else 0.0
            ex = 1.0 if gt_text == pred_text else 0.0
            strict = ex
            soft = 1.0 if c <= E2E_CER_THR else 0.0

            cer_scores.append(c)
            wer_scores.append(w)
            exact_scores.append(ex)
            e2e_strict_scores.append(strict)
            e2e_soft_scores.append(soft)

            img_cers.append(c)
            img_wers.append(w)
            img_exact.append(ex)
            img_e2e_strict.append(strict)
            img_e2e_soft.append(soft)

        used_images += 1

        img_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        img_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        img_f1 = (2 * img_precision * img_recall) / (img_precision + img_recall) if (img_precision + img_recall) > 0 else 0.0

        per_image_rows.append({
            "variant": variant_name,
            "split": split_name,
            "json_file": jf.name,
            "gt_file": str(gt_file),
            "num_gt": len(gt_polys),
            "num_pred": len(pred_polys),
            "num_match": len(matches),
            "det_precision_img": img_precision,
            "det_recall_img": img_recall,
            "det_f1_img": img_f1,
            "cer_img_mean": float(np.mean(img_cers)) if img_cers else None,
            "wer_img_mean": float(np.mean(img_wers)) if img_wers else None,
            "exact_img_mean": float(np.mean(img_exact)) if img_exact else None,
            "e2e_strict_img_mean": float(np.mean(img_e2e_strict)) if img_e2e_strict else None,
            "e2e_soft_img_mean": float(np.mean(img_e2e_soft)) if img_e2e_soft else None,
        })

    precision = det_tp / (det_tp + det_fp) if (det_tp + det_fp) > 0 else 0.0
    recall = det_tp / (det_tp + det_fn) if (det_tp + det_fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    summary = {
        "variant": variant_name,
        "num_eval_images": used_images,
        "det_precision": precision,
        "det_recall": recall,
        "det_f1": f1,
        "det_tp": det_tp,
        "det_fp": det_fp,
        "det_fn": det_fn,
        "rec_CER_mean": float(np.mean(cer_scores)) if cer_scores else None,
        "rec_WER_mean": float(np.mean(wer_scores)) if wer_scores else None,
        "rec_exact_match": float(np.mean(exact_scores)) if exact_scores else None,
        "e2e_strict_acc": float(np.mean(e2e_strict_scores)) if e2e_strict_scores else None,
        "e2e_soft_acc": float(np.mean(e2e_soft_scores)) if e2e_soft_scores else None,
    }

    return summary, per_image_rows

# =====================================================
# MAIN
# =====================================================

def main():
    all_summary = []
    all_per_image = []

    for variant, variant_root in VARIANT_DIRS.items():
        print(f"\n========== Evaluating {variant} ==========")

        json_dir = OUTPUT_DIR / variant / "json"
        if not json_dir.exists():
            print(f"⚠️ Missing json dir: {json_dir}")
            continue

        all_json_files = sorted(json_dir.glob("*.json"))
        if len(all_json_files) == 0:
            print(f"⚠️ No json files in {json_dir}")
            continue

        # split riêng
        for split in SPLITS:
            split_json_files = [jf for jf in all_json_files if jf.stem.startswith(f"{split}__")]

            if len(split_json_files) == 0:
                continue

            summary, per_image_rows = evaluate_json_files(variant, variant_root, split_json_files)
            summary["split"] = split

            all_summary.append(summary)
            all_per_image.extend(per_image_rows)

        # gộp all
        summary_all, per_image_rows_all = evaluate_json_files(variant, variant_root, all_json_files)
        summary_all["split"] = "all"

        all_summary.append(summary_all)

    df_summary = pd.DataFrame(all_summary)
    df_per_image = pd.DataFrame(all_per_image)

    # sắp xếp
    split_order = {"train": 0, "valid": 1, "test": 2, "all": 3}
    df_summary["split_order"] = df_summary["split"].map(split_order)
    df_summary = df_summary.sort_values(["variant", "split_order"]).drop(columns=["split_order"])

    print("\n================ FINAL SUMMARY ================")
    print(df_summary.to_string(index=False))

    summary_path = OUTPUT_DIR / "summary_eval_all_splits.csv"
    per_image_path = OUTPUT_DIR / "per_image_eval_all_splits.csv"

    df_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    df_per_image.to_csv(per_image_path, index=False, encoding="utf-8-sig")

    print(f"\n✅ Saved summary to: {summary_path}")
    print(f"✅ Saved per-image metrics to: {per_image_path}")


if __name__ == "__main__":
    main()