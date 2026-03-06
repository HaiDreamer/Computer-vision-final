import argparse
import os
import random
import re
import shutil
from pathlib import Path

ID_RE = re.compile(r"img_(\d+)\.(jpg|png|jpeg)$", re.IGNORECASE)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def list_ids(img_dir: Path):
    ids = []
    for p in img_dir.iterdir():
        m = ID_RE.match(p.name)
        if m:
            ids.append(int(m.group(1)))
    return sorted(ids)

def find_image(img_dir: Path, i: int) -> Path:
    # try common extensions
    for ext in (".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"):
        p = img_dir / f"img_{i}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"Missing image: {img_dir / f'img_{i}.*'}")

def link_or_copy(src: Path, dst: Path, mode: str):
    ensure_dir(dst.parent)
    if mode == "none":
        return
    if mode == "symlink":
        # On Windows, symlink may require admin or Developer Mode.
        if dst.exists():
            dst.unlink()
        os.symlink(src.resolve(), dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError("mode must be one of: none, symlink, copy")

def write_ids(path: Path, ids):
    ensure_dir(path.parent)
    path.write_text("\n".join(map(str, ids)) + "\n", encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_img_dir", default=r"D:\ic15_raw\ch4_training_images")
    ap.add_argument("--train_gt_dir",  default=r"D:\ic15_raw\ch4_training_localization_transcription_gt")
    ap.add_argument("--test_img_dir",  default=r"D:\ic15_raw\ch4_test_images")
    ap.add_argument("--test_gt_dir",   default=r"D:\ic15_raw\Challenge4_Test_Task4_GT")
    ap.add_argument("--out_root",      default=r"D:\ic15_raw")

    ap.add_argument("--train_n", type=int, default=800)
    ap.add_argument("--valid_n", type=int, default=200)
    ap.add_argument("--test_n",  type=int, default=200)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--materialize", choices=["symlink", "copy"], default="copy",
                    help="Windows recommendation: use copy unless you know symlinks work")
    args = ap.parse_args()

    tr_img_dir = Path(args.train_img_dir)
    tr_gt_dir  = Path(args.train_gt_dir)
    te_img_dir = Path(args.test_img_dir)
    te_gt_dir  = Path(args.test_gt_dir)
    out_root   = Path(args.out_root)

    # sanity checks
    if args.train_n + args.valid_n != 1000:
        raise ValueError("IC15 train has 1000 images; your train_n + valid_n should equal 1000 "
                         f"(got {args.train_n + args.valid_n}).")

    train_ids_all = list_ids(tr_img_dir)
    test_ids_all  = list_ids(te_img_dir)

    if len(train_ids_all) != 1000:
        raise ValueError(f"Expected 1000 training images, found {len(train_ids_all)} in {tr_img_dir}")
    if len(test_ids_all) != 500:
        print(f"Warning: expected 500 test images, found {len(test_ids_all)} in {te_img_dir}")

    rng = random.Random(args.seed)

    # Split train/valid from the 1000 training images
    shuffled_train = train_ids_all[:]
    rng.shuffle(shuffled_train)
    valid_ids = sorted(shuffled_train[:args.valid_n])
    train_ids = sorted(shuffled_train[args.valid_n:args.valid_n + args.train_n])

    # Sample 200 test images (keep full list too)
    shuffled_test = test_ids_all[:]
    rng.shuffle(shuffled_test)
    test_ids_200 = sorted(shuffled_test[:args.test_n])

    # Write id lists
    splits_dir = out_root / "splits"
    write_ids(splits_dir / "train_ids.txt", train_ids)
    write_ids(splits_dir / "valid_ids.txt", valid_ids)
    write_ids(splits_dir / "test_ids_200.txt", test_ids_200)
    write_ids(splits_dir / "test_ids_all.txt", sorted(test_ids_all))

    # Materialize folders 
    def materialize_split(split_name: str, ids, img_dir: Path, gt_dir: Path):
        img_out = out_root / split_name / "images"
        gt_out  = out_root / split_name / "labels"
        for i in ids:
            img_src = find_image(img_dir, i)
            gt_src  = gt_dir / f"gt_img_{i}.txt"

            if not gt_src.exists():
                raise FileNotFoundError(f"Missing GT: {gt_src}")

            link_or_copy(img_src, img_out / img_src.name, args.materialize)
            link_or_copy(gt_src,  gt_out  / gt_src.name, args.materialize)

    materialize_split("train", train_ids, tr_img_dir, tr_gt_dir)
    materialize_split("valid", valid_ids, tr_img_dir, tr_gt_dir)
    materialize_split("test",  test_ids_200, te_img_dir, te_gt_dir)

    print("Done.")
    print(f"Train: {len(train_ids)}  Valid: {len(valid_ids)}  Test(subset): {len(test_ids_200)}")
    print(f"Output root: {out_root}")
    print(f"Split lists: {splits_dir}")

if __name__ == "__main__":
    main()