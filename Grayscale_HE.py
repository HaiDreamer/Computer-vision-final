import cv2
import os
import shutil
from tqdm import tqdm


def apply_he_to_image(image_path, save_path):
    # 1. Đọc ảnh
    img = cv2.imread(image_path)
    if img is None:
        return False

    # 2. Chuyển sang ảnh xám (Grayscale)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Cân bằng Histogram toàn cục (Global Histogram Equalization)
    eq_gray = cv2.equalizeHist(gray)

    # 4. Chuyển ngược về định dạng 3 kênh (BGR) để YOLO không bị lỗi Dimension
    final_img = cv2.cvtColor(eq_gray, cv2.COLOR_GRAY2BGR)

    # 5. Lưu ảnh
    cv2.imwrite(save_path, final_img)
    return True


def process_dataset(input_dir, output_dir):
    # ... (Phần code quản lý thư mục và copy txt của hàm process_dataset GIỮ NGUYÊN NHƯ CŨ) ...
    splits = ['train', 'valid', 'test']
    img_exts = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')

    if not os.path.exists(input_dir):
        print(f" LỖI: Không tìm thấy thư mục gốc:\n{input_dir}")
        return

    for split in splits:
        print(f"\n--- Đang xử lý tập: {split.upper()} ---")
        split_src_dir = os.path.join(input_dir, split)

        if not os.path.exists(split_src_dir):
            continue

        img_dst_dir = os.path.join(output_dir, split, 'images')
        lbl_dst_dir = os.path.join(output_dir, split, 'labels')
        os.makedirs(img_dst_dir, exist_ok=True)
        os.makedirs(lbl_dst_dir, exist_ok=True)

        if os.path.exists(os.path.join(split_src_dir, 'images')):
            img_src_dir = os.path.join(split_src_dir, 'images')
            lbl_src_dir = os.path.join(split_src_dir, 'labels')
        else:
            img_src_dir = split_src_dir
            lbl_src_dir = split_src_dir

        # Chạy thuật toán mới
        images = [f for f in os.listdir(img_src_dir) if f.endswith(img_exts)]
        if images:
            for img_name in tqdm(images, desc=f"Grayscale HE {split}"):
                img_path = os.path.join(img_src_dir, img_name)
                save_path = os.path.join(img_dst_dir, img_name)
                apply_he_to_image(img_path, save_path)  # Gọi hàm mới ở đây

        if os.path.exists(lbl_src_dir):
            labels = [f for f in os.listdir(lbl_src_dir) if f.endswith('.txt') and f != 'classes.txt']
            for lbl_name in tqdm(labels, desc=f"Copy Labels {split}"):
                src_lbl = os.path.join(lbl_src_dir, lbl_name)
                dst_lbl = os.path.join(lbl_dst_dir, lbl_name)
                shutil.copy2(src_lbl, dst_lbl)


if __name__ == "__main__":
    INPUT_DATASET = "./ic15_raw-20260305T085252Z-1-001/ic15_raw"
    OUTPUT_DATASET = "./P1_Grayscale_HE"  # Đổi tên thư mục xuất ra cho chuẩn
    print(" BẮT ĐẦU CHẠY XỬ LÝ DỮ LIỆU P1 MỚI...")
    process_dataset(INPUT_DATASET, OUTPUT_DATASET)
    print("\n HOÀN THÀNH!")