import cv2
import numpy as np

def lab_clahe_bgr(bgr, clip_limit=2.0, tile_grid=8):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=float(clip_limit),
                            tileGridSize=(int(tile_grid), int(tile_grid)))
    L2 = clahe.apply(L)

    lab2 = cv2.merge((L2, a, b))
    out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    return out

img_path = r"D:\ic15_raw\train\images\img_123.jpg"
bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)

out = lab_clahe_bgr(bgr, clip_limit=2.0, tile_grid=8)

# side-by-side (same height/width required; CLAHE doesn't change geometry)
compare = np.hstack([bgr, out])

out_path = r"D:\ic15_raw\compare_img_123.jpg"
cv2.imwrite(out_path, compare)
print("Wrote:", out_path)