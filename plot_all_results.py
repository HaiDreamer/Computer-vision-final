import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =====================================================
# CONFIG
# =====================================================
BASE_DIR = Path(r"C:\Users\Daonq\OneDrive\Documents\USTH\computer vision\Final")
OUTPUT_DIR = BASE_DIR / "outputs"

SUMMARY_EVAL_CSV = OUTPUT_DIR / "summary_eval_all_splits.csv"
RUNTIME_CSV = OUTPUT_DIR / "summary.csv"

PLOT_DIR = OUTPUT_DIR / "plots_final"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

VARIANT_ORDER = ["P0", "P1", "P2", "P3", "P4"]

# =====================================================
# LOAD DATA
# =====================================================
df_eval = pd.read_csv(SUMMARY_EVAL_CSV)
df_runtime = pd.read_csv(RUNTIME_CSV)

df_eval["variant"] = pd.Categorical(df_eval["variant"], categories=VARIANT_ORDER, ordered=True)
df_eval = df_eval.sort_values(["variant", "split"])

df_runtime["variant"] = pd.Categorical(df_runtime["variant"], categories=VARIANT_ORDER, ordered=True)
df_runtime = df_runtime.sort_values("variant")

df_all = df_eval[df_eval["split"] == "all"].copy()
df_all = df_all.sort_values("variant")

df_split = df_eval[df_eval["split"].isin(["train", "valid", "test"])].copy()
df_split = df_split.sort_values(["split", "variant"])

# =====================================================
# HELPER FUNCTIONS
# =====================================================
def add_value_labels(bars, fmt="{:.3f}", fontsize=9):
    for bar in bars:
        h = bar.get_height()
        if pd.isna(h):
            continue
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            h,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=fontsize
        )

def save_single_bar(df, metric_col, title, ylabel, filename, fmt="{:.3f}"):
    plt.figure(figsize=(8, 5))
    bars = plt.bar(df["variant"].astype(str), df[metric_col])
    add_value_labels(bars, fmt=fmt)
    plt.title(title)
    plt.xlabel("Preprocessing Variant")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=220)
    plt.close()

# =====================================================
# 1. DETECTION F1 (ALL)
# =====================================================
save_single_bar(
    df_all,
    "det_f1",
    "Detection F1-score across preprocessing variants",
    "F1-score",
    "01_detection_f1_all.png"
)

# =====================================================
# 2. RECOGNITION CER + EXACT MATCH (ALL)
# =====================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# CER
plt.sca(axes[0])
bars = plt.bar(df_all["variant"].astype(str), df_all["rec_CER_mean"])
add_value_labels(bars)
plt.title("Recognition CER across preprocessing variants")
plt.xlabel("Preprocessing Variant")
plt.ylabel("CER (lower is better)")

# Exact Match
plt.sca(axes[1])
bars = plt.bar(df_all["variant"].astype(str), df_all["rec_exact_match"])
add_value_labels(bars)
plt.title("Recognition Exact Match across preprocessing variants")
plt.xlabel("Preprocessing Variant")
plt.ylabel("Exact Match")

plt.tight_layout()
plt.savefig(PLOT_DIR / "02_recognition_cer_exactmatch_all.png", dpi=220)
plt.close()

# =====================================================
# 3. END-TO-END STRICT ACCURACY (ALL)
# =====================================================
save_single_bar(
    df_all,
    "e2e_strict_acc",
    "End-to-End Strict Accuracy across preprocessing variants",
    "Strict Accuracy",
    "03_e2e_strict_all.png"
)

# =====================================================
# 4. RUNTIME PER IMAGE
# =====================================================
runtime_col = None
for c in ["time_per_image_mean", "time_total_mean_sec", "time_total_sec_mean"]:
    if c in df_runtime.columns:
        runtime_col = c
        break

if runtime_col is not None:
    save_single_bar(
        df_runtime,
        runtime_col,
        "Runtime per image across preprocessing variants",
        "Seconds / image",
        "04_runtime_per_image.png"
    )
else:
    print("⚠️ Runtime column not found in summary.csv")

# =====================================================
# 5. DETECTION F1 BY SPLIT
# =====================================================
split_order = ["train", "valid", "test"]
x = list(range(len(VARIANT_ORDER)))
width = 0.25

plt.figure(figsize=(10, 5))

for i, split in enumerate(split_order):
    sub = df_split[df_split["split"] == split].copy()
    sub["variant"] = pd.Categorical(sub["variant"], categories=VARIANT_ORDER, ordered=True)
    sub = sub.sort_values("variant")

    vals = []
    for v in VARIANT_ORDER:
        row = sub[sub["variant"] == v]
        vals.append(row["det_f1"].values[0] if len(row) > 0 else 0.0)

    xs = [k + (i - 1) * width for k in x]
    bars = plt.bar(xs, vals, width=width, label=split)

    for bar, val in zip(bars, vals):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=8
        )

plt.xticks(x, VARIANT_ORDER)
plt.title("Detection F1-score by split")
plt.xlabel("Preprocessing Variant")
plt.ylabel("F1-score")
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_DIR / "05_detection_f1_by_split.png", dpi=220)
plt.close()

# =====================================================
# 6. DELTA VS BASELINE P0
# =====================================================
baseline = df_all[df_all["variant"] == "P0"].iloc[0]

delta_df = df_all.copy()
delta_df["delta_det_f1"] = delta_df["det_f1"] - baseline["det_f1"]
delta_df["delta_rec_exact"] = delta_df["rec_exact_match"] - baseline["rec_exact_match"]
delta_df["delta_e2e_strict"] = delta_df["e2e_strict_acc"] - baseline["e2e_strict_acc"]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Delta Detection F1
plt.sca(axes[0])
bars = plt.bar(delta_df["variant"].astype(str), delta_df["delta_det_f1"])
for bar, val in zip(bars, delta_df["delta_det_f1"]):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{val:+.3f}",
             ha="center", va="bottom", fontsize=8)
plt.axhline(0, linestyle="--")
plt.title("Delta Detection F1 vs P0")
plt.xlabel("Variant")
plt.ylabel("Δ F1")

# Delta Recognition Exact Match
plt.sca(axes[1])
bars = plt.bar(delta_df["variant"].astype(str), delta_df["delta_rec_exact"])
for bar, val in zip(bars, delta_df["delta_rec_exact"]):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{val:+.3f}",
             ha="center", va="bottom", fontsize=8)
plt.axhline(0, linestyle="--")
plt.title("Delta Recognition Exact Match vs P0")
plt.xlabel("Variant")
plt.ylabel("Δ Exact Match")

# Delta End-to-End Strict
plt.sca(axes[2])
bars = plt.bar(delta_df["variant"].astype(str), delta_df["delta_e2e_strict"])
for bar, val in zip(bars, delta_df["delta_e2e_strict"]):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{val:+.3f}",
             ha="center", va="bottom", fontsize=8)
plt.axhline(0, linestyle="--")
plt.title("Delta End-to-End Strict Accuracy vs P0")
plt.xlabel("Variant")
plt.ylabel("Δ Strict Acc")

plt.tight_layout()
plt.savefig(PLOT_DIR / "06_delta_vs_p0.png", dpi=220)
plt.close()

print(f"✅ All final plots saved to: {PLOT_DIR}")