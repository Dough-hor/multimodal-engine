# -*- coding: utf-8 -*-
"""scale 精细扫描量化：hue_sim（vs 米山舞参考集30张）+ 饱和度均值，与 scale_probe 同口径"""
import os
import glob
import numpy as np
from PIL import Image

REF_DIR = "D:/lora/twiter_sorted/01_彩图_直接用"
FINE_DIR = "D:/multimodal-engine/outputs/scale_fine"

def hue_hist(img_arr):
    hsv = np.asarray(Image.fromarray(img_arr.astype(np.uint8)).convert("HSV"), dtype=np.float32)[:, :, 0]
    h, _ = np.histogram(hsv, bins=np.arange(33) * 8)
    n = np.linalg.norm(h)
    return h / n if n > 0 else h

refs = []
for p in sorted(glob.glob(os.path.join(REF_DIR, "*.jpg")) + glob.glob(os.path.join(REF_DIR, "*.png")))[:30]:
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)
    refs.append(hue_hist(a))

def score(p):
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)
    h = hue_hist(a)
    sims = [float(np.dot(h, r)) for r in refs]
    s = np.asarray(Image.fromarray(a.astype(np.uint8)).convert("HSV"), dtype=np.float32)[:, :, 1]
    return float(np.mean(sims)), float(s.mean() / 255.0)

print(f"{'文件':<38} {'hue_sim':>8} {'sat':>6}   参照(1.5素版锚点)")
print("-" * 78)
rows = []
for p in sorted(glob.glob(os.path.join(FINE_DIR, "*.png"))):
    hs, sat = score(p)
    rows.append((os.path.basename(p), hs, sat))
    print(f"{os.path.basename(p):<38} {hs:>8.4f} {sat:>6.3f}")

print("\n判读要点:")
print("  1. _vivid 版 sat 应明显高于同 scale 素版（目标 >=0.40，贴近参考集）")
print("  2. hue_sim 走向: 若 1.2~1.8 单调降, 说明色彩持续偏移, 但画风可能反而更好(肉眼判)")
print("  3. 肉眼风格判定权重大于本表数字（scale_probe 已证明 hue_sim 测不出画风）")
