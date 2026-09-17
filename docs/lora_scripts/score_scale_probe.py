# -*- coding: utf-8 -*-
"""
scale 探针量化：测 LoRA 对画面的实际影响强度
方法：每个权重下，scale 0/0.5/1.0/1.5 各图 vs scale 0.0（无 LoRA 效果）的
      平均绝对像素差（0-255 标度）+ 色相直方图余弦距离
判读：差值极小(比如 <6) = LoRA 信号微弱；差值大 = 影响强（方向对不对看肉眼）
"""
import os
import numpy as np
from PIL import Image

PROBE = "D:/multimodal-engine/outputs/scale_probe"

def load(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)

def hue_hist_cos_dist(a, b):
    # HSV 色相直方图（32 bins）余弦距离 = 1 - cos相似度
    ha = np.asarray(Image.fromarray(a.astype(np.uint8)).convert("HSV"), dtype=np.float32)[:, :, 0]
    hb = np.asarray(Image.fromarray(b.astype(np.uint8)).convert("HSV"), dtype=np.float32)[:, :, 0]
    bins = np.arange(33) * 8
    ga, _ = np.histogram(ha, bins=bins)
    gb, _ = np.histogram(hb, bins=bins)
    ga = ga / (np.linalg.norm(ga) + 1e-9)
    gb = gb / (np.linalg.norm(gb) + 1e-9)
    return 1.0 - float(np.dot(ga, gb))

for tag in ["v5b_e2", "v5c_e3"]:
    base_path = os.path.join(PROBE, f"{tag}_scale_0.0.png")
    if not os.path.exists(base_path):
        print(f"missing {base_path}")
        continue
    base = load(base_path)
    print(f"\n=== {tag} (基准 = scale 0.0 无LoRA图) ===")
    for sc in ["0.0", "0.5", "1.0", "1.5"]:
        p = os.path.join(PROBE, f"{tag}_scale_{sc}.png")
        if not os.path.exists(p):
            continue
        img = load(p)
        mad = float(np.mean(np.abs(img - base)))          # 平均绝对像素差 0-255
        pct = float(np.mean(np.any(img != base, axis=2)) * 100)  # 像素变化比例
        hd = hue_hist_cos_dist(img, base)
        print(f"  scale={sc}: 平均像素差={mad:6.2f}/255  变化像素占比={pct:5.1f}%  色相分布距离={hd:.3f}")
