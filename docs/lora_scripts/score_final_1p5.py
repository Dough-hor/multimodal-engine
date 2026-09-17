# -*- coding: utf-8 -*-
"""最终验收 final_1p5 量化：hue_sim + sat，vs 参考集30张"""
import os
import glob
import numpy as np
from PIL import Image

REF_DIR = "D:/lora/twiter_sorted/01_彩图_直接用"
FINAL_DIR = "D:/multimodal-engine/outputs/final_1p5"

def hue_hist(img_arr):
    hsv = np.asarray(Image.fromarray(img_arr.astype(np.uint8)).convert("HSV"), dtype=np.float32)[:, :, 0]
    h, _ = np.histogram(hsv, bins=np.arange(33) * 8)
    n = np.linalg.norm(h)
    return h / n if n > 0 else h

refs = []
for p in sorted(glob.glob(os.path.join(REF_DIR, "*.jpg")) + glob.glob(os.path.join(REF_DIR, "*.png")))[:30]:
    refs.append(hue_hist(np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)))

rows = []
for p in sorted(glob.glob(os.path.join(FINAL_DIR, "*.png"))):
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)
    h = hue_hist(a)
    hs = float(np.mean([float(np.dot(h, r)) for r in refs]))
    s = np.asarray(Image.fromarray(a.astype(np.uint8)).convert("HSV"), dtype=np.float32)[:, :, 1]
    rows.append((os.path.basename(p), hs, float(s.mean() / 255.0)))
    print(f"{os.path.basename(p):<32} hue_sim={hs:.4f}  sat={float(s.mean()/255.0):.3f}")

print()
for tag in ["v5b_e2", "v5c_e3"]:
    sub = [r for r in rows if r[0].startswith(tag)]
    if sub:
        print(f"{tag} 均值: hue_sim={np.mean([r[1] for r in sub]):.4f}  sat={np.mean([r[2] for r in sub]):.3f}")
