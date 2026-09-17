"""
v5c (768px + dim64) 效果量化对比 (PIL 版)
比较 v5c e1~e4 服务器采样图 vs 训练集彩色参考 (同口径于 check_v5b_quality.py)
基线: v4 e20 = 0.734, v5 e8 = 0.542, v5b_e2 = 0.754 (甜点)
"""
import os, glob, re
from PIL import Image
import numpy as np

REF_DIR = r"D:/lora/twiter_sorted/01_彩图_直接用"
SAMPLES_DIR = r"D:/lora/output/output/sample"

def hue_stats(path):
    """返回 (hue_hist, mean_saturation)"""
    img = np.array(Image.open(path).convert("RGB").resize((512, 512)))
    img = img.astype(np.float32) / 255.0
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    cmax = np.max(img, axis=-1)
    cmin = np.min(img, axis=-1)
    delta = cmax - cmin
    sat = np.where(cmax > 0, delta / (cmax + 1e-9), 0)
    h = np.zeros_like(cmax)
    mask_r = (cmax == r) & (delta > 0)
    mask_g = (cmax == g) & (delta > 0)
    mask_b = (cmax == b) & (delta > 0)
    h[mask_r] = ((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6
    h[mask_g] = (b[mask_g] - r[mask_g]) / delta[mask_g] + 2
    h[mask_b] = (r[mask_b] - g[mask_b]) / delta[mask_b] + 4
    h = np.clip(h * 30, 0, 179)
    mask = sat >= 0.08
    hist, _ = np.histogram(h[mask], bins=32, range=(0, 180), density=True)
    return hist, float(sat.mean())

def cos_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

# 参考集
ref_paths = sorted(glob.glob(os.path.join(REF_DIR, "*.jpg")) +
                   glob.glob(os.path.join(REF_DIR, "*.png")))[:30]
ref_hists = [hue_stats(p)[0] for p in ref_paths]
ref_avg = np.mean(ref_hists, axis=0)

# v5c 采样图
samples = sorted(glob.glob(os.path.join(SAMPLES_DIR, "yoneya_v5c_e*.png")))
results = []
for p in samples:
    name = os.path.basename(p)
    m = re.search(r"e(\d{6})", name)
    ep = int(m.group(1)) if m else -1
    hist, sat = hue_stats(p)
    sim = cos_sim(hist, ref_avg)
    results.append((ep, name, sim, sat))

results.sort()
print("v5c (768px + dim64) 质量报告")
print(f"参考集: {len(ref_hists)} 张米山舞彩图 (twiter_sorted/01_彩图_直接用)")
print("=" * 78)
print(f"{'epoch':<6}{'file':<48}{'hue_sim':>9}{'sat':>7}")
for ep, name, sim, sat in results:
    print(f"e{ep:<5}{name:<48}{sim:.4f}   {sat:.3f}")

print("")
print("=" * 78)
print("分组均值:")
groups = {}
for ep, name, sim, sat in results:
    groups.setdefault(ep, []).append((sim, sat))
for ep in sorted(groups):
    sims = [s for s, _ in groups[ep]]
    sats = [t for _, t in groups[ep]]
    print(f"  e{ep}: n={len(sims)}  hue_sim mean={np.mean(sims):.4f} (min {min(sims):.4f}, max {max(sims):.4f})  sat mean={np.mean(sats):.3f}")

print("")
print("历史基线: v4 e20=0.734 | v5 e8=0.542 | v5b_e2=0.754(甜点)")
