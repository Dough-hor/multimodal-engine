# -*- coding: utf-8 -*-
"""
scale 精细扫描：用户肉眼判定 scale=1.5 "有点像了" -> 围绕 1.5 精探甜点区间。

设计：
  1) 两版权重 × scale 1.2 / 1.5 / 1.8 —— 找画风跳变的精确档位，1.5 复现锚点
  2) 1.5 + 补饱和度 prompt 变体 —— 量化显示 1.5 档 sat 掉到 0.27（发灰），
     尝试用 "vibrant colors" 把色彩拉回，验证"风格与色彩能否兼得"

判读（用户肉眼）：
  1.2 就像        -> 甜点比 1.5 更低，省强度
  1.5 最像        -> 甜点确认
  1.8 更像        -> 风格区间更靠上，甚至可探 2.0
  彩色版比素版好  -> 发灰问题可被 prompt 修正，操作空间变大
"""
import os
import time
import yaml
import torch

from multimodal_engine.image_gen.pipeline import TextToImage

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_T2I = os.path.join(BASE_DIR, "src", "multimodal_engine", "configs", "sd_config.yaml")
LORA_DIR = "D:/lora/output"
OUT_DIR = os.path.join(BASE_DIR, "outputs", "scale_fine")
os.makedirs(OUT_DIR, exist_ok=True)

WEIGHTS = {
    "v5b_e2": "yoneya_v5b-000002.safetensors",
    "v5c_e3": "output/yoneya_v5c-000003.safetensors",
}
SEED, STEPS, CFG = 42, 30, 7.5
NEG = "lowres, bad anatomy, bad hands, watermark, worst quality, low quality, monochrome, greyscale"
# 与 scale_probe 同款基础 prompt（保证与 1.5 锚点图完全可比）
PROMPT_BASE = "yoneya style, solo, looking at viewer, masterpiece, best quality"
# 补饱和度变体：追加强色彩词 + 负面词里压灰度（见 NEG）
PROMPT_VIVID = "yoneya style, solo, looking at viewer, vibrant colors, vivid, masterpiece, best quality"

# 任务清单：(权重tag, scale, 是否彩色版)
TASKS = []
for tag in WEIGHTS:
    for sc in [1.2, 1.5, 1.8]:
        TASKS.append((tag, sc, False))
    TASKS.append((tag, 1.5, True))  # 彩色修正版锚定 1.5

with open(CONFIG_T2I, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
cfg["lora"]["weight_name"] = WEIGHTS["v5b_e2"]
tmp = os.path.join(BASE_DIR, "outputs", "_tmp_scale_fine.yaml")
with open(tmp, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, allow_unicode=True)

t0 = time.time()
print("=" * 60)
print("[精细扫描] 初始化文生图引擎...")
t2i = TextToImage(config_path=tmp)
adapter = t2i.lora_name
print(f"[精细扫描] 模型加载完成 {time.time()-t0:.1f}s")

for tag, weight in WEIGHTS.items():
    full = os.path.join(LORA_DIR, weight)
    if not os.path.exists(full):
        print(f"[精细扫描] 权重不存在，跳过 {tag}: {full}")
        continue
    try:
        t2i.pipe.delete_adapters(adapter)
    except Exception as e:
        print(f"[精细扫描] delete_adapters 失败(可忽略): {e}")
    t2i.pipe.load_lora_weights(LORA_DIR, weight_name=weight, adapter_name=adapter)
    print(f"\n{'='*60}\n[精细扫描] === {tag} ({weight}) ===")

    for task_tag, sc, vivid in [t for t in TASKS if t[0] == tag]:
        prompt = PROMPT_VIVID if vivid else PROMPT_BASE
        t1 = time.time()
        img = t2i.generate(
            prompt, steps=STEPS, cfg_scale=CFG, negative_prompt=NEG,
            seed=SEED, lora_scale=sc,
        )
        suffix = "_vivid" if vivid else ""
        path = os.path.join(OUT_DIR, f"{task_tag}_scale_{sc:.1f}{suffix}.png")
        img.save(path)
        print(f"[精细扫描] {task_tag} scale={sc}{suffix} ({time.time()-t1:.1f}s) -> {path}")

print("=" * 60)
print(f"[精细扫描] 全部完成！总耗时 {time.time()-t0:.1f}s")
print(f"[精细扫描] device: {t2i.device}, cuda: {torch.cuda.is_available()}")
