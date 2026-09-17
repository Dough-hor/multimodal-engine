# -*- coding: utf-8 -*-
"""
金标准对照：Chanter「米山舞 2D Style LoRA」（SD1.5 / 底模 Anything-5RE，与引擎同底模家族）
同引擎、同 seed、同参数出图，与自训 LoRA 逐张对比。
作者信息：触发词 1girl，权重建议 1.0（0.5~1.4），训练数据 841 张原画。
r1: 与自训 scale_probe 完全同款 prompt（逐字一致，完美可比）
r2: 内容组 g1 同款场景（cyberpunk），看金标准在内容词下的表现
"""
import os
import time
import yaml
import torch

from multimodal_engine.image_gen.pipeline import TextToImage

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_T2I = os.path.join(BASE_DIR, "src", "multimodal_engine", "configs", "sd_config.yaml")
LORA_DIR = "D:/lora/output"
OUT_DIR = os.path.join(BASE_DIR, "outputs", "golden_ref")
os.makedirs(OUT_DIR, exist_ok=True)

GOLDEN = "reference_lora/style_YoneyamaMai_unet_only.safetensors"
SEED, STEPS, CFG = 42, 30, 7.5
SCALES = [0.7, 1.0]
NEG = "lowres, bad anatomy, bad hands, watermark, worst quality, low quality"

PROMPTS = [
    ("r1_same", "yoneya style, solo, looking at viewer, masterpiece, best quality"),
    ("r2_cyber", "1girl, cyberpunk city, neon lights, detailed eyes, masterpiece, best quality"),
]

# 临时 config：加载金标准权重
with open(CONFIG_T2I, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
cfg["lora"]["weight_name"] = GOLDEN
tmp = os.path.join(BASE_DIR, "outputs", "_tmp_golden_ref.yaml")
with open(tmp, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, allow_unicode=True)

t0 = time.time()
print("=" * 60)
print("[金标准] 初始化文生图引擎...")
t2i = TextToImage(config_path=tmp)
print(f"[金标准] 模型加载完成 {time.time()-t0:.1f}s")

for name, prompt in PROMPTS:
    for sc in SCALES:
        t1 = time.time()
        try:
            img = t2i.generate(
                prompt, steps=STEPS, cfg_scale=CFG, negative_prompt=NEG,
                seed=SEED, lora_scale=sc,
            )
            path = os.path.join(OUT_DIR, f"{name}_scale_{sc:.1f}.png")
            img.save(path)
            print(f"[金标准] {name} scale={sc} ({time.time()-t1:.1f}s) -> {path}")
        except Exception as e:
            print(f"[金标准] {name} scale={sc} 失败: {e}")

print("=" * 60)
print(f"[金标准] 完成！总耗时 {time.time()-t0:.1f}s")
