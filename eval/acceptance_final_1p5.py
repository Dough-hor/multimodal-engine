# -*- coding: utf-8 -*-
"""
最终验收：锁定口径 = scale 1.5 + 彩色修正 prompt（正面 vibrant colors，负面 monochrome/greyscale）
跑正式内容场景（与 acceptance_v5b/v5c 的 g1~g4 同场景），v5b_e2 与 v5c_e3 双权重对决。
胜者将写入引擎正式配置。
"""
import os
import time
import yaml
import torch

from multimodal_engine.image_gen.pipeline import TextToImage

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_T2I = os.path.join(BASE_DIR, "src", "multimodal_engine", "configs", "sd_config.yaml")
LORA_DIR = "D:/lora/output"
OUT_DIR = os.path.join(BASE_DIR, "outputs", "final_1p5")
os.makedirs(OUT_DIR, exist_ok=True)

WEIGHTS = {
    "v5b_e2": "yoneya_v5b-000002.safetensors",
    "v5c_e3": "output/yoneya_v5c-000003.safetensors",
}
SEED, STEPS, CFG, SCALE = 42, 30, 7.5, 1.5
NEG = ("lowres, bad anatomy, bad hands, watermark, worst quality, low quality, "
       "monochrome, greyscale")
# 场景 prompt：触发词 + 场景 + 彩色修正 + 质量词（与历轮 g1~g4 同场景，可比）
PROMPTS = [
    ("g1_cyberpunk", "yoneya style, 1girl, cyberpunk city, neon lights, vibrant colors, detailed eyes, masterpiece, best quality"),
    ("g2_flower",    "yoneya style, 1girl, flower field, white dress, vibrant colors, detailed eyes, masterpiece, best quality"),
    ("g3_classroom", "yoneya style, 1girl, classroom, sunset, vibrant colors, detailed eyes, masterpiece, best quality"),
    ("g4_rain",      "yoneya style, 1girl, rain, night city, umbrella, vibrant colors, detailed eyes, masterpiece, best quality"),
]

with open(CONFIG_T2I, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
cfg["lora"]["weight_name"] = WEIGHTS["v5b_e2"]
tmp = os.path.join(BASE_DIR, "outputs", "_tmp_final_1p5.yaml")
with open(tmp, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, allow_unicode=True)

t0 = time.time()
print("[最终验收] 初始化引擎...")
t2i = TextToImage(config_path=tmp)
adapter = t2i.lora_name
print(f"[最终验收] 加载完成 {time.time()-t0:.1f}s")

for tag, weight in WEIGHTS.items():
    full = os.path.join(LORA_DIR, weight)
    if not os.path.exists(full):
        print(f"[最终验收] 权重不存在，跳过 {tag}")
        continue
    try:
        t2i.pipe.delete_adapters(adapter)
    except Exception:
        pass
    t2i.pipe.load_lora_weights(LORA_DIR, weight_name=weight, adapter_name=adapter)
    print(f"\n[最终验收] === {tag} ===")
    for name, prompt in PROMPTS:
        t1 = time.time()
        img = t2i.generate(prompt, steps=STEPS, cfg_scale=CFG, negative_prompt=NEG,
                           seed=SEED, lora_scale=SCALE)
        path = os.path.join(OUT_DIR, f"{tag}_{name}.png")
        img.save(path)
        print(f"[最终验收] {tag} {name} ({time.time()-t1:.1f}s) -> {path}")

print(f"\n[最终验收] 完成，总耗时 {time.time()-t0:.1f}s")
