# -*- coding: utf-8 -*-
"""
v5b (epoch2 checkpoint) 验收测试：4 组文生图 + 1 组图生图
固定 seed 保证可复现，输出到 outputs/acceptance_v5b/
"""
import os
import time
import torch
from PIL import Image

from multimodal_engine.image_gen.pipeline import TextToImage, ImageToImage

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "acceptance_v5b")
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
STEPS = 30
CFG = 7.5
LORA_SCALE = 0.8   # 风格旋钮开到 0.8

NEG = "lowres, bad anatomy, bad hands, watermark, worst quality, low quality"

PROMPTS = [
    ("g1_cyberpunk", "yoneya style, 1girl, cyberpunk city, neon lights, detailed eyes, masterpiece, best quality"),
    ("g2_flower",    "yoneya style, 1girl, white dress, standing in flower field, masterpiece, best quality"),
    ("g3_classroom", "yoneya style, 1girl, school uniform, classroom, sunset, looking at viewer, masterpiece, best quality"),
    ("g4_rain_gray", "yoneya style, 1girl, rainy street at night, umbrella, monochrome atmosphere, masterpiece, best quality"),
]

t0 = time.time()

# ---------- 文生图 4 组 ----------
print("=" * 60)
print("[验收] 初始化文生图引擎 (加载 AnythingV5 + yoneya_v5b-000002)...")
t2i = TextToImage()
print(f"[验收] 模型加载完成，耗时 {time.time()-t0:.1f}s")

for name, prompt in PROMPTS:
    t1 = time.time()
    img = t2i.generate(
        prompt, steps=STEPS, cfg_scale=CFG, negative_prompt=NEG,
        seed=SEED, lora_scale=LORA_SCALE,
    )
    path = os.path.join(OUT_DIR, f"{name}.png")
    img.save(path)
    print(f"[验收] {name} 完成 ({time.time()-t1:.1f}s) -> {path}")

# ---------- 图生图 1 组 ----------
print("=" * 60)
print("[验收] 初始化图生图引擎...")
t2 = time.time()
i2i = ImageToImage()
print(f"[验收] 图生图模型加载完成，耗时 {time.time()-t2:.1f}s")

src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "t2i_1788280427.png")
init_img = Image.open(src).convert("RGB").resize((512, 512))
t2 = time.time()
img = i2i.generate(
    init_img,
    "yoneya style, 1girl, masterpiece, best quality",
    strength=0.6, steps=STEPS, cfg_scale=CFG,
    negative_prompt=NEG, seed=SEED, lora_scale=LORA_SCALE,
)
path = os.path.join(OUT_DIR, "g5_img2img.png")
img.save(path)
print(f"[验收] g5_img2img 完成 ({time.time()-t2:.1f}s) -> {path}")

print("=" * 60)
print(f"[验收] 全部完成！总耗时 {time.time()-t0:.1f}s，输出目录: {OUT_DIR}")
print(f"[验收] LoRA adapter: {t2i.lora_name}, device: {t2i.device}, cuda: {torch.cuda.is_available()}")
