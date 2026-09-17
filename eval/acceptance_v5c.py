# -*- coding: utf-8 -*-
"""
v5c (epoch3 checkpoint) 验收测试：4 组文生图 + 1 组图生图
与 acceptance_v5b.py 完全同口径（同 prompt / seed=42 / steps=30 / cfg=7.5 / lora_scale=0.8）
仅换权重为 v5c e3（yoneya_v5c-000003.safetensors），输出 outputs/acceptance_v5c/
跑完可与 outputs/acceptance_v5b/ 同名图逐张并排对比。
"""
import os
import time
import copy
import yaml
import torch
from PIL import Image

from multimodal_engine.image_gen.pipeline import TextToImage, ImageToImage

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_T2I = os.path.join(BASE_DIR, "src", "multimodal_engine", "configs", "sd_config.yaml")
CONFIG_I2I = os.path.join(BASE_DIR, "src", "multimodal_engine", "configs", "img2img_config.yaml")
OUT_DIR = os.path.join(BASE_DIR, "outputs", "acceptance_v5c")
os.makedirs(OUT_DIR, exist_ok=True)

# v5c e3 权重（相对 lora.dir=D:/lora/output 的子路径）
V5C_WEIGHT = "output/yoneya_v5c-000003.safetensors"


def make_temp_config(src_path, tag):
    """读原 config，把 weight_name 换成 v5c e3，写成临时文件（不污染引擎正式配置）"""
    with open(src_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if "lora" in cfg:
        cfg["lora"]["weight_name"] = V5C_WEIGHT
    else:
        raise RuntimeError(f"{src_path} 没有 lora 段？")
    tmp_path = os.path.join(BASE_DIR, "outputs", f"_tmp_{tag}_v5c_e3.yaml")
    with open(tmp_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)
    print(f"[验收] 临时 config -> {tmp_path}  (weight_name={V5C_WEIGHT})")
    return tmp_path


SEED = 42
STEPS = 30
CFG = 7.5
LORA_SCALE = 0.8

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
print("[验收] 初始化文生图引擎 (AnythingV5 + v5c e3)...")
t2i_cfg = make_temp_config(CONFIG_T2I, "sd")
t2i = TextToImage(config_path=t2i_cfg)
print(f"[验收] 文生图模型加载完成，耗时 {time.time()-t0:.1f}s, adapter={t2i.lora_name}")

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
i2i_cfg = make_temp_config(CONFIG_I2I, "i2i")
i2i = ImageToImage(config_path=i2i_cfg)
print(f"[验收] 图生图模型加载完成，耗时 {time.time()-t0:.1f}s, adapter={i2i.lora_name}")

src = os.path.join(BASE_DIR, "outputs", "t2i_1788280427.png")
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
print(f"[验收] device: {t2i.device}, cuda: {torch.cuda.is_available()}")
