# -*- coding: utf-8 -*-
"""
最终诊断（用户反馈：金标准也崩脸 + 场景图脸崩、v5c_e3 稍好）
两组实验拆干净两个疑点：

组A 金标准公平重测（疑点：是不是 anything-v5 底模天花板？）
  - 上轮金标准测试用了错误触发词 "yoneya style"（不是 Chanter LoRA 的触发词，它的触发词就是 1girl）
  - 本轮用干净 prompt "1girl, solo, looking at viewer" + 作者推荐权重 0.6/1.0
  - 若这轮脸不崩且可辨 → 上轮是测试姿势不公平，底模还有戏
  - 若这轮还崩 → anything-v5 天花板坐实，回炉必须换底模（SDXL/Illustrious）

组B 场景脸崩归因（疑点：是 scale 1.5 太高，还是场景里脸太小？SD1.5 小脸必崩是老毛病）
  - b1: scale 1.5 + "upper body" 构图（脸放大）→ 若不崩 = 构图问题，scale 1.5 无罪
  - b2: scale 1.2 + 原场景 prompt（全身构图不变，只降 scale）→ 若不崩 = scale 问题
  - b3: scale 1.5 + 原场景 + 脸部质量词（detailed face + 负面 bad face）→ prompt 能否救
"""
import os
import time
import yaml
import torch

from multimodal_engine.image_gen.pipeline import TextToImage

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_T2I = os.path.join(BASE_DIR, "src", "multimodal_engine", "configs", "sd_config.yaml")
LORA_DIR = "D:/lora/output"
OUT_DIR = os.path.join(BASE_DIR, "outputs", "diagnosis_face")
os.makedirs(OUT_DIR, exist_ok=True)

GOLDEN = "reference_lora/style_YoneyamaMai_unet_only.safetensors"
V5C = "output/yoneya_v5c-000003.safetensors"
SEED, STEPS, CFG = 42, 30, 7.5
NEG_BASE = "lowres, bad anatomy, bad hands, watermark, worst quality, low quality, monochrome, greyscale"
NEG_FACE = NEG_BASE + ", bad face, deformed face, distorted face"

with open(CONFIG_T2I, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
cfg["lora"]["weight_name"] = V5C
tmp = os.path.join(BASE_DIR, "outputs", "_tmp_diag_face.yaml")
with open(tmp, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, allow_unicode=True)

t0 = time.time()
print("[诊断] 初始化引擎...")
t2i = TextToImage(config_path=tmp)
adapter = t2i.lora_name
print(f"[诊断] 加载完成 {time.time()-t0:.1f}s")


def run(tag, weight, prompt, scale, neg):
    try:
        t2i.pipe.delete_adapters(adapter)
    except Exception:
        pass
    t2i.pipe.load_lora_weights(LORA_DIR, weight_name=weight, adapter_name=adapter)
    t1 = time.time()
    img = t2i.generate(prompt, steps=STEPS, cfg_scale=CFG, negative_prompt=neg,
                       seed=SEED, lora_scale=scale)
    path = os.path.join(OUT_DIR, f"{tag}.png")
    img.save(path)
    print(f"[诊断] {tag} (scale={scale}, {time.time()-t1:.1f}s) -> {path}")


# ---------- 组A：金标准公平重测 ----------
FAIR_PROMPT = "1girl, solo, looking at viewer, masterpiece, best quality"
run("A1_golden_s06_portrait", GOLDEN, FAIR_PROMPT, 0.6, NEG_BASE)
run("A2_golden_s10_portrait", GOLDEN, FAIR_PROMPT, 1.0, NEG_BASE)

# ---------- 组B：场景脸崩归因（v5c_e3） ----------
CYBER_FULL = "yoneya style, 1girl, cyberpunk city, neon lights, vibrant colors, detailed eyes, masterpiece, best quality"
CYBER_FACE_TAGS = ("yoneya style, 1girl, upper body, cyberpunk city, neon lights, vibrant colors, "
                   "detailed face, detailed eyes, masterpiece, best quality")

# b1: 1.5 + 上半身构图（脸放大，scale 不变）
run("B1_v5c_s15_upper_body", V5C, CYBER_FACE_TAGS, 1.5, NEG_FACE)
# b2: 1.2 + 原全身场景 prompt（构图不变，只降 scale）
run("B2_v5c_s12_full_scene", V5C, CYBER_FULL, 1.2, NEG_BASE)
# b3: 1.5 + 原全身场景 + 脸部质量词
run("B3_v5c_s15_face_tags", V5C, CYBER_FULL + ", detailed face", 1.5, NEG_FACE)

print(f"\n[诊断] 完成，总耗时 {time.time()-t0:.1f}s")
