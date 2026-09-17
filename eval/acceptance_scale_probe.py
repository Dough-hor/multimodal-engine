# -*- coding: utf-8 -*-
"""
scale 渐变诊断：LoRA 到底学到了多少"独立风格信号"？

同一 prompt 同一 seed，只拧 lora_scale 旋钮：0 / 0.5 / 1.0 / 1.5
对照两个权重（v5b_e2 旧甜点 + v5c_e3 新甜点）。

判读标准（用户肉眼）：
  四张几乎一样          -> LoRA 信号极弱（基本没学到东西）
  有渐变但只是颜色变浓   -> 学到的是"配色"，不是画风
  某档位出现明显画风跳变 -> 有风格信号，之前只是强度/口径没调对
"""
import os
import time
import yaml
import torch

from multimodal_engine.image_gen.pipeline import TextToImage

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_T2I = os.path.join(BASE_DIR, "src", "multimodal_engine", "configs", "sd_config.yaml")
LORA_DIR = "D:/lora/output"
OUT_DIR = os.path.join(BASE_DIR, "outputs", "scale_probe")
os.makedirs(OUT_DIR, exist_ok=True)

WEIGHTS = {
    "v5b_e2": "yoneya_v5b-000002.safetensors",
    "v5c_e3": "output/yoneya_v5c-000003.safetensors",
}
SCALES = [0.0, 0.5, 1.0, 1.5]
SEED, STEPS, CFG = 42, 30, 7.5
NEG = "lowres, bad anatomy, bad hands, watermark, worst quality, low quality"
# 与风格压力组 s1 完全同款 prompt，保证可比性
PROMPT = "yoneya style, solo, looking at viewer, masterpiece, best quality"

# 临时 config：不污染正式配置，初始加载 v5b_e2
with open(CONFIG_T2I, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
cfg["lora"]["weight_name"] = WEIGHTS["v5b_e2"]
tmp = os.path.join(BASE_DIR, "outputs", "_tmp_scale_probe.yaml")
with open(tmp, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, allow_unicode=True)

t0 = time.time()
print("=" * 60)
print("[scale探针] 初始化文生图引擎...")
t2i = TextToImage(config_path=tmp)
adapter = t2i.lora_name
print(f"[scale探针] 模型加载完成 {time.time()-t0:.1f}s")

for tag, weight in WEIGHTS.items():
    full = os.path.join(LORA_DIR, weight)
    if not os.path.exists(full):
        print(f"[scale探针] 权重不存在，跳过 {tag}: {full}")
        continue
    try:
        t2i.pipe.delete_adapters(adapter)
    except Exception as e:
        print(f"[scale探针] delete_adapters 失败(可忽略): {e}")
    t2i.pipe.load_lora_weights(LORA_DIR, weight_name=weight, adapter_name=adapter)
    print(f"\n{'='*60}\n[scale探针] === {tag} ({weight}) ===")

    for sc in SCALES:
        t1 = time.time()
        img = t2i.generate(
            PROMPT, steps=STEPS, cfg_scale=CFG, negative_prompt=NEG,
            seed=SEED, lora_scale=sc,
        )
        path = os.path.join(OUT_DIR, f"{tag}_scale_{sc:.1f}.png")
        img.save(path)
        print(f"[scale探针] {tag} scale={sc} ({time.time()-t1:.1f}s) -> {path}")

print("=" * 60)
print(f"[scale探针] 全部完成！总耗时 {time.time()-t0:.1f}s")
print(f"[scale探针] device: {t2i.device}, cuda: {torch.cuda.is_available()}")
