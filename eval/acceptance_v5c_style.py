# -*- coding: utf-8 -*-
"""
v5c 风格压力测试组：把"风格信号"拉满，肉眼分辨各 epoch 谁最像米山舞
（内容主导口径下风格词会被场景词稀释，epoch 间差异看不出 → 这组换成极简 prompt + scale 满开）

方法：prompt 只留训练分布核心（触发词 yoneya style + solo/looking at viewer 等训练高频词），
      lora_scale 从 0.8 -> 1.0，让 LoRA 风格完全主导画面。
输出 outputs/acceptance_v5c_style_e{ep}/ 下 s1~s2 两张，同 seed=42 同负prompt。
"""
import os
import time
import yaml
import torch

from multimodal_engine.image_gen.pipeline import TextToImage

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_T2I = os.path.join(BASE_DIR, "src", "multimodal_engine", "configs", "sd_config.yaml")
LORA_DIR = "D:/lora/output"
OUT_ROOT = os.path.join(BASE_DIR, "outputs")

# e1~e3 = checkpoint, e4 = save_last 产物（不带序号）
EPOCH_WEIGHTS = {
    1: "output/yoneya_v5c-000001.safetensors",
    2: "output/yoneya_v5c-000002.safetensors",
    3: "output/yoneya_v5c-000003.safetensors",
    4: "output/yoneya_v5c.safetensors",
}

SEED = 42
STEPS = 30
CFG = 7.5
LORA_SCALE = 1.0   # 风格满开，放大 epoch 差异
NEG = "lowres, bad anatomy, bad hands, watermark, worst quality, low quality"

# 风格压力测试 prompt：内容词极少，让触发词主导
PROMPTS = [
    # s1: 训练集最高频词组合（solo 162次 / looking at viewer 121次）——最贴训练分布
    ("s1_solo", "yoneya style, solo, looking at viewer, masterpiece, best quality"),
    # s2: 只加基本人设，不指定场景/服装/配色，让风格自己说话
    ("s2_girl", "yoneya style, 1girl, long hair, portrait, detailed eyes, masterpiece, best quality"),
]


def make_temp_config(src_path, tag):
    with open(src_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["lora"]["weight_name"] = EPOCH_WEIGHTS[1]
    tmp_path = os.path.join(BASE_DIR, "outputs", f"_tmp_{tag}_v5c_style.yaml")
    with open(tmp_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)
    print(f"[风格组] 临时 config -> {tmp_path}")
    return tmp_path


t0 = time.time()
print("=" * 60)
print("[风格组] 初始化文生图引擎...")
t2i_cfg = make_temp_config(CONFIG_T2I, "sd")
t2i = TextToImage(config_path=t2i_cfg)
adapter = t2i.lora_name
print(f"[风格组] 模型加载完成 {time.time()-t0:.1f}s")

for ep, weight in EPOCH_WEIGHTS.items():
    full_path = os.path.join(LORA_DIR, weight)
    if not os.path.exists(full_path):
        print(f"[风格组] 权重不存在，跳过 e{ep}: {full_path}")
        continue
    try:
        t2i.pipe.delete_adapters(adapter)
    except Exception as e:
        print(f"[风格组] delete_adapters 失败(可忽略): {e}")
    t2i.pipe.load_lora_weights(LORA_DIR, weight_name=weight, adapter_name=adapter)

    out_dir = os.path.join(OUT_ROOT, f"acceptance_v5c_style_e{ep}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*60}\n[风格组] === 切换到 v5c e{ep} ({weight}) ===")

    for name, prompt in PROMPTS:
        t1 = time.time()
        img = t2i.generate(
            prompt, steps=STEPS, cfg_scale=CFG, negative_prompt=NEG,
            seed=SEED, lora_scale=LORA_SCALE,
        )
        path = os.path.join(out_dir, f"{name}.png")
        img.save(path)
        print(f"[风格组] e{ep} {name} 完成 ({time.time()-t1:.1f}s) -> {path}")

print("=" * 60)
print(f"[风格组] 全部完成！总耗时 {time.time()-t0:.1f}s")
print(f"[风格组] device: {t2i.device}, cuda: {torch.cuda.is_available()}")
