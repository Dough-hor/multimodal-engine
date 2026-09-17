# -*- coding: utf-8 -*-
"""
v5c 多 epoch 验收（文生图同口径）：e1 / e2 / e4
（e3 已由 acceptance_v5c.py 产出到 outputs/acceptance_v5c/，此处不再重复）

关键实现：单进程复用同一个 pipe，用"同名 adapter 覆盖"切换权重
（先 delete_adapters 再 load_lora_weights 同名），避免多 adapter 并存撑爆 6GB 显存。
每 epoch 输出 outputs/acceptance_v5c_e{ep}/ 下 g1~g4 四张，与 v5b/e3 同 prompt 同 seed。
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

# epoch -> 权重文件名（相对 lora.dir=D:/lora/output 的子路径）
# e4 是 save_last 产物，文件名不带序号
EPOCH_WEIGHTS = {
    1: "output/yoneya_v5c-000001.safetensors",
    2: "output/yoneya_v5c-000002.safetensors",
    4: "output/yoneya_v5c.safetensors",
}

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


def make_temp_config(src_path, tag):
    """读原 config，把 weight_name 换成 e1（仅用于初始构造），写成临时文件（不污染正式配置）"""
    with open(src_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["lora"]["weight_name"] = EPOCH_WEIGHTS[1]
    tmp_path = os.path.join(BASE_DIR, "outputs", f"_tmp_{tag}_v5c_epochs.yaml")
    with open(tmp_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)
    print(f"[验收] 临时 config -> {tmp_path}")
    return tmp_path


t0 = time.time()
print("=" * 60)
print("[验收] 初始化文生图引擎 (AnythingV5 + v5c)...")
t2i_cfg = make_temp_config(CONFIG_T2I, "sd")
t2i = TextToImage(config_path=t2i_cfg)
adapter = t2i.lora_name
print(f"[验收] 模型加载完成 {time.time()-t0:.1f}s，初始 adapter={adapter}")

for ep, weight in EPOCH_WEIGHTS.items():
    full_path = os.path.join(LORA_DIR, weight)
    if not os.path.exists(full_path):
        print(f"[验收] 权重不存在，跳过 e{ep}: {full_path}")
        continue

    # 同名 adapter 覆盖切换：先删旧的，再载入新权重（防多 adapter 并存爆显存）
    try:
        t2i.pipe.delete_adapters(adapter)
    except Exception as e:
        print(f"[验收] delete_adapters 失败(可忽略): {e}")
    t2i.pipe.load_lora_weights(LORA_DIR, weight_name=weight, adapter_name=adapter)

    out_dir = os.path.join(OUT_ROOT, f"acceptance_v5c_e{ep}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*60}\n[验收] === 切换到 v5c e{ep} ({weight}) ===")

    for name, prompt in PROMPTS:
        t1 = time.time()
        img = t2i.generate(
            prompt, steps=STEPS, cfg_scale=CFG, negative_prompt=NEG,
            seed=SEED, lora_scale=LORA_SCALE,
        )
        path = os.path.join(out_dir, f"{name}.png")
        img.save(path)
        print(f"[验收] e{ep} {name} 完成 ({time.time()-t1:.1f}s) -> {path}")

print("=" * 60)
print(f"[验收] 全部完成！总耗时 {time.time()-t0:.1f}s")
print(f"[验收] device: {t2i.device}, cuda: {torch.cuda.is_available()}")
