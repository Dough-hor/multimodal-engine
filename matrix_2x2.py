"""ControlNet + LoRA 2x2 对比矩阵（作品集 02 板块封面素材）
A: 裸底模 | B: +LoRA | C: +ControlNet | D: 双开（王牌）
同 prompt 同 seed，唯一变量是 LoRA 开关和 ControlNet 开关。
"""
import torch
from PIL import Image, ImageDraw, ImageFont
from multimodal_engine.image_gen.pipeline import TextToImage
from multimodal_engine.image_gen.controlnet import get_controlnet_pipe

# —— 参数 ——
PROMPT = "yoneya style, 1girl, upper body, vibrant colors, masterpiece, best quality"
NEG = ("worst quality, low quality, lowres, bad anatomy, bad hands, text, "
       "watermark, blurry, monochrome, greyscale, bad face, deformed face")
SEED = 123   # 2026-09-15 重跑：seed 42 是 v5c 坏种子（ComfyUI 复测证实），换 123
W, H = 512, 768

# ========== 第一步：文生图管线出 A/B ==========
t2i = TextToImage("src/multimodal_engine/configs/sd_config.yaml")
A = t2i.generate(prompt=PROMPT, negative_prompt=NEG, seed=SEED, width=W, height=H,
                 lora_scale=0.0)   # A: 裸底模
B = t2i.generate(prompt=PROMPT, negative_prompt=NEG, seed=SEED, width=W, height=H,
                 lora_scale=1.2)   # B: +LoRA
print("A/B done")

# ========== 第二步：释放文生图管线，再加载 ControlNet 管线 ==========
# 两条管线同时驻留显存会 OOM，所以先出 A/B 存盘、卸载，再加载 ControlNet
del t2i
torch.cuda.empty_cache()

cn = get_controlnet_pipe()
control = cn.extract_canny(B)          # 线稿源用 B（有风格的构图更贴近成片场景）
control.save("outputs/matrix_canny_control.png")

C = cn.generate(prompt=PROMPT, control_image=control, negative_prompt=NEG, seed=SEED,
                lora_scale=0.0)        # C: +ControlNet
D = cn.generate(prompt=PROMPT, control_image=control, negative_prompt=NEG, seed=SEED,
                lora_scale=1.0)        # D: 双开（王牌）
print("C/D done")

# ========== 第三步：拼 2x2 田字格 + 标签 ==========
def label(img, text):
    """图顶部压一条黑底白字标签，返回同尺寸图"""
    img = img.copy()
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 44], fill=(0, 0, 0))
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    d.text((14, 10), text, fill=(255, 255, 255), font=font)
    return img

canvas = Image.new("RGB", (W * 2, H * 2), (30, 30, 30))
canvas.paste(label(A, "A: base"), (0, 0))
canvas.paste(label(B, "B: +LoRA (scale 1.2)"), (W, 0))
canvas.paste(label(C, "C: +ControlNet (canny)"), (0, H))
canvas.paste(label(D, "D: ControlNet + LoRA"), (W, H))
canvas.save("outputs/controlnet_2x2_matrix.png")
print("saved outputs/controlnet_2x2_matrix.png")
