from PIL import Image
from multimodal_engine.image_gen.controlnet import get_controlnet_pipe

# 随便找张图当线稿源（用你 outputs 里任意一张都行）
src = Image.open("outputs/final_1p5/v5c_e3_g1_cyberpunk.png")
control = get_controlnet_pipe().extract_canny(src)
control.save("outputs/cn_smoke_control.png")

img = get_controlnet_pipe().generate(
    prompt="yoneya style, 1girl, upper body, vibrant colors, masterpiece, best quality",
    control_image=control,
    seed=42,
    lora_scale=1.0,
)
img.save("outputs/cn_smoke_lora.png")
print("done")
