# scale 扫参测试：同 prompt 同 seed，只变 lora_scale
from multimodal_engine.image_gen.pipeline import TextToImage

PROMPT = "yoneya style, 1girl, cyberpunk city, neon lights, detailed eyes, masterpiece, best quality"
NEG = "lowres, bad anatomy, bad hands, watermark, worst quality, low quality"

t2i = TextToImage()  # 只加载一次模型，出 3 张

for scale in [1.0, 1.2]:
    img = t2i.generate(
        PROMPT, steps=30, cfg_scale=7.5, negative_prompt=NEG,
        seed=42, lora_scale=scale,
    )
    img.save(f"outputs/acceptance_v5b/scale_{scale}.png")
    print(f"scale={scale} 完成")
