import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import time
from pathlib import Path
import gradio as gr
from multimodal_engine.image_gen import generate, generate_from_image

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# ==================== 文生图页面 ====================

def gen_text2img(prompt, negative_prompt, steps, cfg_scale, seed):
    start = time.time()
    seed = int(seed) if seed else None
    image = generate(
        prompt=prompt, negative_prompt=negative_prompt,
        steps=steps, cfg_scale=cfg_scale, seed=seed
    )
    elapsed = time.time() - start
    filename = f"t2i_{int(time.time())}.png"
    save_path = OUTPUT_DIR / filename
    image.save(str(save_path))
    return [image], f"生成完成 | 耗时 {elapsed:.1f}s | 已保存 {filename}"


# ==================== 图生图页面 ====================

def gen_img2img(input_image, prompt, negative_prompt, strength, steps, cfg_scale, seed):
    if input_image is None:
        return [], "请先上传一张图片"
    start = time.time()
    seed = int(seed) if seed else None
    image = generate_from_image(
        image=input_image, prompt=prompt, negative_prompt=negative_prompt,
        strength=strength, steps=steps, cfg_scale=cfg_scale, seed=seed
    )
    elapsed = time.time() - start
    filename = f"i2i_{int(time.time())}.png"
    save_path = OUTPUT_DIR / filename
    image.save(str(save_path))
    return [image], f"生成完成 | 耗时 {elapsed:.1f}s | 已保存 {filename}"


# ==================== 界面 ====================

with gr.Blocks(title="AI 多模态创作引擎") as demo:
    gr.Markdown("# AI 多模态创作引擎")

    with gr.Tab("文生图"):
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="正向提示词", placeholder="描述你想要的画面...")
                negative = gr.Textbox(label="反向提示词", value="blurry, ugly, lowres")
                steps = gr.Slider(1, 100, value=50, label="推理步数")
                cfg = gr.Slider(1.0, 20.0, value=7.5, label="引导强度")
                seed = gr.Number(value=42, label="随机种子", precision=0)
                btn_t2i = gr.Button("生成图片", variant="primary")
            with gr.Column():
                output_t2i = gr.Gallery(label="生成结果")
                status_t2i = gr.Textbox(label="状态", interactive=False)
        btn_t2i.click(
            fn=gen_text2img,
            inputs=[prompt, negative, steps, cfg, seed],
            outputs=[output_t2i, status_t2i]
        )

    with gr.Tab("图生图"):
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="上传图片", type="pil")
                prompt_i2i = gr.Textbox(label="正向提示词", placeholder="描述你想要的修改...")
                negative_i2i = gr.Textbox(label="反向提示词", value="blurry, ugly, lowres")
                strength_i2i = gr.Slider(0.1, 1.0, value=0.8, label="变化强度 (越大变化越多)")
                steps_i2i = gr.Slider(1, 100, value=50, label="推理步数")
                cfg_i2i = gr.Slider(1.0, 20.0, value=7.5, label="引导强度")
                seed_i2i = gr.Number(value=42, label="随机种子", precision=0)
                btn_i2i = gr.Button("生成图片", variant="primary")
            with gr.Column():
                output_i2i = gr.Gallery(label="生成结果")
                status_i2i = gr.Textbox(label="状态", interactive=False)
        btn_i2i.click(
            fn=gen_img2img,
            inputs=[input_image, prompt_i2i, negative_i2i, strength_i2i, steps_i2i, cfg_i2i, seed_i2i],
            outputs=[output_i2i, status_i2i]
        )

demo.launch(share=False)
