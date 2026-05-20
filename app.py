import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import gradio as gr
from multimodal_engine.image_gen import generate

def gen_image(prompt, negative_prompt, steps, cfg_scale, seed):
    # 把 Gradio 传过来的参数转成 generate() 需要的格式
    seed = int(seed) if seed else None
    image = generate(
        prompt=prompt,
        negative_prompt=negative_prompt,
        steps=steps,
        cfg_scale=cfg_scale,
        seed=seed
    )
    return image

# 定义界面
with gr.Blocks(title="AI 文生图引擎") as demo:
    gr.Markdown("# AI 多模态创作引擎 - 文生图")
    
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="正向提示词", placeholder="描述你想要的画面...")
            negative = gr.Textbox(label="反向提示词", value="blurry, ugly, lowres")
            steps = gr.Slider(1, 100, value=50, label="推理步数")
            cfg = gr.Slider(1.0, 20.0, value=7.5, label="引导强度")
            seed = gr.Number(value=42, label="随机种子", precision=0)
            btn = gr.Button("生成图片", variant="primary")
        with gr.Column():
            output = gr.Image(label="生成结果", type="pil")
    
    btn.click(fn=gen_image, inputs=[prompt, negative, steps, cfg, seed], outputs=output)

demo.launch(share=False)
