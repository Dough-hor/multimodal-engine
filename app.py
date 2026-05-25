import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import time
from pathlib import Path
import gradio as gr
from multimodal_engine.image_gen import generate, generate_from_image
from multimodal_engine.style_transfer.transfer import StyleTransfer
from multimodal_engine.video_gen import generate_video
from diffusers.utils import export_to_video
from PIL import Image
import torchvision.transforms as T

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# 文生图页面

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


# 图生图页面

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

# 风格迁移页面

# 风格迁移实例（避免重复加载 VGG）
_styler = None

def get_styler():                                         
    global _styler
    if _styler is None:
        _styler = StyleTransfer()   
    return _styler

def gen_style_transfer(content_img, style_img, steps):
    if content_img is None or style_img is None:
        return [], "请上传内容图和风格图"
    # PIL → tensor
    to_tensor = T.ToTensor()
    content_tensor = to_tensor(content_img).unsqueeze(0)
    style_tensor = to_tensor(style_img).unsqueeze(0)
    
    try:
        styler = get_styler()                                           # <-- 修改：使用单例
        result_tensor = styler.transfer(
            content_tensor, style_tensor,
            num_steps=int(steps)                                        # <-- 修改：确保 int
        )
    except Exception as e:
        return [], f"生成失败: {str(e)}"


    # tensor → PIL
    to_pil = T.ToPILImage()
    
    # 如果 result_tensor 有 batch 维，则去掉
    if result_tensor.dim() == 4 and result_tensor.size(0) == 1:
        result_tensor = result_tensor.squeeze(0)
    result_pil = to_pil(result_tensor)
    
    filename = f"style_{int(time.time())}.png"
    save_path = OUTPUT_DIR / filename
    result_pil.save(str(save_path))
    return [result_pil], f"完成 | 已保存 {filename}"

# 图生视频页面
def gen_img2video(image, frames, fps, seed):
        if image is None:
            return None
        start=time.time()
        seed=int(seed) if seed else None
        try:
            frames_output=generate_video(
                image=image,
                frames=int(frames),
                fps=int(fps),
                seed=seed
            )
        except Exception as e:
            return None, f"视频生成失败: {str(e)}"
        elapsed = time.time() - start
        filename = f"video_{int(time.time())}.png"
        save_path = OUTPUT_DIR / filename
        export_to_video(frames_output, str(save_path), fps=int(fps))
        return str(save_path), f"生成完成 | 耗时 {elapsed:.1f}s | 已保存 {filename}"   

# 界面

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

    with gr.Tab("风格迁移"):
        with gr.Row():
            with gr.Column():
                img_content = gr.Image(label="内容图", type="pil")
                img_style = gr.Image(label="风格图", type="pil")
                steps_st = gr.Slider(50, 500, value=300, label="优化步数")
                btn_st = gr.Button("开始风格迁移", variant="primary")
            with gr.Column():
                output_st = gr.Gallery(label="生成结果")
                status_st = gr.Textbox(label="状态", interactive=False)
        btn_st.click(
            fn=gen_style_transfer,
            inputs=[img_content, img_style, steps_st],
            outputs=[output_st, status_st]
        )

    with gr.Tab("图生视频"):
        with gr.Row():
            with gr.Column():
                img_video = gr.Image(label="输入图片", type="pil")
                frames_video = gr.Slider(8, 30, value=14, step=1, label="帧数")
                fps_video = gr.Slider(4, 30, value=7, step=1, label="帧率 (fps)")
                seed_video = gr.Number(value=42, label="随机种子", precision=0)
                btn_video = gr.Button("生成视频", variant="primary")
            with gr.Column():
                output_video = gr.Video(label="生成的视频")
                status_video = gr.Textbox(label="状态", interactive=False)
        btn_video.click(
            fn=gen_img2video,
            inputs=[img_video, frames_video, fps_video, seed_video],
            outputs=[output_video, status_video]
        )

demo.launch(share=False)


