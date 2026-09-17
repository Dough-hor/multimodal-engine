import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ["HF_HUB_OFFLINE"] = "1"



import torch
import time
from pathlib import Path
import gradio as gr
from multimodal_engine.image_gen import generate, generate_from_image
from multimodal_engine.style_transfer.gatys import StyleTransfer
from multimodal_engine.video_gen import generate_video
from diffusers.utils import export_to_video
from PIL import Image
import torchvision.transforms as T
from multimodal_engine.style_transfer import AdaINStyleTransfer, cycleGAN
from multimodal_engine.image_gen.controlnet import ControlNetTextToImage, get_controlnet_pipe


OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# 文生图页面

def gen_text2img(prompt, negative_prompt, steps, cfg_scale, seed, lora_style, lora_scale):
    start = time.time()
    seed = int(seed) if seed else None
    scale = float(lora_scale) if lora_style == "米山舞 LoRA" else 0.0  # 选"无"就归零
    image = generate(
        prompt=prompt, negative_prompt=negative_prompt,
        steps=steps, cfg_scale=cfg_scale, seed=seed,
        lora_scale=scale
    )
    elapsed = time.time() - start
    filename = f"t2i_{int(time.time())}.png"
    save_path = OUTPUT_DIR / filename
    image.save(str(save_path))
    return [image], f"生成完成 | 耗时 {elapsed:.1f}s | 已保存 {filename}"


# 图生图页面

def gen_img2img(input_image, prompt, negative_prompt, strength, steps, cfg_scale, seed, lora_style, lora_scale):
    if input_image is None:
        return [], "请先上传一张图片"
    start = time.time()
    seed = int(seed) if seed else None
    scale = float(lora_scale) if lora_style == "米山舞 LoRA" else 0.0  # 选"无"就归零
    image = generate_from_image(
        image=input_image, prompt=prompt, negative_prompt=negative_prompt,
        strength=strength, steps=steps, cfg_scale=cfg_scale, seed=seed,
        lora_scale=scale
    )
    elapsed = time.time() - start
    filename = f"i2i_{int(time.time())}.png"
    save_path = OUTPUT_DIR / filename
    image.save(str(save_path))
    return [image], f"生成完成 | 耗时 {elapsed:.1f}s | 已保存 {filename}"

# controlnet图生图
def gen_controlnet(input_image, prompt, negative_prompt, cn_scale, steps, cfg_scale, seed):
    if input_image is None:
        return [], "请先上传一张图片"
    start = time.time()
    control_image = ControlNetTextToImage.extract_canny(input_image) 
    image = get_controlnet_pipe().generate(
        prompt=prompt,
        control_image=control_image,
        controlnet_conditioning_scale=cn_scale,
        negative_prompt=negative_prompt,
        steps=steps,
        cfg_scale=cfg_scale,
        seed=seed,
    )
    elapsed = time.time() - start
    filename = f"cn_{int(time.time())}.png"
    save_path = OUTPUT_DIR / filename
    image.save(str(save_path))
    return [control_image, image], f"生成完成 | 耗时 {elapsed:.1f}s | 已保存 {filename}"


# 风格迁移页面

# 单例的控制权应该在应用层，不能写在每个文件的init里，因为有些时候需要两个不同参数的实例，但这里app只需要单例所以才这么写
# 风格迁移实例（避免重复加载 VGG）(gatys)
_styler = None

def get_styler():                                         
    global _styler
    if _styler is None:
        _styler = StyleTransfer()
        if torch.cuda.is_available():
            _styler.extractor.to("cuda")   
    return _styler

# adain单例
_adain=None

def get_adain():
    global _adain
    if _adain is None:
        _adain=AdaINStyleTransfer()
        _adain.eval()
    return _adain

# cyclegan单例
_cyclegan=None

def get_cyclegan(checkpoint_path):
    global _cyclegan
    if _cyclegan is None:
        _cyclegan=cycleGAN()
        if checkpoint_path and os.path.exists(checkpoint_path):
            _cyclegan.load_state_dict(torch.load(checkpoint_path,map_location="cpu"))
        _cyclegan.eval()
    return _cyclegan

# gatys风格迁移推理方法
def gen_style_transfer(content_img, style_img, steps):
    if content_img is None or style_img is None:
        return [], "请上传内容图和风格图"
    # PIL → tensor
    preprocess = T.Compose([
    T.Resize(512),        # 短边缩到512
    T.CenterCrop(512),    # 中心裁成正方形，保证内容/风格图同尺寸
    T.ToTensor(),
    ])
    content_tensor = preprocess(content_img).unsqueeze(0)
    style_tensor = preprocess(style_img).unsqueeze(0)
    
    try:
        styler = get_styler()                                           
        result_tensor = styler.transfer(
            content_tensor, style_tensor,
            num_steps=int(steps)                                       
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

# adain风格迁移推理方法
def gen_adain_transfer(content_img, style_img,alpha=0.8):
    if content_img is None or style_img is None:
        return [], "请上传内容图和风格图"
    # PIL → tensor
    to_tensor = T.ToTensor()
    content_tensor = to_tensor(content_img).unsqueeze(0)
    style_tensor = to_tensor(style_img).unsqueeze(0)
    
    try:
        model = get_adain()                                           
        result_tensor = model(content_tensor,style_tensor,alpha=alpha)
    except Exception as e:
        return [], f"生成失败: {str(e)}"


    # tensor → PIL
    to_pil = T.ToPILImage()
    
    # 如果 result_tensor 有 batch 维，则去掉
    if result_tensor.dim() == 4 and result_tensor.size(0) == 1:
        result_tensor = result_tensor.squeeze(0)
    result_pil = to_pil(result_tensor)
    
    filename = f"adain_{int(time.time())}.png"
    save_path = OUTPUT_DIR / filename
    result_pil.save(str(save_path))
    return [result_pil], f"完成 | 已保存 {filename}"

# cyclegan风格迁移推理方法
def gen_cyclegan_transfer(content_img):
    if content_img is None:
        return [], "请上传内容图"
    # PIL → tensor
    to_tensor = T.ToTensor()
    content_tensor = to_tensor(content_img).unsqueeze(0)
    
    try:
        model = get_cyclegan()                                           
        result_tensor = model(content_tensor)
    except Exception as e:
        return [], f"生成失败: {str(e)}"


    # tensor → PIL
    to_pil = T.ToPILImage()
    
    # 如果 result_tensor 有 batch 维，则去掉
    if result_tensor.dim() == 4 and result_tensor.size(0) == 1:
        result_tensor = result_tensor.squeeze(0)
    result_pil = to_pil(result_tensor)
    
    filename = f"ciclegan_{int(time.time())}.png"
    save_path = OUTPUT_DIR / filename
    result_pil.save(str(save_path))
    return [result_pil], f"完成 | 已保存 {filename}"

# 图生视频页面
def gen_img2video(image, frames, fps, seed):
        if image is not None:
            image = image.resize((512, 288))
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
        filename = f"video_{int(time.time())}.mp4"
        save_path = OUTPUT_DIR / filename
        export_to_video(frames_output, str(save_path), fps=int(fps))
        return str(save_path), f"生成完成 | 耗时 {elapsed:.1f}s | 已保存 {filename}"   



# 界面

with gr.Blocks(title="AI 多模态创作引擎") as demo:
    gr.Markdown("# AI 多模态创作引擎")

    with gr.Tab("文生图"):
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="正向提示词", placeholder="例：yoneya style, 1girl, upper body, vibrant colors（上半身构图脸更稳）")
                negative = gr.Textbox(label="反向提示词", value="worst quality, low quality, normal quality, lowres, bad anatomy, bad hands, missing fingers, extra digits, text, watermark, signature, blurry, ugly, monochrome, greyscale, bad face, deformed face, distorted face")
                steps = gr.Slider(1, 100, value=50, label="推理步数")
                cfg = gr.Slider(1.0, 20.0, value=7.5, label="引导强度")
                seed = gr.Number(value=None, label="随机种子（留空=随机；填固定值可复现）", precision=0)
                lora_style_t2i = gr.Dropdown(
                    choices=["米山舞 LoRA", "无（原版底模）"],
                    value="米山舞 LoRA", label="画风 LoRA"
                )
                lora_scale_t2i = gr.Slider(0.0, 1.2, value=1.0, step=0.05, label="LoRA 强度（甜点 1.0~1.2，实测 >1.2 崩）")
                btn_t2i = gr.Button("生成图片", variant="primary")
            with gr.Column():
                output_t2i = gr.Gallery(label="生成结果")
                status_t2i = gr.Textbox(label="状态", interactive=False)
        btn_t2i.click(
            fn=gen_text2img,
            inputs=[prompt, negative, steps, cfg, seed, lora_style_t2i, lora_scale_t2i],
            outputs=[output_t2i, status_t2i]
        )

    with gr.Tab("ControlNet 可控生成"):
        with gr.Row():
            with gr.Column():
                img_cn = gr.Image(label="原图（自动提取线稿）", type="pil")
                prompt_cn = gr.Textbox(label="正向提示词", placeholder="描述你想要的画面...")
                negative_cn = gr.Textbox(label="反向提示词", value="blurry, ugly, lowres")
                cn_scale = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="控制强度")
                steps_cn = gr.Slider(1, 100, value=30, label="推理步数")
                cfg_cn = gr.Slider(1.0, 20.0, value=7.5, label="引导强度")
                seed_cn = gr.Number(value=None, label="随机种子（留空=随机）", precision=0)
                btn_cn = gr.Button("生成图片", variant="primary")
            with gr.Column():
                output_cn = gr.Gallery(label="生成结果（左线稿 右成品）")
                status_cn = gr.Textbox(label="状态", interactive=False)
        btn_cn.click(
            fn=gen_controlnet,
            inputs=[img_cn,prompt_cn,negative_cn,cn_scale,steps_cn,cfg_cn,seed_cn],          
            outputs=[output_cn, status_cn]
        )

    with gr.Tab("图生图"):
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="上传图片", type="pil")
                prompt_i2i = gr.Textbox(label="正向提示词", placeholder="描述你想要的修改...")
                negative_i2i = gr.Textbox(label="反向提示词", value="worst quality, low quality, normal quality, lowres, bad anatomy, bad hands, missing fingers, extra digits, text, watermark, signature, blurry, ugly, monochrome, greyscale, bad face, deformed face, distorted face")
                strength_i2i = gr.Slider(0.1, 1.0, value=0.8, label="变化强度 (越大变化越多)")
                steps_i2i = gr.Slider(1, 100, value=50, label="推理步数")
                cfg_i2i = gr.Slider(1.0, 20.0, value=7.5, label="引导强度")
                seed_i2i = gr.Number(value=None, label="随机种子（留空=随机）", precision=0)
                lora_style_i2i = gr.Dropdown(
                    choices=["米山舞 LoRA", "无（原版底模）"],
                    value="米山舞 LoRA", label="画风 LoRA"
                )
                lora_scale_i2i = gr.Slider(0.0, 1.2, value=1.0, step=0.05, label="LoRA 强度（甜点 1.0~1.2，实测 >1.2 崩）")
                btn_i2i = gr.Button("生成图片", variant="primary")
            with gr.Column():
                output_i2i = gr.Gallery(label="生成结果")
                status_i2i = gr.Textbox(label="状态", interactive=False)
        btn_i2i.click(
            fn=gen_img2img,
            inputs=[input_image, prompt_i2i, negative_i2i, strength_i2i, steps_i2i, cfg_i2i, seed_i2i, lora_style_i2i, lora_scale_i2i],
            outputs=[output_i2i, status_i2i]
        )

    with gr.Tab("Gatys 风格迁移"):
        with gr.Row():
            with gr.Column():
                img_content_gatys = gr.Image(label="内容图", type="pil")
                img_style_gatys = gr.Image(label="风格图", type="pil")
                steps_gatys = gr.Slider(50, 500, value=300, label="优化步数")
                btn_gatys = gr.Button("开始风格迁移", variant="primary")
            with gr.Column():
                output_gatys = gr.Gallery(label="生成结果")
                status_gatys = gr.Textbox(label="状态", interactive=False)
        btn_gatys.click(
            fn=gen_style_transfer,
            inputs=[img_content_gatys, img_style_gatys, steps_gatys],
            outputs=[output_gatys, status_gatys]
        )
        
    with gr.Tab("AdaIN 风格迁移"):
        with gr.Row():
            with gr.Column():
                img_content_adain = gr.Image(label="内容图", type="pil")
                img_style_adain = gr.Image(label="风格图", type="pil")
                alpha_adain = gr.Slider(0, 1, value=0.8, label="风格强度 (alpha)")
                btn_adain = gr.Button("开始风格迁移", variant="primary")
            with gr.Column():
                output_adain = gr.Gallery(label="生成结果")
                status_adain = gr.Textbox(label="状态", interactive=False)
        btn_adain.click(
            fn=gen_adain_transfer,
            inputs=[img_content_adain, img_style_adain, alpha_adain],
            outputs=[output_adain, status_adain]
        )

    with gr.Tab("CycleGAN 风格迁移"):
        with gr.Row():
            with gr.Column():
                img_content_cycle = gr.Image(label="内容图", type="pil")
                btn_cycle = gr.Button("开始风格迁移", variant="primary")
            with gr.Column():
                output_cycle = gr.Gallery(label="生成结果")
                status_cycle = gr.Textbox(label="状态", interactive=False)
        btn_cycle.click(
            fn=gen_cyclegan_transfer,
            inputs=[img_content_cycle],
            outputs=[output_cycle, status_cycle]
        )

    with gr.Tab("图生视频"):
        with gr.Row():
            with gr.Column():
                img_video = gr.Image(label="输入图片", type="pil")
                frames_video = gr.Slider(8, 25, value=14, step=1, label="帧数")
                fps_video = gr.Slider(4, 30, value=7, step=1, label="帧率 (fps)")
                seed_video = gr.Number(value=None, label="随机种子（留空=随机）", precision=0)
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


