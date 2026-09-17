import os
import logging
import numpy as np
import torch
from PIL import Image
import cv2

from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from .pipeline import TextToImage, _read_config, _build_generator, _load_lora, _set_lora_scale  # 复用现有工具函数

logger = logging.getLogger(__name__)

DEFAULT_CONTROLNET_ID = "lllyasviel/control_v11p_sd15_canny"

class ControlNetTextToImage(TextToImage):
    """在文生图基础上加 ControlNet 条件控制"""

    def __init__(self, config_path=None):
        if config_path is None:
            module_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(module_dir, "..", "configs", "controlnet_config.yaml")
        config = _read_config(config_path)
        cn_cfg=config.get('controlnet',{}) #controlnet模型的id
        model_cfg = config.get('model', {}) # 扩散模型的id
        self.model_id=model_cfg.get('model_id',"runwayml/stable-diffusion-v1-5")
        self.device = model_cfg.get('device', None)
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        infer_cfg = config.get('inference', {})
        self.default_lora_scale = config.get('lora', {}).get('scale', 0.0)
        self.default_steps          = infer_cfg.get('steps', 30)
        self.default_cfg_scale      = infer_cfg.get('guidance_scale', 7.5)
        self.default_negative_prompt = infer_cfg.get('negative_prompt', None)
        self.default_controlnet_conditioning_scale           = infer_cfg.get('controlnet_conditioning_scale', 1.0)
        self.default_seed=infer_cfg.get('seed',42)

        logger.info(f"[图像生成] 加载模型中...")
        try:
            controlnet = ControlNetModel.from_pretrained(
            cn_cfg.get('model_id', DEFAULT_CONTROLNET_ID),
            torch_dtype=torch.float16,   
            )
            self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
                self.model_id,
                controlnet=controlnet,
                torch_dtype=torch.float16,
            )
            self.pipe.enable_model_cpu_offload()
            self.pipe.enable_attention_slicing()
            # 注册米山舞 LoRA（复用 pipeline.py 的工具函数）
            self.lora_name = _load_lora(self.pipe, config)
        except Exception as e:
            logger.error(f"错误：模型加载失败 - {e}")
            raise

    @staticmethod
    def extract_canny(image, low=100, high=200):
        """PIL 图 → canny 线稿 PIL 图"""
        gray = np.asarray(image.convert("L"))
        edges = cv2.Canny(gray, low, high)          # 灰度边缘图，值只有 0/255
        edges = np.stack([edges] * 3, axis=-1)     # 单通道→三通道（pipeline 要求）
        return Image.fromarray(edges)

    def generate(self, prompt, control_image, controlnet_conditioning_scale=None, negative_prompt=None,
                 steps=None, cfg_scale=None, seed=None, lora_scale=None):
        """control_image: 线稿图（canny 输出）"""
        controlnet_conditioning_scale          = float(controlnet_conditioning_scale if controlnet_conditioning_scale is not None else self.default_controlnet_conditioning_scale)
        steps          = steps if steps is not None else self.default_steps
        cfg_scale      = cfg_scale if cfg_scale is not None else self.default_cfg_scale
        negative_prompt = negative_prompt if negative_prompt is not None else self.default_negative_prompt
        seed           = seed if seed is not None else self.default_seed
        generator      = _build_generator(self.device, seed)
        lora_scale = lora_scale if lora_scale is not None else self.default_lora_scale
        _set_lora_scale(self.pipe, self.lora_name, lora_scale)   # 0=关，>0 按强度


        logger.info(f"生成中: {prompt}")
        try:
            image = self.pipe(
                image=control_image,prompt=prompt, negative_prompt=negative_prompt,
                num_inference_steps=steps, guidance_scale=cfg_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale, generator=generator
            ).images[0]
        except Exception as e:
            logger.error(f"错误：图片生成失败 - {e}")
            raise
        return image


_cn_pipe = None

def get_controlnet_pipe():
    """全局缓存，防止每次点按钮都重载模型"""
    global _cn_pipe
    if _cn_pipe is None:
        _cn_pipe = ControlNetTextToImage()
    return _cn_pipe
