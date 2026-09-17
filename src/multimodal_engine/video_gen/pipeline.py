import logging
import os
import yaml
import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video
from PIL import Image

logger = logging.getLogger(__name__)

# 工具函数

def _read_config(config_path):
    """读取 YAML 配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"错误：找不到配置文件 {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"错误：YAML 配置文件解析失败 - {e}")
        raise

class ImageToVideo:
    def __init__(self, config_path=None):
        if config_path is None:
            module_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(module_dir, "..", "configs", "video_config.yaml")
        
        config=_read_config(config_path)
        
        model_cfg = config.get('model', {})
        self.model_id = model_cfg.get('model_id', "stabilityai/stable-video-diffusion-img2vid")
        self.device = model_cfg.get('device', "cpu")
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        infer_cfg = config.get('inference', {})
        self.default_frames       = infer_cfg.get('frames', 14)
        self.default_fps          = infer_cfg.get('fps', 7)
        self.default_chunk_size   = infer_cfg.get('decode_chunk_size', 8)
        self.default_motion       = infer_cfg.get('motion_bucket_id', 127)
        self.default_noise_aug    = infer_cfg.get('noise_aug_strength', 0.1)
        self.default_seed         = infer_cfg.get('seed', 42)
        
        logger.info(f"[视频生成] 加载模型中...")
        try:
            self.pipe = StableVideoDiffusionPipeline.from_pretrained(
                self.model_id, torch_dtype=torch.float16, variant="fp16"
            )
            self.pipe.enable_model_cpu_offload()
            self.pipe.enable_attention_slicing()

        except Exception as e:
            logger.error(f"错误：模型加载失败 - {e}")
            raise
    
    def generate(self, image, frames=None, fps=None, seed=None, output_path=None):
        frames = frames if frames is not None else self.default_frames
        fps = fps if fps is not None else self.default_fps
        seed = seed if seed is not None else self.default_seed
        
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        logger.info(f"生成视频中... ({frames}帧, {fps}fps)")
        try:
            frames_output = self.pipe(
                image,
                num_frames=frames,
                decode_chunk_size=self.default_chunk_size,
                motion_bucket_id=self.default_motion,
                noise_aug_strength=self.default_noise_aug,
                generator=generator
            ).frames[0]
        except Exception as e:
            logger.error(f"错误：视频生成失败 - {e}")
            raise
        
        if output_path:
            export_to_video(frames_output, output_path, fps=fps)
        
        return frames_output

_generator = None

def generate_video(image, **kwargs):
    global _generator
    if _generator is None:
        _generator = ImageToVideo()
    return _generator.generate(image, **kwargs)
