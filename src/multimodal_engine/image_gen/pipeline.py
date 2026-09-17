import logging
import yaml
import os
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from PIL import Image

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
        logger.error(f"错误：配置文件格式有问题：{e}")
        raise

def _build_generator(device, seed):
    """根据 seed 创建 torch.Generator"""
    if seed is not None:
        return torch.Generator(device=device).manual_seed(seed)
    return None


def _load_lora(pipe, config):
    """启动时给 pipe 注册 LoRA；成功返回 adapter 名，没配置或文件缺失返回 None"""
    lora_cfg = config.get("lora", {})
    lora_dir = lora_cfg.get("dir")
    if not lora_dir:
        return None
    weight_name = lora_cfg.get("weight_name", "")
    adapter_name = lora_cfg.get("adapter_name", "style")
    path = os.path.join(lora_dir, weight_name)
    if not os.path.exists(path):
        logger.warning(f"[LoRA] 文件不存在，跳过: {path}")
        return None
    pipe.load_lora_weights(lora_dir, weight_name=weight_name, adapter_name=adapter_name)
    logger.info(f"[LoRA] 已注册 adapter「{adapter_name}」: {path}")
    return adapter_name


def _set_lora_scale(pipe, adapter_name, scale):
    """生成前拧旋钮：scale=0 关闭，>0 按强度生效"""
    if adapter_name:
        pipe.set_adapters([adapter_name], adapter_weights=[float(scale)])


# 文生图 

class TextToImage:
    def __init__(self, config_path=None):
        if config_path is None:
            module_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(module_dir, "..", "configs", "sd_config.yaml")
        
        config = _read_config(config_path)
        
        model_cfg = config.get('model', {})
        self.model_id = model_cfg.get('model_id', "runwayml/stable-diffusion-v1-5")
        self.device = model_cfg.get('device', None)
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        infer_cfg = config.get('inference', {})
        self.default_steps          = infer_cfg.get('steps', 50)
        self.default_cfg_scale      = infer_cfg.get('guidance_scale', 7.5)
        self.default_negative_prompt = infer_cfg.get('negative_prompt', None)
        self.default_width          = infer_cfg.get('width', 512)
        self.default_height         = infer_cfg.get('height', 512)
        self.default_seed           = infer_cfg.get('seed', None)
        # LoRA 默认强度：不传 lora_scale 时用配置值（未配置则 0.0，行为与旧版一致）
        self.default_lora_scale     = config.get('lora', {}).get('scale', 0.0)

        logger.info(f"[文生图] 加载模型中... (设备: {self.device})")
        try:
            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.model_id, torch_dtype=torch.float32
            )
            self.pipe = self.pipe.to(self.device)
        except Exception as e:
            logger.error(f"错误：模型加载失败 - {e}")
            raise

        # LoRA 画风：启动时注册一次（config 没配 lora 段则返回 None，行为与原来完全一致）
        self.lora_name = _load_lora(self.pipe, config)
    
    def generate(self, prompt, steps=None, cfg_scale=None,
                 negative_prompt=None, width=None, height=None, seed=None,
                 lora_scale=None):
        steps          = steps if steps is not None else self.default_steps
        cfg_scale      = cfg_scale if cfg_scale is not None else self.default_cfg_scale
        negative_prompt = negative_prompt if negative_prompt is not None else self.default_negative_prompt
        width          = width if width is not None else self.default_width
        height         = height if height is not None else self.default_height
        seed           = seed if seed is not None else self.default_seed
        lora_scale     = lora_scale if lora_scale is not None else self.default_lora_scale
        generator      = _build_generator(self.device, seed)
        _set_lora_scale(self.pipe, self.lora_name, lora_scale)  # LoRA 强度旋钮
        
        logger.info(f"生成中: {prompt}")
        try:
            image = self.pipe(
                prompt=prompt, negative_prompt=negative_prompt,
                num_inference_steps=steps, guidance_scale=cfg_scale,
                width=width, height=height, generator=generator
            ).images[0]
        except Exception as e:
            logger.error(f"错误：图片生成失败 - {e}")
            raise
        return image


# 图生图

class ImageToImage:
    def __init__(self, config_path=None):
        if config_path is None:
            module_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(module_dir, "..", "configs", "img2img_config.yaml")
        
        config = _read_config(config_path)
        
        model_cfg = config.get('model', {})
        self.model_id = model_cfg.get('model_id', "runwayml/stable-diffusion-v1-5")
        self.device = model_cfg.get('device', None)
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        infer_cfg = config.get('inference', {})
        self.default_steps           = infer_cfg.get('steps', 50)
        self.default_cfg_scale       = infer_cfg.get('guidance_scale', 7.5)
        self.default_strength        = infer_cfg.get('strength', 0.8)
        self.default_negative_prompt = infer_cfg.get('negative_prompt', None)
        self.default_seed            = infer_cfg.get('seed', None)
        # LoRA 默认强度：不传 lora_scale 时用配置值（未配置则 0.0，行为与旧版一致）
        self.default_lora_scale      = config.get('lora', {}).get('scale', 0.0)

        logger.info(f"[图生图] 加载模型中... (设备: {self.device})")
        try:
            self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                self.model_id, torch_dtype=torch.float32
            )
            self.pipe = self.pipe.to(self.device)
        except Exception as e:
            logger.error(f"错误：模型加载失败 - {e}")
            raise

        # LoRA 画风：启动时注册一次（config 没配 lora 段则返回 None，行为与原来完全一致）
        self.lora_name = _load_lora(self.pipe, config)
    
    def generate(self, image, prompt, strength=None, steps=None,
                 cfg_scale=None, negative_prompt=None, seed=None,
                 lora_scale=None):
        strength        = strength if strength is not None else self.default_strength
        steps           = steps if steps is not None else self.default_steps
        cfg_scale       = cfg_scale if cfg_scale is not None else self.default_cfg_scale
        negative_prompt = negative_prompt if negative_prompt is not None else self.default_negative_prompt
        seed            = seed if seed is not None else self.default_seed
        lora_scale      = lora_scale if lora_scale is not None else self.default_lora_scale
        generator       = _build_generator(self.device, seed)
        _set_lora_scale(self.pipe, self.lora_name, lora_scale)  # LoRA 强度旋钮
        
        logger.info(f"图生图中: {prompt}")
        try:
            result = self.pipe(
                prompt=prompt, image=image,
                negative_prompt=negative_prompt,
                strength=strength, num_inference_steps=steps,
                guidance_scale=cfg_scale, generator=generator
            ).images[0]
        except Exception as e:
            logger.error(f"错误：图生图失败 - {e}")
            raise
        return result


# 便捷函数

_t2i = None
def generate(prompt, **kwargs):
    global _t2i
    if _t2i is None:
        _t2i = TextToImage()
    return _t2i.generate(prompt, **kwargs)

_i2i = None
def generate_from_image(image, prompt, **kwargs):
    global _i2i
    if _i2i is None:
        _i2i = ImageToImage()
    return _i2i.generate(image, prompt, **kwargs)
