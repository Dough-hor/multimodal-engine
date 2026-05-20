import logging
import yaml
from diffusers import StableDiffusionPipeline
import torch
import os

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextToImage:
    def __init__(self, config_path=None):
        if config_path is None:
            # 自动定位到同级 configs/sd_config.yaml
            module_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(module_dir, "..", "configs", "sd_config.yaml")
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"错误：找不到配置文件 {config_path}，请检查路径是否正确")
            raise
        except yaml.YAMLError as e:
            logger.error(f"错误：配置文件格式有问题：{e}")
            raise

        # 提取模型配置
        model_cfg = config.get('model', {})
        self.model_id = model_cfg.get('model_id', "runwayml/stable-diffusion-v1-5")
        self.device = model_cfg.get('device', None)

        # 如果 YAML 中没有指定 device，则自动检测
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 提取推理默认值
        self.infer_cfg = config.get('inference', {})
        self.default_steps = self.infer_cfg.get('steps', 50)
        self.default_cfg_scale = self.infer_cfg.get('guidance_scale', 7.5)

        logger.info(f"加载模型中... (设备: {self.device})")
        try:
            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.model_id, torch_dtype=torch.float32
            )
            self.pipe = self.pipe.to(self.device)
        except Exception as e:
            logger.error(f"错误：模型加载失败 - {e}")
            logger.warning(f"提示：请检查模型名称 '{self.model_id}' 是否正确，以及网络是否连通")
            raise
        # 如果显存不够，取消下面一行的注释
        # self.pipe.enable_attention_slicing()

    def generate(self, prompt: str, steps=None, cfg_scale=None):
        steps = steps if steps is not None else self.default_steps
        cfg_scale = cfg_scale if cfg_scale is not None else self.default_cfg_scale

        logger.info(f"生成中: {prompt} (steps={steps}, guidance_scale={cfg_scale})")
        try:
            image = self.pipe(
                prompt,
                num_inference_steps=steps,
                guidance_scale=cfg_scale
            ).images[0]
        except Exception as e:
            logger.error(f"错误：图片生成失败 - {e}")
            raise    
        return image
    
_generator = None

def generate(prompt: str, **kwargs):
    global _generator
    if _generator is None:
        _generator = TextToImage()   # 自动读 sd_config.yaml
    return _generator.generate(prompt, **kwargs)
