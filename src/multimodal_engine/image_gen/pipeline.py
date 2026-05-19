import yaml
from diffusers import StableDiffusionPipeline
import torch

class TextToImage:
    def __init__(self, config_path=None):
        if config_path is None:
            # 自动定位到同级 configs/sd_config.yaml
            module_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(module_dir, "..", "configs", "sd_config.yaml")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

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

        print(f"加载模型中... (设备: {self.device})")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id, torch_dtype=torch.float32
        )
        self.pipe = self.pipe.to(self.device)

        # 如果显存不够，取消下面一行的注释
        # self.pipe.enable_attention_slicing()

    def generate(self, prompt: str, steps=None, cfg_scale=None):
        steps = steps if steps is not None else self.default_steps
        cfg_scale = cfg_scale if cfg_scale is not None else self.default_cfg_scale

        print(f"生成中: {prompt} (steps={steps}, guidance_scale={cfg_scale})")
        image = self.pipe(
            prompt,
            num_inference_steps=steps,
            guidance_scale=cfg_scale
        ).images[0]
        return image
    _generator = None

    def generate(prompt: str, **kwargs):
        global _generator
        if _generator is None:
            _generator = TextToImage()   # 自动读 sd_config.yaml
        return _generator.generate(prompt, **kwargs)
