import logging
import os
import yaml
import torch
import torch.optim as optim
from torchvision import transforms
from .vgg import VGGFeatureExtractor
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

def gram_matrix(tensor):
    """tensor: (B, C, H, W) 或 (C, H, W) -> Gram: (B, C, C) 或 (C, C)"""
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    b, c, h, w = tensor.size()
    features = tensor.view(b,c,-1)
    gram = features @ features.transpose(1,2)
    return gram / (c * h * w)

class StyleTransfer:
    def __init__(self, config_path=None):
        if config_path is None:
            module_dir=os.path.dirname(os.path.abspath(__file__))
            config_path=os.path.join(module_dir,"..","configs","style_transfer_config.yaml")
        
        config=_read_config(config_path)

        vgg_layers=config.get('model',{}).get('vgg_layers', {})
        self.content_layers = vgg_layers.get('content', ['conv4_2'])   # 默认值可选
        self.style_layers = vgg_layers.get('style', [])

        # 读取优化参数
        opt = config.get('optimization', {})
        self.steps = opt.get('steps', 300)
        self.content_weight = opt.get('content_weight', 1.0)
        self.style_weight = float(opt.get('style_weight', 1e6))
        self.tv_weight = opt.get('tv_weight', 0.0)

        self.extractor = VGGFeatureExtractor(self.content_layers, self.style_layers)
        self.extractor.eval()
        for param in self.extractor.parameters():
            param.requires_grad = False

    def transfer(self, content_img, style_img, num_steps=None, 
                 content_weight=None, style_weight=None, tv_weight=None):
        # 使用实例变量或传入参数
        num_steps = num_steps if num_steps is not None else self.steps
        content_weight = content_weight if content_weight is not None else self.content_weight
        style_weight = style_weight if style_weight is not None else self.style_weight
        tv_weight = tv_weight if tv_weight is not None else self.tv_weight

        # 设备
        device=next(self.extractor.parameters()).device
        content_img=content_img.to(device)
        style_img=style_img.to(device)

        # 预处理：归一化到 VGG 所需的均值和标准差
        # VGG 在 ImageNet 上的归一化参数
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(device)

        def preprocess(x):
            return (x-mean)/std
        
        def deprocess(x):
            return torch.clamp(x * std + mean, 0, 1)
        
        content_norm = preprocess(content_img)
        style_norm = preprocess(style_img)

        # 提取风格图的目标 Gram 矩阵
        with torch.no_grad():
            _,style_feats=self.extractor(style_norm)
            style_targets={}
            for name,feat in style_feats.items():
                gram=gram_matrix(feat)
                style_targets[name]=gram.detach()

        # 提取内容图的目标特征
        content_targets = {}
        with torch.no_grad():
            content_feats, _ = self.extractor(content_norm)
            for name, feat in content_feats.items():
                content_targets[name] = feat.detach()

        # 初始化生成图（内容图的副本，开启梯度
        gen=content_norm.clone().requires_grad_(True)

        # 优化器（L-BFGS）
        optimizer=optim.LBFGS([gen],lr=1.0,max_iter=num_steps)

        # 优化循环
        def closure():
            optimizer.zero_grad()
            # 提取当前生成图的特征
            content_feats,style_feats=self.extractor(gen)

            # 内容损失
            content_loss=0
            for name,feat in content_feats.items():
                target=content_targets[name]
                content_loss+=torch.nn.functional.mse_loss(feat,target)

            # 风格损失
            style_loss=0
            for name,feat in style_feats.items():
                target=style_targets[name]
                gram=gram_matrix(feat)
                style_loss+=torch.nn.functional.mse_loss(gram,target) 

            # 总变差损失
            tv_loss=0
            if tv_weight>0:
                tv_loss=torch.mean(torch.abs(gen[:, :, 1:, :] - gen[:, :, :-1, :]))+torch.mean(torch.abs(gen[:, :, :, 1:] - gen[:, :, :, :-1]))      

            # 总loss
            total_loss= content_weight * content_loss + style_weight * style_loss + tv_weight * tv_loss
            total_loss.backward()
            return total_loss # 返回损失值
            
            optimizer.step(closure)

        # 返回生成的图像（逆归一化）
        result=deprocess(gen.detach().cpu())
        return result.squeeze(0) # 去掉batch维度
        