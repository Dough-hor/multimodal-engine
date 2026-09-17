import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .vgg import VGGFeatureExtractor


def calc_mean_std(feat, eps=1e-5):
    """计算特征图每个通道的均值与标准差
    feat: (B, C, H, W) -> mean: (B, C), std: (B, C)
    """
    b, c, h, w = feat.size() # size()返回元组(b, c, h, w)
    feat_flat = feat.view(b, c, -1) # -1自动推断所需要的数，view统一运算时的维度
    mean = feat_flat.mean(dim=2) # 形状: (B, C, H*W)，意思是：在像素那个维度上求平均值
    std = feat_flat.std(dim=2) + eps
    return mean, std


def adain(content_feat, style_feat, eps=1e-5):
    """自适应实例归一化
    将 content_feat 标准化后，用 style_feat 的统计量重新缩放
    """
    c_mean, c_std = calc_mean_std(content_feat, eps)
    s_mean, s_std = calc_mean_std(style_feat, eps)

    c_mean = c_mean.view(content_feat.size(0), -1, 1, 1)
    c_std = c_std.view(content_feat.size(0), -1, 1, 1)
    s_mean = s_mean.view(style_feat.size(0), -1, 1, 1)
    s_std = s_std.view(style_feat.size(0), -1, 1, 1)

    normalized = (content_feat - c_mean) / c_std # 归一化
    return normalized * s_std + s_mean


class AdaINEncoder(nn.Module):
    """用 VGG19 前几层作为编码器，提取特征
    编码阶段冻结所有参数，不参与训练
    """
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features

        enc_layers = list(vgg.children()) # 把 vgg 的所有子模块（每一层）取出来，变成列表,children()返回一个迭代器，包含当前模块的所有直接子模块

        self.encode = nn.Sequential(*enc_layers[: 21]) # *会把列表 enc_layers[:20] 拆解成 20 个独立的元素，然后作为 20 个参数传给 Sequential

        # 冻结参数
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.encode(x)


class AdaINDecoder(nn.Module):
    """解码器：镜像编码器结构，将 AdaIN 处理后的特征图解码回 RGB 图像"""

    def __init__(self):
        super().__init__()

        self.decoder=nn.Sequential( # Sequential是顺序容器
            nn.Conv2d(512,256,3,1,1),nn.ReLU(), #  把 512 个通道的特征图，用 3×3 卷积，变成 256 个通道，尺寸不变，conv2d只的是只在2维做卷积
            nn.Upsample(scale_factor=2,mode="nearest"),
            nn.Conv2d(256,128,3,1,1),nn.ReLU(),
            nn.Upsample(scale_factor=2,mode="nearest"),
            nn.Conv2d(128,64,3,1,1),nn.ReLU(),
            nn.Upsample(scale_factor=2,mode="nearest"),
            nn.Conv2d(64,3,3,1,1),nn.ReLU(), # 第一个3代表RGB
        )

    def forward(self, x):
        return self.decoder(x)


class AdaINStyleTransfer(nn.Module):
    """AdaIN 风格迁移模型
    训练时：前向计算 + 损失
    推理时：直接调用 forward(content, style) 得到结果图
    """
    def __init__(self, content_layer='relu4_1', style_layers=None):
        super().__init__()
        if style_layers is None:
            # 风格损失用了多个层的统计量
            style_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1']

        # 编码器（冻结 VGG 前几层）
        self.encoder = AdaINEncoder()

        # 解码器（可训练）
        self.decoder = AdaINDecoder()

        self.vgg_loss=VGGFeatureExtractor(
            content_layers=[content_layer],
            style_layers=style_layers
        )

        # 损失权重
        self.content_weight = 1.0
        self.style_weight = 10.0
        self.content_layer = content_layer # 内容层只有一层
        self.style_layers = style_layers # 风格层有多层

    def forward(self, content, style, alpha=1.0):
        """推理前向
        返回:
          result: (B, 3, H, W) 风格迁移结果图
        """
        content_feat = self.encoder(content)
        style_feat = self.encoder(style)

        # AdaIN 融合
        t = adain(content_feat, style_feat)
        # alpha 控制风格化强度
        t = alpha * t + (1 - alpha) * content_feat

        return self.decoder(t)

    def calc_content_loss(self, gen_feat, content_feat):
        """计算内容损失
        直接比较生成图和内容图在 VGG 高层特征的 L2 距离
        """
        return F.mse_loss(gen_feat,content_feat)

    def calc_style_loss(self, gen_feats, style_feats):
        """计算风格损失
        比较生成图和风格图在多个 VGG 层上的均值和标准差的 L2 距离
        注意这里是 compare 统计量(mean+std)，不是 gram 矩阵
        """
        # 遍历每个层，计算 ||μ(g) - μ(s)||² + ||σ(g) - σ(s)||²
        loss=0.0
        for layer in self.style_layers:
            gen_f=gen_feats[layer]
            style_f=style_feats[layer]
            gen_mean,gen_std=calc_mean_std(gen_f)
            style_mean,style_std=calc_mean_std(style_f)
            loss+=torch.mean(((gen_mean-style_mean)**2)+((gen_std-style_std)**2))
        return loss

    def compute_loss(self, content, style):
        """训练时的一次完整前向 + 损失计算
        返回:
          total_loss, content_loss, style_loss
        """
        # 编码
        content_feat = self.encoder(content)
        style_feat = self.encoder(style)

        # AdaIN
        t = adain(content_feat, style_feat)

        # 解码
        g_t = self.decoder(t)

        # 内容损失：比较 g_t 和 t 在内容层的特征
        g_t_feat = self.encoder(g_t)
        content_loss = self.calc_content_loss(g_t_feat, t)

        # 风格损失：比较 g_t 和 style 在多个层的统计量
        # 需要用全 VGG（到 relu4_1）提取 g_t 和 style 在多个层的特征
        #   然后对每个层调用 calc_style_loss
        style_loss = 0.0  # placeholder
        with torch.no_grad():
            _,style_feats=self.vgg_loss(style)
        
        _,gen_feats=self.vgg_loss(g_t)
        style_loss = self.calc_style_loss(gen_feats, style_feats)

        # 实现风格损失

        total_loss = (
            self.content_weight * content_loss
            + self.style_weight * style_loss
        )
        return total_loss, content_loss, style_loss
