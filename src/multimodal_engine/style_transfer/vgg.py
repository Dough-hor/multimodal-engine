import torch
import torch.nn as nn
import torchvision.models as models

class VGGFeatureExtractor(nn.Module):
    def __init__(self,content_layers,style_layers):
        super().__init__()
        vgg=models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        for param in vgg.parameters():
            param.requires_grad=False
        
        # VGG19 features 每层的标准名
        vgg_names = [
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
            'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3',
            'conv3_4', 'relu3_4', 'pool3',
            'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3',
            'conv4_4', 'relu4_4', 'pool4',
            'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3',
            'conv5_4', 'relu5_4', 'pool5',
        ]

        self.vgg_layers = nn.ModuleList([vgg[i] for i in range(len(vgg))])
        self.vgg_names = vgg_names[:len(vgg)]
        self.content_layers=content_layers
        self.style_layers=style_layers

    def forward(self,x):
        content_feats={}
        style_feats={}
        for name,layer in zip(self.vgg_names, self.vgg_layers):
            x=layer(x)
            if name in self.content_layers:
                content_feats[name]=x
            if name in self.style_layers:
                style_feats[name]=x
        return content_feats,style_feats

    