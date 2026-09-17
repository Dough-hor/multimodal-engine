import torch.nn as nn
import torch.nn.functional as F


class SRCNN(nn.Module):
    def __init__(self,in_channels=3,out_channels=3):
        super().__init__()
        self.model=nn.Sequential(
            nn.Conv2d(in_channels,64,9,1,4), # padding=(9-1)/2
            nn.ReLU(inplace=True),
            nn.Conv2d(64,32,1,1,0),
            nn.ReLU(inplace=True),
            nn.Conv2d(32,in_channels,5,1,2)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.normal_(m.weight,mean=0.0,std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)

    def forward(self,x):
        return x+self.model(x)


