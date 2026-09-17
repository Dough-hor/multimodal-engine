from PIL import Image
from torchvision import transforms # image与张量之间转换的库
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from .srcnn import SRCNN
import os

class SRDataset(Dataset):
    def __init__(self,image_folder,scale=3,patch_size=33,stride=14): #patch_size设置为33是为了输出时对应的21*21；stride是图片分割时向右移动的步长，切14代表重叠（33-14），重叠越多，训练数据越多，模型学的越好
        self.patches_lr=[]
        self.patches_hr=[]

        # 获取所有图片路径
        paths=[os.path.join(image_folder,f) for f in os.listdir(image_folder) if f.endswith(('.png','.jpg','.jpeg'))]
        for path in paths:
            lr_t,hr_t=self.prepare_pair(path,scale)
            lr_p,hr_p=self.extract_patches(lr_t,hr_t,patch_size,stride)
            self.patches_lr.append(lr_p)
            self.patches_hr.append(hr_p)

        self.patches_lr=torch.cat(self.patches_lr,dim=0) # cat：在现有维度上拼接，不新增维度
        self.patches_hr=torch.cat(self.patches_hr,dim=0)

    def prepare_pair(self,img_path,scale):
        hr=Image.open(img_path).convert("RGB") # 获取图片并转换成RGB
        w,h=hr.size
        w=w-w%scale # 抹去余数让其为scale的整数倍
        h=h-h%scale
        hr=hr.crop((0,0,w,h)) # 将hr图片转换成相同大小的w*h,crop接收一个元组

        # 将hr先缩小scale倍后通过双三次插值放大变成lr图片
        lr=hr.resize((w//scale,h//scale),Image.BICUBIC) # BICUBIC表示双三次插值法,//表示整除
        lr_up=lr.resize((w,h),Image.BICUBIC)
        to_tensor=transforms.ToTensor()
        return to_tensor(lr_up),to_tensor(hr) # 把整张图片的所有像素 → 全部变成数字 → 装进张量里
    
    # 将两种图都切成一模一样位置的 33×33 小块
    def extract_patches(self,lr_t,hr_t,patch_size,stride): # _t表示张量
        _,w,h=lr_t.shape # 维度只是描述张量长什么样，张量本身是数据，在这里是像素值按shape的排列整体，张量 = 可以存放数字的、有维度的 “多维数组”
        lr_patches,hr_patches=[],[] 
        for y in range(0,h-patch_size+1,stride):
            for x in range(0,w-patch_size+1,stride):
                lr_patches.append(lr_t[:,y:y+patch_size,x:x+patch_size]) # 形状为 [3, 33, 33] 的张量
                hr_patches.append(hr_t[:,y:y+patch_size,x:x+patch_size]) # 第一个:前表示所有通道（R/G/B）
        return torch.stack(lr_patches),torch.stack(hr_patches) # stack：新增Batch维度一起打包
    
    # Dataset必须实现的两个方法，否则模型不知道有多少数据,模型没法拿数据训练
    def __len__(self):
        return len(self.patches_lr)
    
    def __getitem__(self,idx):
        return self.patches_lr[idx],self.patches_hr[idx]

if __name__=="__main__":
    dataset = SRDataset("data/t91", scale=3, patch_size=33, stride=14) # Dataset 负责：数据从哪里来（图片→张量）
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True) # DataLoader 负责：怎么把数据喂给模型（打包、搬运、循环）,防止一次性太多数据撑死，同时还能帮忙打乱，自动打包（batch）

    model=SRCNN(in_channels=3)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4) # model.parameters()：模型里所有需要调整的参数；lr=1e-4：学习率 = 每次改多少
    criterion=nn.MSELoss()

    for epoch in range(100): # epoch = 把所有训练数据完整看一遍 叫 1 个 epoch
        for lr_batch,hr_batch in dataloader: # 通过getitem判断返回值，也就是这里的dataloader返回的是getitem的返回值
            output=model(lr_batch)
            loss=criterion(output,hr_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch{epoch+1},LOSS:{loss.item():.6f}") # .6f:表六6位小数

    torch.save(model.state_dict(),"srcnn_t91.pth") # model.state_dict():模型里所有学到的知识点（权重、参数）





