import torch
import torch.nn as nn

# 残差快
class ResidualBlock(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3,1,1), # 输入通道，输出通道，卷积核大小，步长，填充
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),

            # 第二个卷积（无激活）：为了能与最开始的x连接
            nn.Conv2d(channels, channels,3,1,1),
            nn.InstanceNorm2d(channels)
        )
    
    def forward(self,x):
        return x+self.block(x) # 跳跃连接

# 生成器
class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_res=9):
        super().__init__()
        self.down=nn.Sequential(
            # 用大尺寸的卷积提取全局纹理、整体轮廓防止下采样丢失
            nn.Conv2d(in_channels,64,7,1,3), # 约定用7*7的大卷积核
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64,128,3,2,1), # 步长为2，不管padding，尺寸一定缩小；填充为1，原本为抵消3*3卷积核带来的缩小，但因步长导致最终缩小
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128,256,3,2,1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        )

        res_layers=[]
        for _ in range(n_res):
            res_layers.append(ResidualBlock(256))
        self.res_blocks=nn.Sequential(*res_layers)

        self.up=nn.Sequential(
            nn.ConvTranspose2d(256,128,3,2,1,output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128,64,3,2,1,output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.out_conv=nn.Sequential(
            nn.Conv2d(64,out_channels,7,1,3),
            nn.Tanh()
        )

    def forward(self,x):
        x=self.down(x)
        x=self.res_blocks(x)
        x=self.up(x)
        x=self.out_conv(x)
        return x # 最后一层通道为3，表示返回3张RGB的图片

# 判别器        
class Discriminator(nn.Module):
    def __init__(self,in_channels=3):
        super().__init__()
        self.model=nn.Sequential(
            nn.Conv2d(in_channels,64,4,2,1),
            nn.LeakyReLU(0.2,inplace=True), # LeakyReLU与ReLU的差别是在负半轴是αx，这个0.2就是α，generator不用是防止特征被这些x<=0的值脏污

            nn.Conv2d(64,128,4,2,1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(128,256,4,2,1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(256, 512, 4, 1, 1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512,1,4,1,1)
        )

    def forward(self,x):
        return self.model(x) # 最后一层通道是1，表示返回一个数值（[batch, 1, H, W] 的分数图），在0到1区间，接近0表示很假，接近1表示很真
    

class cycleGAN(nn.Module):
    def __init__(self,in_channels=3,out_channels=3,n_res=9):
        super().__init__()
        self.gen_AB=Generator(in_channels,out_channels,n_res)
        self.gen_BA=Generator(out_channels,in_channels,n_res)

        self.disc_A=Discriminator(in_channels)
        self.disc_B=Discriminator(out_channels)

        self.criterion_mse=nn.MSELoss() # GAN 损失（真假判断）
        self.criterion_l1=nn.L1Loss()# 循环损失 + 恒等损失

        # 告诉网络优先听谁的
        self.cycle_weight=10.0 # 循环损失权重：保证图像内容不变
        self.identity_weight=5.0 # 恒等损失权重：保证颜色风格一致

        # 初始化权重:在第一次调用conv2d的时候将卷积里的所有权重优化，相当于让我从一个较好的初始权重开始训练而不是随机的
        self._init_weight()

    def _init_weight(self,mean=0.0,std=0.02):
        for m in self.modules(): # modules是pytorch自带的搜索工具，自动找到所有层
            if isinstance(m,(nn.Conv2d,nn.ConvTranspose2d)): # 判断m是不是，后面的两种层次
                nn.init.normal_(m.weight,mean,std) # normal_设置随机数
                if m.bias is not None:
                    nn.init.constant_(m.bias,0) # constant_设置固定值0
            elif isinstance(m,nn.InstanceNorm2d):
                # InstanceNorm2d 默认 affine=False，此时 weight/bias 为 None，必须先判空
                if m.weight is not None:
                    nn.init.constant_(m.weight,1)
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)

    # 不用于训练，用于生成图片（A->B）
    def forward(self,real_A):
        """输入A类图,输出B风格图"""
        return self.gen_AB(real_A)
    
    # 判别器有最开始的初始权重，计算时需要用到权重，real图的像素以及中间计算网络，但判别器只能改变自己的权重，所以判别器是通过改变权重让最终计算的值接近1或0
    def _train_discriminators(self,real_A,real_B):
        """更新判别器，返回对抗损失"""

        # 生成假图
        fake_B=self.gen_AB(real_A).detach()
        fake_A=self.gen_BA(real_B).detach()

        # 真图标签为1，假图标签为0（LSGAN
        real_labels=torch.ones_like(self.disc_A(real_A)) # 只是根据A的分数图的形状生成的全1图，所以A和B的计算都可以用
        fake_labels=torch.zeros_like(self.disc_A(fake_A))

        # 判别器A损失
        pred_real_A=self.disc_A(real_A)
        pred_fake_A=self.disc_A(fake_A)
        loss_D_A=self.criterion_mse(pred_real_A,real_labels)+self.criterion_mse(pred_fake_A,fake_labels)

        # 判别器B损失
        pred_real_B = self.disc_B(real_B)
        pred_fake_B = self.disc_B(fake_B)
        loss_D_B=self.criterion_mse(pred_real_B,real_labels)+self.criterion_mse(pred_fake_B,fake_labels)

        # 判别器只需要判别真假，仅需要对抗损失
        return loss_D_A + loss_D_B
    
        # 生成器在更新权重后输出被优化的假图，送入判别器后判别器用固定权重计算，输出分数升高
    def _train_generators(self,real_A,real_B):
        """更新生成器,返回总损失（对抗损失+循环损失+恒等损失）"""

        # 生成假图
        fake_B=self.gen_AB(real_A) # 生成器的目标是骗过判别器，所以不能detach，梯度必须回传
        fake_A=self.gen_BA(real_B)

        # 对抗损失
        pred_fake_B = self.disc_B(fake_B)
        pred_fake_A = self.disc_A(fake_A)

        # 只算假图：生成器只关心自己造出来的假图，不需要管真图长什么样
        loss_GAN_AB = self.criterion_mse(pred_fake_B, torch.ones_like(pred_fake_B)) # 生成图只希望越接近1或0，不需要管真图是不是全1，因为接近全1或全0能让判别器识别
        loss_GAN_BA = self.criterion_mse(pred_fake_A, torch.ones_like(pred_fake_A))
        loss_GAN = loss_GAN_AB + loss_GAN_BA

        # 循环一致性损失
        rec_A=self.gen_BA(fake_B)
        rec_B=self.gen_AB(fake_A)
        loss_cycle_A=self.criterion_l1(rec_A,real_A)
        loss_cycle_B=self.criterion_l1(rec_B,real_B)
        loss_cycle = (loss_cycle_A + loss_cycle_B) * self.cycle_weight

        # 恒等损失
        # 核心逻辑：输入本来就是目标域，就不该变
        # 监督生成器：不要改变色彩、色调、亮度、对比度
        idt_B=self.gen_AB(real_B)
        idt_A=self.gen_BA(real_A)
        loss_idt_A = self.criterion_l1(idt_A, real_A)
        loss_idt_B = self.criterion_l1(idt_B, real_B)
        loss_idt = (loss_idt_A + loss_idt_B) * self.identity_weight

        total_loss = loss_GAN + loss_cycle + loss_idt
        return total_loss

    def tranning_step(self,real_A,real_B,optimizer_G,optimizer_D):
        """
        执行一个训练步骤（两个子步骤）
        real_A, real_B: 来自两个域的图像
        optimizer_G: 优化生成器的优化器（通常包含两个生成器的参数）
        optimizer_D: 优化判别器的优化器（通常包含两个判别器的参数）
        返回 losses 字典供记录

        1.冻结生成器
        2.训练判别器（优化器更新判别器）
        3.冻结判别器
        4.训练生成器（优化器更新生成器）
        5.循环……
        """

        # 1.训练判别器

        # 冻结生成器（如果不冻结，PyTorch 会保留生成器的计算图，浪费巨大显存！）
        for param in self.gen_AB.parameters():
            param.requires_grad=False
        for param in self.gen_BA.parameters():
            param.requires_grad=False

        # 训练判别器
        loss_D=self._train_discriminators(real_A,real_B)
        optimizer_D.zero_grad() # 清空判别器上一步的梯度
        loss_D.backward() # pytorch自动记住前向传播的路线，反向传播回【判别器的每一个权重】(梯度自动存到 每一个权重的 .grad 属性里)
        optimizer_D.step()

        # 解冻生成器
        for param in self.gen_AB.parameters():
            param.requires_grad=True
        for param in self.gen_BA.parameters():
            param.requires_grad=True

        # 2.训练生成器
        # 冻结判别器
        for param in self.disc_A.parameters():
            param.requires_grad=False
        for param in self.disc_B.parameters():
            param.requires_grad=False

        loss_G = self._train_generators(real_A, real_B)
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        # 解冻判别器
        for param in self.disc_A.parameters():
            param.requires_grad=True
        for param in self.disc_B.parameters():
            param.requires_grad=True

        # PL框架约定返回字典，以便PL记录loss、画曲线、控制台打印、保存日志
        return {
            'loss_D': loss_D.item(), # item将张量变成数字可以记录在日志中
            'loss_G': loss_G.item()
        }


        