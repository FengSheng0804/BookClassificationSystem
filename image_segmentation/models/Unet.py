import torch
from torch import nn
from torch.nn import functional as F

class Conv_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            # 保留最后一个Dropout，移除中间Dropout
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),  # 仅保留最后一个Dropout
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.layer(x)


class DownSample(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(channel, channel*2, 3, 2, 1, padding_mode='reflect', bias=False),  # 修改下采样通道扩展
            nn.BatchNorm2d(channel*2),
            nn.LeakyReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.layer(x)

class UpSample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channel, out_channel, 1, 1, bias=False)
        )
    
    def forward(self, x, skip):
        x = self.up(x)
        return torch.cat([x, skip], dim=1)

class UNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 编码器 (通道数减半)
        self.c1 = Conv_Block(3, 32)         # 原64
        self.d1 = DownSample(32)            # 输出64
        self.c2 = Conv_Block(64, 64)        # 原128
        self.d2 = DownSample(64)            # 输出128
        self.c3 = Conv_Block(128, 128)      # 原256
        self.d3 = DownSample(128)           # 输出256
        self.c4 = Conv_Block(256, 256)      # 原512
        self.d4 = DownSample(256)           # 输出512
        # 桥接层
        self.c5 = Conv_Block(512, 512)      # 原1024
        # 解码器
        self.u1 = UpSample(512, 256)        # 输入512→输出256
        self.c6 = Conv_Block(512, 256)      # 256 * 2=512输入
        self.u2 = UpSample(256, 128)        # 输入256→输出128
        self.c7 = Conv_Block(256, 128)      # 128 * 2=256输入
        self.u3 = UpSample(128, 64)         # 输入128→输出64
        self.c8 = Conv_Block(128, 64)       # 64 * 2=128输入
        self.u4 = UpSample(64, 32)          # 输入64→输出32
        self.c9 = Conv_Block(64, 32)        # 32 * 2=64输入
        
        # 输出层
        self.out = nn.Sequential(
            nn.Conv2d(32, num_classes, 3, 1, 1),
            nn.LogSoftmax(dim=1)            # 更稳定的输出
        )

    def forward(self, x):
        # 编码器
        s1 = self.c1(x)                     # [B,32,H,W]
        s2 = self.c2(self.d1(s1))           # [B,64,H/2,W/2]
        s3 = self.c3(self.d2(s2))           # [B,128,H/4,W/4]
        s4 = self.c4(self.d3(s3))           # [B,256,H/8,W/8]
        bridge = self.c5(self.d4(s4))       # [B,512,H/16,W/16]
        # 解码器
        d1 = self.u1(bridge, s4)            # [B,512+256=768 → 实际应为256+256=512?]
        d1 = self.c6(d1)
        d2 = self.u2(d1, s3)
        d2 = self.c7(d2)
        d3 = self.u3(d2, s2)
        d3 = self.c8(d3)
        d4 = self.u4(d3, s1)
        d4 = self.c9(d4)
        
        return self.out(d4)

if __name__ == '__main__':
    x=torch.randn(2,3,256,256)
    net=UNet()
    print(net(x).shape)