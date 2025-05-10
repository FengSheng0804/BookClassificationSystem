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
        # 将上采样后的特征图与跳跃连接的特征图拼接
        return torch.cat([x, skip], dim=1)

class UNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 编码器 (通道数减半)
        self.enc1 = Conv_Block(3, 32)        # 原64

        self.down1 = DownSample(32)          # 输出64
        self.enc2 = Conv_Block(64, 64)       # 原128

        self.down2 = DownSample(64)          # 输出128
        self.enc3 = Conv_Block(128, 128)     # 原256

        self.down3 = DownSample(128)         # 输出256
        self.enc4 = Conv_Block(256, 256)     # 原512

        self.down4 = DownSample(256)         # 输出512
        # 桥接层
        self.bridge = Conv_Block(512, 512)   # 原1024
        
        # 解码器
        self.up1 = UpSample(512, 256)        # 输入512→输出256，这里执行完上采样后的通道数为256，再加上skip4的256通道，实际输出为512，所以dec1的输入为512
        self.dec1 = Conv_Block(512, 256)     # 256 * 2=512输入

        self.up2 = UpSample(256, 128)        # 输入256→输出128
        self.dec2 = Conv_Block(256, 128)     # 128 * 2=256输入

        self.up3 = UpSample(128, 64)         # 输入128→输出64
        self.dec3 = Conv_Block(128, 64)      # 64 * 2=128输入

        self.up4 = UpSample(64, 32)          # 输入64→输出32
        self.dec4 = Conv_Block(64, 32)       # 32 * 2=64输入
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Conv2d(32, num_classes, 3, 1, 1),
            nn.LogSoftmax(dim=1)             # 更稳定的输出
        )

    def forward(self, x):
        # 编码器
        # 编码器
        skip1 = self.enc1(x)                 # [B,32,H,W]
        down1 = self.down1(skip1)            # [B,64,H/2,W/2]
        skip2 = self.enc2(down1)             # [B,64,H/2,W/2]
        down2 = self.down2(skip2)            # [B,128,H/4,W/4]
        skip3 = self.enc3(down2)             # [B,128,H/4,W/4]
        down3 = self.down3(skip3)            # [B,256,H/8,W/8]
        skip4 = self.enc4(down3)             # [B,256,H/8,W/8]
        down4 = self.down4(skip4)            # [B,512,H/16,W/16]
        bridge = self.bridge(down4)          # [B,512,H/16,W/16]
        # 解码器
        up1 = self.up1(bridge, skip4)        # [B,512+256=768 → 实际应为256+256=512?]
        up1 = self.dec1(up1)
        up2 = self.up2(up1, skip3)
        up2 = self.dec2(up2)
        up3 = self.up3(up2, skip2)
        up3 = self.dec3(up3)
        up4 = self.up4(up3, skip1)
        up4 = self.dec4(up4)
        
        return self.output_layer(up4)

if __name__ == '__main__':
    x=torch.randn(2,3,256,256)
    net=UNet(5)
    print(net(x).shape)