import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
from image_segmentation.models.Unet import *
from utils import *
from torchvision import transforms
from PIL import Image


transform = transforms.Compose([
    transforms.ToTensor()
])
net=UNet(2).cuda()

weight_path = './image_segmentation/content/params/unet_epoch4.pth'
if os.path.exists(weight_path):
    net.load_state_dict(torch.load(weight_path)['model_state'])
    print('successfully')
else:
    print('no loading')

_input="F:/desktop/1.png"

img=resize_rgb_image(_input)
img_data=transform(img).cuda()
img_data=torch.unsqueeze(img_data,dim=0)
net.eval()
out=net(img_data)
pred_mask = torch.argmax(out, dim=1).squeeze(0)  # [H,W]
# 转换为numpy并调整数据类型
mask_np = pred_mask.byte().cpu().numpy() * 255   # 直接得到0和255的uint8
# 保存为单通道PNG
plt.imshow(mask_np, cmap='gray')
plt.show()