import os
import numpy as np
import torch
from image_segmentation.models.Unet import *
from utils import *
from torchvision import transforms

# 删除小的连通组件
def remove_small_connected_components(mask, min_size):
    # 处理白色小区域，转为黑色
    num_labels_white, labels_white, stats_white, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    mask_white_processed = np.zeros(mask.shape, dtype=np.uint8)
    for i in range(1, num_labels_white):  # 跳过背景（0）
        if stats_white[i, cv2.CC_STAT_AREA] >= min_size:
            mask_white_processed[labels_white == i] = 255  # 保留大面积白色
    
    # show_image(mask_white_processed)

    # 将得到的大白色区域反转，小黑色区域变白
    mask_inv = 255 - mask_white_processed  # 反转图像，黑色区域变为白色
    num_labels_black, labels_black, stats_black, _ = cv2.connectedComponentsWithStats(mask_inv, connectivity=8)
    mask_inv_processed = np.zeros(mask_inv.shape, dtype=np.uint8)
    for i in range(1, num_labels_black):
        if stats_black[i, cv2.CC_STAT_AREA] >= min_size:
            mask_inv_processed[labels_black == i] = 255  # 保留反转后的大面积白色（即原黑色大区域）
    mask_black_processed = 255 - mask_inv_processed  # 反转回来，小黑色区域变白

    # show_image(mask_black_processed)
    
    # 合并结果：保留原大白色 + 原小黑色变白
    final_mask = cv2.bitwise_or(mask_white_processed, mask_black_processed)
    return final_mask


transform = transforms.Compose([
    transforms.ToTensor()
])
net=UNet(2).cuda()

weight_path = './image_segmentation/content/params/unet_epoch1.pth'
if os.path.exists(weight_path):
    net.load_state_dict(torch.load(weight_path)['model_state'])
    print('successfully')
else:
    print('no loading')

_input="images\grass_2.png"

original_img = cv2.imread(_input)

img = resize_rgb_image(_input)
img_data = transform(img).cuda()
img_data = torch.unsqueeze(img_data, dim=0)
net.eval()
out = net(img_data)
pred_mask = torch.argmax(out, dim=1).squeeze(0)  # [H,W]
# 转换为numpy并调整数据类型
mask_np = pred_mask.byte().cpu().numpy() * 255   # 直接得到0和255的uint8

# 应用remove_small_connected_components函数
processed_mask = remove_small_connected_components(mask_np, min_size=500)

# 将掩码恢复成原始图像大小
original_size = original_img.shape[1::-1]
processed_mask_resized = cv2.resize(processed_mask, original_size, interpolation=cv2.INTER_NEAREST)

# 保存处理后的掩码
output_path = 'images\grass_2_0_mask.png'
cv2.imwrite(output_path, processed_mask_resized)