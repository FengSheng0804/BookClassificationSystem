import os
from matplotlib import pyplot as plt
import numpy as np
import torch
import albumentations as A
from torch.utils.data import Dataset
from utils import resize_rgb_image
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor()
])


class MyDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.name = os.listdir(os.path.join(path, 'images'))

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        # 文件名处理
        segment_name = self.name[index]  # xx.png
        mask_name = segment_name.split('.')[0] + '_mask.png'
        
        # 路径构造
        segment_path = os.path.join(self.path, 'masks', mask_name)
        image_path = os.path.join(self.path, 'images', segment_name)
        
        # 加载图像
        image = resize_rgb_image(image_path)
        image_tensor = transform(image)  # 假设transform返回FloatTensor
        
        # 加载并处理mask
        segment_image = resize_rgb_image(segment_path).convert('L')  # 转为单通道
        mask_array = np.array(segment_image)
        
        # 关键修改：二值化处理并转换为LongTensor
        mask_array = (mask_array > 128).astype(np.int64)  # 将255转为1
        mask_tensor = torch.from_numpy(mask_array).long()  # 强制转换为long类型
        
        return image_tensor, mask_tensor


if __name__ == '__main__':
    from torch.nn.functional import one_hot
    data = MyDataset("F:\desktop\dataset")
    print(data[0][0].shape)
    print(data[0][1].shape)
    out=one_hot(data[0][1].long())
    print(out.shape)
