import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.nn.functional import one_hot

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
        image = Image.open(image_path)
        image_tensor = transform(image)
        
        # 加载并处理mask
        segment_image = Image.open(segment_path).convert('L')  # 转为单通道
        mask_array = np.array(segment_image)
        
        # 关键修改：二值化处理并转换为LongTensor
        mask_array = (mask_array > 128).astype(np.int64)  # 将255转为1
        mask_tensor = torch.from_numpy(mask_array).long()  # 强制转换为long类型
        
        return image_tensor, mask_tensor


if __name__ == '__main__':
    data = MyDataset("F:\desktop\dataset")
    print(data[0][0].shape)
    print(data[0][1].shape)
    out=one_hot(data[0][1].long())
    print(out.shape)