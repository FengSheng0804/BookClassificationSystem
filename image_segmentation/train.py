import os
import torch
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
from data import *
from image_segmentation.models.Unet import *
from torchvision.utils import save_image
from torch.cuda.amp import autocast, GradScaler

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

weight_path = './image_segmentation/content/params/unet.pth'
data_path = 'F:/desktop/dataset'
save_path = './image_segmentation/content/train_image'

if __name__ == '__main__':
    num_classes = 1 + 1  # +1是背景也为一类
    data_loader = DataLoader(
        MyDataset(data_path), 
        batch_size=1, 
        shuffle=True,
    )
    net = UNet(num_classes).to(device)
    opt = optim.Adam(net.parameters())
    
    # 初始化梯度缩放器
    scaler = GradScaler()  # 新增代码

    start_epoch = 1
    if os.path.exists(weight_path):
        checkpoint = torch.load(weight_path)
        net.load_state_dict(checkpoint['model_state'])
        opt.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch'] + 1
        print(f'成功从 epoch {start_epoch} 恢复训练！')
    else:
        print('未找到历史权重，开始新训练')

    loss_fun = nn.CrossEntropyLoss()

    for epoch in range(start_epoch, 45):
        net.train()
        for i, (image, segment_image) in enumerate(tqdm(data_loader)):
            image = image.to(device)
            segment_image = segment_image.to(device)
            
            # 使用混合精度前向传播
            with autocast():  # 新增代码
                out_image = net(image)
                train_loss = loss_fun(out_image, segment_image)
            
            # 反向传播和优化
            opt.zero_grad()
            scaler.scale(train_loss).backward()  # 替换原来的 train_loss.backward()
            scaler.step(opt)  # 替换原来的 opt.step()
            scaler.update()  # 新增代码：更新缩放器

            # 每 10 个 batch 清理一次 GPU 缓存
            if i % 10 == 0:
                torch.cuda.empty_cache()

            print(f'{epoch}-{i}-train_loss===>>{train_loss.item()}')
            
            if i % 50 == 0:
                with torch.no_grad():
                    # 获取原始图像（反标准化）
                    _input = image[0].cpu().float()
                    if _input.shape[0] == 3:  # RGB图像反标准化
                        _input = (_input * 0.5) + 0.5  # 假设预处理时使用mean=0.5, std=0.5
                    
                    # 处理真值mask
                    _segment = segment_image[0].cpu().float()
                    _segment = _segment.unsqueeze(0)  # 添加通道维度 [1, H, W]
                    _segment = torch.cat([_segment, _segment, _segment], dim=0)  # 扩展为 3 通道 [3, H, W]
                    
                    # 处理预测结果
                    _pred = torch.argmax(out_image[0], dim=0).cpu().float()
                    _pred = _pred.unsqueeze(0)  # 添加通道维度 [1, H, W]
                    _pred = torch.cat([_pred, _pred, _pred], dim=0)  # 扩展为 3 通道 [3, H, W]
                    
                    # 将输入、真值、预测水平拼接在一起
                    grid = torch.cat([_input, _segment, _pred], dim=-1)  # 沿宽度方向拼接
                    
                    # 保存拼接后的图像
                    save_image(grid, os.path.join(save_path, f'epoch{epoch}_batch{i}.png'))
        # 每个 epoch 结束后清理缓存
        torch.cuda.empty_cache()
        
        # 保存权重
        checkpoint = {
            'epoch': epoch,
            'model_state': net.state_dict(),
            'optimizer_state': opt.state_dict(),
            'loss': train_loss.item() if 'train_loss' in locals() else None
        }
        torch.save(checkpoint, f'./image_segmentation/content/params/unet_epoch{epoch}.pth')
        print(f'Epoch {epoch} 已保存！')