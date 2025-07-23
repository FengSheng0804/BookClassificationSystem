import os
import torch
import numpy as np
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
from data import *
from image_segmentation.models.Unet import *
from torchvision.utils import save_image
from torch.cuda.amp import autocast, GradScaler
import time
import json
from sklearn.metrics import accuracy_score, jaccard_score
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

weight_path = './image_segmentation/content/params/unet.pth'
best_weight_path = './image_segmentation/content/params/best_unet_new.pth'
data_path = 'F:/desktop/dataset'
save_path = './image_segmentation/content/train_image'
log_path = './image_segmentation/content/logs'

# 创建必要的目录
os.makedirs(os.path.dirname(weight_path), exist_ok=True)
os.makedirs(save_path, exist_ok=True)
os.makedirs(log_path, exist_ok=True)

def calculate_segmentation_metrics(pred, target):
    """
    计算图像分割指标
    pred: 预测结果 [B, C, H, W] 或 [B, H, W]
    target: 真实标签 [B, H, W]
    """
    # 确保预测结果是类别索引
    if pred.dim() == 4:  # [B, C, H, W]
        pred = torch.argmax(pred, dim=1)  # [B, H, W]
    
    # 转换为numpy数组
    pred_np = pred.detach().cpu().numpy().flatten()
    target_np = target.detach().cpu().numpy().flatten()
    
    # 计算各种指标
    pixel_accuracy = accuracy_score(target_np, pred_np)
    iou = jaccard_score(target_np, pred_np, average='macro', zero_division=0)
    
    # 计算Dice系数
    pred_binary = (pred_np == 1).astype(np.uint8)
    target_binary = (target_np == 1).astype(np.uint8)
    
    intersection = np.sum(pred_binary * target_binary)
    dice_score = (2.0 * intersection) / (np.sum(pred_binary) + np.sum(target_binary) + 1e-8)
    
    # 计算精确率和召回率
    tp = np.sum(pred_binary * target_binary)
    fp = np.sum(pred_binary * (1 - target_binary))
    fn = np.sum((1 - pred_binary) * target_binary)
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return {
        'pixel_accuracy': pixel_accuracy,
        'iou': iou,
        'dice_score': dice_score,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }

def validate_model(net, val_loader, loss_fun, device):
    """验证模型性能"""
    net.eval()
    val_losses = []
    val_metrics = {
        'pixel_accuracy': [],
        'iou': [],
        'dice_score': [],
        'precision': [],
        'recall': [],
        'f1_score': []
    }
    
    with torch.no_grad():
        for image, segment_image in val_loader:
            image = image.to(device)
            segment_image = segment_image.to(device)
            
            with autocast():
                out_image = net(image)
                val_loss = loss_fun(out_image, segment_image)
            
            val_losses.append(val_loss.item())
            
            # 计算分割指标
            metrics = calculate_segmentation_metrics(out_image, segment_image)
            for key, value in metrics.items():
                val_metrics[key].append(value)
    
    # 计算平均值
    avg_metrics = {}
    for key, values in val_metrics.items():
        avg_metrics[key] = np.mean(values)
    
    avg_metrics['val_loss'] = np.mean(val_losses)
    return avg_metrics

def save_training_log(epoch, train_metrics, val_metrics, log_file):
    """保存训练日志"""
    log_entry = {
        'epoch': epoch,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'train': train_metrics,
        'validation': val_metrics
    }
    
    # 读取现有日志
    logs = []
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:  # 检查文件是否为空
                    logs = json.loads(content)
                else:
                    logs = []
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"Warning: 无法读取日志文件 {log_file}: {e}")
            print("Creating new log file...")
            logs = []
    
    logs.append(log_entry)
    
    # 保存日志
    try:
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Warning: 无法保存日志文件: {e}")

def plot_training_curves(log_file):
    """绘制训练曲线"""
    if not os.path.exists(log_file):
        return
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:  # 检查文件是否为空
                return
            logs = json.loads(content)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"Warning: 无法读取日志文件用于绘图: {e}")
        return
    
    if len(logs) == 0:
        return
    
    epochs = [log['epoch'] for log in logs]
    train_losses = [log['train']['loss'] for log in logs]
    val_losses = [log['validation']['val_loss'] for log in logs]
    val_ious = [log['validation']['iou'] for log in logs]
    val_dice = [log['validation']['dice_score'] for log in logs]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss曲线
    axes[0, 0].plot(epochs, train_losses, label='Train Loss', color='blue')
    axes[0, 0].plot(epochs, val_losses, label='Val Loss', color='red')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # IoU曲线
    axes[0, 1].plot(epochs, val_ious, label='Val IoU', color='green')
    axes[0, 1].set_title('Validation IoU')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('IoU')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Dice Score曲线
    axes[1, 0].plot(epochs, val_dice, label='Val Dice Score', color='purple')
    axes[1, 0].set_title('Validation Dice Score')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Dice Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 综合指标对比
    val_f1 = [log['validation']['f1_score'] for log in logs]
    axes[1, 1].plot(epochs, val_ious, label='IoU', color='green')
    axes[1, 1].plot(epochs, val_dice, label='Dice', color='purple')
    axes[1, 1].plot(epochs, val_f1, label='F1', color='orange')
    axes[1, 1].set_title('Validation Metrics Comparison')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_path, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    print("🚀 开始训练UNet模型...")
    
    num_classes = 1 + 1  # +1是背景也为一类
    
    # 创建数据加载器（假设有训练集和验证集分割）
    full_dataset = MyDataset(data_path)

    # 简单的数据集分割（90% 训练，10% 验证）
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=2, 
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=2
    )
    
    print(f"📊 训练集样本数: {train_size}")
    print(f"📊 验证集样本数: {val_size}")
    
    net = UNet(num_classes).to(device)
    opt = optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=5, verbose=True)
    
    # 初始化梯度缩放器
    scaler = GradScaler()
    
    # 初始化最佳指标
    best_iou = 0.0
    best_dice = 0.0
    best_epoch = 0
    
    start_epoch = 1
    training_history = {
        'best_iou': best_iou,
        'best_dice': best_dice,
        'best_epoch': best_epoch
    }
    
    # 尝试加载历史权重
    if os.path.exists(weight_path):
        try:
            checkpoint = torch.load(weight_path)
            net.load_state_dict(checkpoint['model_state'])
            opt.load_state_dict(checkpoint['optimizer_state'])
            start_epoch = checkpoint['epoch'] + 1
            
            if 'best_iou' in checkpoint:
                best_iou = checkpoint['best_iou']
                best_dice = checkpoint['best_dice']
                best_epoch = checkpoint['best_epoch']
                training_history = {
                    'best_iou': best_iou,
                    'best_dice': best_dice,
                    'best_epoch': best_epoch
                }
            
            print(f'✅ 成功从 epoch {start_epoch-1} 恢复训练！')
            print(f'   当前最佳IoU: {best_iou:.4f} (epoch {best_epoch})')
            print(f'   当前最佳Dice: {best_dice:.4f} (epoch {best_epoch})')
        except Exception as e:
            print(f'⚠️  权重加载失败: {e}，开始新训练')
            start_epoch = 1
    else:
        print('📝 未找到历史权重，开始新训练')

    loss_fun = nn.CrossEntropyLoss()
    log_file = os.path.join(log_path, 'training_log.json')
    
    print(f"🎯 训练目标: epoch {start_epoch} -> 45")
    print("="*60)

    for epoch in range(start_epoch, 45):
        epoch_start_time = time.time()
        
        # ========== 训练阶段 ==========
        net.train()
        train_losses = []
        train_metrics_list = []
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch}/44 [Train]')
        for i, (image, segment_image) in enumerate(train_pbar):
            image = image.to(device)
            segment_image = segment_image.to(device)
            
            # 使用混合精度前向传播
            with autocast():
                out_image = net(image)
                train_loss = loss_fun(out_image, segment_image)
            
            # 反向传播和优化
            opt.zero_grad()
            scaler.scale(train_loss).backward()
            scaler.step(opt)
            scaler.update()
            
            train_losses.append(train_loss.item())
            
            # 计算训练指标（每10个batch计算一次以节省时间）
            if i % 10 == 0:
                with torch.no_grad():
                    metrics = calculate_segmentation_metrics(out_image, segment_image)
                    train_metrics_list.append(metrics)
            
            # 更新进度条
            train_pbar.set_postfix({
                'Loss': f'{train_loss.item():.4f}',
                'LR': f'{opt.param_groups[0]["lr"]:.2e}'
            })
            
            # 每 20 个 batch 清理一次 GPU 缓存
            if i % 20 == 0:
                torch.cuda.empty_cache()
            
            # 只保存每个epoch的第一个batch的训练过程图像
            if i == 0:
                with torch.no_grad():
                    # 获取原始图像（反标准化）
                    _input = image[0].cpu().float()
                    if _input.shape[0] == 3:  # RGB图像反标准化
                        _input = (_input * 0.5) + 0.5
                    
                    # 处理真值mask
                    _segment = segment_image[0].cpu().float()
                    _segment = _segment.unsqueeze(0)
                    _segment = torch.cat([_segment, _segment, _segment], dim=0)
                    
                    # 处理预测结果
                    _pred = torch.argmax(out_image[0], dim=0).cpu().float()
                    _pred = _pred.unsqueeze(0)
                    _pred = torch.cat([_pred, _pred, _pred], dim=0)
                    
                    # 将输入、真值、预测水平拼接在一起
                    grid = torch.cat([_input, _segment, _pred], dim=-1)
                    
                    # 保存拼接后的图像，每个epoch只保存第一个batch
                    save_image(grid, os.path.join(save_path, f'epoch{epoch}_batch0.png'))
        
        # ========== 验证阶段 ==========
        print(f"\n🔍 Epoch {epoch} - 开始验证...")
        val_metrics = validate_model(net, val_loader, loss_fun, device)
        
        # 计算训练阶段平均指标
        avg_train_loss = np.mean(train_losses)
        avg_train_metrics = {}
        if train_metrics_list:
            for key in train_metrics_list[0].keys():
                avg_train_metrics[key] = np.mean([m[key] for m in train_metrics_list])
        avg_train_metrics['loss'] = avg_train_loss
        
        # 学习率调度
        scheduler.step(val_metrics['iou'])
        
        epoch_time = time.time() - epoch_start_time
        
        # ========== 结果显示 ==========
        print(f"\n📊 Epoch {epoch} 结果总结:")
        print(f"   ⏱️  耗时: {epoch_time:.1f}s")
        print(f"   📉 训练损失: {avg_train_loss:.4f}")
        print(f"   📉 验证损失: {val_metrics['val_loss']:.4f}")
        print(f"   🎯 验证IoU: {val_metrics['iou']:.4f}")
        print(f"   🎯 验证Dice: {val_metrics['dice_score']:.4f}")
        print(f"   🎯 验证F1: {val_metrics['f1_score']:.4f}")
        print(f"   🎯 像素准确率: {val_metrics['pixel_accuracy']:.4f}")
        
        # ========== 模型保存 ==========
        is_best = False
        
        # 判断是否为最佳模型（主要看IoU，辅助看Dice）
        if val_metrics['iou'] > best_iou or (val_metrics['iou'] == best_iou and val_metrics['dice_score'] > best_dice):
            best_iou = val_metrics['iou']
            best_dice = val_metrics['dice_score']
            best_epoch = epoch
            is_best = True
            
            print(f"   🌟 发现更好的模型！IoU: {best_iou:.4f}, Dice: {best_dice:.4f}")
        
        # 保存最新权重
        latest_checkpoint = {
            'epoch': epoch,
            'model_state': net.state_dict(),
            'optimizer_state': opt.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'loss': avg_train_loss,
            'val_metrics': val_metrics,
            'best_iou': best_iou,
            'best_dice': best_dice,
            'best_epoch': best_epoch,
            'training_history': training_history
        }
        
        # 只保存最新模型权重
        torch.save(latest_checkpoint, weight_path)
        
        # 保存最佳权重
        if is_best:
            torch.save(latest_checkpoint, best_weight_path)
            print(f"   💾 最佳模型已保存到: {best_weight_path}")
        
        print(f"   💾 最新模型已保存")
        
        # ========== 保存训练日志 ==========
        save_training_log(epoch, avg_train_metrics, val_metrics, log_file)
        
        # 每5个epoch绘制一次训练曲线
        if epoch % 5 == 0:
            plot_training_curves(log_file)
            print(f"   📈 训练曲线已更新")
        
        # 清理GPU缓存
        torch.cuda.empty_cache()
        
        print(f"   🏆 当前最佳: IoU={best_iou:.4f}, Dice={best_dice:.4f} (epoch {best_epoch})")
        print("-" * 60)
    
    # ========== 训练完成 ==========
    print("\n🎉 训练完成！")
    print(f"🏆 最佳性能:")
    print(f"   IoU: {best_iou:.4f}")
    print(f"   Dice: {best_dice:.4f}")
    print(f"   最佳epoch: {best_epoch}")
    print(f"💾 最佳模型保存路径: {best_weight_path}")
    print(f"📊 训练日志保存路径: {log_file}")
    
    # 绘制最终训练曲线
    plot_training_curves(log_file)
    print(f"📈 最终训练曲线: {os.path.join(log_path, 'training_curves.png')}")