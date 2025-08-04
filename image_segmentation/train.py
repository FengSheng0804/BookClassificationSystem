import os
import torch
import numpy as np
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
from data import *
from image_segmentation.models.Unet import *
from torchvision.utils import save_image
from torch.cuda.amp import autocast, GradScaler  # 混合精度训练，节省显存并加速
import time
import json
from sklearn.metrics import accuracy_score, jaccard_score
import matplotlib.pyplot as plt

# ==================== 设备和路径配置 ====================
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 强制CUDA同步执行，便于调试

# 各种文件路径配置
weight_path = './image_segmentation/content/params/unet.pth'          # 最新模型权重保存路径
best_weight_path = './image_segmentation/content/params/best_unet_new.pth'  # 最佳模型权重保存路径
data_path = 'F:/desktop/dataset'                                     # 训练数据集路径
save_path = './image_segmentation/content/train_image'               # 训练过程可视化图片保存路径
log_path = './image_segmentation/content/logs'                       # 训练日志保存路径

# 创建必要的目录结构，避免保存时出现路径不存在的错误
os.makedirs(os.path.dirname(weight_path), exist_ok=True)
os.makedirs(save_path, exist_ok=True)
os.makedirs(log_path, exist_ok=True)

def calculate_segmentation_metrics(pred, target):
    """
    计算图像分割任务的各种评估指标
    
    Args:
        pred: 模型预测结果 [B, C, H, W] 或 [B, H, W]
            - 如果是4维，表示每个类别的概率分布
            - 如果是3维，表示已经是类别索引
        target: 真实标签 [B, H, W]，每个像素的类别索引
    
    Returns:
        dict: 包含各种分割指标的字典
            - pixel_accuracy: 像素级准确率
            - iou: 交并比（Intersection over Union）
            - dice_score: Dice系数，衡量重叠程度
            - precision: 精确率
            - recall: 召回率
            - f1_score: F1分数
    """
    # 如果预测结果是概率分布，转换为类别索引
    if pred.dim() == 4:  # [B, C, H, W] -> [B, H, W]
        pred = torch.argmax(pred, dim=1)  # 选择概率最大的类别
    
    # 转换为numpy数组并展平，便于计算指标
    pred_np = pred.detach().cpu().numpy().flatten()
    target_np = target.detach().cpu().numpy().flatten()
    
    # 计算像素级准确率：正确预测的像素 / 总像素数
    pixel_accuracy = accuracy_score(target_np, pred_np)
    
    # 计算IoU（交并比）：用于衡量分割质量
    # macro平均：对每个类别单独计算IoU，然后求平均
    iou = jaccard_score(target_np, pred_np, average='macro', zero_division=0)
    
    # 计算Dice系数（专门针对前景类别，即类别1）
    pred_binary = (pred_np == 1).astype(np.uint8)    # 转换为二值（前景/背景）
    target_binary = (target_np == 1).astype(np.uint8)
    
    # Dice = 2 * |A ∩ B| / (|A| + |B|)
    intersection = np.sum(pred_binary * target_binary)
    dice_score = (2.0 * intersection) / (np.sum(pred_binary) + np.sum(target_binary) + 1e-8)
    
    # 计算精确率、召回率和F1分数
    tp = np.sum(pred_binary * target_binary)          # 真正例：正确预测为前景的像素
    fp = np.sum(pred_binary * (1 - target_binary))    # 假正例：错误预测为前景的像素
    fn = np.sum((1 - pred_binary) * target_binary)    # 假负例：错误预测为背景的像素
    
    precision = tp / (tp + fp + 1e-8)                 # 精确率 = TP / (TP + FP)
    recall = tp / (tp + fn + 1e-8)                    # 召回率 = TP / (TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)  # F1分数
    
    return {
        'pixel_accuracy': pixel_accuracy,
        'iou': iou,
        'dice_score': dice_score,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }

def validate_model(net, val_loader, loss_fun, device):
    """
    验证模型在验证集上的性能
    
    Args:
        net: 训练好的神经网络模型
        val_loader: 验证集数据加载器
        loss_fun: 损失函数
        device: 计算设备（CPU或GPU）
    
    Returns:
        dict: 包含验证损失和各种指标的字典
    """
    net.eval()  # 设置模型为评估模式（关闭dropout、batch normalization等）
    val_losses = []  # 存储每个batch的验证损失
    
    # 初始化各种指标的存储列表
    val_metrics = {
        'pixel_accuracy': [],
        'iou': [],
        'dice_score': [],
        'precision': [],
        'recall': [],
        'f1_score': []
    }
    
    with torch.no_grad():  # 禁用梯度计算，节省内存并加速验证过程
        for image, segment_image in val_loader:
            # 将数据移动到指定设备（GPU或CPU）
            image = image.to(device)
            segment_image = segment_image.to(device)
            
            # 使用混合精度进行前向传播，提高计算效率
            with autocast():
                out_image = net(image)  # 模型预测
                val_loss = loss_fun(out_image, segment_image)  # 计算验证损失
            
            # 记录验证损失
            val_losses.append(val_loss.item())
            
            # 计算图像分割相关指标（IoU、Dice、精确率、召回率等）
            metrics = calculate_segmentation_metrics(out_image, segment_image)
            # 将各项指标添加到对应的列表中
            for key, value in metrics.items():
                val_metrics[key].append(value)
    
    # 计算所有batch的平均指标
    avg_metrics = {}
    for key, values in val_metrics.items():
        avg_metrics[key] = np.mean(values)
    
    # 添加平均验证损失
    avg_metrics['val_loss'] = np.mean(val_losses)
    return avg_metrics

def save_training_log(epoch, train_metrics, val_metrics, log_file):
    """
    保存训练过程的详细日志到JSON文件
    
    Args:
        epoch: 当前训练轮数
        train_metrics: 训练阶段的指标字典
        val_metrics: 验证阶段的指标字典
        log_file: 日志文件保存路径
    """
    # 构造当前epoch的日志条目
    log_entry = {
        'epoch': epoch,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),  # 添加时间戳
        'train': train_metrics,      # 训练指标
        'validation': val_metrics    # 验证指标
    }
    
    # 读取现有的训练日志（如果存在）
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
    
    # 添加新的日志条目
    logs.append(log_entry)
    
    # 保存更新后的日志
    try:
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Warning: 无法保存日志文件: {e}")

def plot_training_curves(log_file):
    """
    根据训练日志绘制训练曲线图
    包括损失曲线、IoU曲线、Dice分数曲线等
    
    Args:
        log_file: 训练日志文件路径
    """
    if not os.path.exists(log_file):
        return
    
    # 读取训练日志
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
    
    # 提取各种指标数据
    epochs = [log['epoch'] for log in logs]
    train_losses = [log['train']['loss'] for log in logs]
    val_losses = [log['validation']['val_loss'] for log in logs]
    val_ious = [log['validation']['iou'] for log in logs]
    val_dice = [log['validation']['dice_score'] for log in logs]
    
    # 创建2x2的子图布局
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 子图1：训练和验证损失曲线
    axes[0, 0].plot(epochs, train_losses, label='Train Loss', color='blue')
    axes[0, 0].plot(epochs, val_losses, label='Val Loss', color='red')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 子图2：验证IoU曲线
    axes[0, 1].plot(epochs, val_ious, label='Val IoU', color='green')
    axes[0, 1].set_title('Validation IoU')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('IoU')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 子图3：验证Dice分数曲线
    axes[1, 0].plot(epochs, val_dice, label='Val Dice Score', color='purple')
    axes[1, 0].set_title('Validation Dice Score')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Dice Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 子图4：多个验证指标对比
    val_f1 = [log['validation']['f1_score'] for log in logs]
    axes[1, 1].plot(epochs, val_ious, label='IoU', color='green')
    axes[1, 1].plot(epochs, val_dice, label='Dice', color='purple')
    axes[1, 1].plot(epochs, val_f1, label='F1', color='orange')
    axes[1, 1].set_title('Validation Metrics Comparison')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # 调整布局并保存图片
    plt.tight_layout()
    plt.savefig(os.path.join(log_path, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    print("🚀 开始训练UNet模型...")
    
    # ==================== 数据准备 ====================
    num_classes = 1 + 1  # 分割类别数：前景类 + 背景类 = 2类
    
    # 创建完整数据集
    full_dataset = MyDataset(data_path)

    # 简单的数据集分割：90%用于训练，10%用于验证
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # 创建训练数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=2,        # 批次大小：每次处理2张图片
        shuffle=True,        # 打乱数据顺序，提高训练效果，避免模型学习到数据顺序
        num_workers=2        # 数据加载的并行工作进程数
    )
    
    # 创建验证数据加载器
    val_loader = DataLoader(
        val_dataset,
        batch_size=2,        # 验证时也使用相同的批次大小
        shuffle=False,       # 验证时不需要打乱数据
        num_workers=2
    )
    
    print(f"📊 训练集样本数: {train_size}")
    print(f"📊 验证集样本数: {val_size}")
    
    # ==================== 模型和优化器设置 ====================
    # 创建UNet模型并移动到指定设备
    net = UNet(num_classes).to(device)
    
    # Adam优化器：自适应学习率，适合深度学习
    opt = optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # 学习率调度器：当验证IoU不再提升时，自动降低学习率
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt, 
        mode='max',          # 监控指标越大越好
        factor=0.5,          # 学习率衰减因子
        patience=5,          # 连续5个epoch无提升时衰减
        verbose=True         # 打印学习率变化信息
    )
    
    # 初始化混合精度训练的梯度缩放器
    scaler = GradScaler()
    
    # ==================== 训练状态初始化 ====================
    # 初始化最佳指标追踪
    best_iou = 0.0      # 历史最佳IoU
    best_dice = 0.0     # 历史最佳Dice分数
    best_epoch = 0      # 取得最佳性能的epoch
    
    start_epoch = 1     # 训练起始epoch
    training_history = {
        'best_iou': best_iou,
        'best_dice': best_dice,
        'best_epoch': best_epoch
    }
    
    # ==================== 断点续训功能 ====================
    # 尝试从之前保存的权重文件恢复训练
    if os.path.exists(weight_path):
        try:
            checkpoint = torch.load(weight_path)
            # 恢复模型权重
            net.load_state_dict(checkpoint['model_state'])
            # 恢复优化器状态
            opt.load_state_dict(checkpoint['optimizer_state'])
            # 恢复训练轮次
            start_epoch = checkpoint['epoch'] + 1
            
            # 恢复最佳性能记录（如果存在）
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

    # ==================== 损失函数和日志设置 ====================
    # 交叉熵损失：适用于多类分割任务
    loss_fun = nn.CrossEntropyLoss()
    log_file = os.path.join(log_path, 'training_log.json')
    
    print(f"🎯 训练目标: epoch {start_epoch} -> 45")
    print("="*60)

    # ==================== 主训练循环 ====================
    for epoch in range(start_epoch, 45):
        epoch_start_time = time.time()  # 记录epoch开始时间
        
        # ========== 训练阶段 ==========
        net.train()  # 设置模型为训练模式
        train_losses = []        # 存储训练损失
        train_metrics_list = []  # 存储训练指标
        
        # 创建训练进度条
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch}/44 [Train]')
        for i, (image, segment_image) in enumerate(train_pbar):
            # 将数据移动到GPU
            image = image.to(device)
            segment_image = segment_image.to(device)
            
            # 使用混合精度前向传播：提高训练速度，节省显存
            with autocast():
                out_image = net(image)  # 模型预测
                train_loss = loss_fun(out_image, segment_image)  # 计算损失
            
            # 反向传播和参数更新
            opt.zero_grad()                    # 清零梯度
            scaler.scale(train_loss).backward()  # 缩放损失并反向传播
            scaler.step(opt)                   # 更新参数
            scaler.update()                    # 更新缩放因子
            
            train_losses.append(train_loss.item())
            
            # 每10个batch计算一次训练指标（节省计算时间）
            if i % 10 == 0:
                with torch.no_grad():
                    metrics = calculate_segmentation_metrics(out_image, segment_image)
                    train_metrics_list.append(metrics)
            
            # 更新进度条显示信息
            train_pbar.set_postfix({
                'Loss': f'{train_loss.item():.4f}',
                'LR': f'{opt.param_groups[0]["lr"]:.2e}'
            })
            
            # 定期清理GPU缓存，防止内存泄漏
            if i % 20 == 0:
                torch.cuda.empty_cache()
            
            # ========== 训练可视化 ==========
            # 只保存每个epoch的第一个batch的训练过程图像
            if i == 0:
                with torch.no_grad():
                    # 获取原始图像（反标准化处理）
                    _input = image[0].cpu().float()
                    if _input.shape[0] == 3:  # RGB图像需要反标准化
                        _input = (_input * 0.5) + 0.5
                    
                    # 处理真值mask：转换为3通道便于可视化
                    _segment = segment_image[0].cpu().float()
                    _segment = _segment.unsqueeze(0)
                    _segment = torch.cat([_segment, _segment, _segment], dim=0)
                    
                    # 处理预测结果：选择概率最高的类别并转换为3通道
                    _pred = torch.argmax(out_image[0], dim=0).cpu().float()
                    _pred = _pred.unsqueeze(0)
                    _pred = torch.cat([_pred, _pred, _pred], dim=0)
                    
                    # 将输入图像、真值mask、预测结果水平拼接
                    grid = torch.cat([_input, _segment, _pred], dim=-1)
                    
                    # 保存拼接后的对比图像
                    save_image(grid, os.path.join(save_path, f'epoch{epoch}_batch0.png'))
        
        # ========== 验证阶段 ==========
        print(f"\n🔍 Epoch {epoch} - 开始验证...")
        val_metrics = validate_model(net, val_loader, loss_fun, device)
        
        # 计算训练阶段的平均指标
        avg_train_loss = np.mean(train_losses)
        avg_train_metrics = {}
        if train_metrics_list:
            for key in train_metrics_list[0].keys():
                avg_train_metrics[key] = np.mean([m[key] for m in train_metrics_list])
        avg_train_metrics['loss'] = avg_train_loss
        
        # 根据验证IoU作为指标调整学习率
        scheduler.step(val_metrics['iou'])
        
        epoch_time = time.time() - epoch_start_time  # 计算epoch总耗时
        
        # ========== 结果显示 ==========
        print(f"\n📊 Epoch {epoch} 结果总结:")
        print(f"   ⏱️  耗时: {epoch_time:.1f}s")
        print(f"   📉 训练损失: {avg_train_loss:.4f}")
        print(f"   📉 验证损失: {val_metrics['val_loss']:.4f}")
        print(f"   🎯 验证IoU: {val_metrics['iou']:.4f}")
        print(f"   🎯 验证Dice: {val_metrics['dice_score']:.4f}")
        print(f"   🎯 验证F1: {val_metrics['f1_score']:.4f}")
        print(f"   🎯 像素准确率: {val_metrics['pixel_accuracy']:.4f}")
        
        # ========== 最佳模型检测和保存 ==========
        is_best = False
        
        # 判断是否为最佳模型（主要看IoU，辅助看Dice）
        if val_metrics['iou'] > best_iou or (val_metrics['iou'] == best_iou and val_metrics['dice_score'] > best_dice):
            best_iou = val_metrics['iou']
            best_dice = val_metrics['dice_score']
            best_epoch = epoch
            is_best = True
            
            print(f"   🌟 发现更好的模型！IoU: {best_iou:.4f}, Dice: {best_dice:.4f}")
        
        # 准备检查点数据：包含模型状态、优化器状态、训练历史等
        latest_checkpoint = {
            'epoch': epoch,
            'model_state': net.state_dict(),           # 模型参数
            'optimizer_state': opt.state_dict(),       # 优化器状态
            'scheduler_state': scheduler.state_dict(), # 学习率调度器状态
            'loss': avg_train_loss,                    # 训练损失
            'val_metrics': val_metrics,                # 验证指标
            'best_iou': best_iou,                      # 历史最佳IoU
            'best_dice': best_dice,                    # 历史最佳Dice
            'best_epoch': best_epoch,                  # 最佳epoch
            'training_history': training_history       # 训练历史
        }
        
        # 保存最新模型权重（用于断点续训）
        torch.save(latest_checkpoint, weight_path)
        
        # 如果是最佳模型，额外保存一份最佳权重
        if is_best:
            torch.save(latest_checkpoint, best_weight_path)
            print(f"   💾 最佳模型已保存到: {best_weight_path}")
        
        print(f"   💾 最新模型已保存")
        
        # ========== 日志记录和可视化 ==========
        # 保存详细的训练日志
        save_training_log(epoch, avg_train_metrics, val_metrics, log_file)
        
        # 每5个epoch绘制一次训练曲线
        if epoch % 5 == 0:
            plot_training_curves(log_file)
            print(f"   📈 训练曲线已更新")
        
        # 清理GPU缓存，防止内存累积
        torch.cuda.empty_cache()
        
        print(f"   🏆 当前最佳: IoU={best_iou:.4f}, Dice={best_dice:.4f} (epoch {best_epoch})")
        print("-" * 60)
    
    # ==================== 训练完成总结 ====================
    print("\n🎉 训练完成！")
    print(f"🏆 最佳性能:")
    print(f"   IoU: {best_iou:.4f}")
    print(f"   Dice: {best_dice:.4f}")
    print(f"   最佳epoch: {best_epoch}")
    print(f"💾 最佳模型保存路径: {best_weight_path}")
    print(f"📊 训练日志保存路径: {log_file}")
    
    # 绘制最终的完整训练曲线
    plot_training_curves(log_file)
    print(f"📈 最终训练曲线: {os.path.join(log_path, 'training_curves.png')}")