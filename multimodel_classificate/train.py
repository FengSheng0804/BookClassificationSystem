"""
多模态分类模型训练脚本
功能：使用CLIP模型进行图像-文本多模态分类任务的训练
作者：GPJ
日期：2025-07-21
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.clip_finetune import CLIPFineTuner
from data_loader import MultimodalDataset
from config import Config
from utils import (set_seed, accuracy, ensure_dir, save_model_info, 
                   plot_training_history, print_system_info, format_time, RealTimeTrainingVisualizer)
import os
import json
import time
from datetime import datetime
from tqdm import tqdm

def setup_logging(continue_training=False, existing_log_info=None):
    """
    设置日志系统
    参数:
        continue_training: 是否从检查点继续训练
        existing_log_info: 已有的日志信息 (从检查点读取)
    返回日志文件路径和可视化文件夹路径
    """
    # 确保融合策略对应的logs目录存在
    logs_dir = Config.get_logs_path()
    ensure_dir(logs_dir)
    
    # 创建可视化文件夹
    vis_dir = os.path.join(logs_dir, 'visualizations')
    ensure_dir(vis_dir)
    
    if continue_training and existing_log_info:
        # 继续训练：使用现有的日志文件
        log_file = existing_log_info.get('log_file')
        if log_file and os.path.exists(log_file):
            print(f"📄 继续使用现有日志文件: {log_file}")
            return log_file, vis_dir
        else:
            print("⚠️  原日志文件不存在，创建新的日志文件")
    
    # 创建新的日志文件（新训练或原日志丢失）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f'training_{timestamp}.log')
    
    return log_file, vis_dir

def setup_weights_dir():
    """
    设置权重保存目录
    """
    weights_dir = Config.get_save_path()
    ensure_dir(weights_dir)
    print(f"权重将保存到: {weights_dir}")
    return weights_dir

def check_checkpoint():
    """
    检查是否存在checkpoint文件
    
    Returns:
        str: checkpoint文件路径，如果不存在则返回None
    """
    # 使用配置中的检查点路径
    checkpoint_path = Config.get_checkpoint_path()
    if os.path.exists(checkpoint_path):
        return checkpoint_path
    return None

def ask_user_continue_training(checkpoint_path):
    """
    询问用户是否要从checkpoint继续训练
    
    Args:
        checkpoint_path (str): checkpoint文件路径
        
    Returns:
        bool: True表示继续训练，False表示重新开始
    """
    print("\n" + "="*60)
    print("🔍 发现上次训练的检查点文件!")
    print(f"📁 检查点位置: {checkpoint_path}")
    
    # 尝试读取checkpoint信息
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        completed_epoch = checkpoint.get('epoch', -1)
        print(f"📊 检查点信息:")
        print(f"   - 已训练轮数: {completed_epoch + 1}")
        print(f"   - 最佳准确率: {checkpoint.get('best_acc', 0):.4f}")
        
        # 显示训练历史的最后几个epoch
        if 'training_history' in checkpoint and checkpoint['training_history']['val_accuracies']:
            recent_accs = checkpoint['training_history']['val_accuracies'][-3:]
            print(f"   - 最近验证准确率: {[f'{acc:.4f}' for acc in recent_accs]}")
            
    except Exception as e:
        print(f"⚠️  无法读取检查点详细信息: {e}")
    
    print("="*60)
    
    while True:
        choice = input("请选择操作 (c=继续训练, r=重新开始, q=退出): ").lower().strip()
        
        if choice == 'c':
            print("✅ 将从检查点继续训练")
            return True
        elif choice == 'r':
            print("🔄 将重新开始训练")
            return False
        elif choice == 'q':
            print("❌ 退出训练")
            exit(0)
        else:
            print("❗ 请输入 'c'、'r' 或 'q'")

def clean_old_checkpoint():
    """
    清理旧的checkpoint文件
    """
    checkpoint_path = Config.get_checkpoint_path()
    if os.path.exists(checkpoint_path):
        try:
            os.remove(checkpoint_path)
            print(f"🗑️  已清理旧的检查点文件: {checkpoint_path}")
        except Exception as e:
            print(f"⚠️  清理检查点文件失败: {e}")

def clean_optimizer_state(optimizer, device):
    """
    清理优化器状态中的设备问题
    
    Args:
        optimizer: 优化器对象
        device: 目标设备
    """
    try:
        # 清理参数梯度的设备
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad = p.grad.to(device)
        
        # 清理优化器状态的设备
        for param, state in optimizer.state.items():
            if not isinstance(state, dict):
                continue
                
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    # 对于step这类标量张量，始终保持在CPU上
                    # 因为我们没有设置capturable=True
                    if k in ['step']:
                        state[k] = v.cpu()
                    else:
                        # 将其他张量移动到目标设备
                        state[k] = v.to(device)
                        
    except Exception as e:
        print(f"⚠️  清理优化器状态时出错: {e}")
        print("🔄 清空优化器状态，让其重新初始化...")
        # 如果清理失败，清空优化器状态让它重新初始化
        optimizer.state.clear()

def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    """
    从checkpoint恢复训练状态
    
    Args:
        model: 模型对象
        optimizer: 优化器对象
        scheduler: 学习率调度器对象
        checkpoint_path: checkpoint文件路径
        device: 计算设备
        
    Returns:
        tuple: (起始epoch, 最佳准确率, 训练历史, 日志信息)
    """
    print(f"📥 正在加载检查点: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 恢复模型状态
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ 模型状态已恢复")
        
        # 恢复优化器状态 - 使用更安全的方法
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # 清理优化器状态中的设备问题
            clean_optimizer_state(optimizer, device)
            print("✅ 优化器状态已恢复")
        except Exception as opt_error:
            print(f"⚠️  优化器状态恢复失败: {opt_error}")
            print("🔄 将使用新的优化器状态继续训练")
        
        # 恢复学习率调度器状态
        if 'scheduler_state_dict' in checkpoint and scheduler is not None:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print("✅ 学习率调度器状态已恢复")
            except Exception as sched_error:
                print(f"⚠️  学习率调度器状态恢复失败: {sched_error}")
                print("🔄 将使用默认调度器状态")
        
        # 获取训练信息
        completed_epoch = checkpoint.get('epoch', -1)  # 已完成的epoch（0-based）
        start_epoch = completed_epoch + 1  # 下一个要开始训练的epoch
        best_acc = checkpoint.get('best_acc', 0)
        training_history = checkpoint.get('training_history', {
            'train_losses': [],
            'val_accuracies': [],
            'learning_rates': [],
            'epoch_times': [],
            'best_epoch': 0,
            'best_acc': 0
        })
        
        # 获取日志信息
        log_info = checkpoint.get('log_info', {})
        
        print(f"📈 已完成 {completed_epoch + 1} 个epoch，将从第 {start_epoch + 1} 轮开始训练")
        print(f"🏆 当前最佳准确率: {best_acc:.4f}")
        print(f"📚 已恢复 {len(training_history.get('train_losses', []))} 轮的训练历史")
        
        if log_info.get('log_file'):
            print(f"📄 原日志文件: {log_info['log_file']}")
        
        return start_epoch, best_acc, training_history, log_info
        
    except Exception as e:
        print(f"❌ 加载检查点失败: {e}")
        print("🔄 将重新开始训练")
        return 0, 0, {
            # 基础指标
            'train_losses': [],              # 训练集误差
            'train_accuracies': [],          # 训练集准确率
            'val_losses': [],                # 验证集误差
            'val_accuracies': [],            # 验证集准确率
            'learning_rates': [],            # 学习率变化
            'epoch_times': [],               # 每轮训练时间
            
            # 训练详细信息
            'train_batch_losses': [],        # 每批次训练损失
            'train_batch_accs': [],          # 每批次训练准确率
            'val_class_accuracies': [],      # 验证集各类别准确率
            'gradient_norms': [],            # 梯度范数
            'parameter_norms': [],           # 参数范数
            
            # 性能对比指标
            'accuracy_differences': [],      # 训练-验证准确率差异
            'loss_differences': [],          # 训练-验证损失差异
            'overfitting_scores': [],        # 过拟合评分
            
            # 统计信息
            'best_epoch': 0,
            'best_acc': 0,
            'worst_epoch': 0,
            'worst_acc': 1.0,
            'total_training_time': 0,
            'average_epoch_time': 0
        }, {}

def log_print(msg, log_file):
    """
    同时打印到控制台和日志文件
    
    Args:
        msg (str): 要记录的消息
        log_file (str): 日志文件路径
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_msg = f"[{timestamp}] {msg}"
    print(formatted_msg)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(formatted_msg + '\n')

def save_training_config(log_file):
    """
    保存训练配置到日志文件
    
    Args:
        log_file (str): 日志文件路径
    """
    config_info = {
        'num_classes': Config.num_classes,
        'batch_size': Config.batch_size,
        'learning_rate': Config.lr,
        'epochs': Config.epochs,
        'seed': Config.seed,
        'device': Config.device,
        'clip_model': Config.clip_model_name,
        'model_path': Config.get_model_path(),
        'fusion_strategy': Config.fusion_strategy,
        'projection_dim': Config.projection_dim,
        'attention_heads': Config.attention_heads,
        'fusion_dropout': Config.fusion_dropout
    }
    
    log_print("=" * 50, log_file)
    log_print("训练配置信息:", log_file)
    for key, value in config_info.items():
        log_print(f"  {key}: {value}", log_file)
    log_print("=" * 50, log_file)

def train():
    """
    主训练函数
    """
    # 设置随机种子
    set_seed(Config.seed)
    
    # 初始检查是否存在checkpoint以决定日志策略
    checkpoint_path = check_checkpoint()
    continue_training = False
    existing_log_info = {}
    
    if checkpoint_path:
        continue_training = ask_user_continue_training(checkpoint_path)
        if continue_training:
            # 预读取checkpoint中的日志信息
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                existing_log_info = checkpoint.get('log_info', {})
            except Exception as e:
                print(f"⚠️  预读取检查点失败: {e}")
                continue_training = False
    
    # 设置日志和权重目录
    log_file, vis_dir = setup_logging(continue_training, existing_log_info)
    setup_weights_dir()
    
    # 记录训练开始 - 如果是继续训练，添加分隔符
    if continue_training:
        log_print("=" * 80, log_file)
        log_print("🔄 从检查点继续训练", log_file)
        log_print("=" * 80, log_file)
    else:
        log_print(f"🚀 开始新的多模态分类模型训练 - 融合策略: {Config.fusion_strategy}", log_file)
    
    log_print(f"日志保存路径: {Config.get_logs_path()}", log_file)
    log_print(f"权重保存路径: {Config.get_save_path()}", log_file)
    log_print(f"可视化保存路径: {vis_dir}", log_file)
    
    # 打印系统信息
    print_system_info()
    save_training_config(log_file)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_print(f"使用设备: {device}", log_file)
    
    # 初始化模型
    log_print("正在初始化模型...", log_file)
    model = CLIPFineTuner(
        num_classes=Config.num_classes, 
        device=device,
        fusion_strategy=Config.fusion_strategy,
        projection_dim=Config.projection_dim,
        attention_heads=Config.attention_heads,
        fusion_dropout=Config.fusion_dropout
    ).to(device)
    log_print("模型初始化完成", log_file)
    
    # 打印模型信息
    model_size_info = model.get_model_size()
    log_print(f"模型参数统计:", log_file)
    log_print(f"  总参数数: {model_size_info['total_parameters']:,}", log_file)
    log_print(f"  可训练参数数: {model_size_info['trainable_parameters']:,}", log_file)
    log_print(f"  冻结参数数: {model_size_info['frozen_parameters']:,}", log_file)
    log_print(f"  模型大小: {model_size_info['model_size_mb']:.1f} MB", log_file)
    
    # 准备数据集
    log_print("正在加载数据集...", log_file)
    train_dataset = MultimodalDataset(Config.dataset_path, Config, split="train")
    val_dataset = MultimodalDataset(Config.dataset_path, Config, split="val")

    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.batch_size, 
        shuffle=True,
        num_workers=0,  # Windows上设置为0避免多进程问题
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=Config.batch_size,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    log_print(f"训练集样本数: {len(train_dataset)}", log_file)
    log_print(f"验证集样本数: {len(val_dataset)}", log_file)
    log_print(f"训练批次数: {len(train_loader)}", log_file)
    log_print(f"验证批次数: {len(val_loader)}", log_file)
    
    # 获取类别信息
    class_names = train_dataset.get_class_names()
    log_print(f"分类类别: {class_names}", log_file)
    
    # 动态计算类别权重以处理数据不平衡问题
    class_counts_tensor = train_dataset.get_class_counts_tensor()
    class_counts_dict = train_dataset.get_class_counts()
    
    # 计算类别权重（反比例权重）
    class_weights = 1.0 / class_counts_tensor.float()
    class_weights = class_weights.to(device)
    
    log_print(f"类别样本数量: {class_counts_dict}", log_file)
    log_print(f"类别权重: {dict(zip(class_names, [f'{w:.6f}' for w in class_weights.tolist()]))}", log_file)
    
    # 设置损失函数和优化器（使用加权交叉熵损失）
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # 差分学习率设置：为不同的模块设置不同的学习率
    # CLIP backbone (如果未冻结) 使用较小学习率，新增层使用标准学习率
    if hasattr(model, 'clip_model') and any(p.requires_grad for p in model.clip_model.parameters()):
        # 如果CLIP参数未冻结，使用差分学习率
        clip_params = []
        new_params = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'clip_model' in name:
                    clip_params.append(param)
                else:
                    new_params.append(param)
        
        optimizer = optim.Adam([
            {'params': clip_params, 'lr': Config.lr * 0.1},  # CLIP层使用1/10学习率
            {'params': new_params, 'lr': Config.lr}          # 新增层使用标准学习率
        ])
        log_print(f"使用差分学习率 - CLIP: {Config.lr * 0.1:.2e}, 新增层: {Config.lr:.2e}", log_file)
    else:
        # 标准优化器设置
        optimizer = optim.Adam(model.parameters(), lr=Config.lr)
        log_print(f"使用统一学习率: {Config.lr:.2e}", log_file)
    
    # 学习率调度器：余弦退火，对微调更友好
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.epochs, eta_min=Config.lr * 0.01)
    
    log_print("损失函数和优化器设置完成", log_file)
    
    # 检查是否存在checkpoint并处理训练恢复
    start_epoch = 0
    best_acc = 0
    training_history = {
        # 基础指标
        'train_losses': [],              # 训练集误差
        'train_accuracies': [],          # 训练集准确率
        'val_losses': [],                # 验证集误差
        'val_accuracies': [],            # 验证集准确率
        'learning_rates': [],            # 学习率变化
        'epoch_times': [],               # 每轮训练时间
        
        # 训练详细信息
        'train_batch_losses': [],        # 每批次训练损失
        'train_batch_accs': [],          # 每批次训练准确率
        'val_class_accuracies': [],      # 验证集各类别准确率
        'gradient_norms': [],            # 梯度范数
        'parameter_norms': [],           # 参数范数
        
        # 性能对比指标
        'accuracy_differences': [],      # 训练-验证准确率差异
        'loss_differences': [],          # 训练-验证损失差异
        'overfitting_scores': [],        # 过拟合评分
        
        # 统计信息
        'best_epoch': 0,
        'best_acc': 0,
        'worst_epoch': 0,
        'worst_acc': 1.0,
        'total_training_time': 0,
        'average_epoch_time': 0
    }
    
    if continue_training and checkpoint_path:
        # 从之前的检查中我们知道要继续训练
        start_epoch, best_acc, training_history, log_info = load_checkpoint(
            model, optimizer, scheduler, checkpoint_path, device
        )
        log_print(f"✅ 从检查点恢复训练，将从第 {start_epoch + 1} 轮开始, 最佳准确率: {best_acc:.4f}", log_file)
        log_print(f"📊 已恢复训练历史: 损失{len(training_history['train_losses'])}个点, "
                 f"准确率{len(training_history['val_accuracies'])}个点", log_file)
    elif checkpoint_path and not continue_training:
        # 用户选择重新开始
        clean_old_checkpoint()
        log_print("🔄 用户选择重新开始训练，已清理旧检查点", log_file)
    else:
        log_print("🆕 未找到检查点文件，开始新的训练", log_file)
    
    log_print("开始训练循环", log_file)
    total_start_time = time.time()
    
    # 初始化实时可视化器（可选）
    visualizer = None
    if Config.enable_visualization:
        try:
            # 使用已创建的可视化目录
            visualizer = RealTimeTrainingVisualizer(
                save_dir=vis_dir if Config.save_visualization_images else None, 
                update_interval=Config.visualization_update_interval,
                dpi=Config.visualization_dpi,
                show_overfitting_warning=Config.show_overfitting_warning
            )
            
            # 如果是继续训练，恢复之前的训练数据到可视化器
            if continue_training and training_history['train_losses']:
                log_print("📈 正在恢复可视化器的训练历史数据...", log_file)
                try:
                    # 将之前的训练历史传递给可视化器
                    num_points = len(training_history['train_losses'])
                    for i in range(num_points):
                        loss = training_history['train_losses'][i]
                        val_acc = training_history['val_accuracies'][i] if i < len(training_history['val_accuracies']) else 0
                        lr = training_history['learning_rates'][i] if i < len(training_history['learning_rates']) else Config.lr
                        epoch_time = training_history['epoch_times'][i] if i < len(training_history['epoch_times']) else 0
                        
                        # 注意：由于历史数据中没有保存每个epoch的训练准确率，我们传入None
                        # epoch索引使用实际的历史epoch（从0开始）
                        visualizer.update(i, loss, None, val_acc, lr, epoch_time, silent=True)
                    
                    log_print(f"✅ 已恢复 {num_points} 个历史数据点到可视化器", log_file)
                except Exception as restore_error:
                    log_print(f"⚠️  恢复可视化历史数据失败: {restore_error}", log_file)
            
            log_print(f"✅ 实时可视化已启用，更新间隔: {Config.visualization_update_interval} epoch", log_file)
            if Config.save_visualization_images:
                log_print(f"📷 可视化图片保存到: {vis_dir}", log_file)
        except Exception as e:
            log_print(f"❌ 初始化可视化失败: {e}，将继续训练但不显示可视化", log_file)
            visualizer = None
    else:
        log_print("⚪ 实时可视化已禁用", log_file)
    
    # 创建总体进度条
    remaining_epochs = Config.epochs - start_epoch
    epoch_pbar = tqdm(range(start_epoch, Config.epochs), desc="训练进度", unit="epoch")
    
    for epoch in epoch_pbar:
        epoch_start_time = time.time()
        model.train()
        total_loss = 0
        correct_train = 0
        total_train = 0
        
        # 用于记录每个批次的详细指标
        epoch_batch_losses = []
        epoch_batch_accs = []
        epoch_gradient_norms = []
        
        log_print(f"第 {epoch+1}/{Config.epochs} 轮训练", log_file)
        log_print(f"当前学习率: {optimizer.param_groups[0]['lr']:.6f}", log_file)
        
        # 创建批次进度条
        train_pbar = tqdm(
            enumerate(train_loader), 
            total=len(train_loader),
            desc=f"Epoch {epoch+1}",
            unit="batch",
            leave=False
        )
        
        for step, (images, texts, labels) in train_pbar:
            # 数据转移到设备
            images, texts, labels = images.to(device), texts.to(device), labels.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            model_output = model(images, texts)
            
            # 处理不同融合策略的返回值
            if isinstance(model_output, tuple):
                logits, fusion_info = model_output
            else:
                logits = model_output
                fusion_info = None
                
            loss = criterion(logits, labels)
            
            # 反向传播
            loss.backward()
            
            # 计算梯度范数（在梯度裁剪之前）
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            gradient_norm = total_norm ** (1. / 2)
            epoch_gradient_norms.append(gradient_norm)
            
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 统计训练准确率
            preds = torch.argmax(logits, dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)
            
            total_loss += loss.item()
            
            # 记录批次级别的指标
            batch_acc = (preds == labels).float().mean().item()
            epoch_batch_losses.append(loss.item())
            epoch_batch_accs.append(batch_acc)
            
            # 更新进度条显示
            current_acc = correct_train / total_train
            current_loss = total_loss / (step + 1)
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.4f}',
                'Avg_Loss': f'{current_loss:.4f}',
                'Grad_Norm': f'{gradient_norm:.3f}'
            })
            
            # 减少日志输出频率
            if (step + 1) % 50 == 0:  # 每50个batch输出一次详细日志
                log_print(f"  [batch {step+1}/{len(train_loader)}] "
                         f"Loss: {loss.item():.4f}, Accuracy: {current_acc:.4f}, "
                         f"Gradient Norm: {gradient_norm:.4f}", log_file)
        
        # 计算平均损失和训练准确率
        avg_loss = total_loss / len(train_loader)
        train_acc = correct_train / total_train
        
        # 计算参数范数
        total_param_norm = 0
        for p in model.parameters():
            param_norm = p.data.norm(2)
            total_param_norm += param_norm.item() ** 2
        parameter_norm = total_param_norm ** (1. / 2)
        
        # 验证模型并计算验证损失
        log_print("  开始验证...", log_file)
        val_acc, val_loss, val_class_acc = evaluate_detailed_with_loss(
            model, val_loader, criterion, device, class_names, log_file
        )
        
        # 记录基础指标
        training_history['train_losses'].append(avg_loss)
        training_history['train_accuracies'].append(train_acc)
        training_history['val_losses'].append(val_loss)
        training_history['val_accuracies'].append(val_acc)
        training_history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # 记录详细指标
        training_history['train_batch_losses'].append(epoch_batch_losses)
        training_history['train_batch_accs'].append(epoch_batch_accs)
        training_history['val_class_accuracies'].append(val_class_acc)
        training_history['gradient_norms'].append(epoch_gradient_norms)
        training_history['parameter_norms'].append(parameter_norm)
        
        # 计算性能对比指标
        acc_diff = train_acc - val_acc  # 训练准确率与验证准确率的差异
        loss_diff = val_loss - avg_loss  # 验证损失与训练损失的差异
        overfitting_score = max(0, acc_diff * 2 + loss_diff)  # 过拟合评分
        
        training_history['accuracy_differences'].append(acc_diff)
        training_history['loss_differences'].append(loss_diff)
        training_history['overfitting_scores'].append(overfitting_score)
        
        # 更新学习率
        scheduler.step()
        
        # 计算epoch用时
        epoch_time = time.time() - epoch_start_time
        training_history['epoch_times'].append(epoch_time)
        
        # 更新总体进度条
        epoch_pbar.set_postfix({
            'Train_Loss': f'{avg_loss:.4f}',
            'Train_Acc': f'{train_acc:.4f}',
            'Val_Loss': f'{val_loss:.4f}',
            'Val_Acc': f'{val_acc:.4f}',
            'Best_Acc': f'{best_acc:.4f}',
            'Overfit': f'{overfitting_score:.3f}',
            'Time': f'{epoch_time:.1f}s'
        })
        
        # 详细的日志输出
        log_print(f"第 {epoch+1} 轮完成:", log_file)
        log_print(f"  训练: 损失={avg_loss:.4f}, 准确率={train_acc:.4f}", log_file)
        log_print(f"  验证: 损失={val_loss:.4f}, 准确率={val_acc:.4f}", log_file)
        log_print(f"  性能对比: 准确率差异={acc_diff:.4f}, 损失差异={loss_diff:.4f}", log_file)
        log_print(f"  过拟合评分: {overfitting_score:.4f}", log_file)
        log_print(f"  参数范数: {parameter_norm:.4f}", log_file)
        if epoch_gradient_norms:
            avg_gradient_norm = sum(epoch_gradient_norms)/len(epoch_gradient_norms)
            log_print(f"  平均梯度范数: {avg_gradient_norm:.4f}", log_file)
        else:
            log_print(f"  平均梯度范数: 0.0000 (无数据)", log_file)
        log_print(f"  用时: {format_time(epoch_time)}, 学习率: {optimizer.param_groups[0]['lr']:.6f}", log_file)
        
        # 过拟合警告
        if overfitting_score > 0.1:
            log_print(f"  ⚠️  过拟合警告: 评分 {overfitting_score:.4f} (训练准确率过高)", log_file)
        elif acc_diff < -0.05:
            log_print(f"  ⚠️  欠拟合警告: 验证准确率显著高于训练准确率", log_file)
        
        # 更新最差和最佳记录
        if val_acc < training_history['worst_acc']:
            training_history['worst_acc'] = val_acc
            training_history['worst_epoch'] = epoch + 1
        
        # 更新实时可视化（如果启用）
        if visualizer:
            try:
                # 计算梯度范数（安全处理空列表）
                avg_gradient_norm = sum(epoch_gradient_norms)/len(epoch_gradient_norms) if epoch_gradient_norms else 0.0
                
                # 传递更多数据给可视化器
                visualizer.update_detailed(
                    epoch=epoch, 
                    train_loss=avg_loss,
                    train_acc=train_acc,
                    val_loss=val_loss,
                    val_acc=val_acc, 
                    learning_rate=optimizer.param_groups[0]['lr'],
                    epoch_time=epoch_time,
                    gradient_norm=avg_gradient_norm,
                    parameter_norm=parameter_norm,
                    overfitting_score=overfitting_score,
                    acc_diff=acc_diff,
                    loss_diff=loss_diff
                )
            except Exception as e:
                # 如果update_detailed方法不存在，使用原来的方法
                try:
                    visualizer.update(epoch, avg_loss, train_acc, val_acc, 
                                    optimizer.param_groups[0]['lr'], epoch_time)
                except Exception as e2:
                    log_print(f"更新可视化时出错: {e2}", log_file)
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            training_history['best_epoch'] = epoch + 1
            training_history['best_acc'] = best_acc
            
            # 保存模型权重
            torch.save(model.state_dict(), Config.get_model_path())
            
            # 保存模型详细信息
            additional_info = {
                'best_epoch': epoch + 1,
                'best_validation_accuracy': float(best_acc),
                'training_time_minutes': (time.time() - total_start_time) / 60,
                'class_names': class_names,
                'validation_class_accuracies': val_class_acc
            }
            save_model_info(model, Config.get_model_path(), Config, additional_info)
            
            log_print(f"  ✓ 发现更优模型！验证准确率: {val_acc:.4f}，已保存到 {Config.get_model_path()}", log_file)
        
        # 每个epoch都保存最新的checkpoint
        checkpoint_path = Config.get_checkpoint_path()
        checkpoint_data = {
            'epoch': epoch,  # 保存当前epoch
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_acc,
            'training_history': training_history,
            'config_info': {
                'fusion_strategy': Config.fusion_strategy,
                'num_classes': Config.num_classes,
                'batch_size': Config.batch_size,
                'learning_rate': Config.lr
            },
            'log_info': {
                'log_file': log_file,
                'vis_dir': vis_dir,
                'save_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        torch.save(checkpoint_data, checkpoint_path)
        
        # 记录日志
        log_print(f"  最新Checkpoint已更新: {checkpoint_path}", log_file)
    
    # 关闭进度条
    epoch_pbar.close()
    
    # 保存最终的可视化图表（如果启用）
    if visualizer:
        try:
            if Config.save_visualization_images:
                final_plot_path = os.path.join(vis_dir, 'final_training_curves.png')
                visualizer.save_final_plot(final_plot_path)
                log_print(f"最终训练曲线已保存到: {final_plot_path}", log_file)
                
                # 保存详细的指标摘要
                metrics_summary_path = os.path.join(vis_dir, 'training_metrics_summary.json')
                visualizer.save_metrics_summary(metrics_summary_path)
                log_print(f"训练指标摘要已保存到: {metrics_summary_path}", log_file)
        except Exception as e:
            log_print(f"保存最终可视化时出错: {e}", log_file)
        finally:
            # 关闭可视化窗口
            visualizer.close()
    
    # 训练完成
    total_time = time.time() - total_start_time
    
    # 计算统计信息
    training_history['total_training_time'] = total_time
    training_history['average_epoch_time'] = total_time / (Config.epochs - start_epoch)
    
    # 找出最佳和最差的epoch
    if training_history['val_accuracies']:
        best_val_acc = max(training_history['val_accuracies'])
        best_idx = training_history['val_accuracies'].index(best_val_acc)
        training_history['best_epoch'] = best_idx + 1 + start_epoch
        training_history['best_acc'] = best_val_acc
        
        worst_val_acc = min(training_history['val_accuracies'])
        worst_idx = training_history['val_accuracies'].index(worst_val_acc)
        training_history['worst_epoch'] = worst_idx + 1 + start_epoch
        training_history['worst_acc'] = worst_val_acc
    
    log_print("=" * 80, log_file)
    log_print("🎉 训练完成！", log_file)
    log_print("=" * 80, log_file)
    
    # 详细的训练总结
    log_print("📊 训练总结:", log_file)
    log_print(f"  总训练时间: {format_time(total_time)}", log_file)
    log_print(f"  平均每轮时间: {format_time(training_history['average_epoch_time'])}", log_file)
    log_print(f"  训练轮数: {Config.epochs - start_epoch} (从第{start_epoch+1}轮到第{Config.epochs}轮)", log_file)
    
    log_print("🏆 性能指标:", log_file)
    log_print(f"  最佳验证准确率: {training_history['best_acc']:.4f} (第 {training_history['best_epoch']} 轮)", log_file)
    log_print(f"  最差验证准确率: {training_history['worst_acc']:.4f} (第 {training_history['worst_epoch']} 轮)", log_file)
    log_print(f"  最终训练准确率: {training_history['train_accuracies'][-1]:.4f}", log_file)
    log_print(f"  最终验证准确率: {training_history['val_accuracies'][-1]:.4f}", log_file)
    log_print(f"  最终训练损失: {training_history['train_losses'][-1]:.4f}", log_file)
    log_print(f"  最终验证损失: {training_history['val_losses'][-1]:.4f}", log_file)
    
    # 过拟合分析
    if training_history['overfitting_scores']:
        avg_overfit = sum(training_history['overfitting_scores']) / len(training_history['overfitting_scores'])
        max_overfit = max(training_history['overfitting_scores'])
        max_overfit_epoch = training_history['overfitting_scores'].index(max_overfit) + 1 + start_epoch
        
        log_print("🔍 过拟合分析:", log_file)
        log_print(f"  平均过拟合评分: {avg_overfit:.4f}", log_file)
        log_print(f"  最大过拟合评分: {max_overfit:.4f} (第 {max_overfit_epoch} 轮)", log_file)
        
        if avg_overfit > 0.15:
            log_print("  ⚠️  模型存在明显过拟合，建议增加正则化或减少模型复杂度", log_file)
        elif avg_overfit < 0.05:
            log_print("  ✅ 模型拟合良好，无明显过拟合", log_file)
        else:
            log_print("  ⚡ 模型拟合适中，建议继续监控", log_file)
    
    # 学习率分析
    if training_history['learning_rates']:
        initial_lr = training_history['learning_rates'][0]
        final_lr = training_history['learning_rates'][-1]
        lr_decay_ratio = final_lr / initial_lr if initial_lr > 0 else 1.0
        
        log_print("📈 学习率分析:", log_file)
        log_print(f"  初始学习率: {initial_lr:.6f}", log_file)
        log_print(f"  最终学习率: {final_lr:.6f}", log_file)
        log_print(f"  衰减比例: {lr_decay_ratio:.4f}", log_file)
    
    log_print(f"🎯 最佳模型保存路径: {Config.get_model_path()}", log_file)
    log_print("=" * 80, log_file)
    
    # 保存训练历史（包含所有详细指标）
    history_file = log_file.replace('.log', '_history.json')
    
    # 增强的训练历史，包含更多统计信息
    enhanced_history = training_history.copy()
    enhanced_history.update({
        'training_config': {
            'fusion_strategy': Config.fusion_strategy,
            'num_classes': Config.num_classes,
            'batch_size': Config.batch_size,
            'learning_rate': Config.lr,
            'epochs': Config.epochs,
            'seed': Config.seed,
            'clip_model': Config.clip_model_name
        },
        'performance_summary': {
            'best_validation_accuracy': training_history['best_acc'],
            'best_epoch': training_history['best_epoch'],
            'worst_validation_accuracy': training_history['worst_acc'],
            'worst_epoch': training_history['worst_epoch'],
            'final_train_accuracy': training_history['train_accuracies'][-1] if training_history['train_accuracies'] else 0,
            'final_val_accuracy': training_history['val_accuracies'][-1] if training_history['val_accuracies'] else 0,
            'final_train_loss': training_history['train_losses'][-1] if training_history['train_losses'] else 0,
            'final_val_loss': training_history['val_losses'][-1] if training_history['val_losses'] else 0,
            'average_overfitting_score': sum(training_history['overfitting_scores']) / len(training_history['overfitting_scores']) if training_history['overfitting_scores'] else 0,
            'max_overfitting_score': max(training_history['overfitting_scores']) if training_history['overfitting_scores'] else 0
        },
        'class_names': class_names,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(enhanced_history, f, indent=2, ensure_ascii=False)
    log_print(f"增强训练历史已保存到: {history_file}", log_file)
    
    # 绘制训练曲线
    try:
        plot_path = log_file.replace('.log', '_curves.png')
        plot_training_history(training_history['train_losses'], 
                            training_history['val_accuracies'], 
                            plot_path)
        log_print(f"训练曲线已保存到: {plot_path}", log_file)
    except Exception as e:
        log_print(f"绘制训练曲线时出错: {e}", log_file)

def evaluate_detailed_with_loss(model, loader, criterion, device, class_names, log_file=None):
    """
    详细评估模型性能，包括损失和每个类别的准确率
    
    Args:
        model: 要评估的模型
        loader: 数据加载器
        criterion: 损失函数
        device: 计算设备
        class_names: 类别名称列表
        log_file: 日志文件路径（可选）
    
    Returns:
        tuple: (总体准确率, 平均损失, 各类别准确率字典)
    """
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    class_correct = torch.zeros(len(class_names))
    class_total = torch.zeros(len(class_names))
    
    # 创建验证进度条
    val_pbar = tqdm(loader, desc="验证中", unit="batch", leave=False)
    
    with torch.no_grad():
        for images, texts, labels in val_pbar:
            images, texts, labels = images.to(device), texts.to(device), labels.to(device)
            
            model_output = model(images, texts)
            
            # 处理不同融合策略的返回值
            if isinstance(model_output, tuple):
                logits, fusion_info = model_output
            else:
                logits = model_output
                fusion_info = None
                
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # 统计各类别准确率
            for i in range(len(labels)):
                label = labels[i].item()
                class_total[label] += 1
                if preds[i] == labels[i]:
                    class_correct[label] += 1
            
            # 更新进度条
            current_acc = correct / total if total > 0 else 0
            current_loss = total_loss / (val_pbar.n + 1)
            val_pbar.set_postfix({'Acc': f'{current_acc:.4f}', 'Loss': f'{current_loss:.4f}'})
    
    overall_acc = correct / total if total > 0 else 0
    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0
    
    # 计算各类别准确率
    class_accuracies = {}
    if log_file:
        log_print(f"    总体评估结果: {correct}/{total} = {overall_acc:.4f}, 损失: {avg_loss:.4f}", log_file)
        log_print(f"    各类别准确率:", log_file)
    
    for i, class_name in enumerate(class_names):
        if class_total[i] > 0:
            class_acc = class_correct[i] / class_total[i]
            class_accuracies[class_name] = float(class_acc)
            if log_file:
                log_print(f"      {class_name}: {class_acc:.4f} ({int(class_correct[i])}/{int(class_total[i])})", log_file)
        else:
            class_accuracies[class_name] = 0.0
            if log_file:
                log_print(f"      {class_name}: N/A (0 样本)", log_file)
    
    return overall_acc, avg_loss, class_accuracies

def evaluate_detailed(model, loader, device, class_names, log_file=None):
    """
    详细评估模型性能，包括每个类别的准确率
    
    Args:
        model: 要评估的模型
        loader: 数据加载器
        device: 计算设备
        class_names: 类别名称列表
        log_file: 日志文件路径（可选）
    
    Returns:
        tuple: (总体准确率, 各类别准确率字典)
    """
    model.eval()
    correct = 0
    total = 0
    class_correct = torch.zeros(len(class_names))
    class_total = torch.zeros(len(class_names))
    
    # 创建验证进度条
    val_pbar = tqdm(loader, desc="验证中", unit="batch", leave=False)
    
    with torch.no_grad():
        for images, texts, labels in val_pbar:
            images, texts, labels = images.to(device), texts.to(device), labels.to(device)
            
            model_output = model(images, texts)
            
            # 处理不同融合策略的返回值
            if isinstance(model_output, tuple):
                logits, fusion_info = model_output
            else:
                logits = model_output
                fusion_info = None
                
            preds = torch.argmax(logits, dim=1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # 统计各类别准确率
            for i in range(len(labels)):
                label = labels[i].item()
                class_total[label] += 1
                if preds[i] == labels[i]:
                    class_correct[label] += 1
            
            # 更新进度条
            current_acc = correct / total if total > 0 else 0
            val_pbar.set_postfix({'Acc': f'{current_acc:.4f}'})
    
    overall_acc = correct / total if total > 0 else 0
    
    # 计算各类别准确率
    class_accuracies = {}
    if log_file:
        log_print(f"    总体评估结果: {correct}/{total} = {overall_acc:.4f}", log_file)
        log_print(f"    各类别准确率:", log_file)
    
    for i, class_name in enumerate(class_names):
        if class_total[i] > 0:
            class_acc = class_correct[i] / class_total[i]
            class_accuracies[class_name] = float(class_acc)
            if log_file:
                log_print(f"      {class_name}: {class_acc:.4f} ({int(class_correct[i])}/{int(class_total[i])})", log_file)
        else:
            class_accuracies[class_name] = 0.0
            if log_file:
                log_print(f"      {class_name}: N/A (0 样本)", log_file)
    
    return overall_acc, class_accuracies

def evaluate(model, loader, device, log_file=None):
    """
    简单评估模型性能（保持向后兼容）
    
    Args:
        model: 要评估的模型
        loader: 数据加载器
        device: 计算设备
        log_file: 日志文件路径（可选）
    
    Returns:
        float: 准确率
    """
    model.eval()
    correct, total = 0, 0
    
    # 创建评估进度条
    eval_pbar = tqdm(loader, desc="评估中", unit="batch", leave=False)
    
    with torch.no_grad():
        for images, texts, labels in eval_pbar:
            images, texts, labels = images.to(device), texts.to(device), labels.to(device)
            
            model_output = model(images, texts)
            
            # 处理不同融合策略的返回值
            if isinstance(model_output, tuple):
                logits, fusion_info = model_output
            else:
                logits = model_output
                fusion_info = None
                
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # 更新进度条
            current_acc = correct / total if total > 0 else 0
            eval_pbar.set_postfix({'Acc': f'{current_acc:.4f}'})
    
    acc = correct / total if total > 0 else 0
    
    if log_file:
        log_print(f"    评估结果: {correct}/{total} = {acc:.4f}", log_file)
    
    return acc

if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc() 