"""
多模态分类模型训练脚本
功能：使用CLIP模型进行图像-文本多模态分类任务的训练
作者：[您的名字]
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
                   plot_training_history, print_system_info, format_time)
import os
import json
import time
from datetime import datetime
from tqdm import tqdm

def setup_logging():
    """
    设置日志系统
    返回日志文件路径
    """
    # 确保logs目录存在
    logs_dir = os.path.join(os.path.dirname(__file__), 'logs')
    ensure_dir(logs_dir)
    
    # 创建带时间戳的日志文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f'training_{timestamp}.log')
    
    return log_file

def setup_weights_dir():
    """
    设置权重保存目录
    """
    weights_dir = os.path.join(os.path.dirname(__file__), 'models', 'weights')
    ensure_dir(weights_dir)
    print(f"权重将保存到: {weights_dir}")
    return weights_dir

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
        'save_path': Config.save_path
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
    
    # 设置日志和权重目录
    log_file = setup_logging()
    setup_weights_dir()
    
    # 记录训练开始
    log_print("开始多模态分类模型训练", log_file)
    
    # 打印系统信息
    print_system_info()
    save_training_config(log_file)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_print(f"使用设备: {device}", log_file)
    
    # 初始化模型
    log_print("正在初始化模型...", log_file)
    model = CLIPFineTuner(num_classes=Config.num_classes, device=device).to(device)
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
    train_dataset = MultimodalDataset("multimodel_classificate/dataset", Config, split="train")
    val_dataset = MultimodalDataset("multimodel_classificate/dataset", Config, split="val")
    
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
    
    # 设置损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.lr)
    
    # 可选：添加学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
    
    log_print("损失函数和优化器设置完成", log_file)
    
    # 训练记录
    best_acc = 0
    training_history = {
        'train_losses': [],
        'val_accuracies': [],
        'learning_rates': [],
        'epoch_times': [],
        'best_epoch': 0,
        'best_acc': 0
    }
    
    log_print("开始训练循环", log_file)
    total_start_time = time.time()
    
    # 创建总体进度条
    epoch_pbar = tqdm(range(Config.epochs), desc="训练进度", unit="epoch")
    
    for epoch in epoch_pbar:
        epoch_start_time = time.time()
        model.train()
        total_loss = 0
        correct_train = 0
        total_train = 0
        
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
            logits = model(images, texts)
            loss = criterion(logits, labels)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 统计训练准确率
            preds = torch.argmax(logits, dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)
            
            total_loss += loss.item()
            
            # 更新进度条显示
            current_acc = correct_train / total_train
            current_loss = total_loss / (step + 1)
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.4f}',
                'Avg_Loss': f'{current_loss:.4f}'
            })
            
            # 减少日志输出频率
            if (step + 1) % 50 == 0:  # 每50个batch输出一次详细日志
                log_print(f"  [batch_size {step+1}/{len(train_loader)}] "
                         f"Loss: {loss.item():.4f}, Accuracy: {current_acc:.4f}", log_file)
        
        # 计算平均损失和训练准确率
        avg_loss = total_loss / len(train_loader)
        train_acc = correct_train / total_train
        training_history['train_losses'].append(avg_loss)
        training_history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # 验证模型
        log_print("  开始验证...", log_file)
        val_acc, val_class_acc = evaluate_detailed(model, val_loader, device, class_names, log_file)
        training_history['val_accuracies'].append(val_acc)
        
        # 更新学习率
        scheduler.step()
        
        # 计算epoch用时
        epoch_time = time.time() - epoch_start_time
        training_history['epoch_times'].append(epoch_time)
        
        # 更新总体进度条
        epoch_pbar.set_postfix({
            'Train_Acc': f'{train_acc:.4f}',
            'Val_Acc': f'{val_acc:.4f}',
            'Best_Acc': f'{best_acc:.4f}',
            'Time': f'{epoch_time:.1f}s'
        })
        
        log_print(f"第 {epoch+1} 轮完成 - 训练损失: {avg_loss:.4f}, 训练准确率: {train_acc:.4f}, "
                 f"验证准确率: {val_acc:.4f}, 用时: {format_time(epoch_time)}", log_file)
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            training_history['best_epoch'] = epoch + 1
            training_history['best_acc'] = best_acc
            
            # 保存模型权重
            torch.save(model.state_dict(), Config.save_path)
            
            # 保存模型详细信息
            additional_info = {
                'best_epoch': epoch + 1,
                'best_validation_accuracy': float(best_acc),
                'training_time_minutes': (time.time() - total_start_time) / 60,
                'class_names': class_names,
                'validation_class_accuracies': val_class_acc
            }
            save_model_info(model, Config.save_path, Config, additional_info)
            
            log_print(f"  ✓ 发现更优模型！验证准确率: {val_acc:.4f}，已保存到 {Config.save_path}", log_file)
        
        # 每5个epoch保存一次checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = Config.save_path.replace('.pt', f'_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
                'training_history': training_history
            }, checkpoint_path)
            log_print(f"  Checkpoint已保存: {checkpoint_path}", log_file)
    
    # 关闭进度条
    epoch_pbar.close()
    
    # 训练完成
    total_time = time.time() - total_start_time
    log_print("=" * 50, log_file)
    log_print("训练完成！", log_file)
    log_print(f"总训练时间: {format_time(total_time)}", log_file)
    log_print(f"最佳验证准确率: {best_acc:.4f} (第 {training_history['best_epoch']} 轮)", log_file)
    log_print(f"最佳模型保存路径: {Config.save_path}", log_file)
    
    # 保存训练历史
    history_file = log_file.replace('.log', '_history.json')
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(training_history, f, indent=2, ensure_ascii=False)
    log_print(f"训练历史已保存到: {history_file}", log_file)
    
    # 绘制训练曲线
    try:
        plot_path = log_file.replace('.log', '_curves.png')
        plot_training_history(training_history['train_losses'], 
                            training_history['val_accuracies'], 
                            plot_path)
        log_print(f"训练曲线已保存到: {plot_path}", log_file)
    except Exception as e:
        log_print(f"绘制训练曲线时出错: {e}", log_file)

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
            logits = model(images, texts)
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
            logits = model(images, texts)
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