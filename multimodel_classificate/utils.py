"""
实用工具函数模块
包含训练和评估过程中需要的各种工具函数
"""

import random
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from datetime import datetime
import json

def set_seed(seed):
    """
    设置随机种子以确保实验的可重复性
    
    Args:
        seed (int): 随机种子值
    """
    print(f"设置随机种子: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # 确保CUDA计算的确定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def accuracy(preds, labels):
    """
    计算预测准确率
    
    Args:
        preds (torch.Tensor): 预测结果
        labels (torch.Tensor): 真实标签
        
    Returns:
        float: 准确率 (0-1之间)
    """
    return (preds == labels).sum().item() / len(labels)

def count_parameters(model):
    """
    计算模型的参数数量
    
    Args:
        model (torch.nn.Module): 模型
        
    Returns:
        tuple: (总参数数, 可训练参数数)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def format_time(seconds):
    """
    将秒数格式化为易读的时间字符串
    
    Args:
        seconds (float): 秒数
        
    Returns:
        str: 格式化的时间字符串
    """
    if seconds < 60:
        return f"{seconds:.2f}秒"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)}分{secs:.0f}秒"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{int(hours)}小时{int(minutes)}分"

def ensure_dir(dir_path):
    """
    确保目录存在，如果不存在则创建
    
    Args:
        dir_path (str): 目录路径
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        print(f"创建目录: {dir_path}")

def save_model_info(model, save_path, config, additional_info=None):
    """
    保存模型的详细信息
    
    Args:
        model (torch.nn.Module): 模型
        save_path (str): 模型保存路径
        config: 配置对象
        additional_info (dict): 额外信息
    """
    total_params, trainable_params = count_parameters(model)
    
    model_info = {
        'model_path': save_path,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_architecture': str(model),
        'config': {
            'num_classes': config.num_classes,
            'batch_size': config.batch_size,
            'learning_rate': config.lr,
            'epochs': config.epochs,
            'clip_model': config.clip_model_name,
            'device': config.device
        },
        'save_time': datetime.now().isoformat()
    }
    
    if additional_info:
        model_info.update(additional_info)
    
    info_path = save_path.replace('.pt', '_info.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)
    
    print(f"模型信息已保存到: {info_path}")
    return info_path

def plot_training_history(train_losses, val_accuracies, save_path=None):
    """
    绘制训练历史曲线
    
    Args:
        train_losses (list): 训练损失列表
        val_accuracies (list): 验证准确率列表
        save_path (str): 保存路径（可选）
    """
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # 绘制训练损失
        ax1.plot(train_losses, 'b-', label='训练损失')
        ax1.set_title('训练损失曲线')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 绘制验证准确率
        ax2.plot(val_accuracies, 'r-', label='验证准确率')
        ax2.set_title('验证准确率曲线')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"训练曲线已保存到: {save_path}")
        else:
            plt.show()
            
        plt.close()
        
    except Exception as e:
        print(f"绘制训练曲线时出错: {e}")

def get_device_info():
    """
    获取计算设备信息
    
    Returns:
        dict: 设备信息字典
    """
    device_info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': torch.cuda.current_device() if torch.cuda.is_available() else None,
    }
    
    if torch.cuda.is_available():
        device_info['device_name'] = torch.cuda.get_device_name(0)
        device_info['memory_total'] = torch.cuda.get_device_properties(0).total_memory
        device_info['memory_reserved'] = torch.cuda.memory_reserved(0)
        device_info['memory_allocated'] = torch.cuda.memory_allocated(0)
    
    return device_info

def print_system_info():
    """
    打印系统和环境信息
    """
    print("=" * 50)
    print("系统信息")
    print("=" * 50)
    print(f"PyTorch版本: {torch.__version__}")
    print(f"Python版本: {os.sys.version}")
    
    device_info = get_device_info()
    print(f"CUDA可用: {device_info['cuda_available']}")
    
    if device_info['cuda_available']:
        print(f"GPU设备: {device_info['device_name']}")
        print(f"GPU数量: {device_info['device_count']}")
        total_mem = device_info['memory_total'] / 1024**3
        print(f"GPU内存: {total_mem:.2f} GB")
    
    print("=" * 50)

def calculate_class_weights(dataset, num_classes):
    """
    计算类别权重用于处理不平衡数据集
    
    Args:
        dataset: 数据集对象
        num_classes (int): 类别数量
        
    Returns:
        torch.Tensor: 类别权重张量
    """
    # 统计每个类别的样本数
    class_counts = torch.zeros(num_classes)
    
    for _, _, label in dataset.samples:
        class_counts[label] += 1
    
    # 计算权重 (总样本数 / (类别数 * 该类别样本数))
    total_samples = len(dataset.samples)
    class_weights = total_samples / (num_classes * class_counts)
    
    # 避免除零错误
    class_weights = torch.where(class_counts > 0, class_weights, torch.ones_like(class_weights))
    
    print("类别权重:")
    class_names = dataset.get_class_names()
    for i, (name, weight, count) in enumerate(zip(class_names, class_weights, class_counts)):
        print(f"  {name}: {weight:.4f} (样本数: {int(count)})")
    
    return class_weights 