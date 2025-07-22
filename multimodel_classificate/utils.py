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
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches

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

class RealTimeTrainingVisualizer:
    """
    实时训练指标可视化类
    """
    def __init__(self, save_dir=None, update_interval=1, dpi=150, show_overfitting_warning=True):
        """
        初始化可视化器
        
        Args:
            save_dir (str): 图片保存目录
            update_interval (int): 更新间隔（每几个epoch更新一次）
            dpi (int): 保存图片的DPI
            show_overfitting_warning (bool): 是否显示过拟合警告
        """
        self.save_dir = save_dir
        self.update_interval = update_interval
        self.dpi = dpi
        self.show_overfitting_warning = show_overfitting_warning
        self.fig = None
        self.axes = None
        
        # 存储历史数据
        self.epochs = []
        self.train_losses = []
        self.train_accs = []
        self.val_accs = []
        self.learning_rates = []
        self.epoch_times = []
        
        # 设置matplotlib后端为Agg（不显示窗口）
        import matplotlib
        matplotlib.use('Agg')
        
        # 设置中文字体并修复减号显示问题
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False  # 使用ASCII减号而不是unicode减号
        plt.rcParams['font.family'] = 'DejaVu Sans'  # 设置默认字体
        
        self.setup_plots()
    
    def setup_plots(self):
        """设置图表布局"""
        self.fig, self.axes = plt.subplots(2, 3, figsize=(18, 10))
        self.fig.suptitle('Training Process Real-time Monitor', fontsize=16, fontweight='bold')
        
        # 设置子图标题（使用英文避免字体问题）
        titles = [
            'Training Loss', 'Training Accuracy', 'Validation Accuracy',
            'Learning Rate', 'Epoch Time', 'Accuracy Comparison'
        ]
        
        for i, ax in enumerate(self.axes.flat):
            ax.set_title(titles[i], fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Epoch')
        
        # 设置y轴标签
        self.axes[0, 0].set_ylabel('Loss')
        self.axes[0, 1].set_ylabel('Accuracy')
        self.axes[0, 2].set_ylabel('Accuracy')
        self.axes[1, 0].set_ylabel('Learning Rate')
        self.axes[1, 1].set_ylabel('Time (seconds)')
        self.axes[1, 2].set_ylabel('Value')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
    
    def update(self, epoch, train_loss, train_acc, val_acc, lr, epoch_time):
        """
        更新训练指标
        
        Args:
            epoch (int): 当前轮次
            train_loss (float): 训练损失
            train_acc (float): 训练准确率
            val_acc (float): 验证准确率
            lr (float): 学习率
            epoch_time (float): 本轮训练时间
        """
        # 添加数据
        self.epochs.append(epoch + 1)
        self.train_losses.append(train_loss)
        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)
        self.learning_rates.append(lr)
        self.epoch_times.append(epoch_time)
        
        # 只在指定间隔更新图表
        if (epoch + 1) % self.update_interval == 0:
            self._update_plots()
    
    def _update_plots(self):
        """更新所有子图"""
        if len(self.epochs) == 0:
            return
        
        # 清空所有子图
        for ax in self.axes.flat:
            ax.clear()
        
        # 重新设置标题和网格（使用英文）
        titles = [
            'Training Loss', 'Training Accuracy', 'Validation Accuracy',
            'Learning Rate', 'Epoch Time', 'Accuracy Comparison'
        ]
        
        for i, ax in enumerate(self.axes.flat):
            ax.set_title(titles[i], fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Epoch')
        
        # 1. 训练损失
        self.axes[0, 0].plot(self.epochs, self.train_losses, 'b-', linewidth=2, marker='o', markersize=4)
        self.axes[0, 0].set_ylabel('Loss')
        
        # 2. 训练准确率
        self.axes[0, 1].plot(self.epochs, self.train_accs, 'g-', linewidth=2, marker='s', markersize=4)
        self.axes[0, 1].set_ylabel('Accuracy')
        self.axes[0, 1].set_ylim(0, 1)
        
        # 3. 验证准确率
        self.axes[0, 2].plot(self.epochs, self.val_accs, 'r-', linewidth=2, marker='^', markersize=4)
        self.axes[0, 2].set_ylabel('Accuracy')
        self.axes[0, 2].set_ylim(0, 1)
        
        # 标注最佳验证准确率
        if self.val_accs:
            best_val_acc = max(self.val_accs)
            best_epoch = self.epochs[self.val_accs.index(best_val_acc)]
            self.axes[0, 2].axhline(y=best_val_acc, color='r', linestyle='--', alpha=0.7)
            self.axes[0, 2].text(0.02, 0.98, f'Best: {best_val_acc:.4f}\n(Epoch {best_epoch})', 
                               transform=self.axes[0, 2].transAxes, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 4. 学习率变化
        self.axes[1, 0].plot(self.epochs, self.learning_rates, 'm-', linewidth=2, marker='d', markersize=4)
        self.axes[1, 0].set_ylabel('Learning Rate')
        self.axes[1, 0].set_yscale('log')
        
        # 5. 每轮训练时间
        self.axes[1, 1].bar(self.epochs, self.epoch_times, color='orange', alpha=0.7, width=0.6)
        self.axes[1, 1].set_ylabel('Time (seconds)')
        
        # 添加平均时间线
        if self.epoch_times:
            avg_time = sum(self.epoch_times) / len(self.epoch_times)
            self.axes[1, 1].axhline(y=avg_time, color='red', linestyle='--', alpha=0.8, linewidth=2)
            self.axes[1, 1].text(0.02, 0.98, f'Avg: {avg_time:.1f}s', 
                               transform=self.axes[1, 1].transAxes, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 6. 总体对比（训练vs验证准确率）
        self.axes[1, 2].plot(self.epochs, self.train_accs, 'g-', linewidth=2, marker='s', 
                           markersize=4, label='Train Accuracy')
        self.axes[1, 2].plot(self.epochs, self.val_accs, 'r-', linewidth=2, marker='^', 
                           markersize=4, label='Val Accuracy')
        self.axes[1, 2].set_ylabel('Accuracy')
        self.axes[1, 2].set_ylim(0, 1)
        self.axes[1, 2].legend()
        
        # 添加过拟合检测
        if len(self.train_accs) > 5 and len(self.val_accs) > 5 and self.show_overfitting_warning:
            train_trend = np.mean(self.train_accs[-3:]) - np.mean(self.train_accs[-6:-3]) if len(self.train_accs) >= 6 else 0
            val_trend = np.mean(self.val_accs[-3:]) - np.mean(self.val_accs[-6:-3]) if len(self.val_accs) >= 6 else 0
            
            if train_trend > 0.01 and val_trend < -0.01:
                # 可能过拟合
                self.axes[1, 2].text(0.02, 0.02, 'Warning: Overfitting', 
                                   transform=self.axes[1, 2].transAxes, 
                                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        # 保存图片，每次都覆盖原来的文件
        if self.save_dir:
            save_path = os.path.join(self.save_dir, 'training_progress.png')  # 固定文件名，不带epoch
            self.fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
    
    def save_final_plot(self, save_path):
        """保存最终的训练曲线图"""
        try:
            self._update_plots()
            self.fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"最终训练曲线已保存到: {save_path}")
        except Exception as e:
            print(f"保存最终训练曲线时出错: {e}")
    
    def close(self):
        """关闭图表窗口"""
        if self.fig:
            plt.close(self.fig)
        # 不需要plt.ioff()因为我们使用的是Agg后端

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