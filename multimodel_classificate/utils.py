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
        self.val_losses = []
        self.val_accs = []
        self.learning_rates = []
        self.epoch_times = []
        
        # 扩展的详细指标
        self.gradient_norms = []
        self.parameter_norms = []
        self.overfitting_scores = []
        self.accuracy_differences = []
        self.loss_differences = []
        
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
        self.fig, self.axes = plt.subplots(3, 4, figsize=(24, 15))
        self.fig.suptitle('Training Process Comprehensive Monitor', fontsize=18, fontweight='bold')
        
        # 设置子图标题（使用英文避免字体问题）
        titles = [
            'Training Loss', 'Training Accuracy', 'Validation Loss', 'Validation Accuracy',
            'Learning Rate', 'Epoch Time', 'Loss Comparison', 'Accuracy Comparison', 
            'Gradient Norms', 'Parameter Norms', 'Overfitting Score', 'Performance Metrics'
        ]
        
        for i, ax in enumerate(self.axes.flat):
            ax.set_title(titles[i], fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Epoch')
        
        # 设置y轴标签
        self.axes[0, 0].set_ylabel('Training Loss')
        self.axes[0, 1].set_ylabel('Training Accuracy')
        self.axes[0, 2].set_ylabel('Validation Loss')
        self.axes[0, 3].set_ylabel('Validation Accuracy')
        self.axes[1, 0].set_ylabel('Learning Rate')
        self.axes[1, 1].set_ylabel('Time (seconds)')
        self.axes[1, 2].set_ylabel('Loss Value')
        self.axes[1, 3].set_ylabel('Accuracy Value')
        self.axes[2, 0].set_ylabel('Gradient Norm')
        self.axes[2, 1].set_ylabel('Parameter Norm')
        self.axes[2, 2].set_ylabel('Overfitting Score')
        self.axes[2, 3].set_ylabel('Difference Value')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
    
    def update(self, epoch, train_loss, train_acc, val_acc, lr, epoch_time, silent=False):
        """
        更新训练指标
        
        Args:
            epoch (int): 当前轮次
            train_loss (float): 训练损失
            train_acc (float): 训练准确率
            val_acc (float): 验证准确率
            lr (float): 学习率
            epoch_time (float): 本轮训练时间
            silent (bool): 是否静默模式（不保存图片，用于恢复历史数据）
        """
        # 添加数据
        self.epochs.append(epoch + 1)
        self.train_losses.append(train_loss)
        if train_acc is not None:
            self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)
        self.learning_rates.append(lr)
        self.epoch_times.append(epoch_time)
        
        # 只在非静默模式且指定间隔更新图表
        if not silent and (epoch + 1) % self.update_interval == 0:
            self._update_plots()
    
    def update_detailed(self, epoch, train_loss, train_acc, val_loss, val_acc, learning_rate, 
                       epoch_time, gradient_norm=None, parameter_norm=None, overfitting_score=None,
                       acc_diff=None, loss_diff=None, silent=False):
        """
        更新详细训练指标
        
        Args:
            epoch (int): 当前轮次
            train_loss (float): 训练损失
            train_acc (float): 训练准确率
            val_loss (float): 验证损失
            val_acc (float): 验证准确率
            learning_rate (float): 学习率
            epoch_time (float): 本轮训练时间
            gradient_norm (float): 梯度范数
            parameter_norm (float): 参数范数
            overfitting_score (float): 过拟合评分
            acc_diff (float): 准确率差异
            loss_diff (float): 损失差异
            silent (bool): 是否静默模式（不保存图片，用于恢复历史数据）
        """
        try:
            # 参数验证和安全处理
            epoch = int(epoch) if epoch is not None else 0
            train_loss = float(train_loss) if train_loss is not None else 0.0
            train_acc = float(train_acc) if train_acc is not None else 0.0
            val_loss = float(val_loss) if val_loss is not None else 0.0
            val_acc = float(val_acc) if val_acc is not None else 0.0
            learning_rate = float(learning_rate) if learning_rate is not None else 0.0
            epoch_time = float(epoch_time) if epoch_time is not None else 0.0
            
            # 可选参数的安全处理
            gradient_norm = float(gradient_norm) if gradient_norm is not None and gradient_norm != 0.0 else None
            parameter_norm = float(parameter_norm) if parameter_norm is not None else None
            overfitting_score = float(overfitting_score) if overfitting_score is not None else None
            acc_diff = float(acc_diff) if acc_diff is not None else None
            loss_diff = float(loss_diff) if loss_diff is not None else None
            
        except (ValueError, TypeError) as e:
            print(f"参数转换错误: {e}")
            return
            
        # 添加基础数据
        self.epochs.append(epoch + 1)
        self.train_losses.append(train_loss)
        self.train_accs.append(train_acc)
        self.val_losses.append(val_loss)
        self.val_accs.append(val_acc)
        self.learning_rates.append(learning_rate)
        self.epoch_times.append(epoch_time)
        
        # 添加详细数据（确保不超过基础数据长度）
        current_length = len(self.epochs)
        
        if gradient_norm is not None:
            self.gradient_norms.append(gradient_norm)
            # 确保长度一致
            if len(self.gradient_norms) > current_length:
                self.gradient_norms = self.gradient_norms[-current_length:]
                
        if parameter_norm is not None:
            self.parameter_norms.append(parameter_norm)
            if len(self.parameter_norms) > current_length:
                self.parameter_norms = self.parameter_norms[-current_length:]
                
        if overfitting_score is not None:
            self.overfitting_scores.append(overfitting_score)
            if len(self.overfitting_scores) > current_length:
                self.overfitting_scores = self.overfitting_scores[-current_length:]
                
        if acc_diff is not None:
            self.accuracy_differences.append(acc_diff)
            if len(self.accuracy_differences) > current_length:
                self.accuracy_differences = self.accuracy_differences[-current_length:]
                
        if loss_diff is not None:
            self.loss_differences.append(loss_diff)
            if len(self.loss_differences) > current_length:
                self.loss_differences = self.loss_differences[-current_length:]
        
        # 只在非静默模式且指定间隔更新图表
        if not silent and (epoch + 1) % self.update_interval == 0:
            try:
                self._update_detailed_plots()
            except Exception as e:
                print(f"更新详细图表时出错: {e}")
                # 降级到简单模式
                try:
                    self._update_plots()
                except Exception as e2:
                    print(f"更新简单图表也失败: {e2}")
                    # 完全忽略图表更新，继续训练
    
    def _validate_data_dimensions(self):
        """验证和修复数据维度一致性"""
        base_length = len(self.epochs)
        if base_length == 0:
            return False
            
        # 修复基础数据长度
        if len(self.train_losses) > base_length:
            self.train_losses = self.train_losses[:base_length]
        if len(self.val_accs) > base_length:
            self.val_accs = self.val_accs[:base_length]
        if len(self.learning_rates) > base_length:
            self.learning_rates = self.learning_rates[:base_length]
        if len(self.epoch_times) > base_length:
            self.epoch_times = self.epoch_times[:base_length]
            
        # 修复可选数据长度
        if len(self.train_accs) > base_length:
            self.train_accs = self.train_accs[:base_length]
        if len(self.val_losses) > base_length:
            self.val_losses = self.val_losses[:base_length]
        if len(self.gradient_norms) > base_length:
            self.gradient_norms = self.gradient_norms[:base_length]
        if len(self.parameter_norms) > base_length:
            self.parameter_norms = self.parameter_norms[:base_length]
        if len(self.overfitting_scores) > base_length:
            self.overfitting_scores = self.overfitting_scores[:base_length]
        if len(self.accuracy_differences) > base_length:
            self.accuracy_differences = self.accuracy_differences[:base_length]
        if len(self.loss_differences) > base_length:
            self.loss_differences = self.loss_differences[:base_length]
            
        return True

    def _update_plots(self):
        """更新所有子图"""
        if not self._validate_data_dimensions():
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
        if len(self.train_losses) <= len(self.epochs):
            epochs_for_train_loss = self.epochs[:len(self.train_losses)]
            self.axes[0, 0].plot(epochs_for_train_loss, self.train_losses, 'b-', linewidth=2, marker='o', markersize=4)
        self.axes[0, 0].set_ylabel('Loss')
        
        # 2. 训练准确率
        if self.train_accs and len(self.train_accs) > 0 and len(self.train_accs) <= len(self.epochs):
            valid_train_epochs = self.epochs[:len(self.train_accs)]  # 确保长度匹配
            self.axes[0, 1].plot(valid_train_epochs, self.train_accs, 'g-', linewidth=2, marker='s', markersize=4)
        self.axes[0, 1].set_ylabel('Accuracy')
        self.axes[0, 1].set_ylim(0, 1)
        
        # 3. 验证准确率
        if len(self.val_accs) <= len(self.epochs):
            epochs_for_val_acc = self.epochs[:len(self.val_accs)]
            self.axes[0, 2].plot(epochs_for_val_acc, self.val_accs, 'r-', linewidth=2, marker='^', markersize=4)
        self.axes[0, 2].set_ylabel('Accuracy')
        self.axes[0, 2].set_ylim(0, 1)
        
        # 标注最佳验证准确率
        if self.val_accs and len(self.val_accs) <= len(self.epochs):
            best_val_acc = max(self.val_accs)
            best_val_idx = self.val_accs.index(best_val_acc)
            if best_val_idx < len(self.epochs):
                best_epoch = self.epochs[best_val_idx]
                self.axes[0, 2].axhline(y=best_val_acc, color='r', linestyle='--', alpha=0.7)
                self.axes[0, 2].text(0.02, 0.98, f'Best: {best_val_acc:.4f}\n(Epoch {best_epoch})', 
                                   transform=self.axes[0, 2].transAxes, verticalalignment='top',
                                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 4. 学习率变化
        if len(self.learning_rates) <= len(self.epochs):
            epochs_for_lr = self.epochs[:len(self.learning_rates)]
            self.axes[1, 0].plot(epochs_for_lr, self.learning_rates, 'm-', linewidth=2, marker='d', markersize=4)
        self.axes[1, 0].set_ylabel('Learning Rate')
        self.axes[1, 0].set_yscale('log')
        
        # 5. 每轮训练时间
        if len(self.epoch_times) <= len(self.epochs):
            epochs_for_time = self.epochs[:len(self.epoch_times)]
            self.axes[1, 1].bar(epochs_for_time, self.epoch_times, color='orange', alpha=0.7, width=0.6)
        self.axes[1, 1].set_ylabel('Time (seconds)')
        
        # 添加平均时间线
        if self.epoch_times:
            avg_time = sum(self.epoch_times) / len(self.epoch_times)
            self.axes[1, 1].axhline(y=avg_time, color='red', linestyle='--', alpha=0.8, linewidth=2)
            self.axes[1, 1].text(0.02, 0.98, f'Avg: {avg_time:.1f}s', 
                               transform=self.axes[1, 1].transAxes, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 6. 总体对比（训练vs验证准确率）
        if self.train_accs and len(self.train_accs) > 0 and len(self.train_accs) <= len(self.epochs):
            valid_train_epochs = self.epochs[:len(self.train_accs)]  # 确保长度匹配
            self.axes[1, 2].plot(valid_train_epochs, self.train_accs, 'g-', linewidth=2, marker='s', 
                               markersize=4, label='Train Accuracy')
        if len(self.val_accs) <= len(self.epochs):
            valid_val_epochs = self.epochs[:len(self.val_accs)]
            self.axes[1, 2].plot(valid_val_epochs, self.val_accs, 'r-', linewidth=2, marker='^', 
                               markersize=4, label='Val Accuracy')
        self.axes[1, 2].set_ylabel('Accuracy')
        self.axes[1, 2].set_ylim(0, 1)
        self.axes[1, 2].legend()
        
        # 添加过拟合检测（只在有训练准确率数据时）
        if (len(self.train_accs) > 5 and len(self.val_accs) > 5 and 
            self.show_overfitting_warning and len(self.train_accs) == len(self.val_accs)):
            train_trend = np.mean(self.train_accs[-3:]) - np.mean(self.train_accs[-6:-3]) if len(self.train_accs) >= 6 else 0
            val_trend = np.mean(self.val_accs[-3:]) - np.mean(self.val_accs[-6:-3]) if len(self.val_accs) >= 6 else 0
            
            if train_trend > 0.01 and val_trend < -0.01:
                # 可能过拟合
                self.axes[1, 2].text(0.02, 0.02, 'Warning: Overfitting', 
                                   transform=self.axes[1, 2].transAxes, 
                                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    def _update_detailed_plots(self):
        """更新所有详细子图"""
        if not self._validate_data_dimensions():
            return
        
        # 清空所有子图
        for ax in self.axes.flat:
            ax.clear()
        
        # 重新设置标题和网格（使用英文）
        titles = [
            'Training Loss', 'Training Accuracy', 'Validation Loss', 'Validation Accuracy',
            'Learning Rate', 'Epoch Time', 'Loss Comparison', 'Accuracy Comparison', 
            'Gradient Norms', 'Parameter Norms', 'Overfitting Score', 'Performance Metrics'
        ]
        
        for i, ax in enumerate(self.axes.flat):
            ax.set_title(titles[i], fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Epoch')
        
        # 第一行：基础损失和准确率
        # 1. 训练损失
        if len(self.train_losses) == len(self.epochs):
            self.axes[0, 0].plot(self.epochs, self.train_losses, 'b-', linewidth=2, marker='o', markersize=3)
        self.axes[0, 0].set_ylabel('Training Loss')
        
        # 2. 训练准确率
        if self.train_accs and len(self.train_accs) <= len(self.epochs):
            epochs_for_train_acc = self.epochs[:len(self.train_accs)]
            self.axes[0, 1].plot(epochs_for_train_acc, self.train_accs, 'g-', linewidth=2, marker='s', markersize=3)
        self.axes[0, 1].set_ylabel('Training Accuracy')
        self.axes[0, 1].set_ylim(0, 1)
        
        # 3. 验证损失
        if self.val_losses and len(self.val_losses) <= len(self.epochs):
            epochs_for_val_loss = self.epochs[:len(self.val_losses)]
            self.axes[0, 2].plot(epochs_for_val_loss, self.val_losses, 'orange', linewidth=2, marker='d', markersize=3)
        self.axes[0, 2].set_ylabel('Validation Loss')
        
        # 4. 验证准确率
        if len(self.val_accs) <= len(self.epochs):
            epochs_for_val_acc = self.epochs[:len(self.val_accs)]
            self.axes[0, 3].plot(epochs_for_val_acc, self.val_accs, 'r-', linewidth=2, marker='^', markersize=3)
        self.axes[0, 3].set_ylabel('Validation Accuracy')
        self.axes[0, 3].set_ylim(0, 1)
        
        # 标注最佳验证准确率
        if self.val_accs and len(self.val_accs) == len(self.epochs):
            best_val_acc = max(self.val_accs)
            best_val_idx = self.val_accs.index(best_val_acc)
            if best_val_idx < len(self.epochs):
                best_epoch = self.epochs[best_val_idx]
                self.axes[0, 3].axhline(y=best_val_acc, color='r', linestyle='--', alpha=0.7)
                self.axes[0, 3].text(0.02, 0.98, f'Best: {best_val_acc:.4f}\n(Epoch {best_epoch})', 
                                   transform=self.axes[0, 3].transAxes, verticalalignment='top',
                                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 第二行：学习率、时间和对比
        # 5. 学习率变化
        if len(self.learning_rates) <= len(self.epochs):
            epochs_for_lr = self.epochs[:len(self.learning_rates)]
            self.axes[1, 0].plot(epochs_for_lr, self.learning_rates, 'm-', linewidth=2, marker='d', markersize=3)
        self.axes[1, 0].set_ylabel('Learning Rate')
        self.axes[1, 0].set_yscale('log')
        
        # 6. 每轮训练时间
        if len(self.epoch_times) <= len(self.epochs):
            epochs_for_time = self.epochs[:len(self.epoch_times)]
            self.axes[1, 1].bar(epochs_for_time, self.epoch_times, color='orange', alpha=0.7, width=0.6)
        self.axes[1, 1].set_ylabel('Time (seconds)')
        
        # 添加平均时间线
        if self.epoch_times:
            avg_time = sum(self.epoch_times) / len(self.epoch_times)
            self.axes[1, 1].axhline(y=avg_time, color='red', linestyle='--', alpha=0.8, linewidth=2)
            self.axes[1, 1].text(0.02, 0.98, f'Avg: {avg_time:.1f}s', 
                               transform=self.axes[1, 1].transAxes, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 7. 损失对比
        if len(self.train_losses) <= len(self.epochs):
            epochs_for_train_loss = self.epochs[:len(self.train_losses)]
            self.axes[1, 2].plot(epochs_for_train_loss, self.train_losses, 'b-', linewidth=2, marker='o', 
                               markersize=3, label='Train Loss')
        if self.val_losses and len(self.val_losses) <= len(self.epochs):
            epochs_for_val_loss = self.epochs[:len(self.val_losses)]
            self.axes[1, 2].plot(epochs_for_val_loss, self.val_losses, 'orange', linewidth=2, marker='d', 
                               markersize=3, label='Val Loss')
        self.axes[1, 2].set_ylabel('Loss Value')
        self.axes[1, 2].legend()
        
        # 8. 准确率对比
        if self.train_accs and len(self.train_accs) <= len(self.epochs):
            epochs_for_train_acc = self.epochs[:len(self.train_accs)]
            self.axes[1, 3].plot(epochs_for_train_acc, self.train_accs, 'g-', linewidth=2, marker='s', 
                               markersize=3, label='Train Accuracy')
        if len(self.val_accs) <= len(self.epochs):
            epochs_for_val_acc = self.epochs[:len(self.val_accs)]
            self.axes[1, 3].plot(epochs_for_val_acc, self.val_accs, 'r-', linewidth=2, marker='^', 
                               markersize=3, label='Val Accuracy')
        self.axes[1, 3].set_ylabel('Accuracy Value')
        self.axes[1, 3].set_ylim(0, 1)
        self.axes[1, 3].legend()
        
        # 第三行：高级指标
        # 9. 梯度范数
        if self.gradient_norms and len(self.gradient_norms) <= len(self.epochs):
            epochs_for_grads = self.epochs[:len(self.gradient_norms)]
            self.axes[2, 0].plot(epochs_for_grads, self.gradient_norms, 
                               'purple', linewidth=2, marker='*', markersize=3)
            self.axes[2, 0].set_ylabel('Gradient Norm')
            self.axes[2, 0].set_yscale('log')
        
        # 10. 参数范数
        if self.parameter_norms and len(self.parameter_norms) <= len(self.epochs):
            epochs_for_params = self.epochs[:len(self.parameter_norms)]
            self.axes[2, 1].plot(epochs_for_params, self.parameter_norms, 
                               'brown', linewidth=2, marker='x', markersize=4)
            self.axes[2, 1].set_ylabel('Parameter Norm')
        
        # 11. 过拟合评分
        if self.overfitting_scores and len(self.overfitting_scores) <= len(self.epochs):
            epochs_subset = self.epochs[:len(self.overfitting_scores)]
            self.axes[2, 2].plot(epochs_subset, self.overfitting_scores, 
                               'red', linewidth=2, marker='v', markersize=3)
            self.axes[2, 2].axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, label='Warning Line')
            self.axes[2, 2].set_ylabel('Overfitting Score')
            self.axes[2, 2].legend()
        
        # 12. 性能差异指标
        if (self.accuracy_differences and self.loss_differences and 
            len(self.accuracy_differences) <= len(self.epochs) and 
            len(self.loss_differences) <= len(self.epochs)):
            epochs_for_acc_diff = self.epochs[:len(self.accuracy_differences)]
            epochs_for_loss_diff = self.epochs[:len(self.loss_differences)]
            
            self.axes[2, 3].plot(epochs_for_acc_diff, self.accuracy_differences, 
                               'cyan', linewidth=2, marker='h', markersize=3, label='Acc Diff')
            self.axes[2, 3].plot(epochs_for_loss_diff, self.loss_differences, 
                               'magenta', linewidth=2, marker='p', markersize=3, label='Loss Diff')
            self.axes[2, 3].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            self.axes[2, 3].set_ylabel('Difference Value')
            self.axes[2, 3].legend()
        
        # 添加过拟合检测
        if (len(self.train_accs) > 5 and len(self.val_accs) > 5 and 
            self.show_overfitting_warning and len(self.train_accs) == len(self.val_accs) and
            len(self.train_accs) >= 6 and len(self.val_accs) >= 6):
            try:
                train_trend = np.mean(self.train_accs[-3:]) - np.mean(self.train_accs[-6:-3])
                val_trend = np.mean(self.val_accs[-3:]) - np.mean(self.val_accs[-6:-3])
                
                if train_trend > 0.01 and val_trend < -0.01:
                    # 可能过拟合
                    self.axes[1, 3].text(0.02, 0.02, 'Warning: Overfitting', 
                                       transform=self.axes[1, 3].transAxes, 
                                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
            except Exception as e:
                # 忽略过拟合检测错误，继续绘图
                pass
        
        # 保存图片，每次都覆盖原来的文件
        if self.save_dir:
            save_path = os.path.join(self.save_dir, 'training_progress.png')  # 固定文件名，不带epoch
            self.fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
    
    def save_final_plot(self, save_path):
        """保存最终的训练曲线图"""
        try:
            # 如果有详细数据，使用详细绘图
            if self.val_losses or self.gradient_norms or self.overfitting_scores:
                self._update_detailed_plots()
            else:
                self._update_plots()
            self.fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"最终训练曲线已保存到: {save_path}")
        except Exception as e:
            print(f"保存最终训练曲线时出错: {e}")
    
    def save_metrics_summary(self, save_path):
        """
        保存训练指标摘要到JSON文件
        
        Args:
            save_path (str): 保存路径
        """
        try:
            metrics_summary = {
                'training_summary': {
                    'total_epochs': len(self.epochs),
                    'best_validation_accuracy': max(self.val_accs) if self.val_accs else 0,
                    'final_training_loss': self.train_losses[-1] if self.train_losses else 0,
                    'final_validation_accuracy': self.val_accs[-1] if self.val_accs else 0,
                    'total_training_time': sum(self.epoch_times) if self.epoch_times else 0,
                    'average_epoch_time': sum(self.epoch_times) / len(self.epoch_times) if self.epoch_times else 0
                },
                'detailed_metrics': {
                    'epochs': self.epochs,
                    'train_losses': self.train_losses,
                    'train_accuracies': self.train_accs,
                    'val_losses': self.val_losses,
                    'val_accuracies': self.val_accs,
                    'learning_rates': self.learning_rates,
                    'epoch_times': self.epoch_times,
                    'gradient_norms': self.gradient_norms,
                    'parameter_norms': self.parameter_norms,
                    'overfitting_scores': self.overfitting_scores,
                    'accuracy_differences': self.accuracy_differences,
                    'loss_differences': self.loss_differences
                }
            }
            
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(metrics_summary, f, indent=2, ensure_ascii=False)
            print(f"训练指标摘要已保存到: {save_path}")
        except Exception as e:
            print(f"保存训练指标摘要时出错: {e}")
    
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