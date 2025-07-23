"""
多模态分类模型评估脚本
功能：加载训练好的模型并在测试集上进行性能评估
输出详细的评估报告和混淆矩阵
"""

import torch
from torch.utils.data import DataLoader
from models.clip_finetune import CLIPFineTuner
from data_loader import MultimodalDataset
from config import Config
from utils import set_seed, accuracy
import numpy as np
import os
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

def log_print(msg, log_file=None):
    """
    同时打印到控制台和日志文件
    
    Args:
        msg (str): 要记录的消息
        log_file (str): 日志文件路径（可选）
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_msg = f"[{timestamp}] {msg}"
    print(formatted_msg)
    
    if log_file:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(formatted_msg + '\n')

def setup_evaluation_logging():
    """
    设置评估日志系统
    
    Returns:
        tuple: (日志文件路径, 结果保存目录)
    """
    # 根据融合策略创建对应的结果目录
    base_result_dir = os.path.join(os.path.dirname(__file__), 'test_result')
    strategy_result_dir = os.path.join(base_result_dir, Config.fusion_strategy)
    os.makedirs(strategy_result_dir, exist_ok=True)
    
    # 创建带时间戳的评估日志文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(strategy_result_dir, f'evaluation_{timestamp}.log')
    
    return log_file, strategy_result_dir

def plot_confusion_matrix(cm, class_names, save_path, title='混淆矩阵'):
    """
    绘制并保存混淆矩阵
    
    Args:
        cm: 混淆矩阵
        class_names: 类别名称列表
        save_path: 保存路径
        title: 图表标题
    """
    plt.figure(figsize=(10, 8))
    
    # 使用seaborn绘制热力图
    sns.heatmap(cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': '样本数量'})
    
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel('预测标签', fontsize=12)
    plt.ylabel('真实标签', fontsize=12)
    
    # 旋转x轴标签以避免重叠
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形以释放内存
    
    return save_path

def plot_classification_metrics(cm, class_names, save_path):
    """
    绘制分类指标图表
    
    Args:
        cm: 混淆矩阵
        class_names: 类别名称列表
        save_path: 保存路径
    """
    # 计算每个类别的指标
    precision = cm.diagonal() / cm.sum(axis=0)
    recall = cm.diagonal() / cm.sum(axis=1)
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    # 处理可能的除零错误
    precision = np.nan_to_num(precision)
    recall = np.nan_to_num(recall)
    f1_score = np.nan_to_num(f1_score)
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 左侧：各类别指标柱状图
    x = np.arange(len(class_names))
    width = 0.25
    
    ax1.bar(x - width, precision, width, label='精确率', alpha=0.8)
    ax1.bar(x, recall, width, label='召回率', alpha=0.8)
    ax1.bar(x + width, f1_score, width, label='F1分数', alpha=0.8)
    
    ax1.set_xlabel('类别')
    ax1.set_ylabel('分数')
    ax1.set_title('各类别分类指标')
    ax1.set_xticks(x)
    ax1.set_xticklabels(class_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.1)
    
    # 右侧：准确率饼图
    support = cm.sum(axis=1)
    colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
    
    ax2.pie(support, labels=class_names, autopct='%1.1f%%', colors=colors)
    ax2.set_title('各类别样本分布')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path

def evaluate():
    """
    主评估函数
    """
    # 设置随机种子
    set_seed(Config.seed)
    
    # 设置日志和结果目录
    log_file, result_dir = setup_evaluation_logging()
    log_print("开始模型评估", log_file)
    log_print(f"融合策略: {Config.fusion_strategy}", log_file)
    log_print(f"结果保存目录: {result_dir}", log_file)
    
    # 检查模型文件是否存在
    model_path = Config.get_model_path()
    if not os.path.exists(model_path):
        log_print(f"错误: 找不到模型文件 {model_path}", log_file)
        log_print("请先运行 train.py 训练模型", log_file)
        return
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_print(f"使用设备: {device}", log_file)
    
    # 加载模型
    log_print("正在加载训练好的模型...", log_file)
    model = CLIPFineTuner(
        num_classes=Config.num_classes, 
        device=device,
        fusion_strategy=Config.fusion_strategy,
        projection_dim=Config.projection_dim,
        attention_heads=Config.attention_heads,
        fusion_dropout=Config.fusion_dropout
    ).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        log_print(f"模型加载成功: {model_path}", log_file)
    except Exception as e:
        log_print(f"模型加载失败: {e}", log_file)
        return
    
    # 准备测试数据集
    log_print("正在加载测试数据集...", log_file)
    test_dataset = MultimodalDataset("multimodel_classificate/dataset", Config, split="test")
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size)
    
    log_print(f"测试集样本数: {len(test_dataset)}", log_file)
    log_print(f"测试批次数: {len(test_loader)}", log_file)
    
    # 获取类别名称
    class_names = test_dataset.get_class_names()
    log_print(f"类别列表: {class_names}", log_file)
    
    # 开始评估
    log_print("开始模型评估...", log_file)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    correct = 0
    total = 0
    
    # 创建评估进度条
    eval_pbar = tqdm(test_loader, desc="模型评估进度", unit="batch")
    
    with torch.no_grad():
        for batch_idx, (images, texts, labels) in enumerate(eval_pbar):
            images, texts, labels = images.to(device), texts.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(images, texts)
            
            # 处理不同融合策略的返回值
            if isinstance(outputs, tuple):
                logits, fusion_info = outputs
            else:
                logits = outputs
                
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            # 统计结果
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # 更新进度条显示
            current_acc = correct / total if total > 0 else 0
            eval_pbar.set_postfix({
                'Acc': f'{current_acc:.4f}',
                'Samples': f'{total}'
            })
            
            # 减少日志输出频率
            if (batch_idx + 1) % 20 == 0:
                log_print(f"  已评估: {total} 样本, 当前准确率: {current_acc:.4f}", log_file)
    
    # 计算整体准确率
    overall_accuracy = correct / total if total > 0 else 0
    
    # 生成分类报告
    log_print("=" * 50, log_file)
    log_print("评估结果", log_file)
    log_print("=" * 50, log_file)
    log_print(f"总体准确率: {overall_accuracy:.4f} ({correct}/{total})", log_file)
    
    # 详细分类报告
    try:
        # 生成分类报告
        report = classification_report(
            all_labels, 
            all_preds, 
            target_names=class_names,
            digits=4,
            zero_division=0
        )
        log_print("\n详细分类报告:", log_file)
        log_print(report, log_file)
        
        # 生成混淆矩阵
        cm = confusion_matrix(all_labels, all_preds)
        log_print("\n混淆矩阵:", log_file)
        log_print("行=真实标签, 列=预测标签", log_file)
        
        # 打印混淆矩阵头部
        header = "真实\\预测".ljust(12)
        for class_name in class_names:
            header += class_name[:8].center(10)
        log_print(header, log_file)
        
        # 打印混淆矩阵内容
        for i, class_name in enumerate(class_names):
            row = class_name[:10].ljust(12)
            for j in range(len(class_names)):
                row += str(cm[i, j]).center(10)
            log_print(row, log_file)
        
        # 计算每个类别的准确率
        log_print("\n各类别准确率:", log_file)
        class_accuracies = cm.diagonal() / cm.sum(axis=1)
        for i, (class_name, acc) in enumerate(zip(class_names, class_accuracies)):
            support = cm[i].sum()
            log_print(f"  {class_name}: {acc:.4f} (支持样本: {support})", log_file)
        
    except Exception as e:
        log_print(f"生成详细报告时出错: {e}", log_file)
    
    # 绘制并保存混淆矩阵
    try:
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 支持中文显示
        plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
        
        # 创建时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 绘制混淆矩阵
        cm_save_path = os.path.join(result_dir, f'confusion_matrix_{timestamp}.png')
        plot_confusion_matrix(cm, class_names, cm_save_path, 
                             f'混淆矩阵 - {Config.fusion_strategy}策略')
        log_print(f"混淆矩阵已保存到: {cm_save_path}", log_file)
        
        # 绘制分类指标图
        metrics_save_path = os.path.join(result_dir, f'classification_metrics_{timestamp}.png')
        plot_classification_metrics(cm, class_names, metrics_save_path)
        log_print(f"分类指标图已保存到: {metrics_save_path}", log_file)
        
        # 创建评估摘要
        summary = {
            'fusion_strategy': Config.fusion_strategy,
            'overall_accuracy': f"{overall_accuracy:.4f}",
            'total_samples': total,
            'correct_predictions': correct,
            'model_path': model_path,
            'evaluation_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'projection_dim': Config.projection_dim,
            'attention_heads': Config.attention_heads,
            'fusion_dropout': Config.fusion_dropout
        }
        
        # 保存详细的评估摘要
        summary_file = os.path.join(result_dir, f'evaluation_summary_{timestamp}.txt')
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write(f"模型评估摘要 - {Config.fusion_strategy}策略\n")
            f.write("="*60 + "\n")
            f.write(f"融合策略: {summary['fusion_strategy']}\n")
            f.write(f"投影维度: {summary['projection_dim']}\n")
            f.write(f"注意力头数: {summary['attention_heads']}\n")
            f.write(f"融合层Dropout: {summary['fusion_dropout']}\n")
            f.write(f"总体准确率: {summary['overall_accuracy']}\n")
            f.write(f"正确预测: {summary['correct_predictions']}/{summary['total_samples']}\n")
            f.write(f"模型路径: {summary['model_path']}\n")
            f.write(f"评估时间: {summary['evaluation_time']}\n")
            f.write("="*60 + "\n")
            
            # 添加各类别准确率
            f.write("\n各类别准确率:\n")
            f.write("-" * 40 + "\n")
            for i, (class_name, acc) in enumerate(zip(class_names, class_accuracies)):
                support = cm[i].sum()
                f.write(f"  {class_name:8s}: {acc:.4f} (支持样本: {support:3d})\n")
            
            # 添加混淆矩阵的文本版本
            f.write("\n混淆矩阵:\n")
            f.write("-" * 40 + "\n")
            f.write("行=真实标签, 列=预测标签\n")
            
            # 打印混淆矩阵头部
            header = "真实\\预测".ljust(12)
            for class_name in class_names:
                header += class_name[:8].center(10)
            f.write(header + "\n")
            
            # 打印混淆矩阵内容
            for i, class_name in enumerate(class_names):
                row = class_name[:10].ljust(12)
                for j in range(len(class_names)):
                    row += str(cm[i, j]).center(10)
                f.write(row + "\n")
        
        log_print(f"评估摘要已保存到: {summary_file}", log_file)
        
        # 保存简化的JSON结果用于后续分析
        json_result_file = os.path.join(result_dir, f'evaluation_results_{timestamp}.json')
        json_summary = {
            'fusion_strategy': Config.fusion_strategy,
            'overall_accuracy': float(overall_accuracy),
            'total_samples': total,
            'correct_predictions': correct,
            'class_accuracies': {class_names[i]: float(acc) for i, acc in enumerate(class_accuracies)},
            'class_support': {class_names[i]: int(cm[i].sum()) for i in range(len(class_names))},
            'model_config': {
                'projection_dim': Config.projection_dim,
                'attention_heads': Config.attention_heads,
                'fusion_dropout': Config.fusion_dropout,
                'num_classes': Config.num_classes
            },
            'evaluation_time': summary['evaluation_time'],
            'model_path': model_path
        }
        
        with open(json_result_file, 'w', encoding='utf-8') as f:
            json.dump(json_summary, f, indent=2, ensure_ascii=False)
        log_print(f"JSON结果已保存到: {json_result_file}", log_file)
        
    except Exception as e:
        log_print(f"保存可视化结果时出错: {e}", log_file)
    
    log_print("=" * 50, log_file)
    log_print("评估完成！", log_file)

if __name__ == "__main__":
    evaluate() 