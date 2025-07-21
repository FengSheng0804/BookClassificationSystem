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
        str: 日志文件路径
    """
    # 确保logs目录存在
    logs_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # 创建带时间戳的评估日志文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f'evaluation_{timestamp}.log')
    
    return log_file

def evaluate():
    """
    主评估函数
    """
    # 设置随机种子
    set_seed(Config.seed)
    
    # 设置日志
    log_file = setup_evaluation_logging()
    log_print("开始模型评估", log_file)
    
    # 检查模型文件是否存在
    if not os.path.exists(Config.save_path):
        log_print(f"错误: 找不到模型文件 {Config.save_path}", log_file)
        log_print("请先运行 train.py 训练模型", log_file)
        return
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_print(f"使用设备: {device}", log_file)
    
    # 加载模型
    log_print("正在加载训练好的模型...", log_file)
    model = CLIPFineTuner(num_classes=Config.num_classes, device=device).to(device)
    
    try:
        model.load_state_dict(torch.load(Config.save_path, map_location=device))
        log_print(f"模型加载成功: {Config.save_path}", log_file)
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
            logits = model(images, texts)
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
    
    # 保存评估结果
    results = {
        'overall_accuracy': float(overall_accuracy),
        'total_samples': total,
        'correct_predictions': correct,
        'class_names': class_names,
        'predictions': [int(p) for p in all_preds],
        'true_labels': [int(l) for l in all_labels],
        'model_path': Config.save_path,
        'evaluation_time': datetime.now().isoformat()
    }
    
    # 保存结果到JSON文件
    results_file = log_file.replace('.log', '_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log_print(f"评估结果已保存到: {results_file}", log_file)
    
    log_print("=" * 50, log_file)
    log_print("评估完成！", log_file)

if __name__ == "__main__":
    evaluate() 