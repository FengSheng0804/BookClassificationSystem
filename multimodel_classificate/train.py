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

def check_checkpoint():
    """
    检查是否存在checkpoint文件
    
    Returns:
        str: checkpoint文件路径，如果不存在则返回None
    """
    # 确保路径正确
    checkpoint_path = os.path.join(Config.save_path, 'lastest_checkpoint.pt')
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
        print(f"📊 检查点信息:")
        print(f"   - 已训练轮数: {checkpoint.get('epoch', '未知')}")
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
    checkpoint_path = os.path.join(Config.save_path, 'lastest_checkpoint.pt')
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
        tuple: (起始epoch, 最佳准确率, 训练历史)
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
        start_epoch = checkpoint.get('epoch', 0)
        best_acc = checkpoint.get('best_acc', 0)
        training_history = checkpoint.get('training_history', {
            'train_losses': [],
            'val_accuracies': [],
            'learning_rates': [],
            'epoch_times': [],
            'best_epoch': 0,
            'best_acc': 0
        })
        
        print(f"📈 训练将从第 {start_epoch + 1} 轮开始")
        print(f"🏆 当前最佳准确率: {best_acc:.4f}")
        print(f"📚 已恢复 {len(training_history.get('train_losses', []))} 轮的训练历史")
        
        return start_epoch, best_acc, training_history
        
    except Exception as e:
        print(f"❌ 加载检查点失败: {e}")
        print("🔄 将重新开始训练")
        return 0, 0, {
            'train_losses': [],
            'val_accuracies': [],
            'learning_rates': [],
            'epoch_times': [],
            'best_epoch': 0,
            'best_acc': 0
        }

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
        'model_path': Config.model_path,
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
    
    # 设置损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    # 移除capturable=True以避免CUDA张量要求的问题
    optimizer = optim.Adam(model.parameters(), lr=Config.lr)
    
    # 可选：添加学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
    
    log_print("损失函数和优化器设置完成", log_file)
    
    # 检查是否存在checkpoint并询问用户
    start_epoch = 0
    best_acc = 0
    training_history = {
        'train_losses': [],
        'val_accuracies': [],
        'learning_rates': [],
        'epoch_times': [],
        'best_epoch': 0,
        'best_acc': 0
    }
    
    checkpoint_path = check_checkpoint()
    if checkpoint_path:
        continue_training = ask_user_continue_training(checkpoint_path)
        if continue_training:
            start_epoch, best_acc, training_history = load_checkpoint(
                model, optimizer, scheduler, checkpoint_path, device
            )
            log_print(f"从检查点恢复训练，起始epoch: {start_epoch + 1}, 最佳准确率: {best_acc:.4f}", log_file)
        else:
            clean_old_checkpoint()
            log_print("用户选择重新开始训练，已清理旧检查点", log_file)
    else:
        log_print("未找到检查点文件，开始新的训练", log_file)
    
    log_print("开始训练循环", log_file)
    total_start_time = time.time()
    
    # 初始化实时可视化器（可选）
    visualizer = None
    if Config.enable_visualization:
        try:
            vis_save_dir = os.path.join(os.path.dirname(log_file), 'visualizations') if Config.save_visualization_images else None
            if vis_save_dir:
                ensure_dir(vis_save_dir)
            visualizer = RealTimeTrainingVisualizer(
                save_dir=vis_save_dir, 
                update_interval=Config.visualization_update_interval,
                dpi=Config.visualization_dpi,
                show_overfitting_warning=Config.show_overfitting_warning
            )
            log_print(f"实时可视化已启用，更新间隔: {Config.visualization_update_interval} epoch", log_file)
            if vis_save_dir:
                log_print(f"可视化图片保存到: {vis_save_dir}", log_file)
        except Exception as e:
            log_print(f"初始化可视化失败: {e}，将继续训练但不显示可视化", log_file)
            visualizer = None
    else:
        log_print("实时可视化已禁用", log_file)
    
    # 创建总体进度条
    remaining_epochs = Config.epochs - start_epoch
    epoch_pbar = tqdm(range(start_epoch, Config.epochs), desc="训练进度", unit="epoch")
    
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
        
        # 更新实时可视化（如果启用）
        if visualizer:
            try:
                visualizer.update(epoch, avg_loss, train_acc, val_acc, 
                                optimizer.param_groups[0]['lr'], epoch_time)
            except Exception as e:
                log_print(f"更新可视化时出错: {e}", log_file)
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            training_history['best_epoch'] = epoch + 1
            training_history['best_acc'] = best_acc
            
            # 保存模型权重
            torch.save(model.state_dict(), Config.model_path)
            
            # 保存模型详细信息
            additional_info = {
                'best_epoch': epoch + 1,
                'best_validation_accuracy': float(best_acc),
                'training_time_minutes': (time.time() - total_start_time) / 60,
                'class_names': class_names,
                'validation_class_accuracies': val_class_acc
            }
            save_model_info(model, Config.model_path, Config, additional_info)
            
            log_print(f"  ✓ 发现更优模型！验证准确率: {val_acc:.4f}，已保存到 {Config.model_path}", log_file)
        
        # 每个epoch都保存最新的checkpoint
        checkpoint_path = os.path.join(Config.save_path, 'lastest_checkpoint.pt')
        torch.save({
            'epoch': epoch,  # 保存当前epoch
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_acc,
            'training_history': training_history
        }, checkpoint_path)
        
        # 记录日志
        log_print(f"  最新Checkpoint已更新: {checkpoint_path}", log_file)
    
    # 关闭进度条
    epoch_pbar.close()
    
    # 保存最终的可视化图表（如果启用）
    if visualizer:
        try:
            if Config.save_visualization_images:
                vis_save_dir = os.path.join(os.path.dirname(log_file), 'visualizations')
                final_plot_path = os.path.join(vis_save_dir, 'final_training_curves.png')
                visualizer.save_final_plot(final_plot_path)
                log_print(f"最终训练曲线已保存到: {final_plot_path}", log_file)
        except Exception as e:
            log_print(f"保存最终可视化时出错: {e}", log_file)
        finally:
            # 关闭可视化窗口
            visualizer.close()
    
    # 训练完成
    total_time = time.time() - total_start_time
    log_print("=" * 50, log_file)
    log_print("训练完成！", log_file)
    log_print(f"总训练时间: {format_time(total_time)}", log_file)
    log_print(f"最佳验证准确率: {best_acc:.4f} (第 {training_history['best_epoch']} 轮)", log_file)
    log_print(f"最佳模型保存路径: {Config.model_path}", log_file)
    
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