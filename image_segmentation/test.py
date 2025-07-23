import os
import numpy as np
import torch
from image_segmentation.models.Unet import *
from utils import *
from torchvision import transforms
import cv2
import time
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, jaccard_score
import glob
from tqdm import tqdm

# 删除小的连通组件
def remove_small_connected_components(mask, min_size):
    # 处理白色小区域，转为黑色
    num_labels_white, labels_white, stats_white, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    mask_white_processed = np.zeros(mask.shape, dtype=np.uint8)
    for i in range(1, num_labels_white):  # 跳过背景（0）
        if stats_white[i, cv2.CC_STAT_AREA] >= min_size:
            mask_white_processed[labels_white == i] = 255  # 保留大面积白色
    
    # 将得到的大白色区域反转，小黑色区域变白
    mask_inv = 255 - mask_white_processed  # 反转图像，黑色区域变为白色
    num_labels_black, labels_black, stats_black, _ = cv2.connectedComponentsWithStats(mask_inv, connectivity=8)
    mask_inv_processed = np.zeros(mask_inv.shape, dtype=np.uint8)
    for i in range(1, num_labels_black):
        if stats_black[i, cv2.CC_STAT_AREA] >= min_size:
            mask_inv_processed[labels_black == i] = 255  # 保留反转后的大面积白色（即原黑色大区域）
    mask_black_processed = 255 - mask_inv_processed  # 反转回来，小黑色区域变白

    # 合并结果：保留原大白色 + 原小黑色变白
    final_mask = cv2.bitwise_or(mask_white_processed, mask_black_processed)
    return final_mask

def calculate_segmentation_metrics(pred_mask, gt_mask=None):
    """
    计算图像分割评价指标
    pred_mask: 预测的掩码 (0-255)
    gt_mask: 真实标签掩码 (0-255)，如果没有则计算基本统计指标
    """
    # 将掩码转换为二进制 (0, 1)
    pred_binary = (pred_mask > 127).astype(np.uint8)
    
    metrics = {}
    
    # 基本统计指标
    total_pixels = pred_mask.shape[0] * pred_mask.shape[1]
    foreground_pixels = np.sum(pred_binary)
    background_pixels = total_pixels - foreground_pixels
    
    metrics['total_pixels'] = total_pixels
    metrics['foreground_pixels'] = foreground_pixels
    metrics['background_pixels'] = background_pixels
    metrics['foreground_ratio'] = foreground_pixels / total_pixels
    
    # 连通组件分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(pred_binary, connectivity=8)
    
    # 去除背景组件
    component_areas = stats[1:, cv2.CC_STAT_AREA]  # 跳过背景
    
    metrics['num_components'] = num_labels - 1  # 减去背景
    if len(component_areas) > 0:
        metrics['largest_component_area'] = np.max(component_areas)
        metrics['smallest_component_area'] = np.min(component_areas)
        metrics['average_component_area'] = np.mean(component_areas)
        metrics['component_area_std'] = np.std(component_areas)
    else:
        metrics['largest_component_area'] = 0
        metrics['smallest_component_area'] = 0
        metrics['average_component_area'] = 0
        metrics['component_area_std'] = 0
    
    # 如果有真实标签，计算分割精度指标
    if gt_mask is not None:
        gt_binary = (gt_mask > 127).astype(np.uint8)
        
        # 展平数组用于计算指标
        pred_flat = pred_binary.flatten()
        gt_flat = gt_binary.flatten()
        
        # 计算各种指标
        metrics['pixel_accuracy'] = accuracy_score(gt_flat, pred_flat)
        metrics['precision'] = precision_score(gt_flat, pred_flat, zero_division=0)
        metrics['recall'] = recall_score(gt_flat, pred_flat, zero_division=0)
        metrics['f1_score'] = f1_score(gt_flat, pred_flat, zero_division=0)
        metrics['iou'] = jaccard_score(gt_flat, pred_flat, zero_division=0)
        
        # 计算Dice系数
        intersection = np.sum(pred_binary * gt_binary)
        dice_coefficient = (2.0 * intersection) / (np.sum(pred_binary) + np.sum(gt_binary) + 1e-8)
        metrics['dice_coefficient'] = dice_coefficient
        
        # 计算特异性
        tn = np.sum((1 - pred_binary) * (1 - gt_binary))
        fp = np.sum(pred_binary * (1 - gt_binary))
        specificity = tn / (tn + fp + 1e-8)
        metrics['specificity'] = specificity
    
    return metrics

def print_single_image_metrics(metrics, image_name, gt_available=False):
    """打印单张图像的评价指标"""
    print(f"\n📷 图像: {image_name}")
    print(f"   前景占比: {metrics['foreground_ratio']:.2%}")
    print(f"   连通组件数量: {metrics['num_components']}")
    
    if gt_available:
        print(f"   像素准确率: {metrics['pixel_accuracy']:.4f}")
        print(f"   IoU: {metrics['iou']:.4f}")
        print(f"   Dice系数: {metrics['dice_coefficient']:.4f}")
        print(f"   F1分数: {metrics['f1_score']:.4f}")

def print_average_metrics(all_metrics, gt_available=False):
    """打印平均评价指标"""
    print("\n" + "="*70)
    print("                    测试集整体评价指标报告")
    print("="*70)
    
    # 计算平均值 - 只处理数值型指标
    avg_metrics = {}
    
    # 定义需要计算统计的数值型指标
    numeric_metrics = [
        'total_pixels', 'foreground_pixels', 'background_pixels', 'foreground_ratio',
        'num_components', 'largest_component_area', 'smallest_component_area', 
        'average_component_area', 'component_area_std'
    ]
    
    # 如果有真实标签，添加精度指标
    if gt_available:
        numeric_metrics.extend([
            'pixel_accuracy', 'precision', 'recall', 'f1_score', 
            'iou', 'dice_coefficient', 'specificity'
        ])
    
    for metric_name in numeric_metrics:
        if metric_name in all_metrics[0]:
            values = []
            for m in all_metrics:
                if metric_name in m and isinstance(m[metric_name], (int, float, np.number)):
                    values.append(m[metric_name])
            
            if len(values) > 0:
                avg_metrics[metric_name] = np.mean(values)
                avg_metrics[f'{metric_name}_std'] = np.std(values)
    
    # 基本统计信息
    print(f"\n📊 测试集基本信息:")
    print(f"   测试图像数量: {len(all_metrics)}")
    if 'total_pixels' in avg_metrics:
        print(f"   平均图像尺寸: {avg_metrics['total_pixels']:.0f} 像素")
    if 'foreground_ratio' in avg_metrics:
        print(f"   平均前景占比: {avg_metrics['foreground_ratio']:.2%} ± {avg_metrics['foreground_ratio_std']:.2%}")
    if 'num_components' in avg_metrics:
        print(f"   平均连通组件数: {avg_metrics['num_components']:.1f} ± {avg_metrics['num_components_std']:.1f}")
    
    if gt_available and 'pixel_accuracy' in avg_metrics:
        print(f"\n🎯 分割精度指标 (平均值 ± 标准差):")
        print(f"   像素准确率: {avg_metrics['pixel_accuracy']:.4f} ± {avg_metrics['pixel_accuracy_std']:.4f}")
        print(f"   精确率 (Precision): {avg_metrics['precision']:.4f} ± {avg_metrics['precision_std']:.4f}")
        print(f"   召回率 (Recall): {avg_metrics['recall']:.4f} ± {avg_metrics['recall_std']:.4f}")
        print(f"   F1分数: {avg_metrics['f1_score']:.4f} ± {avg_metrics['f1_score_std']:.4f}")
        print(f"   IoU (Jaccard): {avg_metrics['iou']:.4f} ± {avg_metrics['iou_std']:.4f}")
        print(f"   Dice系数: {avg_metrics['dice_coefficient']:.4f} ± {avg_metrics['dice_coefficient_std']:.4f}")
        print(f"   特异性: {avg_metrics['specificity']:.4f} ± {avg_metrics['specificity_std']:.4f}")
        
        # 性能等级评估
        avg_iou = avg_metrics['iou']
        if avg_iou >= 0.8:
            performance = "优秀 🌟"
        elif avg_iou >= 0.6:
            performance = "良好 ✅"
        elif avg_iou >= 0.4:
            performance = "一般 ⚠️"
        else:
            performance = "需要改进 ❌"
        
        print(f"\n📈 整体性能评估: {performance}")
        
        # 分布统计
        iou_values = []
        for m in all_metrics:
            if 'iou' in m and isinstance(m['iou'], (int, float, np.number)):
                iou_values.append(m['iou'])
        
        if len(iou_values) > 0:
            excellent_count = sum(1 for iou in iou_values if iou >= 0.8)
            good_count = sum(1 for iou in iou_values if 0.6 <= iou < 0.8)
            fair_count = sum(1 for iou in iou_values if 0.4 <= iou < 0.6)
            poor_count = sum(1 for iou in iou_values if iou < 0.4)
            
            print(f"\n📊 性能分布:")
            print(f"   优秀 (IoU≥0.8): {excellent_count}/{len(iou_values)} ({excellent_count/len(iou_values)*100:.1f}%)")
            print(f"   良好 (0.6≤IoU<0.8): {good_count}/{len(iou_values)} ({good_count/len(iou_values)*100:.1f}%)")
            print(f"   一般 (0.4≤IoU<0.6): {fair_count}/{len(iou_values)} ({fair_count/len(iou_values)*100:.1f}%)")
            print(f"   需改进 (IoU<0.4): {poor_count}/{len(iou_values)} ({poor_count/len(iou_values)*100:.1f}%)")
    
    print("="*70)
    return avg_metrics

def test_on_dataset(test_images_path, test_masks_path, model, output_dir="test_results", min_component_size=500):
    """
    在测试集上进行评估
    test_images_path: 测试图像路径
    test_masks_path: 测试掩码路径  
    model: 训练好的模型
    output_dir: 输出结果目录
    min_component_size: 最小连通组件大小
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有测试图像
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(test_images_path, ext)))
    
    if len(image_files) == 0:
        print(f"❌ 在 {test_images_path} 中未找到图像文件")
        return None, None
    
    print(f"📁 找到 {len(image_files)} 张测试图像")
    
    all_metrics = []
    inference_times = []
    transform = transforms.Compose([transforms.ToTensor()])
    
    print("🔄 开始批量测试...")
    
    for image_file in tqdm(image_files, desc="处理图像"):
        image_name = os.path.basename(image_file)
        base_name = os.path.splitext(image_name)[0]
        
        try:
            # 加载原始图像
            original_img = cv2.imread(image_file)
            if original_img is None:
                print(f"⚠️  跳过无法读取的图像: {image_name}")
                continue
            
            # 预处理
            img = resize_rgb_image(image_file)
            img_data = transform(img).cuda()
            img_data = torch.unsqueeze(img_data, dim=0)
            
            # 推理
            start_inference = time.time()
            model.eval()
            with torch.no_grad():
                out = model(img_data)
                pred_mask = torch.argmax(out, dim=1).squeeze(0)
            inference_time = time.time() - start_inference
            inference_times.append(inference_time)
            
            # 后处理
            mask_np = pred_mask.byte().cpu().numpy() * 255
            processed_mask = remove_small_connected_components(mask_np, min_component_size)
            
            # 调整到原始尺寸
            original_size = original_img.shape[1::-1]
            processed_mask_resized = cv2.resize(processed_mask, original_size, interpolation=cv2.INTER_NEAREST)
            
            # 保存预测掩码
            output_mask_path = os.path.join(output_dir, f"{base_name}_pred_mask.png")
            cv2.imwrite(output_mask_path, processed_mask_resized)
            
            # 加载真实标签
            gt_mask = None
            gt_available = False
            
            # 尝试多种可能的标签文件名
            possible_gt_names = [
                f"{base_name}.png",
                f"{base_name}.jpg", 
                f"{base_name}.jpeg",
                f"{base_name}_mask.png",
                f"{base_name}_gt.png"
            ]
            
            for gt_name in possible_gt_names:
                gt_path = os.path.join(test_masks_path, gt_name)
                if os.path.exists(gt_path):
                    gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                    gt_mask_resized = cv2.resize(gt_mask, original_size, interpolation=cv2.INTER_NEAREST)
                    gt_available = True
                    break
            
            # 计算指标
            metrics = calculate_segmentation_metrics(processed_mask_resized, gt_mask_resized if gt_available else None)
            metrics['image_name'] = image_name  # 保存文件名用于记录
            metrics['inference_time'] = inference_time  # 保存推理时间
            all_metrics.append(metrics)
            
            # 打印单张图像结果
            print_single_image_metrics(metrics, image_name, gt_available)
            
        except Exception as e:
            print(f"❌ 处理图像 {image_name} 时发生错误: {str(e)}")
            continue
    
    if len(all_metrics) == 0:
        print("❌ 没有成功处理任何图像")
        return None, None
    
    # 计算并显示平均指标
    gt_available = any('pixel_accuracy' in m for m in all_metrics)
    avg_metrics = print_average_metrics(all_metrics, gt_available)
    
    # 计算推理性能统计
    if len(inference_times) > 0:
        avg_inference_time = np.mean(inference_times)
        print(f"\n⏱️  推理性能统计:")
        print(f"   平均推理时间: {avg_inference_time:.3f} 秒")
        print(f"   最快推理时间: {np.min(inference_times):.3f} 秒")
        print(f"   最慢推理时间: {np.max(inference_times):.3f} 秒")
        print(f"   推理FPS: {1.0/avg_inference_time:.1f}")
    
    # 保存详细结果到文件
    results_file = os.path.join(output_dir, "test_results_summary.txt")
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("测试集评价指标详细报告\n")
        f.write("="*50 + "\n\n")
        f.write(f"测试图像数量: {len(all_metrics)}\n")
        f.write(f"测试图像路径: {test_images_path}\n")
        f.write(f"测试标签路径: {test_masks_path}\n")
        f.write(f"输出结果路径: {output_dir}\n\n")
        
        if gt_available and 'pixel_accuracy' in avg_metrics:
            f.write("平均分割精度指标:\n")
            f.write(f"  像素准确率: {avg_metrics['pixel_accuracy']:.4f} ± {avg_metrics['pixel_accuracy_std']:.4f}\n")
            f.write(f"  精确率: {avg_metrics['precision']:.4f} ± {avg_metrics['precision_std']:.4f}\n")
            f.write(f"  召回率: {avg_metrics['recall']:.4f} ± {avg_metrics['recall_std']:.4f}\n")
            f.write(f"  F1分数: {avg_metrics['f1_score']:.4f} ± {avg_metrics['f1_score_std']:.4f}\n")
            f.write(f"  IoU: {avg_metrics['iou']:.4f} ± {avg_metrics['iou_std']:.4f}\n")
            f.write(f"  Dice系数: {avg_metrics['dice_coefficient']:.4f} ± {avg_metrics['dice_coefficient_std']:.4f}\n\n")
        
        if len(inference_times) > 0:
            f.write("推理性能统计:\n")
            f.write(f"  平均推理时间: {avg_inference_time:.3f} 秒\n")
            f.write(f"  推理FPS: {1.0/avg_inference_time:.1f}\n\n")
        
        f.write("单张图像详细结果:\n")
        f.write("-" * 50 + "\n")
        for i, metrics in enumerate(all_metrics):
            f.write(f"{i+1:3d}. {metrics.get('image_name', 'unknown')}\n")
            f.write(f"     前景占比: {metrics['foreground_ratio']:.2%}\n")
            f.write(f"     连通组件: {metrics['num_components']}\n")
            if 'inference_time' in metrics:
                f.write(f"     推理时间: {metrics['inference_time']:.3f}s\n")
            if 'iou' in metrics and isinstance(metrics['iou'], (int, float, np.number)):
                f.write(f"     IoU: {metrics['iou']:.4f}\n")
            if 'dice_coefficient' in metrics and isinstance(metrics['dice_coefficient'], (int, float, np.number)):
                f.write(f"     Dice: {metrics['dice_coefficient']:.4f}\n")
            f.write("\n")
    
    print(f"\n📄 详细结果已保存到: {results_file}")
    
    return all_metrics, avg_metrics

# 主程序
if __name__ == "__main__":
    # 设置测试数据路径
    test_images_path = "F:/desktop/test_dataset/images"
    test_masks_path = "F:/desktop/test_dataset/masks"
    output_dir = "F:/desktop/test_results"
    min_component_size = 500
    
    print("🚀 开始测试集批量评估...")
    start_time = time.time()
    
    # 检查路径是否存在
    if not os.path.exists(test_images_path):
        print(f"❌ 测试图像路径不存在: {test_images_path}")
        exit()
    
    if not os.path.exists(test_masks_path):
        print(f"⚠️  测试标签路径不存在: {test_masks_path}")
        print("   将只计算基本统计指标")
    
    # 加载模型
    net = UNet(2).cuda()
    weight_path = './image_segmentation/content/params/best_unet_new.pth'
    
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path)['model_state'])
        print('✅ 模型权重加载成功')
    else:
        print('❌ 模型权重加载失败')
        exit()
    
    # 运行测试
    all_metrics, avg_metrics = test_on_dataset(
        test_images_path=test_images_path,
        test_masks_path=test_masks_path,
        model=net,
        output_dir=output_dir,
        min_component_size=min_component_size
    )
    
    total_time = time.time() - start_time
    
    if all_metrics is not None:
        print(f"\n⏱️  总测试耗时: {total_time:.3f} 秒")
        print(f"📁 预测结果已保存到: {output_dir}")
        print("\n🎉 测试集评估完成!")
    else:
        print("\n❌ 测试失败!")