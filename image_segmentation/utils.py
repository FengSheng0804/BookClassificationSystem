from pathlib import Path
import random
import shutil
from PIL import Image
import os
import cv2
from glob import glob
import albumentations as A
import numpy as np
from tqdm import tqdm

# 批量处理文件夹，旋转竖版图片
def batch_process_shape(input_folder, output_folder):
    """批量处理文件夹"""
    def process_image(image_path, output_folder):
        """处理单个图像文件"""
        try:
            with Image.open(image_path) as img:
                # 获取原始尺寸
                width, height = img.size
                
                # 仅处理目标尺寸的图片
                if (width, height) not in [(4096, 3072), (3072, 4096)]:
                    print(f"跳过非标准尺寸文件: {os.path.basename(image_path)}")
                    return

                # 旋转竖版图片 (3072x4096 -> 4096x3072)
                if width == 3072 and height == 4096:
                    img = img.transpose(Image.ROTATE_90)

                # 保存处理结果
                output_path = os.path.join(output_folder, os.path.basename(image_path))
                img.save(output_path, quality=95, subsampling=0)  # 保持JPEG高质量
                print(f"已旋转: {os.path.basename(image_path)}")

        except Exception as e:
            print(f"处理失败: {os.path.basename(image_path)} - {str(e)}")

    # 创建输出目录
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # 支持的文件格式
    valid_ext = ('.jpg', '.jpeg', '.png', '.webp')
    
    # 遍历处理文件
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(valid_ext):
                process_image(
                    image_path=os.path.join(root, file),
                    output_folder=output_folder
                )

# 压缩文件夹中的图像，覆盖原文件
import os
from PIL import Image

def compress_images(folder_path, quality=90, max_size=None, preserve_format=True):
    """
    压缩文件夹中的图像，覆盖原文件
    
    :param folder_path: 图像文件夹路径
    :param quality: JPEG压缩质量 (1-100)，默认90
    :param max_size: 最大尺寸限制，如(1024, 1024)，None表示不限制
    :param preserve_format: 是否保持原格式，默认True
    """
    # 支持的文件格式
    valid_ext = ('.jpg', '.jpeg', '.png', '.webp')
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(valid_ext):
            file_path = os.path.join(folder_path, filename)
            
            try:
                with Image.open(file_path) as img:
                    original_format = img.format
                    original_mode = img.mode
                    
                    # 如果设置了最大尺寸，进行缩放
                    if max_size:
                        img.thumbnail(max_size, Image.LANCZOS)
                    
                    # 根据原格式决定保存方式
                    if preserve_format and original_format in ['PNG', 'WEBP']:
                        # 保持原格式
                        save_kwargs = {'optimize': True}
                        if original_format == 'PNG':
                            save_kwargs['compress_level'] = 6  # PNG压缩级别
                        img.save(file_path, original_format, **save_kwargs)
                    else:
                        # 转换为JPEG
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        img.save(file_path, 'JPEG', quality=quality, optimize=True)
                    
                    print(f"已压缩: {filename}")
                    
            except Exception as e:
                print(f"压缩失败: {filename} - {str(e)}")

# 处理RGB图像，使用左上角颜色填充
def resize_rgb_image(path, size=(1024, 1024)):
    """直接拉伸图像到目标尺寸，不保持宽高比"""
    # 打开图像并确保RGB模式
    img = Image.open(path).convert('RGB')
    
    # 直接拉伸到目标尺寸
    resized_img = img.resize(size, Image.LANCZOS)
    print(f"已处理图像: {path} -> {size}")
    
    return resized_img

# 增强数据集，生成更多数据
def get_more_dataset(input_img_dir, input_mask_dir, enhance_img_dir, enhance_mask_dir, size=(1024, 1024)):
    def augment_pair(img_path, mask_path, save_idx):
        # 读取图像和掩码
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转为RGB格式
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 灰度模式读取
        
        # 生成增强版本
        augmented = aug_pipeline(image=img, mask=mask)
        aug_img = augmented['image']
        aug_mask = augmented['mask']
        
        # 调整到指定大小
        aug_img = cv2.resize(aug_img, size, interpolation=cv2.INTER_LINEAR)
        aug_mask = cv2.resize(aug_mask, size, interpolation=cv2.INTER_NEAREST)
        
        # 生成保存路径
        base_name = os.path.basename(img_path).split(".")[0]
        img_save_path = os.path.join(enhance_img_dir, f"{base_name}_aug{save_idx}.png")
        mask_save_path = os.path.join(enhance_mask_dir, f"{base_name}_aug{save_idx}_mask.png")
        
        # 保存增强结果
        cv2.imwrite(img_save_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(mask_save_path, aug_mask)

    
    aug_times = 10  # 每对数据增强次数（2次 * 5种方法）

    # 创建输出目录
    os.makedirs(enhance_img_dir, exist_ok=True)
    os.makedirs(enhance_mask_dir, exist_ok=True)

    # 定义同步增强管道
    aug_pipeline = A.Compose([
        # 随机90度旋转（无填充问题）
        A.RandomRotate90(p=1.0),
        
        # 任意角度旋转（设置边缘填充）
        A.Rotate(
            limit=30,
            border_mode=cv2.BORDER_CONSTANT,  # 关键修改点
            value=None,                        # 禁用颜色填充
            mask_value=None,                   # 掩码禁用填充
            p=1.0
        ),
        
        # 平移缩放旋转（已设置border_mode）
        A.ShiftScaleRotate(                         
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=30,
            border_mode=cv2.BORDER_CONSTANT,   # 黑色填充
            value=None,                        # 确保不覆盖
            p=1.0
        ),
        
        # 弹性形变（同步设置）
        A.ElasticTransform(                         
            alpha=120,
            sigma=120 * 0.05,
            alpha_affine=120 * 0.03,
            border_mode=cv2.BORDER_CONSTANT,
            value=None,
            mask_value=None,
            p=1.0
        ),

        A.HorizontalFlip(p=1.0),                    # 水平翻转
        A.VerticalFlip(p=1.0),                      # 垂直翻转
        A.RandomGamma(gamma_limit=(80, 120), p=1.0) # 伽马变换
    ], additional_targets={'mask': 'mask'})         # 声明掩码使用相同变换

    # 获取原始文件列表
    img_files = sorted(glob(os.path.join(input_img_dir, "*.png")))
    mask_files = [
        os.path.join(input_mask_dir, os.path.basename(f).replace(".png", "_mask.png"))
        for f in img_files
    ]

    # 批量处理
    for idx in tqdm(range(len(img_files))):
        img_path = img_files[idx]
        mask_path = mask_files[idx]
        
        # 保存原始数据副本并调整到指定大小
        base_name = os.path.basename(img_path).split(".")[0]
        orig_img = cv2.resize(cv2.imread(img_path), size, interpolation=cv2.INTER_LINEAR)
        orig_mask = cv2.resize(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE), size, interpolation=cv2.INTER_NEAREST)
        
        cv2.imwrite(os.path.join(enhance_img_dir, f"{base_name}_aug0.png"), orig_img)
        cv2.imwrite(os.path.join(enhance_mask_dir, f"{base_name}_aug0_mask.png"), orig_mask)
        
        # 生成增强数据（每种处理执行2次）
        for aug_idx in range(1, aug_times+1):
            augment_pair(img_path, mask_path, aug_idx)

    print(f"增强完成！原始数据量: {len(img_files)}，增强后总量: {len(img_files)*(aug_times+1)}")

def process_image_pairs(images_path, masks_path, ext='.png'):
    """
    处理图像-掩码配对并重新编号
    
    :param images_path: 原图像文件夹路径
    :param masks_path: 掩码文件夹路径
    :param ext: 文件扩展名，默认为.png
    """
    # 收集有效文件对
    pairs = []
    for filename in os.listdir(images_path):
        if filename.endswith(ext):
            base_name = filename[:-len(ext)]
            mask_name = f"{base_name}_mask{ext}"
            mask_path = os.path.join(masks_path, mask_name)
            
            # 验证掩码文件存在
            if os.path.exists(mask_path):
                pairs.append((filename, mask_name))
    
    # 打乱文件顺序
    random.shuffle(pairs)
    
    # 重新编号并重命名文件
    for idx, (img_file, mask_file) in enumerate(pairs, start=1):
        # 生成新文件名
        new_img = f"{idx}{ext}"
        new_mask = f"{idx}_mask{ext}"
        
        # 原文件路径
        old_img_path = os.path.join(images_path, img_file)
        old_mask_path = os.path.join(masks_path, mask_file)
        
        # 新文件路径
        new_img_path = os.path.join(images_path, new_img)
        new_mask_path = os.path.join(masks_path, new_mask)
        
        # 重命名文件
        os.rename(old_img_path, new_img_path)
        os.rename(old_mask_path, new_mask_path)
        
        print(f"Renamed: {img_file} => {new_img}")
        print(f"Renamed: {mask_file} => {new_mask}")
    
    print(f"Total processed pairs: {len(pairs)}")

if __name__ == '__main__':
    input_img_dir = "F:\desktop\dataset\images_old"
    input_mask_dir = "F:\desktop\dataset\masks_old"
    enhance_img_dir = "F:\desktop\dataset\images_pro"
    enhance_mask_dir = "F:\desktop\dataset\masks_pro"

    # 创建输出目录
    os.makedirs(enhance_img_dir, exist_ok=True)
    os.makedirs(enhance_mask_dir, exist_ok=True)

    # 1. 图像压缩
    compress_images(input_img_dir)
    compress_images(input_mask_dir)

    # 2. 增强数据集
    get_more_dataset(input_img_dir, input_mask_dir, enhance_img_dir, enhance_mask_dir)

    # 3. 处理图像-掩码配对并重新编号
    process_image_pairs(enhance_img_dir, enhance_mask_dir)
