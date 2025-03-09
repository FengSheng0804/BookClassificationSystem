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
                print(f"已处理: {os.path.basename(image_path)}")

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

# 处理RGB图像，使用左上角颜色填充
def resize_rgb_image(path, size=(1024, 1024)):
    """直接拉伸图像到目标尺寸，不保持宽高比"""
    # 打开图像并确保RGB模式
    img = Image.open(path).convert('RGB')
    
    # 直接拉伸到目标尺寸
    resized_img = img.resize(size, Image.LANCZOS)
    
    return resized_img

# 增强数据集，生成更多数据
def get_more_dataset(input_img_dir, input_mask_dir, output_img_dir, output_mask_dir):
    def augment_pair(img_path, mask_path, save_idx):
        # 读取图像和掩码
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转为RGB格式
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 灰度模式读取
        
        # 生成增强版本
        augmented = aug_pipeline(image=img, mask=mask)
        aug_img = augmented['image']
        aug_mask = augmented['mask']
        
        # 生成保存路径
        base_name = os.path.basename(img_path).split(".")[0]
        img_save_path = os.path.join(output_img_dir, f"{base_name}_aug{save_idx}.png")
        mask_save_path = os.path.join(output_mask_dir, f"{base_name}_aug{save_idx}_mask.png")
        
        # 保存增强结果
        cv2.imwrite(img_save_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(mask_save_path, aug_mask)

    
    aug_times = 5  # 每对数据增强次数

    # 创建输出目录
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    # 定义同步增强管道
    aug_pipeline = A.Compose([
        # 随机90度旋转（无填充问题）
        A.RandomRotate90(p=0.5),                   
        
        # 任意角度旋转（设置边缘填充）
        A.Rotate(
            limit=30,
            border_mode=cv2.BORDER_REPLICATE,  # 关键修改点
            value=None,                        # 禁用颜色填充
            mask_value=None,                   # 掩码禁用填充
            p=0.5
        ),
        
        # 平移缩放旋转（已设置border_mode）
        A.ShiftScaleRotate(                         
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=30,
            border_mode=cv2.BORDER_REPLICATE,  # 边缘复制
            value=None,                        # 确保不覆盖
            p=0.8
        ),
        
        # 弹性形变（同步设置）
        A.ElasticTransform(                         
            alpha=120,
            sigma=120 * 0.05,
            alpha_affine=120 * 0.03,
            border_mode=cv2.BORDER_REPLICATE,  # 添加边界模式
            value=None,
            mask_value=None,
            p=0.5
        ),
        A.RandomBrightnessContrast(                 # 亮度对比度调整
            brightness_limit=(-0.2, 0.2), 
            contrast_limit=(-0.2, 0.2), 
            p=0.3
        ),
        A.HorizontalFlip(p=0.5),                    # 水平翻转
        A.VerticalFlip(p=0.5),                      # 垂直翻转
        A.RandomGamma(gamma_limit=(80, 120), p=0.3) # 伽马变换
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
        
        # 保存原始数据副本
        base_name = os.path.basename(img_path).split(".")[0]
        cv2.imwrite(os.path.join(output_img_dir, f"{base_name}_aug0.png"), cv2.imread(img_path))
        cv2.imwrite(os.path.join(output_mask_dir, f"{base_name}_aug0_mask.png"), cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE))
        
        # 生成增强数据
        for aug_idx in range(1, aug_times+1):
            augment_pair(img_path, mask_path, aug_idx)

    print(f"增强完成！原始数据量: {len(img_files)}，增强后总量: {len(img_files)*(aug_times+1)}")

def process_image_pairs(src_folder, dst_folder, ext='.png'):
    """
    处理图像-掩码配对并重新编号
    
    :param src_folder: 源文件夹路径
    :param dst_folder: 目标文件夹路径
    :param ext: 文件扩展名，默认为.png
    """
    # 创建目标文件夹
    os.makedirs(dst_folder, exist_ok=True)
    
    # 收集有效文件对
    pairs = []
    for filename in os.listdir(src_folder):
        # 筛选基础图像文件
        if filename.endswith(ext) and '_mask' not in filename:
            base_name = filename[:-len(ext)]
            mask_name = f"{base_name}_mask{ext}"
            mask_path = os.path.join(src_folder, mask_name)
            
            # 验证掩码文件存在
            if os.path.exists(mask_path):
                pairs.append( (filename, mask_name) )
    
    # 打乱文件顺序
    random.shuffle(pairs)
    
    # 重新编号并复制文件
    for idx, (img_file, mask_file) in enumerate(pairs, start=1):
        # 生成新文件名
        new_img = f"{idx}{ext}"
        new_mask = f"{idx}_mask{ext}"
        
        # 源文件路径
        src_img = os.path.join(src_folder, img_file)
        src_mask = os.path.join(src_folder, mask_file)
        
        # 目标文件路径
        dst_img = os.path.join(dst_folder, new_img)
        dst_mask = os.path.join(dst_folder, new_mask)
        
        # 复制文件并保留元数据
        shutil.copy2(src_img, dst_img)
        shutil.copy2(src_mask, dst_mask)
        print(f"Processed: {img_file: <20} => {new_img}")
        print(f"Processed: {mask_file: <20} => {new_mask}")

if __name__ == '__main__':
    input_img_dir = "F:\desktop\dataset\images"
    input_mask_dir = "F:\desktop\dataset\masks"
    output_img_dir = "F:\desktop\dataset\images_pro"
    output_mask_dir = "F:\desktop\dataset\masks_pro"

    # # 1. 增强数据集
    # get_more_dataset(input_img_dir, input_mask_dir, output_img_dir, output_mask_dir)

    # # 2. 批处理文件夹，旋转竖版图片
    # batch_process_shape(input_img_dir, input_img_dir)
    # batch_process_shape(input_mask_dir, input_mask_dir)

    # # 3. 处理图像-变成正方形
    # images_paths = os.listdir(input_img_dir)
    # # 处理原图像
    # for image_path in images_paths:
    #     image_path = os.path.join(input_img_dir, image_path)
    #     img = resize_rgb_image(image_path)
    #     image_path = os.path.join(output_img_dir, image_path)
    #     img.save(image_path)
    # # 处理mask图像
    # masks_paths = os.listdir(input_mask_dir)
    # for mask_path in masks_paths:
    #     mask_path = os.path.join(input_mask_dir, mask_path)
    #     img = resize_rgb_image(mask_path)
    #     mask_path = os.path.join(output_mask_dir, mask_path)
    #     img.save(mask_path)

    # 4. 处理图像-掩码配对并重新编号
    src_folder = "F:/desktop/dataset/images"
    dst_folder = "F:/desktop/dataset/masks"
    process_image_pairs(src_folder, dst_folder) 


