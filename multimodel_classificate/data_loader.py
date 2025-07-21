"""
多模态数据加载器
功能：加载图像和文本数据用于多模态分类任务
支持数据集自动划分（训练集/验证集/测试集）
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import clip
import random
from glob import glob
from tqdm import tqdm

class MultimodalDataset(Dataset):
    """
    多模态数据集类
    
    Args:
        root_dir (str): 数据集根目录路径
        config: 配置对象
        split (str): 数据集划分类型 ('train', 'val', 'test')
        seed (int): 随机种子
    """
    
    def __init__(self, root_dir, config, split="train", seed=42):
        self.samples = []                # 存储所有样本 (图像路径, 文本, 标签)
        self.class_to_idx = {}          # 类别名称到索引的映射
        self.split = split              # 数据集划分类型
        self.config = config            # 配置对象
        self.device = config.device     # 计算设备
        
        print(f"正在初始化 {split} 数据集...")
        
        # 加载CLIP模型和预处理器
        print(f"正在加载CLIP模型: {config.clip_model_name}")
        self.clip_model, self.preprocess = clip.load(
            config.clip_model_name,
            device=self.device,
            download_root="./multimodel_classificate/models/weights"
        )
        
        # 扫描所有类别目录
        self._load_samples(root_dir, seed)
        
        print(f"{split} 数据集加载完成，共 {len(self.samples)} 个样本")
        print(f"类别数量: {len(self.class_to_idx)}")
        print(f"类别列表: {list(self.class_to_idx.keys())}")
    
    def _load_samples(self, root_dir, seed):
        """
        加载数据样本
        
        Args:
            root_dir (str): 数据集根目录
            seed (int): 随机种子
        """
        # 遍历所有类别子目录
        all_classes = sorted([d for d in os.listdir(root_dir) 
                             if os.path.isdir(os.path.join(root_dir, d))])
        
        print(f"发现 {len(all_classes)} 个类别: {all_classes}")
        
        # 为每个类别创建索引映射，使用进度条显示
        for idx, class_name in enumerate(tqdm(all_classes, desc="加载类别数据", unit="类别")):
            self.class_to_idx[class_name] = idx
            class_dir = os.path.join(root_dir, class_name)
            
            # 读取类别对应的标题文件
            titles_path = os.path.join(class_dir, "titles.txt")
            if not os.path.exists(titles_path):
                print(f"警告: 未找到 {class_name} 类别的 titles.txt 文件")
                continue
            
            # 解析标题文件，格式：图像文件名:标题文本
            class_samples = 0
            with open(titles_path, encoding="utf-8") as f:
                lines = f.readlines()
                
                # 使用进度条显示文件解析进度
                for line_num, line in enumerate(tqdm(lines, desc=f"解析 {class_name}", leave=False, unit="行"), 1):
                    line = line.strip()
                    if not line or ":" not in line:
                        continue
                    
                    try:
                        img_name, title = line.split(":", 1)
                        img_path = os.path.join(class_dir, img_name.strip())
                        
                        # 验证图像文件是否存在
                        if os.path.exists(img_path):
                            self.samples.append((img_path, title.strip(), idx))
                            class_samples += 1
                        else:
                            print(f"警告: 图像文件不存在 {img_path}")
                    except ValueError:
                        print(f"警告: {titles_path} 第 {line_num} 行格式错误: {line}")
            
            print(f"  {class_name}: {class_samples} 个样本")
        
        # 根据划分比例分割数据集
        self._split_dataset(seed)
    
    def _split_dataset(self, seed):
        """
        按比例划分数据集
        
        Args:
            seed (int): 随机种子
        """
        # 设置随机种子确保划分结果一致
        random.seed(seed)
        random.shuffle(self.samples)
        
        # 数据集划分比例：训练集80%，验证集10%，测试集10%
        n = len(self.samples)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        
        if self.split == "train":
            self.samples = self.samples[:n_train]
        elif self.split == "val":
            self.samples = self.samples[n_train:n_train+n_val]
        else:  # test
            self.samples = self.samples[n_train+n_val:]
        
        print(f"数据集划分完成 - {self.split}: {len(self.samples)} 样本")

    def __len__(self):
        """返回数据集大小"""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        获取指定索引的数据样本
        
        Args:
            idx (int): 样本索引
            
        Returns:
            tuple: (processed_image, tokenized_text, label)
        """
        if idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.samples)}")
        
        img_path, text, label = self.samples[idx]
        
        try:
            # 加载并预处理图像
            image = Image.open(img_path).convert("RGB")
            image = self.preprocess(image)
            
            # 对文本进行tokenization
            text = clip.tokenize([text], truncate=True)[0]
            
            return image, text, label
            
        except Exception as e:
            print(f"错误: 无法加载样本 {idx} ({img_path}): {e}")
            # 返回一个默认样本避免训练中断
            return self.__getitem__(0)
    
    def get_class_names(self):
        """
        获取所有类别名称列表
        
        Returns:
            list: 类别名称列表
        """
        return [class_name for class_name, _ in sorted(self.class_to_idx.items(), key=lambda x: x[1])]
    
    def get_sample_info(self, idx):
        """
        获取样本的详细信息
        
        Args:
            idx (int): 样本索引
            
        Returns:
            dict: 样本信息
        """
        if idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range")
        
        img_path, text, label = self.samples[idx]
        class_name = self.get_class_names()[label]
        
        return {
            'index': idx,
            'image_path': img_path,
            'text': text,
            'label': label,
            'class_name': class_name
        } 