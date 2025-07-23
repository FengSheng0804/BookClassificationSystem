"""
数据增强工具
功能：对数据集进行智能增强，解决类别不平衡问题
- 图像增强：随机旋转、裁剪、平移
- 文本增强：同义词替换、语序调整
目标：将少于10000个样本的类别增强到10000个
作者：Assistant
日期：2025-07-23
"""

import os
import random
import shutil
from PIL import Image, ImageEnhance
import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import jieba
import jieba.posseg as pseg
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

class ImageAugmentor:
    """
    图像数据增强器
    支持旋转、缩放、平移、亮度调整等
    """
    
    def __init__(self):
        self.augment_transforms = [
            self._random_rotation,
            self._random_scale_crop, 
            self._random_translate,
            self._random_brightness,
            self._random_contrast,
            self._random_flip
        ]
    
    def _random_rotation(self, image):
        """随机旋转 ±15°"""
        angle = random.uniform(-15, 15)
        return F.rotate(image, angle, fill=255)
    
    def _random_scale_crop(self, image):
        """随机缩放裁剪 0.8-1.2"""
        scale = random.uniform(0.8, 1.2)
        width, height = image.size
        
        # 计算新尺寸
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # 缩放
        image = image.resize((new_width, new_height), Image.LANCZOS)
        
        # 如果放大了，需要裁剪到原始尺寸
        if scale > 1.0:
            left = (new_width - width) // 2
            top = (new_height - height) // 2
            image = image.crop((left, top, left + width, top + height))
        # 如果缩小了，需要填充到原始尺寸
        else:
            new_img = Image.new('RGB', (width, height), (255, 255, 255))
            paste_x = (width - new_width) // 2
            paste_y = (height - new_height) // 2
            new_img.paste(image, (paste_x, paste_y))
            image = new_img
            
        return image
    
    def _random_translate(self, image):
        """随机平移 10%偏移"""
        width, height = image.size
        max_dx = int(width * 0.1)
        max_dy = int(height * 0.1)
        
        dx = random.randint(-max_dx, max_dx)
        dy = random.randint(-max_dy, max_dy)
        
        return F.affine(image, angle=0, translate=(dx, dy), scale=1, shear=0, fill=255)
    
    def _random_brightness(self, image):
        """随机亮度调整"""
        factor = random.uniform(0.8, 1.2)
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)
    
    def _random_contrast(self, image):
        """随机对比度调整"""
        factor = random.uniform(0.8, 1.2)
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)
    
    def _random_flip(self, image):
        """随机水平翻转（适用于某些图像）"""
        if random.random() < 0.3:  # 30%概率翻转
            return F.hflip(image)
        return image
    
    def augment_image(self, image_path, num_augmentations=1):
        """
        对单张图像进行增强
        
        Args:
            image_path: 原始图像路径
            num_augmentations: 生成的增强图像数量
            
        Returns:
            list: 增强后的PIL图像列表
        """
        try:
            original_image = Image.open(image_path).convert('RGB')
            augmented_images = []
            
            for i in range(num_augmentations):
                # 随机选择2-4个增强操作
                num_ops = random.randint(2, 4)
                selected_ops = random.sample(self.augment_transforms, num_ops)
                
                # 应用增强操作
                augmented = original_image.copy()
                for op in selected_ops:
                    augmented = op(augmented)
                
                augmented_images.append(augmented)
            
            return augmented_images
            
        except Exception as e:
            print(f"图像增强失败 {image_path}: {e}")
            return []

class TextAugmentor:
    """
    文本数据增强器
    支持同义词替换、语序调整等
    """
    
    def __init__(self):
        # 常见的同义词映射（可扩展）
        self.synonym_dict = {
            # 教育相关
            "教程": ["指南", "教材", "手册", "课程"],
            "学习": ["研习", "学会", "掌握", "修习"],
            "方法": ["技巧", "方式", "途径", "手段"],
            "技能": ["技术", "本领", "能力", "技艺"],
            "知识": ["学问", "学识", "见识", "智慧"],
            
            # 书籍相关
            "大全": ["集锦", "汇编", "合集", "全书"],
            "手册": ["指南", "教程", "说明", "指导"],
            "指南": ["指导", "手册", "教程", "导引"],
            "全书": ["大全", "全集", "汇编", "总集"],
            "丛书": ["系列", "套书", "文库", "选集"],
            
            # 医学相关
            "疾病": ["病症", "病患", "疾患", "病变"],
            "治疗": ["医治", "诊治", "救治", "疗治"],
            "症状": ["病症", "表现", "征象", "病象"],
            "药物": ["药品", "药剂", "医药", "药材"],
            
            # 艺术相关
            "绘画": ["画画", "作画", "画图", "图画"],
            "技法": ["技巧", "手法", "方法", "技术"],
            "作品": ["著作", "创作", "艺术品", "画作"],
            "风格": ["样式", "格调", "派头", "特色"],
            
            # 历史相关
            "历史": ["史学", "史实", "历程", "沿革"],
            "古代": ["古时", "上古", "远古", "古昔"],
            "文明": ["文化", "文明史", "历史文化", "人文"],
            "朝代": ["王朝", "代次", "时代", "年代"],
            
            # 文学相关
            "小说": ["故事", "传奇", "演义", "话本"],
            "文学": ["文艺", "文学作品", "文学创作", "文艺作品"],
            "作家": ["文学家", "作者", "文人", "文学创作者"],
            "经典": ["名著", "杰作", "佳作", "传世之作"],
            
            # 儿童相关
            "儿童": ["孩子", "小朋友", "幼儿", "孩童"],
            "少儿": ["儿童", "小孩", "青少年", "未成年"],
            "启蒙": ["入门", "基础", "初级", "启发"],
            "趣味": ["有趣", "好玩", "生动", "活泼"]
        }
        
        # 标点符号模式
        self.punctuation_patterns = [
            ("：", "——"),
            ("——", "："),
            ("，", "、"),
            ("、", "，")
        ]
    
    def _replace_synonyms(self, text):
        """同义词替换"""
        words = jieba.lcut(text)
        new_words = []
        
        for word in words:
            if word in self.synonym_dict and random.random() < 0.3:  # 30%概率替换
                synonyms = self.synonym_dict[word]
                new_word = random.choice(synonyms)
                new_words.append(new_word)
            else:
                new_words.append(word)
        
        return ''.join(new_words)
    
    def _adjust_word_order(self, text):
        """调整语序"""
        # 检测常见模式并调整
        patterns = [
            # "XX大全" -> "大全：XX版"
            (r'(.+)大全', r'大全：\1版'),
            # "XX教程" -> "教程：XX"  
            (r'(.+)教程', r'教程：\1'),
            # "XX指南" -> "指南：XX"
            (r'(.+)指南', r'指南：\1'),
            # "XX学习" -> "学习XX"
            (r'(.+)学习', r'学习\1'),
        ]
        
        import re
        for pattern, replacement in patterns:
            if re.search(pattern, text) and random.random() < 0.4:  # 40%概率调整
                text = re.sub(pattern, replacement, text)
                break
        
        return text
    
    def _adjust_punctuation(self, text):
        """调整标点符号"""
        for old_punct, new_punct in self.punctuation_patterns:
            if old_punct in text and random.random() < 0.2:  # 20%概率调整
                text = text.replace(old_punct, new_punct, 1)  # 只替换一次
                break
        return text
    
    def _add_descriptive_words(self, text):
        """添加描述性词汇"""
        descriptive_words = {
            'art': ['精美', '经典', '实用', '详细'],
            'children': ['有趣', '生动', '益智', '启发'],
            'history': ['权威', '详实', '深入', '全面'],
            'literature': ['经典', '优秀', '精选', '名家'],
            'medicine': ['专业', '实用', '权威', '详细'],
            'novel': ['精彩', '热门', '经典', '畅销']
        }
        
        # 根据内容特征选择合适的描述词
        for category, words in descriptive_words.items():
            if any(keyword in text for keyword in [category, 
                'art' if category == 'art' else category[:3]]):
                if random.random() < 0.3:  # 30%概率添加
                    desc_word = random.choice(words)
                    if desc_word not in text:
                        text = desc_word + text
                break
        
        return text
    
    def augment_text(self, text, num_augmentations=1):
        """
        对文本进行增强
        
        Args:
            text: 原始文本
            num_augmentations: 生成的增强文本数量
            
        Returns:
            list: 增强后的文本列表
        """
        augmented_texts = []
        
        for i in range(num_augmentations):
            augmented = text
            
            # 随机应用增强策略
            strategies = [
                self._replace_synonyms,
                self._adjust_word_order, 
                self._adjust_punctuation,
                self._add_descriptive_words
            ]
            
            # 随机选择1-3个策略
            num_strategies = random.randint(1, 3)
            selected_strategies = random.sample(strategies, num_strategies)
            
            for strategy in selected_strategies:
                augmented = strategy(augmented)
            
            # 确保增强后的文本不为空且与原文不同
            if augmented and augmented != text:
                augmented_texts.append(augmented)
            else:
                # 如果增强失败，使用原文本
                augmented_texts.append(text)
        
        return augmented_texts

class DatasetEnhancer:
    """
    数据集增强主类
    """
    
    def __init__(self, dataset_path, target_samples=10000):
        self.dataset_path = dataset_path
        self.target_samples = target_samples
        self.image_augmentor = ImageAugmentor()
        self.text_augmentor = TextAugmentor()
        
        # 统计原始数据（区分原始和增强数据）
        self.dataset_stats = self._get_dataset_stats()
        self.original_stats = self.dataset_stats['original']
        self.enhanced_stats = self.dataset_stats['enhanced']
        self.total_stats = self.dataset_stats['total']
        
    def _get_dataset_stats(self):
        """获取数据集统计信息（区分原始数据和增强数据）"""
        stats = {
            'total': {},
            'original': {},
            'enhanced': {}
        }
        
        for category in os.listdir(self.dataset_path):
            category_path = os.path.join(self.dataset_path, category)
            if not os.path.isdir(category_path):
                continue
                
            titles_file = os.path.join(category_path, "titles.txt")
            if os.path.exists(titles_file):
                original_count = 0
                enhanced_count = 0
                
                with open(titles_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                for line in lines:
                    line = line.strip()
                    if line and ':' in line:
                        try:
                            img_name, _ = line.split(':', 1)
                            img_name = img_name.strip()
                            img_path = os.path.join(category_path, img_name)
                            
                            if os.path.exists(img_path):
                                if '_aug_' in img_name:
                                    enhanced_count += 1
                                else:
                                    original_count += 1
                        except:
                            continue
                
                stats['original'][category] = original_count
                stats['enhanced'][category] = enhanced_count
                stats['total'][category] = original_count + enhanced_count
            else:
                stats['original'][category] = 0
                stats['enhanced'][category] = 0
                stats['total'][category] = 0
        
        return stats
    
    def _get_original_entries(self, category):
        """获取原始数据条目（排除已增强的数据）"""
        category_path = os.path.join(self.dataset_path, category)
        titles_file = os.path.join(category_path, "titles.txt")
        
        original_entries = []
        enhanced_entries = []
        
        with open(titles_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if line and ':' in line:
                try:
                    img_name, title = line.split(':', 1)
                    img_name = img_name.strip()
                    title = title.strip()
                    img_path = os.path.join(category_path, img_name)
                    
                    if os.path.exists(img_path):
                        # 检查是否是增强数据（文件名包含_aug_）
                        if '_aug_' in img_name:
                            enhanced_entries.append((img_name, title))
                        else:
                            original_entries.append((img_name, title))
                except:
                    continue
        
        return original_entries, enhanced_entries
    
    def _calculate_enhancement_strategy(self, category, original_entries, enhanced_entries):
        """计算增强策略"""
        current_total = len(original_entries) + len(enhanced_entries)
        needed_samples = self.target_samples - current_total
        
        if needed_samples <= 0:
            return None, 0, 0
        
        original_count = len(original_entries)
        
        # 策略1：每个原始样本增强1次
        round1_possible = original_count
        
        # 策略2：如果第一轮不够，继续第二轮增强
        if needed_samples <= round1_possible:
            # 第一轮就够了
            return "single_round", needed_samples, 1
        else:
            # 需要多轮增强
            remaining_after_round1 = needed_samples - round1_possible
            round2_needed = min(remaining_after_round1, original_count)
            
            if remaining_after_round1 <= original_count:
                return "two_rounds", needed_samples, 2
            else:
                # 需要更多轮次
                max_rounds = (needed_samples + original_count - 1) // original_count
                return "multi_rounds", needed_samples, max_rounds
    
    def _enhance_category(self, category):
        """增强单个类别的数据（优化版）"""
        category_path = os.path.join(self.dataset_path, category)
        titles_file = os.path.join(category_path, "titles.txt")
        
        if not os.path.exists(titles_file):
            print(f"警告：{category} 类别缺少 titles.txt 文件")
            return
        
        # 获取原始数据和已增强数据
        original_entries, enhanced_entries = self._get_original_entries(category)
        current_total = len(original_entries) + len(enhanced_entries)
        
        if current_total >= self.target_samples:
            print(f"📊 {category}: {current_total} 样本，无需增强")
            return
        
        if not original_entries:
            print(f"警告：{category} 类别没有原始数据可供增强")
            return
        
        # 计算增强策略
        strategy, needed_samples, max_rounds = self._calculate_enhancement_strategy(
            category, original_entries, enhanced_entries
        )
        
        if strategy is None:
            print(f"📊 {category}: 已达到目标，无需增强")
            return
        
        print(f"📈 {category}: {current_total} -> {self.target_samples} (+{needed_samples})")
        print(f"  原始样本: {len(original_entries)}, 已增强: {len(enhanced_entries)}")
        print(f"  增强策略: {strategy}, 最大轮次: {max_rounds}")
        
        # 执行分轮增强
        new_entries = []
        total_augmented = 0
        
        with tqdm(total=needed_samples, desc=f"增强 {category}") as pbar:
            for round_num in range(1, max_rounds + 1):
                if total_augmented >= needed_samples:
                    break
                
                print(f"  🔄 第 {round_num} 轮增强...")
                
                # 计算这一轮需要增强的数量
                remaining_needed = needed_samples - total_augmented
                round_target = min(remaining_needed, len(original_entries))
                
                # 为这一轮选择样本（可以是全部原始样本的子集）
                if round_target < len(original_entries):
                    # 随机选择部分样本
                    selected_entries = random.sample(original_entries, round_target)
                else:
                    # 使用全部原始样本
                    selected_entries = original_entries.copy()
                
                # 对选中的样本进行增强
                round_augmented = 0
                for original_img, original_title in selected_entries:
                    if total_augmented >= needed_samples:
                        break
                    
                    original_img_path = os.path.join(category_path, original_img)
                    
                    # 图像增强（每个样本增强1次）
                    try:
                        augmented_images = self.image_augmentor.augment_image(
                            original_img_path, 1
                        )
                        if not augmented_images:
                            continue
                    except Exception as e:
                        print(f"图像增强失败 {original_img}: {e}")
                        continue
                    
                    # 文本增强
                    try:
                        augmented_titles = self.text_augmentor.augment_text(
                            original_title, 1
                        )
                        if not augmented_titles:
                            augmented_titles = [f"增强版{original_title}"]
                    except Exception as e:
                        print(f"文本增强失败: {e}")
                        augmented_titles = [f"增强版{original_title}"]
                    
                    # 生成新文件名（包含轮次信息）
                    base_name = os.path.splitext(original_img)[0]
                    ext = os.path.splitext(original_img)[1]
                    new_img_name = f"{base_name}_aug_r{round_num}_{round_augmented:04d}{ext}"
                    new_img_path = os.path.join(category_path, new_img_name)
                    
                    # 保存增强图像
                    try:
                        augmented_images[0].save(new_img_path, quality=95)
                    except Exception as e:
                        print(f"保存图像失败: {e}")
                        continue
                    
                    # 记录新条目
                    new_entries.append(f"{new_img_name}:{augmented_titles[0]}")
                    total_augmented += 1
                    round_augmented += 1
                    pbar.update(1)
                
                print(f"    第 {round_num} 轮完成，增强 {round_augmented} 个样本")
        
        if new_entries:
            # 备份原始文件
            backup_file = titles_file + f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copy2(titles_file, backup_file)
            print(f"📁 原始文件已备份到: {backup_file}")
            
            # 更新 titles.txt 文件
            with open(titles_file, 'a', encoding='utf-8') as f:
                f.write('\n')
                for entry in new_entries:
                    f.write(entry + '\n')
            
            print(f"✅ {category} 增强完成，新增 {len(new_entries)} 个样本")
        else:
            print(f"⚠️ {category} 增强失败，未生成新样本")
    
    def enhance_dataset(self):
        """增强整个数据集（优化版）"""
        print("🚀 开始数据集增强（智能分轮策略）")
        print("=" * 60)
        
        print("📊 当前数据统计:")
        total_original = 0
        total_enhanced = 0
        total_all = 0
        
        for category in self.original_stats.keys():
            original = self.original_stats[category]
            enhanced = self.enhanced_stats[category]
            total = self.total_stats[category]
            
            print(f"  {category}:")
            print(f"    原始: {original:,}, 已增强: {enhanced:,}, 总计: {total:,}")
            
            total_original += original
            total_enhanced += enhanced
            total_all += total
        
        print(f"  整体统计:")
        print(f"    原始总计: {total_original:,}")
        print(f"    已增强总计: {total_enhanced:,}")
        print(f"    数据总计: {total_all:,}")
        
        print(f"\n🎯 目标: 将少于 {self.target_samples:,} 的类别增强到 {self.target_samples:,}")
        
        # 计算需要增强的类别
        categories_to_enhance = {}
        enhancement_plan = {}
        
        for category in self.total_stats.keys():
            current_total = self.total_stats[category]
            if current_total < self.target_samples:
                needed = self.target_samples - current_total
                original_count = self.original_stats[category]
                
                categories_to_enhance[category] = current_total
                
                # 计算增强策略
                if needed <= original_count:
                    strategy = f"单轮增强 ({needed}/{original_count})"
                elif needed <= original_count * 2:
                    strategy = f"两轮增强 (全部原始样本增强2次)"
                else:
                    rounds = (needed + original_count - 1) // original_count
                    strategy = f"多轮增强 ({rounds}轮)"
                
                enhancement_plan[category] = {
                    'current': current_total,
                    'needed': needed,
                    'original_count': original_count,
                    'strategy': strategy
                }
        
        if not categories_to_enhance:
            print("✅ 所有类别都已达到目标样本数，无需增强")
            return
        
        print(f"\n📈 增强计划 ({len(categories_to_enhance)} 个类别):")
        for category, plan in enhancement_plan.items():
            print(f"  {category}: {plan['current']:,} -> {self.target_samples:,} (+{plan['needed']:,})")
            print(f"    原始样本: {plan['original_count']:,}, 策略: {plan['strategy']}")
        
        # 计算总的增强量
        total_to_add = sum(plan['needed'] for plan in enhancement_plan.values())
        print(f"\n📊 总计需要新增: {total_to_add:,} 个样本")
        
        # 逐个增强
        for category in categories_to_enhance:
            print(f"\n{'='*50}")
            try:
                self._enhance_category(category)
            except Exception as e:
                print(f"❌ {category} 增强失败: {e}")
                continue
        
        # 最终统计
        print(f"\n{'='*60}")
        print("🎉 数据集增强完成！")
        
        final_stats = self._get_dataset_stats()
        print("\n📊 最终数据统计:")
        total_final_original = 0
        total_final_enhanced = 0
        total_final_all = 0
        
        for category in self.original_stats.keys():
            original_old = self.original_stats[category]
            enhanced_old = self.enhanced_stats[category]
            total_old = self.total_stats[category]
            
            original_new = final_stats['original'][category]
            enhanced_new = final_stats['enhanced'][category]
            total_new = final_stats['total'][category]
            
            added_enhanced = enhanced_new - enhanced_old
            
            print(f"  {category}:")
            print(f"    原始: {original_new:,} (无变化)")
            print(f"    增强: {enhanced_new:,} (+{added_enhanced:,})")
            print(f"    总计: {total_new:,} (+{total_new - total_old:,})")
            
            total_final_original += original_new
            total_final_enhanced += enhanced_new
            total_final_all += total_new
        
        total_added = total_final_all - total_all
        print(f"  整体统计:")
        print(f"    原始总计: {total_final_original:,} (无变化)")
        print(f"    增强总计: {total_final_enhanced:,} (+{total_final_enhanced - total_enhanced:,})")
        print(f"    数据总计: {total_final_all:,} (+{total_added:,})")
        
        # 保存增强报告
        self._save_enhancement_report(final_stats)
    
    def _save_enhancement_report(self, final_stats):
        """保存增强报告（优化版）"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'target_samples': self.target_samples,
            'original_stats': {
                'original': self.original_stats,
                'enhanced': self.enhanced_stats,
                'total': self.total_stats
            },
            'final_stats': final_stats,
            'enhancement_summary': {},
            'strategy_used': 'intelligent_multi_round'
        }
        
        for category in self.original_stats.keys():
            old_original = self.original_stats[category]
            old_enhanced = self.enhanced_stats[category]
            old_total = self.total_stats[category]
            
            new_original = final_stats['original'][category]
            new_enhanced = final_stats['enhanced'][category]
            new_total = final_stats['total'][category]
            
            report['enhancement_summary'][category] = {
                'before': {
                    'original': old_original,
                    'enhanced': old_enhanced,
                    'total': old_total
                },
                'after': {
                    'original': new_original,
                    'enhanced': new_enhanced,
                    'total': new_total
                },
                'added': {
                    'enhanced': new_enhanced - old_enhanced,
                    'total': new_total - old_total
                },
                'enhancement_ratio': new_total / old_total if old_total > 0 else 0,
                'target_reached': new_total >= self.target_samples
            }
        
        report_file = os.path.join(self.dataset_path, 
                                  f"enhancement_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"📝 增强报告已保存: {report_file}")

def main():
    """主函数"""
    dataset_path = "multimodel_classificate/dataset"
    target_samples = 10000
    
    if not os.path.exists(dataset_path):
        print(f"❌ 数据集路径不存在: {dataset_path}")
        return
    
    print("🔧 数据集增强工具")
    print("=" * 60)
    print(f"数据集路径: {dataset_path}")
    print(f"目标样本数: {target_samples:,}")
    print("增强策略:")
    print("  🖼️  图像: 旋转(±15°)、缩放(0.8-1.2)、平移(10%)、亮度/对比度调整")
    print("  📝 文本: 同义词替换、语序调整、标点调整、描述词添加")
    print("=" * 60)
    
    # 确认执行
    confirm = input("\n继续执行数据增强？(y/N): ").lower().strip()
    if confirm != 'y':
        print("❌ 操作已取消")
        return
    
    # 执行增强
    enhancer = DatasetEnhancer(dataset_path, target_samples)
    enhancer.enhance_dataset()

if __name__ == "__main__":
    main()
