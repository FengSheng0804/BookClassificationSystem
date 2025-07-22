"""
CLIP模型微调类
基于预训练的CLIP模型进行多模态分类任务的微调
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import os
import math

class MultimodalFusion(nn.Module):
    """
    多模态特征融合模块
    支持多种融合策略：concat, attention, cross_attention
    """
    
    def __init__(self, image_dim, text_dim, projection_dim, strategy="concat", 
                 attention_heads=8, dropout=0.1):
        super(MultimodalFusion, self).__init__()
        
        self.strategy = strategy
        self.projection_dim = projection_dim
        self.attention_heads = attention_heads
        self.dropout = dropout
        
        # 特征投影层
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 根据融合策略初始化相应的层
        if strategy == "concat":
            self.output_dim = projection_dim * 2
        elif strategy == "attention":
            self.output_dim = projection_dim
            self.attention = nn.MultiheadAttention(
                embed_dim=projection_dim,
                num_heads=attention_heads,
                dropout=dropout,
                batch_first=True
            )
            self.norm1 = nn.LayerNorm(projection_dim)
            self.norm2 = nn.LayerNorm(projection_dim)
            self.ffn = nn.Sequential(
                nn.Linear(projection_dim, projection_dim * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(projection_dim * 4, projection_dim)
            )
        elif strategy == "cross_attention":
            self.output_dim = projection_dim * 2
            self.cross_attention_img = nn.MultiheadAttention(
                embed_dim=projection_dim,
                num_heads=attention_heads,
                dropout=dropout,
                batch_first=True
            )
            self.cross_attention_text = nn.MultiheadAttention(
                embed_dim=projection_dim,
                num_heads=attention_heads,
                dropout=dropout,
                batch_first=True
            )
            self.norm_img = nn.LayerNorm(projection_dim)
            self.norm_text = nn.LayerNorm(projection_dim)
        else:
            raise ValueError(f"Unsupported fusion strategy: {strategy}")
    
    def forward(self, image_features, text_features):
        """
        融合图像和文本特征
        
        Args:
            image_features: 图像特征 [batch_size, image_dim]
            text_features: 文本特征 [batch_size, text_dim]
            
        Returns:
            torch.Tensor: 融合后的特征
        """
        # 特征投影
        img_proj = self.image_proj(image_features.float())  # [B, projection_dim]
        text_proj = self.text_proj(text_features.float())   # [B, projection_dim]
        
        if self.strategy == "concat":
            # 简单拼接
            fused = torch.cat([img_proj, text_proj], dim=1)
        
        elif self.strategy == "attention":
            # 自注意力融合
            # 将特征堆叠为序列 [batch_size, 2, projection_dim]
            features = torch.stack([img_proj, text_proj], dim=1)
            
            # 自注意力
            attn_out, _ = self.attention(features, features, features)
            attn_out = self.norm1(features + attn_out)
            
            # FFN
            ffn_out = self.ffn(attn_out)
            features_out = self.norm2(attn_out + ffn_out)
            
            # 平均池化得到最终特征
            fused = torch.mean(features_out, dim=1)
            
        elif self.strategy == "cross_attention":
            # 交叉注意力融合
            # 图像特征作为query，文本特征作为key和value
            img_attended, _ = self.cross_attention_img(
                img_proj.unsqueeze(1), 
                text_proj.unsqueeze(1), 
                text_proj.unsqueeze(1)
            )
            img_attended = self.norm_img(img_proj.unsqueeze(1) + img_attended).squeeze(1)
            
            # 文本特征作为query，图像特征作为key和value
            text_attended, _ = self.cross_attention_text(
                text_proj.unsqueeze(1), 
                img_proj.unsqueeze(1), 
                img_proj.unsqueeze(1)
            )
            text_attended = self.norm_text(text_proj.unsqueeze(1) + text_attended).squeeze(1)
            
            # 拼接增强后的特征
            fused = torch.cat([img_attended, text_attended], dim=1)
        
        return fused


class CLIPFineTuner(nn.Module):
    """
    CLIP微调模型
    
    该模型基于预训练的CLIP模型，通过添加分类头进行微调
    支持图像和文本的多模态输入
    
    Args:
        clip_model_name (str): CLIP模型名称，默认为 "ViT-B/32"
        num_classes (int): 分类类别数，默认为 10
        device (str): 计算设备，默认为 "cpu"
        freeze_clip (bool): 是否冻结CLIP参数，默认为 True
    """
    
    def __init__(self, clip_model_name="ViT-B/32", num_classes=10, device="cpu", freeze_clip=True,
                 fusion_strategy="concat", projection_dim=512, attention_heads=8, fusion_dropout=0.1): 
        super(CLIPFineTuner, self).__init__()
        
        self.device = device
        self.num_classes = num_classes
        self.freeze_clip = freeze_clip
        self.fusion_strategy = fusion_strategy
        
        print(f"正在初始化CLIP模型: {clip_model_name}")
        print(f"使用融合策略: {fusion_strategy}")
        
        self.clip_model, _ = clip.load(
            clip_model_name,
            device=self.device,
            download_root="./multimodel_classificate/models/weights"
        )
        
        # 获取特征维度
        self.image_feature_dim = self.clip_model.visual.output_dim
        self.text_feature_dim = self.clip_model.transformer.width
        
        print(f"图像特征维度: {self.image_feature_dim}")
        print(f"文本特征维度: {self.text_feature_dim}")
        
        # 是否冻结CLIP模型参数
        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False
            print("CLIP模型参数已冻结")
        else:
            print("CLIP模型参数将参与训练")
        
        # 初始化多模态融合模块
        self.fusion_module = MultimodalFusion(
            image_dim=self.image_feature_dim,
            text_dim=self.text_feature_dim,
            projection_dim=projection_dim,
            strategy=fusion_strategy,
            attention_heads=attention_heads,
            dropout=fusion_dropout
        )
        
        # 定义分类器
        fusion_output_dim = self.fusion_module.output_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_output_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        print(f"融合特征维度: {fusion_output_dim}")
        print(f"分类器输出维度: {num_classes}")
        self._print_model_info()

    def _print_model_info(self):
        """打印模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"模型总参数数: {total_params:,}")
        print(f"可训练参数数: {trainable_params:,}")
        print(f"参数冻结比例: {(total_params - trainable_params) / total_params * 100:.1f}%")

    def forward(self, image, text):
        """
        前向传播
        
        Args:
            image (torch.Tensor): 图像张量 [batch_size, 3, 224, 224]
            text (torch.Tensor): 文本token张量 [batch_size, 77]
            
        Returns:
            torch.Tensor: 分类logits [batch_size, num_classes]
        """
        # 确保输入在正确的设备上
        image = image.to(self.device)
        text = text.to(self.device)
        
        # 提取CLIP特征
        with torch.set_grad_enabled(not self.freeze_clip):
            # 编码图像特征
            image_features = self.clip_model.encode_image(image)
            # 编码文本特征
            text_features = self.clip_model.encode_text(text)
        
        # 多模态特征融合
        fused_features = self.fusion_module(image_features, text_features)
        
        # 分类预测
        logits = self.classifier(fused_features)
        
        return logits
    
    def get_features(self, image, text):
        """
        获取融合后的特征向量（用于特征分析）
        
        Args:
            image (torch.Tensor): 图像张量
            text (torch.Tensor): 文本token张量
            
        Returns:
            torch.Tensor: 融合特征向量
        """
        image = image.to(self.device)
        text = text.to(self.device)
        
        with torch.no_grad():
            # 提取CLIP特征
            image_features = self.clip_model.encode_image(image)
            text_features = self.clip_model.encode_text(text)
            
            # 多模态特征融合
            fused_features = self.fusion_module(image_features, text_features)
            
        return fused_features
    
    def freeze_clip_model(self):
        """冻结CLIP模型参数"""
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.freeze_clip = True
        print("CLIP模型参数已冻结")
    
    def unfreeze_clip_model(self):
        """解冻CLIP模型参数"""
        for param in self.clip_model.parameters():
            param.requires_grad = True
        self.freeze_clip = False
        print("CLIP模型参数已解冻")
    
    def get_model_size(self):
        """
        获取模型大小信息
        
        Returns:
            dict: 模型大小信息
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # 估算模型大小（MB）
        model_size_mb = total_params * 4 / (1024 * 1024)  # 假设float32
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': total_params - trainable_params,
            'model_size_mb': model_size_mb,
            'trainable_ratio': trainable_params / total_params
        } 