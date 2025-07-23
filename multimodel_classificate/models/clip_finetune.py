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

from config import Config

class DynamicResidualGateBlock(nn.Module):
    """
    动态残差门控块
    """
    def __init__(self, hidden_dim, dropout=0.3):
        super(DynamicResidualGateBlock, self).__init__()
        self.hidden_dim = hidden_dim
        
        # 门控机制
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # 当前特征 + 图像特征 + 文本特征
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),  # 输出三个门控权重
            nn.Softmax(dim=-1)
        )
        
        # 特征变换网络
        self.transform_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 残差连接的归一化
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, current_features, image_features, text_features, layer_weight):
        """
        前向传播
        
        Args:
            current_features: 当前特征 [B, hidden_dim]
            image_features: 图像特征 [B, hidden_dim]
            text_features: 文本特征 [B, hidden_dim]
            layer_weight: 层权重 [B, 1]
            
        Returns:
            processed_features: 处理后的特征
            gate_weights: 门控权重
        """
        # 拼接特征用于门控计算
        gate_input = torch.cat([current_features, image_features, text_features], dim=-1)
        gate_weights = self.gate_network(gate_input)  # [B, 3]
        
        # 加权组合特征
        weighted_current = gate_weights[:, 0:1] * current_features
        weighted_image = gate_weights[:, 1:2] * image_features
        weighted_text = gate_weights[:, 2:3] * text_features
        
        combined_features = weighted_current + weighted_image + weighted_text
        
        # 特征变换
        transformed = self.transform_network(combined_features)
        
        # 残差连接
        residual_output = self.norm(current_features + transformed)
        
        # 应用层权重
        final_output = layer_weight * residual_output + (1 - layer_weight) * current_features
        
        return final_output, gate_weights


class DynamicResidualGatedFusion(nn.Module):
    """
    动态残差门控融合：根据输入动态调整网络结构
    """
    def __init__(self, image_dim, text_dim, hidden_dim=512, num_classes=10,
                 max_layers=5, dropout=0.3):
        super(DynamicResidualGatedFusion, self).__init__()
        self.hidden_dim = hidden_dim
        self.max_layers = max_layers
        
        # 特征投影
        self.image_projection = nn.Linear(image_dim, hidden_dim)
        self.text_projection = nn.Linear(text_dim, hidden_dim)
        
        # 动态层深度预测器
        self.depth_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_layers),
            nn.Softmax(dim=-1)  # 输出每层的使用概率
        )
        
        # 多个残差门控层
        self.residual_gate_layers = nn.ModuleList([
            DynamicResidualGateBlock(hidden_dim, dropout)
            for _ in range(max_layers)
        ])
        
        # 自适应权重融合
        self.adaptive_fusion = nn.Sequential(
            nn.Linear(hidden_dim * max_layers, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # 输出维度
        self.output_dim = hidden_dim
        
    def forward(self, image_features, text_features):
        """
        前向传播
        
        Args:
            image_features: 图像特征 [batch_size, image_dim]
            text_features: 文本特征 [batch_size, text_dim]
            
        Returns:
            final_features: 融合后的特征
            fusion_info: 融合过程信息
        """
        batch_size = image_features.size(0)
        
        # 1. 特征投影
        proj_image = self.image_projection(image_features)
        proj_text = self.text_projection(text_features)
        
        # 2. 预测最优层数
        initial_concat = torch.cat([proj_image, proj_text], dim=-1)
        layer_weights = self.depth_predictor(initial_concat)  # [B, max_layers]
        
        # 3. 初始特征
        current_features = proj_image + proj_text
        
        # 4. 动态层处理
        layer_outputs = []
        gate_weights_history = []
        
        for layer_idx, gate_layer in enumerate(self.residual_gate_layers):
            # 使用层权重决定是否处理这一层
            layer_weight = layer_weights[:, layer_idx:layer_idx+1]  # [B, 1]
            
            if layer_weight.mean() > 0.1:  # 阈值过滤
                processed_features, gate_weights = gate_layer(
                    current_features, proj_image, proj_text, layer_weight
                )
                layer_outputs.append(processed_features)
                gate_weights_history.append(gate_weights)
                current_features = processed_features
            else:
                # 跳过这一层
                layer_outputs.append(current_features)
                gate_weights_history.append(torch.zeros(batch_size, 3, device=image_features.device))
        
        # 5. 自适应融合所有层的输出
        if layer_outputs:
            all_outputs = torch.cat(layer_outputs, dim=-1)
            final_features = self.adaptive_fusion(all_outputs)
        else:
            final_features = current_features
        
        # 融合信息
        fusion_info = {
            'gate_weights': gate_weights_history,
            'layer_weights': layer_weights,
            'active_layers': (layer_weights > 0.1).sum(dim=1).float().mean()
        }
        
        return final_features, fusion_info

class MultimodalFusion(nn.Module):
    """
    多模态特征融合模块
    支持多种融合策略：concat, attention, dynamic_residual_gated
    """
    
    def __init__(self, image_dim, text_dim, projection_dim, strategy="concat", 
                 attention_heads=8, dropout=0.1, max_layers=5):
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
        elif strategy == "dynamic_residual_gated":
            self.dynamic_fusion = DynamicResidualGatedFusion(
                image_dim=image_dim,
                text_dim=text_dim,
                hidden_dim=projection_dim,
                max_layers=max_layers,
                dropout=dropout
            )
            self.output_dim = self.dynamic_fusion.output_dim
        else:
            raise ValueError(f"Unsupported fusion strategy: {strategy}")
    
    def forward(self, image_features, text_features):
        """
        融合图像和文本特征
        
        Args:
            image_features: 图像特征 [batch_size, image_dim]
            text_features: 文本特征 [batch_size, text_dim]
            
        Returns:
            融合后的特征或(特征, 融合信息)的元组
        """
        if self.strategy == "concat":
            # 特征投影
            img_proj = self.image_proj(image_features.float())  # [B, projection_dim]
            text_proj = self.text_proj(text_features.float())   # [B, projection_dim]
            # 简单拼接
            fused = torch.cat([img_proj, text_proj], dim=1)
            return fused
        
        elif self.strategy == "attention":
            # 特征投影
            img_proj = self.image_proj(image_features.float())  # [B, projection_dim]
            text_proj = self.text_proj(text_features.float())   # [B, projection_dim]
            
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
            return fused
            
        elif self.strategy == "dynamic_residual_gated":
            # 动态残差门控融合
            fused_features, fusion_info = self.dynamic_fusion(image_features.float(), text_features.float())
            return fused_features, fusion_info


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
                 fusion_strategy="concat", projection_dim=512, attention_heads=8, fusion_dropout=0.1,
                 max_layers=5): 
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
            download_root=Config.base_weights_path
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
            dropout=fusion_dropout,
            max_layers=max_layers
        )
        
        # 确保融合模块在正确的设备上
        self.fusion_module.to(self.device)
        
        # 定义分类器
        fusion_output_dim = self.fusion_module.output_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_output_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        # 确保分类器在正确的设备上
        self.classifier.to(self.device)
        
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
            torch.Tensor or tuple: 
                - 如果是动态残差门控策略，返回 (logits, fusion_info)
                - 否则返回 logits [batch_size, num_classes]
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
        fusion_result = self.fusion_module(image_features, text_features)
        
        # 处理不同融合策略的返回值
        if self.fusion_strategy == "dynamic_residual_gated":
            fused_features, fusion_info = fusion_result
            # 分类预测
            logits = self.classifier(fused_features)
            return logits, fusion_info
        else:
            fused_features = fusion_result
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
            torch.Tensor or tuple: 融合特征向量，动态残差门控策略会额外返回融合信息
        """
        image = image.to(self.device)
        text = text.to(self.device)
        
        with torch.no_grad():
            # 提取CLIP特征
            image_features = self.clip_model.encode_image(image)
            text_features = self.clip_model.encode_text(text)
            
            # 多模态特征融合
            fusion_result = self.fusion_module(image_features, text_features)
            
            # 处理不同融合策略的返回值
            if self.fusion_strategy == "dynamic_residual_gated":
                fused_features, fusion_info = fusion_result
                return fused_features, fusion_info
            else:
                fused_features = fusion_result
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