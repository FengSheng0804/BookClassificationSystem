import os

class Config:
    """
    配置类：包含训练和评估的所有超参数设置
    """
    # 数据集路径配置
    dataset_path = "multimodel_classificate/dataset/"
    
    # 基础路径配置
    base_weights_path = 'multimodel_classificate/models/weights/'
    base_logs_path = 'multimodel_classificate/logs/'
    
    # 根据融合策略动态设置路径
    @classmethod
    def get_save_path(cls):
        """权重保存路径（根据融合策略分类）"""
        strategy_path = os.path.join(cls.base_weights_path, cls.fusion_strategy)
        return strategy_path
    
    @classmethod
    def get_logs_path(cls):
        """日志保存路径（根据融合策略分类）"""
        strategy_path = os.path.join(cls.base_logs_path, cls.fusion_strategy)
        return strategy_path
    
    @classmethod
    def get_model_path(cls):
        """模型权重文件路径"""
        return os.path.join(cls.get_save_path(), "best_clip_model.pt")
    
    @classmethod
    def get_checkpoint_path(cls):
        """检查点文件路径"""
        return os.path.join(cls.get_save_path(), "lastest_checkpoint.pt")
    
    # 为了向后兼容，保留属性访问方式
    @property
    def save_path(self):
        return self.get_save_path()
    
    @property
    def logs_path(self):
        return self.get_logs_path()
    
    @property
    def model_path(self):
        return self.get_model_path()
    
    @property
    def checkpoint_path(self):
        return self.get_checkpoint_path()

    # CLIP预训练模型配置
    clip_model_name = "ViT-B/32"       # 使用的CLIP模型名称
    clip_model_path = "multimodel_classificate/models/weights/ViT-B-32.pt"  # 预训练CLIP权重路径
    
    # 模型参数配置
    num_classes = 6                     # 分类类别数
    batch_size = 256                    # 批处理大小
    lr = 5e-5                           # 学习率 (微调任务推荐值)
    epochs = 50                         # 训练轮数
    seed = 42                           # 随机种子
    
    # 设备配置
    device = "cuda" if __import__('torch').cuda.is_available() else "cpu" 
    
    # 特征融合策略配置
    fusion_strategy = "dynamic_residual_gated"       # 融合策略: "concat", "attention", "dynamic_residual_gated"
    projection_dim = 256                # 特征投影维度
    attention_heads = 8                 # 注意力头数（用于attention策略）
    fusion_dropout = 0.3                # 融合层dropout率
    
    # 可视化配置
    enable_visualization = True         # 是否启用实时可视化（不显示窗口，只保存图片）
    visualization_update_interval = 1   # 可视化更新间隔（每几个epoch更新一次）
    save_visualization_images = True    # 是否保存可视化图片
    visualization_dpi = 150             # 保存图片的DPI
    show_overfitting_warning = True     # 是否显示过拟合警告 