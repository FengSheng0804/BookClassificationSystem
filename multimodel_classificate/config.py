class Config:
    """
    配置类：包含训练和评估的所有超参数设置
    """
    # 数据集路径配置
    dataset_path = "multimodel_classificate/dataset/"
    
    # 模型保存路径配置
    save_path = 'multimodel_classificate/models/weights/'

    model_path = save_path + "best_clip_model.pt"  # 训练得到的最佳模型权重

    # CLIP预训练模型配置
    clip_model_name = "ViT-B/32"       # 使用的CLIP模型名称
    clip_model_path = "multimodel_classificate/models/weights/ViT-B-32.pt"  # 预训练CLIP权重路径
    
    # 模型参数配置
    num_classes = 10                    # 分类类别数
    batch_size = 64                     # 批处理大小
    lr = 1e-4                           # 学习率
    epochs = 100                        # 训练轮数
    seed = 42                           # 随机种子
    
    # 设备配置
    device = "cuda" if __import__('torch').cuda.is_available() else "cpu" 
    
    # 特征融合策略配置
    fusion_strategy = "attention"       # 融合策略: "concat", "attention", "cross_attention"
    projection_dim = 512                # 特征投影维度
    attention_heads = 8                 # 注意力头数（用于attention策略）
    fusion_dropout = 0.1                # 融合层dropout率
    
    # 可视化配置
    enable_visualization = True         # 是否启用实时可视化（不显示窗口，只保存图片）
    visualization_update_interval = 1   # 可视化更新间隔（每几个epoch更新一次）
    save_visualization_images = True    # 是否保存可视化图片
    visualization_dpi = 150             # 保存图片的DPI
    show_overfitting_warning = True     # 是否显示过拟合警告 