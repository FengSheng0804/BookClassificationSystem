class Config:
    """
    配置类：包含训练和评估的所有超参数设置
    """
    # 数据集路径配置
    train_file = "multimodel_classificate/dataset/train.txt"
    val_file = "multimodel_classificate/dataset/val.txt"
    test_file = "multimodel_classificate/dataset/test.txt"
    
    # 模型保存路径配置
    save_path = "multimodel_classificate/models/weights/best_clip_model.pt"  # 训练得到的最佳模型权重
    
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