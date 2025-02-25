import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='Chinese Text Classification')
# 添加一个参数：选择模型
parser.add_argument('--model', default='TextRNN', type=str, required=True, help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
# 添加一个参数：选择embedding：random/pre_trained
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
# 添加一个参数：不是我们自己规定的
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
args = parser.parse_args()


if __name__ == '__main__':
    dataset = 'content'  # 数据集

    # 搜狗:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
    embedding = 'embedding_SougouNews.npz'
    if args.embedding == 'random':
        embedding = 'random'
    # 将参数中的args.model传给model_name
    model_name = args.model  #TextCNN, TextRNN,
    if model_name == 'FastText':
        from utils_fasttext import build_dataset, build_iterator, get_time_dif
        embedding = 'random'
    else:
        # 导入数据预处理的函数：加载数据集、
        from utils import build_dataset, build_iterator, get_time_dif

    # 得到训练出来的模型
    x = import_module('models.' + model_name)
    config = x.Config(dataset, embedding)
    # 随机出来相同的数字，保证每次结果一样
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    # 计时开始
    start_time = time.time()
    print("Loading data...")
    # 获取到训练集、验证集、测试集
    vocab, train_data, dev_data, test_data = build_dataset(config, args.word)
    # 获得迭代器
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    # 计时结束
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    config.n_vocab = len(vocab)
    model = x.Model(config).to(config.device)
    if model_name != 'Transformer':
        init_network(model)
    print(model.parameters)
    train(config, model, train_iter, dev_iter, test_iter)