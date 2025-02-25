import random
"""
合成数据集并将其划分为训练集、测试集和验证集。
参数:
data_path (str): 数据文件的路径。
class_name_list (list): 包含类别名称的列表，每个类别对应一个数据文件。
功能:
1. 读取每个类别的数据文件，并将其内容添加到一个列表中。
2. 使用固定的种子打乱列表，以确保每次运行结果一致。
3. 将打乱后的数据划分为训练集、测试集和验证集，并分别写入对应的文件。
文件输出:
- './text_classificate/data/train.txt': 包含80%数据的训练集文件。
- './text_classificate/data/test.txt': 包含10%数据的测试集文件。
- './text_classificate/data/dev.txt': 包含10%数据的验证集文件。
"""

def synthesis_dataset(data_path, class_name_list):
    all_data = []
    for class_name in class_name_list:
        with open(data_path + class_name + '.txt', 'r', encoding='utf-8') as f:
            contents = f.readlines()
            for content in contents:
                all_data.append(content)
    # 使用固定的种子打乱列表
    random.seed(520)
    random.shuffle(all_data)
    with (open('./text_classificate/data/train.txt', 'w', encoding='utf-8') as f_train,
          open('./text_classificate/data/test.txt', 'w', encoding='utf-8') as f_test,
          open('./text_classificate/data/dev.txt', 'w', encoding='utf-8') as f_valid):
        f_train.writelines(all_data[:int(len(all_data)*0.8)])
        f_test.writelines(all_data[int(len(all_data)*0.8)+1:int(len(all_data)*0.9)])
        f_valid.writelines(all_data[int(len(all_data)*0.9)+1:])