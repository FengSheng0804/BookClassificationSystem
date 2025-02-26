import time
from importlib import import_module
import torch
from PIL import Image
import pytesseract
import pickle as pkl

def replace_to_blank(text, str_list):
    for i in range(len(str_list)):
        text = text.replace(str_list[i], '')
    return text

# 修改后的 split_text 函数（过滤空句子）
def split_text(text):
    split_list1 = text.split('。')
    final_list = []
    for sentence1 in split_list1:
        split_list2 = sentence1.split('，')
        valid_sentences = []
        for sentence_str in split_list2:
            # 过滤空字符串和长度过短的句子
            if len(sentence_str.strip()) >= 2:  # 调整为至少2个字符
                valid_sentences.append(sentence_str.strip())
        if valid_sentences:
            final_list.append(valid_sentences)
    return final_list


# 获取模型
x = import_module('models.TextCNN')
config = x.Config('./text_classificate/content', 'embedding_SougouNews.npz')
model = x.Model(config)
model.load_state_dict(torch.load('./text_classificate/content/saved_dict/TextCNN.ckpt', map_location='cpu'))
model.eval()
    
# 获取原图片地址并二值化处理
threshold = 128
pic_path = './text_classificate/images/pic.jpg'
image = Image.open(pic_path)
image = image.convert('L')  # 转换为灰度图像
binary_image = image.point(lambda p: p > threshold and 255)

# 获取二值化图片的文字
text = pytesseract.image_to_string(binary_image, lang='chi_sim', config='--psm 6 --oem 3')
# 获取停用词：
stopwords_list = ['“', '”', '(', ')', ':', '.', '、', "'", '`', '【', '】', '{', '}',
                      '〔', '〕', '〈', '〉', ';', '《', '》', '%', '!', '！', '-', '?', '？',
                      '@', '#', '$', '……', '^', '&', '*', '|', '/', '\\', '"',
                      '~', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
# 将停用词添加到这些符号中
with open('./text_classificate/content/data/cn_stopwords.txt', 'r', encoding='utf-8') as f:
    stopwords = f.readlines()
    for word in stopwords:
        stopwords_list.append(word.strip())
# 将所有遇到的','转成'，'方便后续切割
text = replace_to_blank("".join(text.split()), stopwords_list).replace(',', '，')

# 获取语句列表
content = []
text_lists = split_text(text)

for text_list in text_lists:
    for text in text_list:
        content.append(text)

# 使用分词形式
ues_word = False
if ues_word:
    tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
# 使用分字形式
else:
    tokenizer = lambda x: [y for y in x]  # char-level

# 获取vocab.pkl
vocab = pkl.load(open('./text_classificate/content/data/vocab.pkl', 'rb'))

# 定义变量
pad_size = 14
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号
contents = []

for text in content:
    words_line = []
    token = tokenizer(text)
    seq_len = len(token)
    
    # 过滤空序列
    if seq_len == 0:
        continue  # 跳过空句子
        
    if pad_size:
        if seq_len < pad_size:
            token.extend([vocab.get(PAD)] * (pad_size - seq_len))
        else:
            token = token[:pad_size]
            seq_len = pad_size  # 确保 seq_len 不超过 pad_size
            
    for word in token:
        words_line.append(vocab.get(word, vocab.get(UNK)))
    
    contents.append((words_line, 0, seq_len))

# 再次过滤空数据
if not contents:
    print("No valid sentences after preprocessing!")
    exit()

# 构造输入张量
x = torch.tensor([item[0] for item in contents], dtype=torch.long)
y = torch.tensor([item[2] for item in contents], dtype=torch.long)
data = (x, y)

# 写入日志
with open('./text_classificate/log.txt', mode='a') as f:
    f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    f.write(f'\t\tGet input successfully\n')

# 模型预测
with torch.no_grad():  # 确保不会计算梯度
    outputs = model(data)
    _, predicted = torch.max(outputs, 1)  # 获取最大值索引
    predicted = predicted.tolist()

    # 获取到个数对应的字典
    num_dict = {
        'movie': predicted.count(0),
        'classics': predicted.count(1),
        'education': predicted.count(2),
        'travel': predicted.count(3),
        'biology': predicted.count(4)
    }
    text_class = max(num_dict, key=num_dict.get)
    print('The pic_class is ' + text_class)