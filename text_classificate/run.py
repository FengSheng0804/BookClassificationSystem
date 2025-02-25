from importlib import import_module
import torch
from PIL import Image
import pytesseract
import pickle as pkl

def replace_to_blank(text, str_list):
    for i in range(len(str_list)):
        text = text.replace(str_list[i], '')
    return text

def split_text(text):
    # 先使用'。'分割
    split_list1 = text.split('。')
    final_list = []
    for sentence1 in split_list1:
        # 再使用'，'分割
        split_list2 = sentence1.split('，')
        for sentence_str in split_list2:
            # 删除比较少的字符
            if len(sentence_str) < 4:
                split_list2.remove(sentence_str)
        if split_list2:
            final_list.append(split_list2)
    return final_list

# 获取模型
x = import_module('models.TextRNN')
config = x.Config('content', 'embedding_SougouNews.npz')
model = x.Model(config)
model.load_state_dict(torch.load('./content/saved_dict/TextRNN.ckpt'))
model.eval()

# 获取原图片地址并二值化处理
threshold=128
pic_path = './images/source.jpg'
image = Image.open(pic_path)
image = image.convert('L')  # 转换为灰度图像
binary_image = image.point(lambda p: p > threshold and 255)

# 获取二值化图片的文字
text = pytesseract.image_to_string(binary_image, lang='chi_sim', config='--psm 6 --oem 3')
print('图片文字提取完成')
print(text)
# 获取停用词：
stopwords_list = ['“','”','(',')',':','.','、',"'",'`','【','】','{','}',
                  '〔','〕','〈','〉',';','《','》','%','!','！','-','?','？',
                  '@', '#', '$', '……', '^', '&', '*', '|', '/','\\','"',
                  '~','0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
# 将停用词添加到这些符号中
with open('./content/data/cn_stopwords.txt', 'r', encoding='utf-8') as f:
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
vocab = pkl.load(open('./content/data/vocab.pkl', 'rb'))
print(f"Vocab size: {len(vocab)}")

# 定义变量
pad_size = 14
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号
contents = []

for text in content:
    words_line = []
    # 使用tokenizer函数对文本内容进行分词
    token = tokenizer(text)
    # 计算分词后的序列长度
    seq_len = len(token)

    if pad_size:
        # 如果长度比pad_size小，则补长
        if len(token) < pad_size:
            token.extend([vocab.get(PAD)] * (pad_size - len(token)))
        # 否则，截取前pad_size个
        else:
            token = token[:pad_size]
            seq_len = pad_size
    # 根据vocab.pkl将word转化成对应的id
    for word in token:
        words_line.append(vocab.get(word, vocab.get(UNK)))
    contents.append((words_line, 0, seq_len))

x = []
y = []
for tup in contents:
    x.append(tup[0])
    y.append(tup[2])


x = torch.tensor(x)
y = torch.tensor(y)
data = (x, y)

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
    print('该文本内容的预测结果是：' + max(num_dict,key=num_dict.get))