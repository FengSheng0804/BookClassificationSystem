#!/usr/bin/python3
# -*- coding:utf-8 -*-

# 作者：高培骏
# 创建日期：2024年10月9日
# 版本：1.0


import RPi.GPIO as GPIO
import time
from importlib import import_module
import torch
from PIL import Image
import pytesseract
import pickle as pkl
import cv2
import traceback            #获取报错信息并写入文档
from aligo import Aligo


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

# 定义中断事件回调函数：执行拍照事件
def button_callback1(channel):
    print('Button1 pressed!')
    
    # 写入日志
    with open('/home/pi/dc/log.txt', mode='a') as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        f.write(f'\t\tButton1 pressed\n')

    # 使用LED进行显示
    GPIO.output(4, GPIO.HIGH)
    GPIO.output(17, GPIO.HIGH)
    GPIO.output(27, GPIO.HIGH)
    GPIO.output(18, GPIO.HIGH)
    GPIO.output(22, GPIO.HIGH)
    time.sleep(0.2)
    GPIO.output(4, GPIO.LOW)
    GPIO.output(17, GPIO.LOW)
    GPIO.output(27, GPIO.LOW)
    GPIO.output(18, GPIO.LOW)
    GPIO.output(22, GPIO.LOW)
    
    # 延时实现按键防抖效果
    # time.sleep(1)

    # 将按键状态写入文档中
    # 初始化摄像头，0通常是默认的USB摄像头
    cap = cv2.VideoCapture(0)
    # 检查摄像头是否成功打开
    if not cap.isOpened():
        # 写入日志
        with open('/home/pi/dc/log.txt', mode='a') as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            f.write(f'\t\tCamera open failed\n')
        exit()

    # 捕获一帧图像
    ret, frame = cap.read()

    # 如果成功捕获图像，ret会为True
    if ret:
        # 定义图片的文件名和路径
        filename = '/home/pi/dc/images/pic.jpg'

        # 保存图片
        cv2.imwrite(filename, frame)
        print(f"picture saved as {filename}")
        # 写入日志
        with open('/home/pi/dc/log.txt', mode='a') as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            f.write(f"\t\tpicture saved as {filename}\n")
        
        ali = Aligo()
        user_name = ali.get_user().user_name
        # 写入日志
        with open('/home/pi/dc/log.txt', mode='a') as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            f.write(f"\t\taligo {user_name} login successfully\n")
        # 上传图片
        local_file = r"/home/pi/dc/images/pic.jpg"
        up_file = ali.upload_file(local_file)
        # 写入日志
        with open('/home/pi/dc/log.txt', mode='a') as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            f.write(f"\t\tpicture upload successfully\n")

    else:
        # 写入日志
        print('get no picture')
        with open('/home/pi/dc/log.txt', mode='a') as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            f.write(f"\t\tget no picture\n")

    # 释放摄像头资源
    cap.release()

# 定义中断事件回调函数：识别文字、判断类型、控制LED
def button_callback2(channel):
    print("Button2 pressed!")
    
    # 写入日志
    with open('/home/pi/dc/log.txt', mode='a') as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        f.write(f"\t\tButton2 pressed\n")

    # 使用LED进行显示
    for i in range(0, 2):
        GPIO.output(4, GPIO.HIGH)
        GPIO.output(17, GPIO.HIGH)
        GPIO.output(27, GPIO.HIGH)
        GPIO.output(18, GPIO.HIGH)
        GPIO.output(22, GPIO.HIGH)
        time.sleep(0.2)
        GPIO.output(4, GPIO.LOW)
        GPIO.output(17, GPIO.LOW)
        GPIO.output(27, GPIO.LOW)
        GPIO.output(18, GPIO.LOW)
        GPIO.output(22, GPIO.LOW)
        time.sleep(0.2)
    
    # 写入日志
    with open('/home/pi/dc/log.txt', mode='a') as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        f.write(f"\t\tSet led successfully\n")

    # 获取模型
    x = import_module('models.TextRNN')
    config = x.Config('/home/pi/dc/content', 'embedding_SougouNews.npz')
    model = x.Model(config)
    model.load_state_dict(torch.load('/home/pi/dc/content/saved_dict/TextRNN.ckpt', map_location='cpu', weights_only=True))
    model.eval()
        
    # 获取原图片地址并二值化处理
    threshold = 128
    pic_path = '/home/pi/dc/images/pic.jpg'
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
    with open('/home/pi/dc/content/data/cn_stopwords.txt', 'r', encoding='utf-8') as f:
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
    
    # 写入日志
    with open('/home/pi/dc/log.txt', mode='a') as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        f.write(f"\t\tGet text successfully\n")

    # 使用分词形式
    ues_word = False
    if ues_word:
        tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
    # 使用分字形式
    else:
        tokenizer = lambda x: [y for y in x]  # char-level

    # 获取vocab.pkl
    vocab = pkl.load(open('/home/pi/dc/content/data/vocab.pkl', 'rb'))

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
    
    # 写入日志
    with open('/home/pi/dc/log.txt', mode='a') as f:
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
        with open('/home/pi/dc/log.txt', mode='a') as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            f.write(f"\t\tThe pic_class is {text_class}\n")

        if text_class == 'movie':
            GPIO.output(4, GPIO.HIGH)  # 点亮LED
        elif text_class == 'classics':
            GPIO.output(17, GPIO.HIGH)  # 点亮LED
        elif text_class == 'educatioon':
            GPIO.output(27, GPIO.HIGH)  # 点亮LED
        elif text_class == 'travel':
            GPIO.output(18, GPIO.HIGH)  # 点亮LED
        elif text_class == 'biology':
            GPIO.output(22, GPIO.HIGH)  # 点亮LED


# 设置使用BCM编码
GPIO.setmode(GPIO.BCM)

# 设置GPIO
# 设置LED灯为输出模式
GPIO.setup(4, GPIO.OUT)
GPIO.setup(17, GPIO.OUT)
GPIO.setup(27, GPIO.OUT)
GPIO.setup(18, GPIO.OUT)
GPIO.setup(22, GPIO.OUT)
# 设置按键为输入模式
GPIO.setup(20, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(21, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# 默认低电平
GPIO.output(4, GPIO.LOW)  # 点亮LED
GPIO.output(17, GPIO.LOW)  # 点亮LED
GPIO.output(27, GPIO.LOW)  # 点亮LED
GPIO.output(18, GPIO.LOW)  # 点亮LED
GPIO.output(22, GPIO.LOW)  # 点亮LED

# 闪烁两次表示开机自启动成功
for i in range(0, 3):
    GPIO.output(4, GPIO.HIGH)
    GPIO.output(17, GPIO.HIGH)
    GPIO.output(27, GPIO.HIGH)
    GPIO.output(18, GPIO.HIGH)
    GPIO.output(22, GPIO.HIGH)
    time.sleep(0.2)
    GPIO.output(4, GPIO.LOW)
    GPIO.output(17, GPIO.LOW)
    GPIO.output(27, GPIO.LOW)
    GPIO.output(18, GPIO.LOW)
    GPIO.output(22, GPIO.LOW)
    time.sleep(0.2)

time.sleep(2)

# 为按钮引脚添加中断检测，当按钮被按下时（从高电平到低电平），调用button_callback函数
GPIO.add_event_detect(20, GPIO.FALLING, callback=button_callback1, bouncetime=200)
GPIO.add_event_detect(21, GPIO.FALLING, callback=button_callback2, bouncetime=200)

with open('/home/pi/dc/log.txt', mode='a') as f:
    f.write("\n\n")
    f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    f.write(f"\t\tInitialize successfully\n")

warning_str = 'Waiting for button'

try:
    while True:
        time.sleep(1)
        print(warning_str)
except KeyboardInterrupt:
    GPIO.cleanup()
