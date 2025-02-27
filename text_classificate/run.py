#!/usr/bin/python3
# -*- coding:utf-8 -*-

# 作者：高培骏
# 创建日期：2024年10月9日
# 版本：1.0
# 用途：树莓派中断事件实现拍照和识别文字
# 适用环境：树莓派4B
# 适用语言：Python3

# 作者：高培骏
# 创建日期：2025年2月25日
# 版本：1.1
# 用途：将函数的功能进行拆分，使得代码更加清晰

# 作者：高培骏
# 创建日期：2025年2月26日
# 版本：1.2
# 用途：添加语音播报功能


import RPi.GPIO as GPIO
import time
import torch
import pytesseract
import pickle as pkl
import cv2
from JQ8900Controller import JQ8900Controller
from aligo import Aligo
from importlib import import_module
from PIL import Image


def replace_to_blank(text, str_list):
    for i in range(len(str_list)):
        text = text.replace(str_list[i], '')
    return text

def split_text(text):
    # 先使用'。'分割
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

# 定义中断事件回调函数：执行拍照事件
def button_callback1(channel):
    print('Button1 pressed!')
    # 播放按键1声音：7
    controller.uart2_play(7)
    time.sleep(1)
    # 使用LED进行显示
    GPIO.output(4, GPIO.HIGH)
    GPIO.output(17, GPIO.HIGH)
    GPIO.output(27, GPIO.HIGH)
    GPIO.output(22, GPIO.HIGH)
    GPIO.output(18, GPIO.HIGH)
    time.sleep(0.2)
    GPIO.output(4, GPIO.LOW)
    GPIO.output(17, GPIO.LOW)
    GPIO.output(27, GPIO.LOW)
    GPIO.output(22, GPIO.LOW)
    GPIO.output(18, GPIO.LOW)
    time.sleep(0.2)

    # 写入日志
    with open('/home/pi/dc/log.txt', mode='a') as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        f.write(f'\t\tButton1 pressed\n')

    # 将按键状态写入文档中
    # 初始化摄像头，0通常是默认的USB摄像头
    cap = cv2.VideoCapture(0)
    # 检查摄像头是否成功打开
    if not cap.isOpened():
        # 写入日志
        with open('/home/pi/dc/log.txt', mode='a') as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            f.write(f'\t\tCamera open failed\n')
        # 播放错误声音：8
        controller.uart2_play(8)

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
        
        # 播放拍照成功声音：9
        controller.uart2_play(9)
        time.sleep(1)
        
        # 上传图片
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

# 图片的预处理
def preprocess_image(pic_path):
    threshold = 128
    image = Image.open(pic_path)
    image = image.convert('L')  # 转换为灰度图像
    binary_image = image.point(lambda p: p > threshold and 255)

    return binary_image

# 获取语句列表
def get_sentences(binary_image):
    text = pytesseract.image_to_string(binary_image, lang='chi_sim', config='--psm 6 --oem 3')
    # 获取停用词：
    stopwords_list = ['“','”','(',')','【','】','{','}','[',']','〔','〕','〈','〉',
                  '《','》','（','）','〖','〗','『','』','「','」','<','>',
                  '@', '#', '$', '…', '^', '&', '*', '|', '/','\\','"','：','；',
                  '~','+','=','_',',','·','丨','‖','〇','″',':','.','、',"'",'`',
                  '′',';','%','!','！','-','?','？','\'','"',
                  '0','1','2','3','4','5','6','7','8','9',
                  'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q',
                  'r','s','t','u','v','w','x','y','z','A','B','C','D','E','F','G','H',
                  'I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
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
        token = tokenizer(text)
        seq_len = len(token)
        
        # 过滤空序列
        if seq_len == 0:
            continue  # 跳过空句子
        
        # 截断和填充
        if pad_size:
            if seq_len < pad_size:
                token.extend([vocab.get(PAD)] * (pad_size - seq_len))
            else:
                token = token[:pad_size]
                seq_len = pad_size  # 确保 seq_len 不超过 pad_size
        
        # 将token转换为数字
        for word in token:
            words_line.append(vocab.get(word, vocab.get(UNK)))
        
        contents.append((words_line, 0, seq_len))
    return contents

# 定义中断事件回调函数：识别文字、判断类型、控制LED
def button_callback2(channel):
    print("Button2 pressed!")
    # 播放按键2声音：10
    controller.uart2_play(10)
    time.sleep(1)
    # 使用LED进行显示
    for i in range(0, 2):
        GPIO.output(4, GPIO.HIGH)
        GPIO.output(17, GPIO.HIGH)
        GPIO.output(27, GPIO.HIGH)
        GPIO.output(22, GPIO.HIGH)
        GPIO.output(18, GPIO.HIGH)
        time.sleep(0.2)
        GPIO.output(4, GPIO.LOW)
        GPIO.output(17, GPIO.LOW)
        GPIO.output(27, GPIO.LOW)
        GPIO.output(22, GPIO.LOW)
        GPIO.output(18, GPIO.LOW)
        time.sleep(0.2)
    
    # 写入日志
    with open('/home/pi/dc/log.txt', mode='a') as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        f.write(f"\t\tButton2 pressed\n")

    # 获取模型
    x = import_module('models.TextRNN')
    config = x.Config('/home/pi/dc/content', 'embedding_SougouNews.npz')
    model = x.Model(config)
    model.load_state_dict(torch.load('/home/pi/dc/content/saved_dict/TextRNN.ckpt', map_location='cpu', weights_only=True))
    model.eval()
        
    # 获取原图片地址并二值化处理
    pic_path = '/home/pi/dc/images/pic.jpg'

    # 图片的预处理
    binary_image = preprocess_image(pic_path)

    # 获取图片中的文字
    contents = get_sentences(binary_image)

    # 再次过滤空数据
    if not contents:
        print("No valid sentences after preprocessing!")
        # 播放错误声音：11
        controller.uart2_play(11)
        time.sleep(1)
        # 设置流水线LED来表示出错
        GPIO.output(4, GPIO.HIGH)
        time.sleep(0.1)
        GPIO.output(4, GPIO.LOW)
        time.sleep(0.1)
        GPIO.output(17, GPIO.HIGH)
        time.sleep(0.1)
        GPIO.output(17, GPIO.LOW)
        time.sleep(0.1)
        GPIO.output(27, GPIO.HIGH)
        time.sleep(0.1)
        GPIO.output(27, GPIO.LOW)
        time.sleep(0.1)
        GPIO.output(22, GPIO.HIGH)
        time.sleep(0.1)
        GPIO.output(22, GPIO.LOW)
        time.sleep(0.1)
        GPIO.output(18, GPIO.HIGH)
        time.sleep(0.1)
        GPIO.output(18, GPIO.LOW)
        time.sleep(0.1)

        # 写入日志
        with open('/home/pi/dc/log.txt', mode='a') as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            f.write(f"\t\tNo valid sentences after preprocessing!\n")

    # 构造输入张量
    x = torch.tensor([item[0] for item in contents], dtype=torch.long)
    y = torch.tensor([item[2] for item in contents], dtype=torch.long)
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
            # 播报movie
            controller.uart2_play(1)
            time.sleep(1)
        elif text_class == 'classics':
            GPIO.output(17, GPIO.HIGH)  # 点亮LED
            # 播报classics
            controller.uart2_play(2)
            time.sleep(1)
        elif text_class == 'educatioon':
            GPIO.output(27, GPIO.HIGH)  # 点亮LED
            # 播报education
            controller.uart2_play(3)
            time.sleep(1)
        elif text_class == 'travel':
            GPIO.output(22, GPIO.HIGH)  # 点亮LED
            # 播报travel
            controller.uart2_play(4)
            time.sleep(1)
        elif text_class == 'biology':
            GPIO.output(18, GPIO.HIGH)  # 点亮LED
            # 播报biology
            controller.uart2_play(5)
            time.sleep(1)


# 设置使用BCM编码
GPIO.setmode(GPIO.BCM)
# 控制语音播报
controller = JQ8900Controller(port='/dev/ttyUSB0', baudrate=9600)
# 设置音量（20级）
controller.set_volume(20)

time.sleep(1)

# 设置GPIO
# 设置LED灯为输出模式
GPIO.setup(4, GPIO.OUT)
GPIO.setup(17, GPIO.OUT)
GPIO.setup(27, GPIO.OUT)
GPIO.setup(22, GPIO.OUT)
GPIO.setup(18, GPIO.OUT)
# 设置按键为输入模式
GPIO.setup(20, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(21, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# 默认低电平
GPIO.output(4, GPIO.LOW)  # 点亮LED
GPIO.output(17, GPIO.LOW)  # 点亮LED
GPIO.output(27, GPIO.LOW)  # 点亮LED
GPIO.output(22, GPIO.LOW)  # 点亮LED
GPIO.output(18, GPIO.LOW)  # 点亮LED


# 播放初始化声音：6
controller.uart2_play(6)
# 闪烁两次表示开机自启动成功
for i in range(0, 3):
    GPIO.output(4, GPIO.HIGH)
    GPIO.output(17, GPIO.HIGH)
    GPIO.output(27, GPIO.HIGH)
    GPIO.output(22, GPIO.HIGH)
    GPIO.output(18, GPIO.HIGH)
    time.sleep(0.2)
    GPIO.output(4, GPIO.LOW)
    GPIO.output(17, GPIO.LOW)
    GPIO.output(27, GPIO.LOW)
    GPIO.output(22, GPIO.LOW)
    GPIO.output(18, GPIO.LOW)
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