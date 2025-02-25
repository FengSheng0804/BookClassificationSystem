import os
import pytesseract
from PIL import Image

"""
    将字符串中的指定字符替换为空
"""
def replace_to_blank(text, str_list):
    for i in range(len(str_list)):
        text = text.replace(str_list[i], '')
    return text

"""
    获取图片中的文字
"""
def get_pic_text(path):
    image = Image.open(path)
    text = pytesseract.image_to_string(image, lang='chi_sim', config='--psm 6 --oem 3')
    print(path + '图片文字提取完成')
    # 获取停用词：
    stopwords_list = ['“','”','(',')',':','.','、',"'",'`','【','】','{','}',
                      '〔','〕','〈','〉',';','《','》','%','!','！','-','?','？',
                      '@', '#', '$', '……', '^', '&', '*', '|', '/','\\','"',
                      '~','0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    # 将停用词添加到这些符号中
    with open('./dataset_get/cn_stopwords.txt', 'r', encoding='utf-8') as f:
        stopwords = f.readlines()
        for word in stopwords:
            stopwords_list.append(word.strip())
    # 将所有遇到的','转成'，'方便后续切割
    text = replace_to_blank("".join(text.split()), stopwords_list).replace(',', '，')
    return text

"""
    获取目录下所有图片的文字
"""
def get_pic_text_main(dir_path):
    # 调用get_pic_text函数将所有的照片拼接起来
    text = ""
    for filename in os.listdir(dir_path):
        if filename.lower().endswith(('.png', '.jpg')):
            image_path = os.path.join(dir_path, filename)
            text += get_pic_text(image_path)
    return text

"""
    将文字分割成句子
"""
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
