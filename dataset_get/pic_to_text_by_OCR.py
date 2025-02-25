import os
import random
import pytesseract
from PIL import Image


"""
    本文件用于将图片转换为文字
    1. 使用pytesseract将图片转换为文字
    2. 将文字分割成句子
    3. 将句子写入文件
    4. 合成数据集
"""


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

"""
    合成数据集
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
    with (open('../text_classificate/content/data/train.txt', 'w', encoding='utf-8') as f_train,
          open('../text_classificate/content/data/test.txt', 'w', encoding='utf-8') as f_test,
          open('../text_classificate/content/data/dev.txt', 'w', encoding='utf-8') as f_valid):
        f_train.writelines(all_data[:int(len(all_data)*0.8)])
        f_test.writelines(all_data[int(len(all_data)*0.8)+1:int(len(all_data)*0.9)])
        f_valid.writelines(all_data[int(len(all_data)*0.9)+1:])


if __name__ == '__main__':
    class_name_list = ['movie','classics','education','travel','biology']
    for class_name in class_name_list:
        output_directory = 'F:/desktop/图像素材/二值化图片/' + class_name  # 替换为你希望保存二值化图片的目录

        # 获取二值化图片，并处理成文字
        print(class_name + '文字提取开始')
        text = get_pic_text_main(output_directory)
        print(class_name + '文字提取完成')
        split_list = split_text(text)
        with open(f'./dataset_get/class/{class_name}.txt', 'w', encoding='utf-8') as f:
            for list1 in split_list:
                for sentence in list1:
                    if len(sentence) >= 4:
                        f.write(sentence + '\t' + str(class_name_list.index(class_name)) + '\n')
        print(class_name + '文字分割完成')
    synthesis_dataset('',class_name_list)
    print('数据集合成完成')


    # # 使用项目中的../text_classificate/images/source.jpg进行测试
    # print(get_pic_text('../text_classificate/images/source.jpg'))
