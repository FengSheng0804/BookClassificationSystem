from pic_to_text_by_OCR import get_pic_text_main, split_text
from synthetic_dataset import synthesis_dataset


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
    synthesis_dataset('./dataset_get/class/', class_name_list)
    print("数据集合成完成")