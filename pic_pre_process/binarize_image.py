import os
from PIL import Image


def binarize_image(image_path, threshold=128):
    image = Image.open(image_path)
    image = image.convert('L')  # 转换为灰度图像
    binary_image = image.point(lambda p: p > threshold and 255)
    return binary_image

"""
    遍历目录下的所有图片，将图片二值化并保存
"""
def process_directory(directory, output_directory):
    # 确保输出目录存在
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 遍历目录下的所有文件
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg')):
            image_path = os.path.join(directory, filename)
            binary_image = binarize_image(image_path)

            # 保存二值化后的图片
            binary_image_path = os.path.join(output_directory, filename)
            binary_image.save(binary_image_path)
            print(f"图片已二值化并保存: {binary_image_path}")

if __name__ == "__main__":
    class_name_list = ['movie','classics','education','travel','biology']
    for class_name in class_name_list:
        directory = 'F:/desktop/图像素材/原图片/' + class_name  # 替换为你的图片目录
        output_directory = 'F:/desktop/图像素材/二值化图片/' + class_name  # 替换为你希望保存二值化图片的目录
        # 图像二值化处理
        process_directory(directory, output_directory)
        print(class_name + "二值化处理完成")