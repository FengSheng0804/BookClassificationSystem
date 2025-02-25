import cv2

# 初始化摄像头，0通常是默认的USB摄像头
cap = cv2.VideoCapture(0)

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# 捕获一帧图像
ret, frame = cap.read()

# 如果成功捕获图像，ret会为True
if ret:
    # 定义图片的文件名和路径
    filename = '/home/pi/dc/images/source.jpg'
    
    # 保存图片
    cv2.imwrite(filename, frame)
    print(f"图片已保存为 {filename}")
else:
    print("无法捕获图像")

# 释放摄像头资源
cap.release()
