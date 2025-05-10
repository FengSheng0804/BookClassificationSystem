import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from math import atan2, degrees, sqrt
from itertools import combinations
from sklearn.cluster import DBSCAN, KMeans
from scipy.interpolate import CubicSpline
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d
from torchvision import transforms
from image_segmentation.models.Unet import UNet
from dataset_get.pic_to_text_by_OCR import get_pic_text
from image_segmentation.utils import resize_rgb_image

# 显示图像
def show_image(img):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# 处理数据集
def process_dataset():
    class_name_list = ['movie','classics','education','travel','biology']
    for class_name in class_name_list:
        directory = 'F:/desktop/图像素材/原图片/' + class_name  # 替换为你的图片目录
        output_directory = 'F:/desktop/图像素材/二值化图片/' + class_name  # 替换为你希望保存二值化图片的目录
        # 图像二值化处理
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

# 二值化图像
def binarize_image(image_path, threshold=128):
    image = Image.open(image_path)
    image = image.convert('L')  # 转换为灰度图像
    binary_image = image.point(lambda p: p > threshold and 255)

    return binary_image

# OCR前预处理图片
def process_before_OCR(image):
    # 灰度化
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 降噪
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    # 自适应二值化
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 3)
    # 形态学操作
    kernel = np.ones((2,2), np.uint8)
    processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return processed

# 删除小的连通组件
def remove_small_connected_components(mask, min_size):
    # 处理白色小区域，转为黑色
    num_labels_white, labels_white, stats_white, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    mask_white_processed = np.zeros(mask.shape, dtype=np.uint8)
    for i in range(1, num_labels_white):  # 跳过背景（0）
        if stats_white[i, cv2.CC_STAT_AREA] >= min_size:
            mask_white_processed[labels_white == i] = 255  # 保留大面积白色
    
    # show_image(mask_white_processed)

    # 将得到的大白色区域反转，小黑色区域变白
    mask_inv = 255 - mask_white_processed  # 反转图像，黑色区域变为白色
    num_labels_black, labels_black, stats_black, _ = cv2.connectedComponentsWithStats(mask_inv, connectivity=8)
    mask_inv_processed = np.zeros(mask_inv.shape, dtype=np.uint8)
    for i in range(1, num_labels_black):
        if stats_black[i, cv2.CC_STAT_AREA] >= min_size:
            mask_inv_processed[labels_black == i] = 255  # 保留反转后的大面积白色（即原黑色大区域）
    mask_black_processed = 255 - mask_inv_processed  # 反转回来，小黑色区域变白

    # show_image(mask_black_processed)
    
    # 合并结果：保留原大白色 + 原小黑色变白
    final_mask = cv2.bitwise_or(mask_white_processed, mask_black_processed)
    return final_mask

# 使用Unet进行图像分割
def predict_by_unet(origin_path, net, transform):
    img = cv2.imread(origin_path)
    resize_img = resize_rgb_image(origin_path)
    img_data=transform(resize_img).cuda()
    img_data=torch.unsqueeze(img_data,dim=0)
    net.eval()
    out=net(img_data)
    pred_mask = torch.argmax(out, dim=1).squeeze(0)  # [H,W]
    # 转换为numpy并调整数据类型
    mask_np = pred_mask.byte().cpu().numpy() * 255

    # 删除小的连通域
    mask_np = remove_small_connected_components(mask_np, 2000)

    # 将掩码恢复成原图大小
    mask_np_after = cv2.resize(mask_np, (img.shape[1], img.shape[0]))

    # 先闭运算填补边缘缝隙，再开运算消除毛刺
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(mask_np_after, cv2.MORPH_CLOSE, kernel, iterations=4)
    mask = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=10)

    # show_image(smoothed)

    return mask

# 应用掩码
def apply_mask(image_path, mask_path):
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image

# 自适应光照增强
def adaptive_lighting_enhancement(img_path):
    """与现有UNet预处理流程集成"""
    img = cv2.imread(img_path)
    # 确保输入为BGR格式的uint8图像
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    
    # 自动处理灰度图情况
    if len(img.shape) == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # 确保为3通道
    if img.shape[2] > 3:
        img = img[:,:,:3]

    # 使用LAB颜色空间优化光照补偿
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # CLAHE自适应直方图均衡
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    # 亮度通道融合
    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

# =========================裁剪黑边+图像倾斜纠正=========================
# 自动裁剪黑边
def auto_remove_black_border(img):
    """基于内容检测自动裁剪黑边"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                                            # 灰度化
    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)                             # 二值化

    # show_image(thresh)
    
    # 寻找内容边界
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]      # 查找轮廓
    cnt = max(contours, key=cv2.contourArea)                                                # 找到最大轮廓
    x,y,w,h = cv2.boundingRect(cnt)                                                         # 获取最小外接矩形

    # # ==================================== 可视化调试 ====================================
    # debug_img = img.copy()
    # cv2.drawContours(debug_img, [cnt], -1, (0,255,0), 2)
    # cv2.rectangle(debug_img, (x,y), (x+w,y+h), (0,255,0), 2)
    # show_image(debug_img)
    
    # 扩展5像素保留边缘
    x = max(0, x-5)                                                                         # 防止越界
    y = max(0, y-5)                                                                         # 防止越界
    w = min(w+10, img.shape[1]-x)                                                           # 防止越界
    h = min(h+10, img.shape[0]-y)                                                           # 防止越界
    
    return img[y:y+h, x:x+w]

# 纠正图像倾斜
def correct_book_rotation(img_path):
    # 1. 读取图像
    img = cv2.imread(img_path)
    height, width = img.shape[:2]                                                           # 获取图像尺寸
    
    # 2. 预处理
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                                            # 灰度化
    blur = cv2.GaussianBlur(gray, (7,7), 0)                                                 # 高斯模糊
    thresh = cv2.adaptiveThreshold(blur, 255,                                               # 自适应二值化
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 21, 10)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))                               # 创建结构元素
    gradient = cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, kernel)                         # 形态学梯度

    # 3. 查找轮廓
    contours, _ = cv2.findContours(gradient, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    # 查找轮廓
    
    # 4. 找到最大轮廓（书本主体）
    max_contour = max(contours, key=cv2.contourArea)                                        # 找到最大轮廓
    
    # 5. 获取最小外接矩形
    rect = cv2.minAreaRect(max_contour)                                                     # 获取最小外接矩形
    (cx, cy), (w, h), angle = rect                                                          # 获取中心点、宽高、旋转角度

    # 6. 角度修正逻辑（处理OpenCV的角度表示问题）
    if w < h:
        angle -= 90

    # 7. 旋转图像
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)                                       # 计算旋转矩阵
    rotated = cv2.warpAffine(img, M, (width, height),                                       # 执行旋转
                           flags=cv2.INTER_CUBIC, 
                           borderMode=cv2.BORDER_REPLICATE)
    
    # 8. 删除黑边
    rotated_after = auto_remove_black_border(rotated)                                       # 自动裁剪黑边

    return rotated_after

# =========================分页处理+分界线倾斜矫正=========================
# 选择最佳分界点
def find_vertical_spine_points(points, img_size):
    """
    检测垂直中缝的上下端点
    :param points: 候选角点集合 (N,2)
    :param img_size: 图像尺寸 (width, height)
    :return: (upper_point, lower_point)
    """
    w, h = img_size                                                                         # 获取图像尺寸
    center_x = w // 2                                                                       # 获取图像中心坐标
    
    # ========== 1. 筛选位于垂直中线附近15%区域的点 ==========
    vertical_mask = (points[:,0] > center_x - w*0.075) & (points[:,0] < center_x + w*0.075)
    vertical_points = points[vertical_mask]                                                 # 筛选位于垂直中线附近15%区域的点

    print('vertical_points的个数：',len(vertical_points))

    # ========== 2. 降级处理：若无足够点，选择x最接近中心的2个点 ==========
    if len(vertical_points) < 2:
        print("vertical_points的个数小于2，降级处理")
        vertical_points = sorted(points, key=lambda p: abs(p[0]-center_x))[:2]              # 选择x最接近中心的2个点
    
    # ========== 3. 按垂直位置排序并选择上下端点 ==========
    sorted_points = sorted(vertical_points, key=lambda p: p[1])                             # 按垂直位置排序

    return sorted_points[0], sorted_points[-1]

# 计算旋转角度
def calculate_rotation_angle(upper, lower):
    """计算需要旋转到垂直的角度"""
    dx = lower[0] - upper[0]                                                                # 计算x方向差值
    dy = lower[1] - upper[1]                                                                # 计算y方向差值
    current_angle = degrees(atan2(dy, dx))                                                  # 计算当前角度
    return current_angle - 90                                                               # 调整到垂直方向

# 书页分割
def find_book_corners_and_split(img_path):
    # ========== 1. 图像预处理 ==========
    img = cv2.imread(img_path)

    h, w = img.shape[:2]
    
    # 多尺度预处理
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                                            # 灰度化
    blur = cv2.bilateralFilter(gray, 9, 30, 30)                                             # 双边滤波
    edges = cv2.Canny(blur, 30, 100)                                                        # 边缘检测

    # 形态学强化轮廓
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))                               # 创建结构元素
    dilated = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)                # 闭运算

    # ========== 2. 轮廓提取与处理 ==========
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)     # 查找轮廓
    if not contours:
        raise ValueError("未检测到有效轮廓")
    
    main_contour = max(contours, key=cv2.contourArea)                                       # 找到最大轮廓
    points = np.squeeze(main_contour, axis=1)                                               # 提取轮廓点集

    # ========== 3. 候选角点检测 ==========
    def smooth_points(points, sigma=2):
        """高斯平滑点云坐标"""
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        x_smoothed = gaussian_filter1d(x, sigma=sigma)
        y_smoothed = gaussian_filter1d(y, sigma=sigma)
        return list(zip(x_smoothed, y_smoothed))

    def curvature_angle(points, idx, window):
        """改进的曲率角度计算（基于滑动窗口向量方向）"""
        start = max(0, idx - window)
        end = min(len(points), idx + window)
        prev_segment = points[start:idx]
        next_segment = points[idx:end]
        
        # 至少需要两个点计算方向
        if len(prev_segment) < 1 or len(next_segment) < 1:
            return 180.0
        
        # 计算前向和后向向量（起点到终点）
        v1 = np.array(points[idx]) - np.array(prev_segment[0])
        v2 = np.array(next_segment[-1]) - np.array(points[idx])
        
        # 处理零向量
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 180.0
        
        # 计算夹角（0~180度）
        cos_theta = np.dot(v1, v2) / (norm_v1 * norm_v2)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)  # 避免数值误差
        return np.degrees(np.arccos(cos_theta))

    # 生成平滑后的点云
    smoothed_points = smooth_points(points, sigma=1)

    # 多尺度窗口检测（小窗口+大窗口）
    candidates = []
    window_sizes = [
        max(3, int(0.003 * len(points))),  # 小窗口：敏感局部特征
        max(5, int(0.01 * len(points)))    # 大窗口：稳定大范围变化
    ]

    for i in range(len(smoothed_points)):
        for ws in window_sizes:
            angle = curvature_angle(smoothed_points, i, ws)
            # 放宽角度条件并允许不同尺度检测
            if 30 < angle < 150:
                candidates.append(points[i])  # 保留原始坐标
                break  # 满足任一尺度即加入候选

    # 去重（如果同一点被多个尺度检测到）
    candidates = list({tuple(p): p for p in candidates}.values())
    
    # ========== 4. 角点聚类筛选 ==========
    clustering = DBSCAN(eps=w*0.03, min_samples=3).fit(candidates)                          # DBSCAN聚类
    labels = clustering.labels_                                                             # 获取标签
    
    # 提取有效聚类中心
    cluster_centers = []                                                                    # 聚类中心
    for label in set(labels):
        if label == -1: continue                                                            # 忽略噪声点
        cluster = np.array(candidates)[labels == label]                                     # 提取聚类
        cluster_centers.append(np.median(cluster, axis=0))                                  # 计算中心点
    
    # 按垂直位置排序
    final_corners = sorted(cluster_centers, key=lambda p: p[1])                             

    # ========== 5. 中缝检测与旋转 ==========
    try:
        upper, lower = find_vertical_spine_points(np.array(final_corners), (w, h))          # 寻找垂直中缝的上下端点
    except:
        # 降级处理：使用轮廓极值点
        upper = points[points[:,1].argmin()]                                                # 寻找最上方点
        lower = points[points[:,1].argmax()]                                                # 寻找最下方点

    rotate_angle = calculate_rotation_angle(upper, lower)                                   # 计算旋转角度
    
    # 执行旋转
    M = cv2.getRotationMatrix2D((w//2, h//2), rotate_angle, 1)                              # 计算旋转矩阵
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)                         # 执行旋转

    # ========== 6. 精确分页 ==========
    # 计算旋转后的中缝位置
    pts = np.array([upper, lower], dtype=np.float32).reshape(-1,1,2)                        # 构建中缝点集
    rotated_pts = cv2.transform(pts, M).squeeze()                                           # 计算旋转后的中缝位置
    
    # 取x坐标中值并限制范围
    split_x = int(np.median(rotated_pts[:,0]))                                              # 计算x坐标中值
    split_x = np.clip(split_x, int(w*0.45), int(w*0.55))                                    # 限制范围

    # 分割与边框处理
    border = 15                                                                             # 边框大小
    left = cv2.copyMakeBorder(rotated[:, :split_x], border, border, border, border,         # 左侧
                            cv2.BORDER_CONSTANT, value=(0,0,0))
    right = cv2.copyMakeBorder(rotated[:, split_x:], border, border, border, border,        # 右侧
                             cv2.BORDER_CONSTANT, value=(0,0,0))

    # # ==================================== 可视化调试 ====================================
    # # 绘制聚类中心，中缝，旋转后中缝
    # debug_img = img.copy()
    # for pt in candidates:
    #     cv2.circle(debug_img, tuple(pt.astype(int)), 5, (0,255,0), -1)
    # for pt in final_corners:
    #     cv2.circle(debug_img, tuple(pt.astype(int)), 10, (0,0,255), -1)
    # cv2.line(debug_img, tuple(upper.astype(int)), tuple(lower.astype(int)),
    #         (255,0,0), 3)
    # cv2.putText(debug_img, f"Rotate: {rotate_angle:.2f}°", (20,40),
    #            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 4)
    # show_image(debug_img)

    return left, right

# =========================水平矫正+垂直矫正=========================
# 得到有序的四个角点
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")                                                # 初始化坐标矩阵
    s = pts.sum(axis=1)                                                                     # 计算坐标和
    rect[0] = pts[np.argmin(s)]                                                             # 左上
    rect[2] = pts[np.argmax(s)]                                                             # 右下
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]                                                          # 右上
    rect[3] = pts[np.argmax(diff)]                                                          # 左下
    return rect

# 自动查找书页四个角点
def auto_detect_page_corners(img):
    # ========== 1. 图像预处理 ==========
    # 预处理强化文字对比度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                                            # 灰度化
    blur = cv2.bilateralFilter(gray, 15, 75, 75)                                            # 保边滤波
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16,16))                            # 创建CLAHE对象
    enhanced = clahe.apply(blur)                                                            # 应用CLAHE

    # 动态参数计算
    h, w = img.shape[:2]                                                                    # 获取图像尺寸

    # 动态计算窗口和C值：经过调整后，这个参数适合在光照自动补偿后的图像上使用
    block_size = max(25, int(min(h,w)/15*2.5)+1) | 1

    c_value = max(2, int(min(h,w)/200)) | 1
    
    # 二值化强化边缘
    thresh = cv2.adaptiveThreshold(enhanced, 255,                                           
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, block_size, c_value)

    # show_image(thresh)

    # 形态学重建（针对密集文字优化）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))                            # 创建结构元素
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)                # 闭运算
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)                 # 开运算

    # show_image(opened)

    # ========== 2. 轮廓提取与处理 ==========
    contours, hierarchy = cv2.findContours(opened, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 查找轮廓
    
    # 筛选有效轮廓
    valid_contours = []
    for i, cnt in enumerate(contours):
        # 层级校验（仅最外层父轮廓）
        if hierarchy[0][i][3] != -1:
            continue
        
        # 几何特征校验
        area = cv2.contourArea(cnt)                                                         # 计算面积
        perimeter = cv2.arcLength(cnt, True)                                                # 计算周长
        if area < (h*w*0.5) or perimeter < (h+w)*1.5:
            continue
            
        # 矩形度验证
        rect = cv2.minAreaRect(cnt)                                                         # 获取最小外接矩形
        box = cv2.boxPoints(rect)                                                           # 获取矩形四个角点
        box_area = cv2.contourArea(box)                                                     # 计算矩形面积
        if abs(1 - area/box_area) > 0.3:
            continue
            
        valid_contours.append(cnt)

    # # ==================================== 可视化调试 ====================================
    # # 绘制有效轮廓
    # debug_img = img.copy()
    # cv2.drawContours(debug_img, valid_contours, -1, (0,255,0), 4)
    # show_image(debug_img)

    # 选择最佳候选
    if not valid_contours:
        raise ValueError("未找到有效书页轮廓")
        
    page_contour = max(valid_contours, key=cv2.contourArea)                                 # 选择最大轮廓

    # ========== 3. 亚像素级角点精修 ==========
    # 逼近多边形，提取角点，通过调节epsilon参数可以控制精度，越小与原图像贴合度越高，角点越多
    epsilon = 0.005 * cv2.arcLength(page_contour, True)                                     # 计算周长
    approx = cv2.approxPolyDP(page_contour, epsilon, True)                                  # 多边形逼近
    corners = cv2.cornerSubPix(gray, np.float32(approx.reshape(-1,2)),                      # 亚像素级角点精修
                             (5,5), (-1,-1), 
                             (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    
    # # ==================================== 可视化调试 ====================================
    # # 绘制角点
    # debug_img = img.copy()
    # cv2.drawContours(debug_img, [corners.astype(int)], -1, (0,255,0), 4)
    # for pt in corners:
    #     cv2.circle(debug_img, tuple(pt.astype(int)), 10, (0,0,255), -1)
    # show_image(debug_img)

    # 获取全局垂直极值点，将上面两个角点的纵坐标调整为书页的最高点，下面两个角点的纵坐标调整为书页的最低点
    all_points = page_contour.reshape(-1,2)                                                 # 获取所有轮廓点
    top_point = all_points[np.argmin(all_points[:,1])]                                      # 最高点(Y最小)
    bottom_point = all_points[np.argmax(all_points[:,1])]                                   # 最低点(Y最大)
    
    (tl, tr, br, bl) = order_points(corners)
    adjusted_points = np.array([
        [tl[0], top_point[1] - 10],                                                         # 左上保持X，Y设为全局最高并预留10像素
        [tr[0], top_point[1] - 10],                                                         # 右上保持X，Y设为全局最高并预留10像素
        [br[0], bottom_point[1] + 10],                                                      # 右下保持X，Y设为全局最低并预留10像素
        [bl[0], bottom_point[1] + 10]                                                       # 左下保持X，Y设为全局最低并预留10像素
    ], dtype=np.float32)
    
    return adjusted_points

# 透视变换
def horizontal_warp_image(img, src_points):
    # ========== 1. 坐标排序验证 ==========
    if src_points.shape != (4, 2):                                                          # 验证坐标点数量
        raise ValueError("必须提供4个二维坐标点")

    # ========== 2. 智能尺寸计算 ==========
    (tl, tr, br, bl) = src_points                                                           # 获取四个角点
    width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))                           # 计算宽度
    height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))                          # 计算高度

    # ========== 3. 构建目标矩形（上下严格等宽） ==========
    dst = np.array([                                                                        # 构建目标矩形
        [0, 0],
        [width - 1, 0],  # 确保不越界
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")

    # ========== 4. 计算变换矩阵 ==========
    M = cv2.getPerspectiveTransform(src_points, dst)                                        # 计算变换矩阵
    
    # ========== 5. 执行变换（带抗锯齿处理） ==========
    warped = cv2.warpPerspective(                                                           # 执行透视变换
        img, M, 
        (int(width), int(height)),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REPLICATE
    )
    
    return warped


# 书页在垂直方向上展开
def vertical_warp_image(img, which_side, num_cells=40, k=1.3):
    # 如果是左边的页面，删除左边的10%
    if which_side == "left":
        img = img[:, int(img.shape[1] * 0.1):]
    # 如果是右边的页面，删除右边的10%
    elif which_side == "right":
        img = img[:, :-int(img.shape[1] * 0.1)]

    # show_image(img)

    # ========== 1. 图像预处理 ==========
    # 预处理强化文字对比度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                                                # 灰度化
    denoised = cv2.fastNlMeansDenoising(gray, h=15, templateWindowSize=7, searchWindowSize=7)   # 去噪
    blur = cv2.bilateralFilter(denoised, 20, 75, 75)                                            # 保边滤波
    clahe = cv2.createCLAHE(clipLimit=0.1, tileGridSize=(8,8))                                  # 创建CLAHE对象
    enhanced = clahe.apply(blur)                                                                # 应用CLAHE

    # show_image(enhanced)

    # 动态参数计算
    h, w = img.shape[:2]                                                                    # 获取图像尺寸
    scale = 0.07                                                                            # 缩放比例，经验值
    block_size = max(25, min(int(min(h, w) * scale) | 1, 101))                              # 计算块大小，确保为奇数
    C = max(2, int(min(h, w) / 150))                                                        # 计算C值，经验值
    
    # 二值化强化边缘
    thresh = cv2.adaptiveThreshold(enhanced, 255,                                           # 自适应二值化
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, block_size, C)

    # show_image(thresh)

    # 形态学重建（针对密集文字优化）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))                            # 创建结构元素
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)                # 闭运算
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)                 # 开运算

    # show_image(opened)

    # 多层轮廓分析
    # 使用cv2.CHAIN_APPROX_SIMPLE，只保留终点坐标，所以如果是矩形，只会返回4个坐标点
    # 使用cv2.CHAIN_APPROX_NONE，会存储所有的边界点，这样就会返回所有的轮廓点
    contours, hierarchy = cv2.findContours(opened, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # 查找轮廓

    # 筛选有效轮廓
    valid_contours = []
    for i, cnt in enumerate(contours):
        # 层级校验（仅最外层父轮廓）
        if hierarchy[0][i][3] != -1:                                                        # 仅最外层父轮廓
            continue
        
        # 几何特征校验
        area = cv2.contourArea(cnt)                                                         # 计算面积
        perimeter = cv2.arcLength(cnt, True)                                                # 计算周长
        if area < (h*w*0.5) or perimeter < (h+w)*1.5:                                       # 面积和周长校验
            continue
            
        # 矩形度验证
        rect = cv2.minAreaRect(cnt)                                                         # 获取最小外接矩形
        box = cv2.boxPoints(rect)                                                           # 获取矩形四个角点
        box_area = cv2.contourArea(box)                                                     # 计算矩形面积
        if abs(1 - area/box_area) > 0.3:                                                    # 矩形度校验
            continue
            
        valid_contours.append(cnt)

    # # ==================================== 可视化调试 ====================================
    # # 绘制有效轮廓
    # debug_img = img.copy()
    # cv2.drawContours(debug_img, valid_contours, -1, (0, 255, 0), 4)
    # show_image(debug_img)

    # ========== 3. 单元格处理 ==========
    cells_output = []                                                                       # 单元格输出
    remainder = w % num_cells                                                               # 计算余数
    
    for i in range(num_cells):
        if i == 0:
            cell_width = max(10, w // num_cells)                                            # 左侧单元格最小宽度
            x_left = 0                                                                      # 左侧边界
            x_right = cell_width                                                            # 右侧边界
        else:
            cell_width = w // num_cells
            x_left = i * cell_width
            x_right = x_left + cell_width
        
        # 防止越界
        x_left = max(0, x_left)
        x_right = min(w, x_right)

        # print('h:',h , 'w:',w)
        # print(f"单元格{i}: {x_left}-{x_right}")

        if x_left >= x_right:
            print('由于x_left >= x_right，跳过该单元格')
            continue
        
        # 新增有效性校验
        if x_left > w - 10 or x_right < 10:
            print(f"跳过单元格{i}: {x_left}-{x_right}")
            continue
        
        # 提取上下边界
        def extract_boundary_points(contours, img_height, threshold=0.1):
            """提取上下边界点"""
            upper_points = []                                                                   # 上边界点
            lower_points = []                                                                   # 下边界点
            for cnt in contours:
                for p in cnt[:,0]:
                    if p[1] < img_height * threshold:                                           # 上边界点，阈值0.1
                        upper_points.append(p)
                    elif p[1] > img_height * (1 - threshold):                                   # 下边界点，阈值0.9
                        lower_points.append(p)
            return upper_points, lower_points

        def get_y_at_x(contours, target_x):
            """优化后的坐标获取函数"""
            all_points = np.vstack([c.squeeze() for c in contours]) if contours else np.empty((0,2))
            if len(all_points) == 0:
                return []
            
            # 邻近点检测
            x_diffs = np.abs(all_points[:,0] - target_x)
            nearby_mask = x_diffs <= 2
            if np.any(nearby_mask):
                return sorted(all_points[nearby_mask][:,1].tolist(), reverse=True)  # 降序排列
            
            # 范围插值
            x_sorted = np.sort(all_points[:,0])
            y_sorted = all_points[np.argsort(all_points[:,0]),1]
            return [np.interp(target_x, x_sorted, y_sorted)]

        try:
            upper_points, lower_points = extract_boundary_points(valid_contours, h)
            # 获取边界曲线值（浮点坐标）
            # 上边界取最大值
            y_top_left = max(get_y_at_x(upper_points, x_left) or [0])
            y_top_right = max(get_y_at_x(upper_points, x_right) or [0])
            
            # 下边界取最小值
            y_bottom_left = min(get_y_at_x(lower_points, x_left) or [0])
            y_bottom_right = min(get_y_at_x(lower_points, x_right) or [0])
        except:
            print(f"存在异常，跳过单元格{i}: {x_left}-{x_right}")
            continue  # 跳过无效单元格
        
        # 计算目标尺寸
        h_left = y_bottom_left - y_top_left                                                 # 左侧高度
        h_right = y_bottom_right - y_top_right                                              # 右侧高度

        # 处理边缘高度差异过大的情况
        if abs(h_left - h_right) > 10:                                                      # 高度差异过大
            if which_side == "left":
                h_left = h_right
            else:
                h_right = h_left

        target_width = sqrt((abs(h_left - h_right) * k) ** 2 + (x_right - x_left) ** 2) # 目标宽度
        target_height = max(h_left, h_right)                                                                    # 目标高度

        # print(f"Cell {i}: {x_left}-{x_right}, {y_top_left}-{y_bottom_left} / {y_top_right}-{y_bottom_right}")
        # print(f"\tTarget size: {target_width} x {target_height}")
        
        if target_height <= 1e-9 or target_width <= 1e-9:
            print(f"目标尺寸过小，跳过单元格{i}")
            continue  # 忽略无效变换
        
        # 定义源点和目标点（浮点坐标）
        src = [                                                                             # 源点
            (x_left, y_top_left),
            (x_right, y_top_right),
            (x_right, y_bottom_right),
            (x_left, y_bottom_left)
        ]

        dst = [                                                                             # 目标点
            (0.0, 0.0),
            (target_width, 0.0),
            (target_width, target_height),
            (0.0, target_height)
        ]
        
        # 提取原始单元格区域（整数坐标）
        cell_region = img[0:h, x_left:x_right]                                              # 提取单元格区域

        # # ==================================== 可视化调试 ====================================
        # # 绘制单元格区域
        # print(f'单元格区域：{x_left}-{x_right}')
        # show_image(cell_region)
        
        # 计算变换矩阵
        src = np.array(src, dtype=np.float32)                                               # 转换为浮点型
        dst = np.array(dst, dtype=np.float32)                                               # 转换为浮点型
        matrix = cv2.getPerspectiveTransform(src, dst)                                      # 计算变换矩阵

        # 动态选择插值方法
        if target_width > cell_region.shape[1]:
            warp_method = cv2.INTER_CUBIC                                                   # 放大时使用三次插值
        else:
            warp_method = cv2.INTER_AREA                                                    # 缩小时使用面积插值
        
        # 应用透视变换并验证有效性
        try:
            # print(f"\nProcessing cell {i}: x_left={x_left}, x_right={x_right}")
            # print(f"  src:\n{src}, \ndst:\n{dst}")
            # print(f"  matrix:\n{matrix}")
            
            warped = cv2.warpPerspective(cell_region, matrix, (int(target_width), int(target_height)), flags=warp_method, borderMode=cv2.BORDER_WRAP)

            # # 这里存在一些问题，虽然看起来变换成功了，但是实际上存在越界的问题，只是使用了BORDER_WRAP模式避免出现黑边
            # warped = cv2.warpPerspective(cell_region, matrix, (int(target_width), int(target_height)), flags=warp_method)
            # show_image(warped)
        except cv2.error as e:
            print(f"\nWarp failed for cell {i}:\n{str(e)}")
            continue

        if warped.size == 0 or warped.shape[0] <= 0 or warped.shape[1] <= 0:                # 无效变换
            print(f"变换无效，跳过单元格{i}")
            continue  # 跳过无效图像
        cells_output.append(warped)
    
    # ========== 4. 单元格拼接 ==========
    # 统一拉伸所有单元格到基准高度
    if not cells_output:
        return img
    base_h = cells_output[0].shape[0]                                                       # 基准高度
    for i in range(len(cells_output)):
        curr_h = cells_output[i].shape[0]                                                   # 当前高度
        if curr_h != base_h:
            # 保持宽度不变，仅调整高度
            cells_output[i] = cv2.resize(cells_output[i],                                   # 调整大小
                                    (cells_output[i].shape[1], base_h),
                                    interpolation=cv2.INTER_AREA)
    
    # 拼接处理后的单元格
    final_image = cv2.hconcat(cells_output) if cells_output else img                        # 拼接单元格
    return final_image

# 书页矫正主函数
def book_page_rectifier(img_path, which_side):
    img = cv2.imread(img_path)
    # 获取浮点型坐标（例如：[[123.4, 56.7], ...]）
    corners = auto_detect_page_corners(img)                                                 # 自动检测角点

    # # ==================================== 可视化调试 ====================================
    # # 绘制角点和四边形
    # int_corners = corners.astype(int)
    # debug_img = img.copy()
    # for pt in int_corners:
    #     # 确保坐标格式为Python原生整数元组
    #     center = (int(pt[0]), int(pt[1]))  # 双重转换确保类型安全
    #     cv2.circle(debug_img, center, 10, (0,0,255), -1)
    # # 将角点坐标连接为四边形
    # for i in range(4):
    #     cv2.line(debug_img, tuple(int_corners[i]), 
    #             tuple(int_corners[(i+1)%4]), (0,255,0), 4)
    # show_image(debug_img)

    # ========== 1. 水平方向矫正 ==========
    horizontal_img =  horizontal_warp_image(img, corners)                                   # 水平矫正

    # ========== 2. 垂直方向矫正 ==========
    vertical_img = vertical_warp_image(horizontal_img, which_side)                          # 垂直矫正

    # # ==================================== 可视化调试 ====================================
    # # 显示矫正后的图像
    # show_image(horizontal_img)
    # show_image(vertical_img)

    return vertical_img

# =========================文字方向矫正=========================
def rotate_text_image(img_path, max_angle=10):
    img = cv2.imread(img_path)
    # ========== 1. 灰度化 + 高斯降噪 ==========
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # ========== 2. 自适应阈值分割 ==========
    thresh = cv2.adaptiveThreshold(blurred, 255, 
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 21, 10)

    # show_image(thresh)

    # ========== 3. 形态学操作（连接字符间隙+去除孤立噪声） ==========
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.dilate(cleaned, kernel, iterations=5)

    # show_image(cleaned)

    # ========== 4. 霍夫直线检测 ==========
    lines = cv2.HoughLinesP(cleaned, 1, np.pi / 180, 200, minLineLength=200, maxLineGap=3)
    if lines is None:
        return img
    
    # # ==================================== 可视化调试 ====================================
    # # 绘制检测到的直线
    # debug_img = img.copy()
    # for line in lines:
    #     x1, y1, x2, y2 = line[0]
    #     cv2.line(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    # show_image(debug_img)

    # ========== 5. 计算所有直线的角度 ==========
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = degrees(atan2(y2 - y1, x2 - x1))
        if -max_angle < angle < max_angle and angle != 0:
            angles.append(angle)

    if not angles:
        return img

    # ========== 6. 角度聚类==========
    # 将角度排序后按阈值聚类
    sorted_angles = sorted(angles)                                                          # 排序
    clusters = []                                                                           # 角度簇
    current_cluster = [sorted_angles[0]]                                                    # 当前簇
    angle_threshold = 2.0                                                                   # 同一簇允许的最大角度差
    
    for angle in sorted_angles[1:]:
        if abs(angle - current_cluster[-1]) <= angle_threshold:
            current_cluster.append(angle)
        else:
            clusters.append(current_cluster)
            current_cluster = [angle]
    clusters.append(current_cluster)
    
    # 选择最大的角度簇
    max_cluster = max(clusters, key=len)
    print(f"检测到 {len(clusters)} 个角度簇，最大簇包含 {len(max_cluster)} 条直线")
    
    # ========== 7. 计算最终角度 ==========
    angle = np.mean(max_cluster)
    print(f"最终矫正角度：{angle:.2f} 度")

    # ========== 8. 旋转图像 ==========
    center = (img.shape[1] // 2, img.shape[0] // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 这里使用BORDER_REFLECT填充边界，避免黑边导致后面对文本区域进行分割时出现顶部有凸起的问题
    rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

    return rotated

# =========================文字区域切割=========================
# 获取文字块
def get_text_block(img_path, black_tolerance=0.05):   
    img = cv2.imread(img_path)
    
    # ========== 1. 灰度化 + 高斯降噪 ==========
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    # show_image(blurred)

    # ========== 2. 自适应阈值分割 ==========
    thresh = cv2.adaptiveThreshold(blurred, 255, 
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 21, 10)

    # ========== 3. 形态学操作（连接字符间隙+去除孤立噪声） ==========
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.dilate(cleaned, kernel, iterations=25)

    # ========== 4. 边缘保护处理 ==========
    border_size = 20
    cleaned[:border_size, :] = 0                                                            # 上边缘
    cleaned[-border_size:, :] = 0                                                           # 下边缘
    cleaned[:, :border_size] = 0                                                            # 左边缘
    cleaned[:, -border_size:] = 0                                                           # 右边缘

    # show_image(cleaned)

    # ========== 5. 轮廓检测 ==========
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)     # 查找轮廓
    
    # 有效性验证（面积阈值调整为5%）
    valid_contours = []
    for cnt in contours:
        if len(cnt) >= 5:
            area = cv2.contourArea(cnt)                                                     # 计算面积
            if area > img.shape[0]*img.shape[1]*0.05:                                       # 面积阈值：5%
                valid_contours.append(cnt)
    
    if not valid_contours:
        raise ValueError("未检测到有效轮廓")
    
    # ========== 6. 执行局部异常凸起检测算法 ==========
    repaired_contours = []
    for cnt in valid_contours:
        try:
            # 6.1 前置检查
            if len(cnt) < 4:                                                                # 至少需要4个点才能形成有效轮廓
                repaired_contours.append(cnt)                                               # 保留原始轮廓
                continue

            # 6.2 类型转换与凸包计算
            cnt = cnt.astype(np.int32)                                                      # 类型转换
            hull_indices = cv2.convexHull(cnt, returnPoints=False)                          # 计算凸包索引
            
            if hull_indices is None or len(hull_indices) < 3:
                print("凸包计算失败，保留原始轮廓")
                repaired_contours.append(cnt)                                               # 保留原始轮廓
                continue

            # 6.3 凸性缺陷检测
            defects = cv2.convexityDefects(cnt, hull_indices)                               # 计算凸性缺陷
            if defects is None:
                print("凸性缺陷检测失败，保留原始轮廓")
                repaired_contours.append(cnt)                                               # 保留原始轮廓
                continue

            # 6.4 缺陷点分析
            defect_segments = []
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]                                                  # 起始点、结束点、远点、到凸包的距离
                # 索引有效性验证
                if all(0 <= x < len(cnt) for x in [s, e, f]):
                    distance = d / 256.0                                                    # 距离归一化
                    if distance > 50:                                                       # 凸起敏感度阈值：50
                        start = tuple(cnt[s][0])                                            # 起始点
                        end = tuple(cnt[e][0])                                              # 结束点
                        far = tuple(cnt[f][0])                                              # 远点
                        print(f"检测到凸起缺陷：{start} -> {end} (Far: {far}, Distance: {distance})")
                        defect_segments.append( (start, end, far) )                         # 保存凸起段

            # 6.5 无显著缺陷则跳过
            if not defect_segments:
                print("未检测到有效凸起缺陷")
                repaired_contours.append(cnt)                                               # 保留原始轮廓
                continue

            # 6.6 轮廓修复处理
            contour_points = cnt.squeeze()                                                  # 轮廓坐标点
            mask = np.ones(len(contour_points), dtype=bool)                                 # 创建掩码

            # 创建坐标查找字典（加速匹配）
            coord_dict = {(x, y): idx for idx, (x, y) in enumerate(contour_points)}         # 创建坐标查找字典

            for s_p, e_p, f_p in defect_segments:
                # 使用容差匹配查找索引（±2像素范围）
                start_idx = next( (coord_dict.get((x,y)) for x in range(s_p[0]-2, s_p[0]+3)             # 查找起始点索引
                                 for y in range(s_p[1]-2, s_p[1]+3) if (x,y) in coord_dict), None)      
                end_idx = next( (coord_dict.get((x,y)) for x in range(e_p[0]-2, e_p[0]+3)               # 查找结束点索引
                               for y in range(e_p[1]-2, e_p[1]+3) if (x,y) in coord_dict), None)

                if start_idx is None or end_idx is None:                                    # 无效索引
                    continue

                # 标记需要删除的点（保留端点）
                if start_idx < end_idx:                                                     # 起始点在前
                    mask[start_idx+1:end_idx] = False
                else:
                    mask[start_idx+1:] = False
                    mask[:end_idx] = False

            # 保留有效点
            preserved_points = contour_points[mask]                                         # 保留有效点

            # # ==================================== 可视化调试 ====================================
            # debug_img = img.copy()
            # cv2.drawContours(debug_img, [preserved_points.reshape(-1, 1, 2)], -1, (0, 255, 0), 2)
            # show_image(debug_img)

            # 6.7 凸包段插入
            for s_p, e_p, _ in defect_segments:                                             # 遍历凸起段
                # 生成凸包连接段
                segment_points = np.array([s_p, e_p], dtype=np.int32)                       # 凸包连接段
                hull_segment = cv2.convexHull(segment_points).squeeze()                     # 计算凸包

                # 查找插入位置
                try:
                    insert_pos = np.where((preserved_points == s_p).all(axis=1))[0][0]      # 查找起始点位置
                    # 插入凸包点（排除重复端点）
                    preserved_points = np.insert(                                           # 插入凸包点
                        preserved_points,
                        insert_pos + 1,
                        hull_segment[1:-1],
                        axis=0
                    )
                except IndexError:
                    continue

            # 6.8 生成最终轮廓
            repaired_cnt = preserved_points.reshape(-1, 1, 2).astype(np.int32)              # 生成最终轮廓
            repaired_contours.append(repaired_cnt)                                          # 保存修复轮廓

            # # ==================================== 可视化调试 ====================================
            # # 生成最终轮廓
            # debug_img = img.copy()
            # cv2.drawContours(debug_img, [repaired_cnt], -1, (0, 255, 0), 2)
            # show_image(debug_img)

        except Exception as e:
            print(f"轮廓处理异常: {str(e)}，保留原始轮廓")
            repaired_contours.append(cnt)                                                   # 保留原始轮廓

    # ========== 7. 主区域合并逻辑 ==========
    if repaired_contours:
        # 合并所有有效轮廓的坐标点
        merged_points = np.vstack(repaired_contours)                                        # 合并所有轮廓
        x, y, w, h = cv2.boundingRect(merged_points)                                        # 计算外接矩形
    else:
        h, w = img.shape[:2]                                                                # 获取图像尺寸
        x, y = int(w*0.1), int(h*0.1)                                                       # 计算初始坐标
        w, h = int(w*0.8), int(h*0.8)                                                       # 计算初始尺寸
    
    # 安全边界扩展
    x = max(0, x-10)                                                                        # 扩展左侧
    y = max(0, y-10)                                                                        # 扩展上侧
    w = min(img.shape[1]-x, w+20)                                                           # 扩展右侧
    h = min(img.shape[0]-y, h+20)                                                           # 扩展下侧

    # ========== 8. 最终裁剪与验证 ==========
    cropped = img[y:y+h, x:x+w]                                                             # 裁剪图像
    gray_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)                                # 灰度化
    black_ratio = np.sum(gray_cropped < 50) / gray_cropped.size                             # 计算黑色区域占比
    
    if black_ratio > black_tolerance:
        print(f"警告：黑色区域占比 {black_ratio:.2%}，超过阈值 {black_tolerance}")
    
    return cropped

# =========================文字区域分块=========================
def smart_horizontal_split(img_path, min_gap=5):
    # 读取图像并获取尺寸信息
    img = cv2.imread(img_path)
    H, W = img.shape[:2]
    min_h = int(H * 0.25)                                                                   # 最小允许高度
    max_h = int(H * 0.45)                                                                   # 最大允许高度
    
    # ========== 1. 预处理阶段 ==========
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, 
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 21, 10)
    
    # ========== 2. 文本保护处理 ==========
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.dilate(cleaned, kernel, iterations=10)

    # show_image(cleaned)

    # ========== 3. 间隙检测算法 ==========
    def detect_text_regions(cleaned):
        window_h = 1                                                                        # 将窗口高度设为1，逐行检测
        img_area = H * W                                                                    # 计算图像面积
        white_pixel_ratio = np.sum(cleaned == 255) / img_area                               # 计算白色像素比例
        density_th = max(0.1, min(0.3, white_pixel_ratio * 0.5))                            # 降低密度阈值
        
        text_mask = np.zeros(H, dtype=np.uint8)                                             # 创建文本掩码
        for y in range(H - window_h + 1):                                                   # 滑动窗口
            window = cleaned[y:y+window_h, :]                                               # 获取窗口
            if np.sum(window == 255) / (window_h * W) > density_th:                         # 密度阈值检测
                text_mask[y:y+window_h] = 1                                                 # 标记当前行
        
        # 区域合并（放宽合并条件）
        regions, start = [], None
        for y in range(H):
            if text_mask[y]:                                                                # 白色像素
                if start is None:
                    start = y                                                               # 记录起始位置
                end = y                                                                     # 更新结束位置
            else:
                if start is not None:
                    if regions and start - regions[-1][1] <= 5:                             # 合并间隔扩大至5行
                        regions[-1] = (regions[-1][0], end)                                 # 更新区域
                    else:
                        regions.append((start, end))                                        # 添加区域
                    start = None
        # 处理最后未闭合的区域
        if start is not None:                                                               # 处理最后未闭合的区域
            if regions and start - regions[-1][1] <= 5:                                     # 合并间隔扩大至5行
                regions[-1] = (regions[-1][0], end)
            else:
                regions.append((start, end))                                                # 添加区域
        
        return [r for r in regions if r[1]-r[0] >= 1]                                       # 允许最小高度为1

    white_regions = detect_text_regions(cleaned)                                            # 检测文本区域

    # ========== 4. 间隙分析 ==========
    valid_gaps = []
    for i in range(1, len(white_regions)):
        gap_start = white_regions[i-1][1] + 1                                               # 间隙起始
        gap_end = white_regions[i][0] - 1                                                   # 间隙结束
        if gap_end - gap_start >= min_gap:                                                  # 有效间隙
            valid_gaps.append({
                'start': gap_start,                                                         # 间隙起始
                'end': gap_end,                                                             # 间隙结束
                'center': (gap_start + gap_end) // 2,                                       # 间隙中心
                'size': gap_end - gap_start                                                 # 间隙大小
            })
    
    # # ==================================== 可视化调试 ====================================
    # # 绘制间隙和文本区域
    # debug_img = img.copy()
    # for start, end in white_regions:
    #     cv2.rectangle(debug_img, (0, start), (W, end), (0, 0, 255), 4)
    # for gap in valid_gaps:
    #     cv2.rectangle(debug_img, (0, gap['start']), (W, gap['end']), (0, 255, 0), 2)
    # show_image(debug_img)

    # ========== 5. 智能分割算法 ==========
    def find_optimal_splits(gaps):
        H = img.shape[0]
        min_h = int(H * 0.25)
        max_h = int(H * 0.45)
        
        if not gaps:
            return None
        
        # 生成候选分割点（使用间隙首尾+图像边界）
        candidates = {0, H}  # 用集合自动去重
        for gap in gaps:
            candidates.add(gap['start'])                                                    # 间隙起始
            candidates.add(gap['end'])                                                      # 间隙结束
        candidates = sorted(list(candidates))                                               # 转换为有序列表
        
        # 动态规划寻找最优路径
        dp = [{'score': -float('inf'), 'path': []} for _ in range(H+1)]                     # 动态规划表
        dp[0] = {'score': 0, 'path': [0]}                                                   # 初始状态
        
        for y in range(1, H+1):
            for s in range(max(0, y-max_h), y-min_h+1):
                if dp[s]['score'] == -float('inf'):                                         # 无效状态
                    continue
                    
                # 计算当前段得分
                current_score = dp[s]['score']
                # 间隙位置奖励（当s是某个间隙的结束，y是下个间隙的开始时获得奖励）
                if any(g['end'] == s for g in gaps) and any(g['start'] == y for g in gaps): # 间隙位置奖励
                    current_score += 15                                                     # 连续间隙奖励
                elif s in candidates or y in candidates:                                    # 间隙位置惩罚
                    current_score += 10                                                     # 候选点基础奖励
                
                # 更新最优路径
                if current_score > dp[y]['score']:                                          # 更新最优路径
                    dp[y] = {
                        'score': current_score,
                        'path': dp[s]['path'] + [y]
                    }
        
        # 提取最佳路径
        best_splits = dp[H]['path'] if dp[H]['score'] != -float('inf') else None            # 最佳路径
        
        # 有效性验证
        def validate(splits):
            if not splits or splits[-1] != H:                                               # 无效路径
                return False
            segments = [splits[i+1]-splits[i] for i in range(len(splits)-1)]                # 计算段长度
            return all(min_h <= h <= max_h for h in segments) and (3 <= len(segments) <=4)  # 段长度有效性
        
        if validate(best_splits):                                                           # 最佳路径有效
            return best_splits
        
        # 保底策略：全排列搜索
        for split_num in [3,4]:                                                             # 尝试3-4段分割
            for points in combinations([c for c in candidates if 0 < c < H], split_num-1):  # 组合搜索
                trial = sorted([0] + list(points) + [H])                                    # 排序并添加边界
                if validate(trial):                                                         # 有效性验证
                    return trial
        return None

    # ========== 5. 执行分割方案 ==========
    final_split = find_optimal_splits(valid_gaps)                                           # 执行分割方案

    # ========== 6. 后处理验证 ==========
    segments = []
    for i in range(len(final_split)-1):
        y1, y2 = final_split[i], final_split[i+1]                                           # 段起始和结束
        # 最终高度校验
        if (y2 - y1) >= min_h:                                                              # 最终高度校验
            segments.append(img[y1:y2])                                                     # 添加分割段
    return segments

# =========================主处理流程=========================
def process_main(fold_path, img_name, net, transform):
    # 使用Unet进行图像分割
    img_name = img_name.split(".")[0]

    print(f"开始Unet图像分割{img_name}.png...")
    mask = predict_by_unet(fold_path + img_name + '.png', net, transform)
    cv2.imwrite(f"{fold_path}/{img_name}_0_mask.png", mask)

    # 图像掩膜
    print(f"开始图像掩膜{img_name}.png...")
    masked_img = apply_mask(f"{fold_path}/{img_name}.png", f"{fold_path}/{img_name}_0_mask.png")
    cv2.imwrite(f"{fold_path}/{img_name}_1_masked.png", masked_img)

    # 动态光照补偿
    print(f"开始动态光照补偿{img_name}_1_masked.png...")
    enhanced_img = adaptive_lighting_enhancement(f"{fold_path}/{img_name}_1_masked.png")
    cv2.imwrite(f"{fold_path}/{img_name}_2_enhanced.png", enhanced_img)

    # 旋转校正
    print(f"开始旋转校正{img_name}_2_enhanced.png...")
    rotated = correct_book_rotation(f'{fold_path}/{img_name}_2_enhanced.png')
    cv2.imwrite(f"{fold_path}/{img_name}_3_rotated.png", rotated)

    # 分页处理
    print(f"开始分页处理{img_name}_3_rotated.png...")
    left_page, right_page = find_book_corners_and_split(f"{fold_path}/{img_name}_3_rotated.png")
    cv2.imwrite(f"{fold_path}/{img_name}_4_left_page.png", left_page)
    cv2.imwrite(f"{fold_path}/{img_name}_4_right_page.png", right_page)

    # 书页矫正
    print(f"开始书页矫正{img_name}_4_left_page.png...")
    corrected_left = book_page_rectifier(f"{fold_path}/{img_name}_4_left_page.png", 'left')
    print(f"开始书页矫正{img_name}_4_right_page.png...")
    corrected_right = book_page_rectifier(f"{fold_path}/{img_name}_4_right_page.png", 'right')
    cv2.imwrite(f"{fold_path}/{img_name}_5_corrected_left.png", corrected_left)
    cv2.imwrite(f"{fold_path}/{img_name}_5_corrected_right.png", corrected_right)

    # 文字方向矫正
    print(f"开始文字方向矫正{img_name}_5_corrected_left.png...")
    text_corrected_left = rotate_text_image(f"{fold_path}/{img_name}_5_corrected_left.png")
    print(f"开始文字方向矫正{img_name}_4_corrected_right.png...")
    text_corrected_right = rotate_text_image(f"{fold_path}/{img_name}_5_corrected_right.png")
    cv2.imwrite(f"{fold_path}/{img_name}_6_text_corrected_left.png", text_corrected_left)
    cv2.imwrite(f"{fold_path}/{img_name}_6_text_corrected_right.png", text_corrected_right)

    # 文字区域切割
    print(f"开始文字区域切割{img_name}_6_text_corrected_left.png...")
    text_block_left = get_text_block(f"{fold_path}/{img_name}_6_text_corrected_left.png")
    print(f"开始文字区域切割{img_name}_5_text_corrected_right.png...")
    text_block_right = get_text_block(f"{fold_path}/{img_name}_6_text_corrected_right.png")
    cv2.imwrite(f"{fold_path}/{img_name}_7_text_block_left.png", text_block_left)
    cv2.imwrite(f"{fold_path}/{img_name}_7_text_block_right.png", text_block_right)

    # 文字区域分块
    print(f"开始文字区域分块{img_name}_7_text_block_left.png...")
    text_blocks_left = smart_horizontal_split(f"{fold_path}/{img_name}_7_text_block_left.png")
    print(f"开始文字区域分块{img_name}_7_text_block_right.png...")
    text_blocks_right = smart_horizontal_split(f"{fold_path}/{img_name}_7_text_block_right.png")
    print(f"左页分块数量：{len(text_blocks_left)}，右页分块数量：{len(text_blocks_right)}")
    for j, block in enumerate(text_blocks_left):
        block_process = process_before_OCR(block)
        cv2.imwrite(f"{fold_path}/{img_name}_8_text_block_left_{j}.png", block_process)
    for j, block in enumerate(text_blocks_right):
        block_process = process_before_OCR(block)
        cv2.imwrite(f"{fold_path}/{img_name}_8_text_block_right_{j}.png", block_process)

if __name__ == "__main__":
    # 创建Unet模型
    net=UNet(2).cuda()
    # 加载预训练权重，可以从0-5中选择
    weight_path = './image_segmentation/content/params/unet_epoch1.pth'
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path)['model_state'])
        print('successfully load model')
    else:
        print('no loading')
    # 加载transform
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # for i in range(1, 11):
    #     process_main('F:/desktop/images/', f'grass_{i}.png', net, transform)

    process_main('./images/', 'grass_1', net, transform)