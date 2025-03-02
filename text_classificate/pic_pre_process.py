import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from math import atan2, degrees, sqrt
from sklearn.cluster import DBSCAN
from scipy.interpolate import CubicSpline
from dataset_get.pic_to_text_by_OCR import get_pic_text

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

# 获取摄像头图像
def get_picture():
    # 显示摄像头的图像
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # 如果按下s键，保存图像
        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite("./frame.png", frame)

# 显示图像
def show_image(img):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# ===============================================裁剪黑边+图像倾斜纠正+===============================================
# 自动裁剪黑边
def auto_remove_black_border(img):
    """基于内容检测自动裁剪黑边"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    # 寻找内容边界
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    cnt = max(contours, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(cnt)
    
    # 扩展5像素保留边缘
    x = max(0, x-5)
    y = max(0, y-5)
    w = min(w+10, img.shape[1]-x)
    h = min(h+10, img.shape[0]-y)
    
    return img[y:y+h, x:x+w]

# 纠正图像倾斜
def correct_book_rotation(img_path):
    # 1. 读取图像
    img = cv2.imread(img_path)
    height, width = img.shape[:2]
    
    # 2. 预处理
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, 
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 21, 10)
    
    edges = cv2.Canny(blur, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    dilated = cv2.dilate(edges, kernel, iterations=3)
    gradient = cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, kernel)

    # 3. 查找轮廓
    contours, _ = cv2.findContours(gradient, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 4. 找到最大轮廓（书本主体）
    max_contour = max(contours, key=cv2.contourArea)
    
    # 5. 获取最小外接矩形
    rect = cv2.minAreaRect(max_contour)
    (cx, cy), (w, h), angle = rect

    # 6. 角度修正逻辑（处理OpenCV的角度表示问题）
    if w < h:
        angle -= 90

    # 7. 旋转图像
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (width, height), 
                           flags=cv2.INTER_CUBIC, 
                           borderMode=cv2.BORDER_REPLICATE)
    
    # 8. 删除黑边
    rotated_after = auto_remove_black_border(rotated)

    return rotated_after

# ===============================================分页处理+分界线倾斜矫正===============================================
# 选择最佳分界点
def find_vertical_spine_points(points, img_center, vertical_margin=0.1):
    """
    寻找垂直中缝的上下端点
    :param points: 候选角点集合 (N,2)
    :param img_center: 图像中心坐标 (cx, cy)
    :param vertical_margin: 垂直方向中间区域比例
    :return: (upper_point, lower_point)
    """
    cx, cy = img_center
    img_width = 2 * cx

    # 筛选位于垂直中线附近的点（宽度10%范围内）
    vertical_points = [p for p in points if abs(p[0] - cx) < img_width * 0.05]

    if len(vertical_points) < 2:
        # 降级处理：选择x坐标最接近中心的两个点
        vertical_points = sorted(points, key=lambda p: abs(p[0] - cx))[:2]

    # 按垂直位置排序
    sorted_points = sorted(vertical_points, key=lambda p: p[1])
    
    # 确定中间区域边界
    upper_bound = cy * (1 - vertical_margin)
    lower_bound = cy * (1 + vertical_margin)

    # 选择中间区域最上/下的点
    upper_candidates = [p for p in sorted_points if p[1] < upper_bound]
    lower_candidates = [p for p in sorted_points if p[1] > lower_bound]

    upper = min(upper_candidates, key=lambda p: p[1]) if upper_candidates else sorted_points[0]
    lower = max(lower_candidates, key=lambda p: p[1]) if lower_candidates else sorted_points[-1]

    return upper, lower

# 计算旋转角度
def find_vertical_spine_points(points, img_size):
    """
    检测垂直中缝的上下端点
    :param points: 候选角点集合 (N,2)
    :param img_size: 图像尺寸 (width, height)
    :return: (upper_point, lower_point)
    """
    w, h = img_size
    center_x = w // 2
    
    # 筛选位于垂直中线附近15%区域的点
    vertical_mask = (points[:,0] > center_x - w*0.075) & (points[:,0] < center_x + w*0.075)
    vertical_points = points[vertical_mask]
    
    # 降级处理：若无足够点，选择x最接近中心的2个点
    if len(vertical_points) < 2:
        vertical_points = sorted(points, key=lambda p: abs(p[0]-center_x))[:2]
    
    # 按垂直位置排序并选择上下端点
    sorted_points = sorted(vertical_points, key=lambda p: p[1])
    return sorted_points[0], sorted_points[-1]

# 计算旋转角度
def calculate_rotation_angle(upper, lower):
    """计算需要旋转到垂直的角度"""
    dx = lower[0] - upper[0]
    dy = lower[1] - upper[1]
    current_angle = degrees(atan2(dy, dx))
    return current_angle - 90  # 调整到垂直方向

# 书页分割
def find_book_corners_and_split(img_path):
    # ========== 1. 图像预处理 ==========
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    
    # 多尺度预处理
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 9, 30, 30)
    edges = cv2.Canny(blur, 30, 100)
    
    # 形态学强化轮廓
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    dilated = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    # ========== 2. 轮廓提取与处理 ==========
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("未检测到有效轮廓")
    
    main_contour = max(contours, key=cv2.contourArea)
    points = np.squeeze(main_contour, axis=1)

    # ========== 3. 候选角点检测 ==========
    def curvature_angle(points, idx, window=15):
        """改进的曲率角度检测"""
        prev = points[max(0, idx-window):idx]
        next = points[idx:min(len(points), idx+window)]
        
        if len(prev)<2 or len(next)<2:
            return 180
        
        v1 = np.mean(prev[-2:] - prev[-1], axis=0)
        v2 = np.mean(next[:2] - next[0], axis=0)
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        mod = np.linalg.norm(v1) * np.linalg.norm(v2)
        return degrees(np.arccos(dot/(mod + 1e-7)))

    candidates = []
    window_size = max(15, int(0.005 * len(points)))
    for i in range(len(points)):
        if 80 < curvature_angle(points, i, window_size) < 100:  # 严格直角范围
            candidates.append(points[i])
    
    # ========== 4. 角点聚类筛选 ==========
    clustering = DBSCAN(eps=w*0.03, min_samples=3).fit(candidates)
    labels = clustering.labels_
    
    # 提取有效聚类中心
    cluster_centers = []
    for label in set(labels):
        if label == -1: continue
        cluster = np.array(candidates)[labels == label]
        cluster_centers.append(np.median(cluster, axis=0))
    
    # 按垂直位置排序
    final_corners = sorted(cluster_centers, key=lambda p: p[1])

    # ========== 5. 中缝检测与旋转 ==========
    try:
        upper, lower = find_vertical_spine_points(np.array(final_corners), (w, h))
    except:
        # 降级处理：使用轮廓极值点
        upper = points[points[:,1].argmin()]
        lower = points[points[:,1].argmax()]

    rotate_angle = calculate_rotation_angle(upper, lower)
    
    # 执行旋转
    M = cv2.getRotationMatrix2D((w//2, h//2), rotate_angle, 1)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)

    # ========== 6. 精确分页 ==========
    # 计算旋转后的中缝位置
    pts = np.array([upper, lower], dtype=np.float32).reshape(-1,1,2)
    rotated_pts = cv2.transform(pts, M).squeeze()
    
    # 取x坐标中值并限制范围
    split_x = int(np.median(rotated_pts[:,0]))
    split_x = np.clip(split_x, int(w*0.45), int(w*0.55))

    # 分割与边框处理
    border = 15
    left = cv2.copyMakeBorder(rotated[:, :split_x], border, border, border, border,
                            cv2.BORDER_CONSTANT, value=(0,0,0))
    right = cv2.copyMakeBorder(rotated[:, split_x:], border, border, border, border,
                             cv2.BORDER_CONSTANT, value=(0,0,0))

    # # ==================================================== 可视化调试 ====================================================
    # debug_img = img.copy()
    # for pt in final_corners:
    #     cv2.circle(debug_img, tuple(pt.astype(int)), 10, (0,0,255), -1)
    # for pt in candidates:
    #     cv2.circle(debug_img, tuple(pt.astype(int)), 5, (0,255,0), -1)
    # cv2.line(debug_img, tuple(upper.astype(int)), tuple(lower.astype(int)),
    #         (255,0,0), 3)
    # cv2.putText(debug_img, f"Rotate: {rotate_angle:.2f}°", (20,40),
    #            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 4)

    # show_image(debug_img)

    return left, right

# ===============================================水平矫正+垂直矫正===============================================

# 得到有序的四个角点
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # 左上
    rect[2] = pts[np.argmax(s)]  # 右下
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # 右上
    rect[3] = pts[np.argmax(diff)]  # 左下
    return rect

# 自动查找书页四个角点
def auto_detect_page_corners(img):
    # 预处理强化文字对比度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 15, 75, 75)  # 保边滤波
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16,16))
    enhanced = clahe.apply(blur)

    # 动态参数计算
    h, w = img.shape[:2]
    block_size = max(31, int(min(h,w)/15)*2+1)  # 确保足够大的窗口
    c_value = max(7, int(min(h,w)/100))
    
    # 二值化强化边缘
    thresh = cv2.adaptiveThreshold(enhanced, 255, 
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, block_size, c_value)

    # 形态学重建（针对密集文字优化）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=2)

    # 多层轮廓分析
    contours, hierarchy = cv2.findContours(opened, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # 筛选有效轮廓
    valid_contours = []
    for i, cnt in enumerate(contours):
        # 层级校验（仅最外层父轮廓）
        if hierarchy[0][i][3] != -1:
            continue
        
        # 几何特征校验
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if area < (h*w*0.5) or perimeter < (h+w)*1.5:
            continue
            
        # 矩形度验证
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box_area = cv2.contourArea(box)
        if abs(1 - area/box_area) > 0.3:
            continue
            
        valid_contours.append(cnt)

    # # ==================================================== 可视化调试 ====================================================
    # cv2.drawContours(img, valid_contours, -1, (0,255,0), 4)
    # show_image(img)

    # 选择最佳候选
    if not valid_contours:
        raise ValueError("未找到有效书页轮廓")
        
    page_contour = max(valid_contours, key=cv2.contourArea)

    # 亚像素级角点精修
    epsilon = 0.01 * cv2.arcLength(page_contour, True)
    approx = cv2.approxPolyDP(page_contour, epsilon, True)
    corners = cv2.cornerSubPix(gray, np.float32(approx.reshape(-1,2)), 
                             (5,5), (-1,-1), 
                             (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    
    # 获取全局垂直极值点，将上面两个角点的纵坐标调整为书页的最高点，下面两个角点的纵坐标调整为书页的最低点
    all_points = page_contour.reshape(-1,2)
    top_point = all_points[np.argmin(all_points[:,1])]  # 最高点(Y最小)
    bottom_point = all_points[np.argmax(all_points[:,1])]  # 最低点(Y最大)
    
    (tl, tr, br, bl) = order_points(corners)
    adjusted_points = np.array([
        [tl[0], top_point[1] - 10],    # 左上保持X，Y设为全局最高并预留10像素
        [tr[0], top_point[1] - 10],    # 右上保持X，Y设为全局最高并预留10像素
        [br[0], bottom_point[1] + 10], # 右下保持X，Y设为全局最低并预留10像素
        [bl[0], bottom_point[1] + 10]  # 左下保持X，Y设为全局最低并预留10像素
    ], dtype=np.float32)
    
    return adjusted_points

# 透视变换
def horizontal_warp_image(img, src_points):
# 坐标排序验证
    if src_points.shape != (4, 2):
        raise ValueError("必须提供4个二维坐标点")

    # 智能尺寸计算
    (tl, tr, br, bl) = src_points
    width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))

    # 构建目标矩形（上下严格等宽）
    dst = np.array([
        [0, 0],
        [width - 1, 0],  # 确保不越界
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")

    # 计算变换矩阵
    M = cv2.getPerspectiveTransform(src_points, dst)
    
    # 执行变换（带抗锯齿处理）
    warped = cv2.warpPerspective(
        img, M, 
        (int(width), int(height)),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REPLICATE
    )
    
    return warped

# 书页矫正主函数
def book_page_rectifier(img_path):
    img = cv2.imread(img_path)
    # 获取浮点型坐标（例如：[[123.4, 56.7], ...]）
    corners = auto_detect_page_corners(img)  

    # # ==================================================== 可视化调试 ====================================================
    # # 转换为整数坐标
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

    horizontal_img =  horizontal_warp_image(img, corners)

    # show_image(horizontal_img)

    return horizontal_img


if __name__ == "__main__":
    for i in range(1, 4):
        origin_path = f'./text_classificate/content/images/{i}.png'
        # 旋转校正
        rotated = correct_book_rotation(origin_path)
        cv2.imwrite(f"./text_classificate/content/images/{i}_rotated.png", rotated)

        # 分页处理
        left_page, right_page = find_book_corners_and_split(f"./text_classificate/content/images/{i}_rotated.png")
        cv2.imwrite(f"./text_classificate/content/images/{i}_left_page.png", left_page)
        cv2.imwrite(f"./text_classificate/content/images/{i}_right_page.png", right_page)

        # 书页矫正
        corrected_left = book_page_rectifier(f"./text_classificate/content/images/{i}_left_page.png")
        corrected_right = book_page_rectifier(f"./text_classificate/content/images/{i}_right_page.png")
        cv2.imwrite(f"./text_classificate/content/images/{i}_corrected_left.png", corrected_left)
        cv2.imwrite(f"./text_classificate/content/images/{i}_corrected_right.png", corrected_right)