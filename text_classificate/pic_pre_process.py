import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from math import atan2, degrees, sqrt
from itertools import combinations
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

# OCR前预处理图片
def process_before_OCR(image):
    # 灰度化
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 降噪
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    # 自适应二值化
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 7)
    # 形态学操作
    kernel = np.ones((2,2), np.uint8)
    processed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return processed

# 显示图像
def show_image(img):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# =========================裁剪黑边+图像倾斜纠正=========================
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
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
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

# =========================分页处理+分界线倾斜矫正=========================
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

    # ========== 1. 筛选位于垂直中线附近的点（宽度10%范围内） ==========
    vertical_points = [p for p in points if abs(p[0] - cx) < img_width * 0.05]

    if len(vertical_points) < 2:
        # 降级处理：选择x坐标最接近中心的两个点
        vertical_points = sorted(points, key=lambda p: abs(p[0] - cx))[:2]

    # ========== 2. 按垂直位置排序 ==========
    sorted_points = sorted(vertical_points, key=lambda p: p[1])
    
    # ========== 3. 确定中间区域边界 ==========
    upper_bound = cy * (1 - vertical_margin)
    lower_bound = cy * (1 + vertical_margin)

    # ========== 4.选择中间区域最上/下的点 ==========
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
    
    # ========== 1. 筛选位于垂直中线附近15%区域的点 ==========
    vertical_mask = (points[:,0] > center_x - w*0.075) & (points[:,0] < center_x + w*0.075)
    vertical_points = points[vertical_mask]
    
    # ========== 2. 降级处理：若无足够点，选择x最接近中心的2个点 ==========
    if len(vertical_points) < 2:
        vertical_points = sorted(points, key=lambda p: abs(p[0]-center_x))[:2]
    
    # ========== 3. 按垂直位置排序并选择上下端点 ==========
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

    # # ==================================== 可视化调试 ====================================
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

# =========================水平矫正+垂直矫正=========================
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
    # ========== 1. 图像预处理 ==========
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

    # ========== 2. 轮廓提取与处理 ==========
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

    # # ==================================== 可视化调试 ====================================
    # cv2.drawContours(img, valid_contours, -1, (0,255,0), 4)
    # show_image(img)

    # 选择最佳候选
    if not valid_contours:
        raise ValueError("未找到有效书页轮廓")
        
    page_contour = max(valid_contours, key=cv2.contourArea)

    # ========== 3. 亚像素级角点精修 ==========
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
    # ========== 1. 坐标排序验证 ==========
    if src_points.shape != (4, 2):
        raise ValueError("必须提供4个二维坐标点")

    # ========== 2. 智能尺寸计算 ==========
    (tl, tr, br, bl) = src_points
    width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))

    # ========== 3. 构建目标矩形（上下严格等宽） ==========
    dst = np.array([
        [0, 0],
        [width - 1, 0],  # 确保不越界
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")

    # ========== 4. 计算变换矩阵 ==========
    M = cv2.getPerspectiveTransform(src_points, dst)
    
    # ========== 5. 执行变换（带抗锯齿处理） ==========
    warped = cv2.warpPerspective(
        img, M, 
        (int(width), int(height)),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REPLICATE
    )
    
    return warped


# 书页在垂直方向上展开
def vertical_warp_image(img, num_cells=30, k=1.3):
    # ========== 1. 图像预处理 ==========
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

    # # ==================================== 可视化调试 ====================================
    # debug_img = img.copy()
    # cv2.drawContours(debug_img, valid_contours, -1, (0, 255, 0), 4)
    # show_image(debug_img)

    # ========== 2. 处理四边界曲线 ==========
    def extract_boundary_points(contours, img_height, threshold=0.3):
        """提取上下边界点"""
        upper_points = []
        lower_points = []
        for cnt in contours:
            for p in cnt[:,0]:
                if p[1] < img_height * threshold:
                    upper_points.append(p)
                elif p[1] > img_height * (1 - threshold):
                    lower_points.append(p)
        return upper_points, lower_points

    def create_spline_curve(points, img_width, img_height):
        """创建覆盖整个宽度的三次样条曲线（带严格递增校验）"""
        if not points:
            return lambda x: np.zeros_like(x)
        
        # 按x坐标排序并去重
        points = sorted(points, key=lambda p: p[0])
        unique_points = []
        seen_x = set()
        for p in reversed(points):  # 保留每个x最后出现的点
            if p[0] not in seen_x:
                seen_x.add(p[0])
                unique_points.append(p)
        unique_points = sorted(unique_points, key=lambda p: p[0])

        # 提取坐标
        x = np.array([p[0] for p in unique_points])
        y = np.array([p[1] for p in unique_points])
        
        # 智能添加边界约束
        if len(x) > 0:
            if x[0] > 0:
                x = np.concatenate([[0], x])
                y = np.concatenate([[y[0]], y])
            if x[-1] < img_width - 1:
                x = np.concatenate([x, [img_width - 1]])
                y = np.concatenate([y, [y[-1]]])
        
        # 最终校验
        if len(x) < 2:
            return lambda x: np.zeros_like(x)
        
        # 确保严格递增
        assert np.all(np.diff(x) > 0), "x坐标必须严格递增"
        
        cubic_spline = CubicSpline(x, y)
        return lambda x: np.clip(cubic_spline(x), 0, img_height)

    # 提取上下边界点并创建曲线
    h, w = img.shape[:2]
    upper_points, lower_points = extract_boundary_points(valid_contours, h)

    # # ==================================== 可视化调试 ====================================
    # debug_img = img.copy()
    # for p in upper_points:
    #     cv2.circle(debug_img, tuple(p), 5, (0,0,255), -1)
    # for p in lower_points:
    #     cv2.circle(debug_img, tuple(p), 5, (255,0,0), -1)
    # show_image(debug_img)

    upper_curve = create_spline_curve(upper_points, w, h)
    lower_curve = create_spline_curve(lower_points, w, h)

    # # ==================================== 可视化调试 ====================================
    # debug_img = img.copy()
    # # 生成采样点
    # x_samples = np.linspace(0, w-1, 2000)
    # y_upper = upper_curve(x_samples)
    # y_lower = lower_curve(x_samples)
    # for x, y in zip(x_samples.astype(int), y_upper.astype(int)):
    #     cv2.circle(debug_img, (x, y), 5, (0,0,255), -1)
    # for x, y in zip(x_samples.astype(int), y_lower.astype(int)):
    #     cv2.circle(debug_img, (x, y), 5, (255,0,0), -1)
    # show_image(debug_img)

    # ========== 3. 单元格处理 ==========
    cells_output = []
    for i in range(num_cells):
        if i == 0:
            cell_width = max(10, w // num_cells)  # 左侧单元格最小宽度
            remainder = w % num_cells
            x_left = 0
            x_right = cell_width
        else:
            cell_width = w // num_cells
            remainder = w % num_cells
            x_left = i * cell_width + (1 if i < remainder else 0)
            x_right = x_left + cell_width
        
        # 防止越界
        x_left = max(0, x_left)
        x_right = min(w, x_right)
        if x_left >= x_right:
            continue
        
        # 新增有效性校验
        if x_left > w - 10 or x_right < 10:
            print(f"Skipping edge cell {i}: {x_left}-{x_right}")
            continue
        
        try:
            # 获取边界曲线值（浮点坐标）
            y_top_left = upper_curve(x_left)
            y_top_right = upper_curve(x_right)
            y_bottom_left = lower_curve(x_left)
            y_bottom_right = lower_curve(x_right)
        except:
            continue  # 跳过无效单元格
        
        # 计算目标尺寸
        h_left = y_bottom_left - y_top_left
        h_right = y_bottom_right - y_top_right

        target_width = sqrt(((max(h_left, h_right) - min(h_left, h_right)) * k) ** 2 + (x_right - x_left) ** 2)
        target_height = max(h_left, h_right)

        # print(f"Cell {i}: {x_left}-{x_right}, {y_top_left}-{y_bottom_left} / {y_top_right}-{y_bottom_right}")
        # print(f"\tTarget size: {target_width} x {target_height}")
        
        if target_height <= 1e-9 or target_width <= 1e-9:
            continue  # 忽略无效变换
        
        # 定义源点和目标点（浮点坐标）
        src = [
            (x_left, y_top_left),
            (x_right, y_top_right),
            (x_right, y_bottom_right),
            (x_left, y_bottom_left)
        ]

        dst = [
            (0.0, 0.0),
            (target_width, 0.0),
            (target_width, target_height),
            (0.0, target_height)
        ]
        
        # 提取原始单元格区域（整数坐标）
        cell_region = img[0:h, x_left:x_right]
        # 可视化调试
        # show_image(cell_region)
        
        # 计算变换矩阵
        src = np.array(src, dtype=np.float32)
        dst = np.array(dst, dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(src, dst)

        # 动态选择插值方法
        if target_width > cell_region.shape[1]:
            warp_method = cv2.INTER_CUBIC  # 放大时使用三次插值
        else:
            warp_method = cv2.INTER_AREA   # 缩小时使用面积插值
        
        # 应用透视变换并验证有效性
        try:
            # print(f"\nProcessing cell {i}: x_left={x_left}, x_right={x_right}")
            # print(f"  src:\n{src}, \ndst:\n{dst}")
            # print(f"  matrix:\n{matrix}")
            
            warped = cv2.warpPerspective(cell_region, matrix, (int(target_width), int(target_height)), flags=warp_method, borderMode=cv2.BORDER_WRAP)
            # warped = cv2.warpPerspective(cell_region, matrix, (int(target_width), int(target_height)), flags=warp_method)
            # show_image(warped)
        except cv2.error as e:
            print(f"\nWarp failed for cell {i}:\n{str(e)}")
            continue
        

        if warped.size == 0 or warped.shape[0] <= 0 or warped.shape[1] <= 0:
            continue  # 跳过无效图像
        cells_output.append(warped)
    
    # 统一拉伸所有单元格到基准高度
    if not cells_output:
        return img
    base_h = cells_output[0].shape[0]
    for i in range(len(cells_output)):
        curr_h = cells_output[i].shape[0]
        if curr_h != base_h:
            # 保持宽度不变，仅调整高度
            cells_output[i] = cv2.resize(cells_output[i], 
                                    (cells_output[i].shape[1], base_h),
                                    interpolation=cv2.INTER_AREA)
    
    # 拼接处理后的单元格
    final_image = cv2.hconcat(cells_output) if cells_output else img
    return final_image

# 书页矫正主函数
def book_page_rectifier(img_path):
    img = cv2.imread(img_path)
    # 获取浮点型坐标（例如：[[123.4, 56.7], ...]）
    corners = auto_detect_page_corners(img)  

    # # ==================================== 可视化调试 ====================================
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

    # ========== 1. 水平方向矫正 ==========
    horizontal_img =  horizontal_warp_image(img, corners)

    # ========== 2. 垂直方向矫正 ==========
    vertical_img = vertical_warp_image(horizontal_img)

    # # ==================================== 可视化调试 ====================================
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
        if angle > -max_angle and angle < max_angle and angle != 0:
            angles.append(angle)

    # ========== 6. 计算最终角度 ==========
    if not angles:
        return img
    angle = np.mean(angles)

    # print(f"文字方向矫正角度：{angle}")

    # ========== 7. 旋转图像 ==========
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
    cleaned[:border_size, :] = 0  # 上边缘
    cleaned[-border_size:, :] = 0  # 下边缘
    cleaned[:, :border_size] = 0  # 左边缘
    cleaned[:, -border_size:] = 0  # 右边缘

    # ========== 5. 轮廓检测 ==========
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 有效性验证（面积阈值调整为5%）
    valid_contours = []
    for cnt in contours:
        if len(cnt) >= 5:
            area = cv2.contourArea(cnt)
            if area > img.shape[0]*img.shape[1]*0.05:  # 修改为5%
                valid_contours.append(cnt)
    
    if not valid_contours:
        raise ValueError("未检测到有效轮廓")
    
    # ========== 6. 执行局部异常凸起检测算法 ==========
    repaired_contours = []
    for cnt in valid_contours:
        try:
            # 6.1 前置检查
            if len(cnt) < 4:  # 至少需要4个点才能形成有效轮廓
                repaired_contours.append(cnt)
                continue

            # 6.2 类型转换与凸包计算
            cnt = cnt.astype(np.int32)
            hull_indices = cv2.convexHull(cnt, returnPoints=False)
            
            if hull_indices is None or len(hull_indices) < 3:
                print("凸包计算失败，保留原始轮廓")
                repaired_contours.append(cnt)
                continue

            # 6.3 凸性缺陷检测
            defects = cv2.convexityDefects(cnt, hull_indices)
            if defects is None:
                print("凸性缺陷检测失败，保留原始轮廓")
                repaired_contours.append(cnt)
                continue

            # 6.4 缺陷点分析
            defect_segments = []
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                # 索引有效性验证
                if all(0 <= x < len(cnt) for x in [s, e, f]):
                    distance = d / 256.0
                    if distance > 50:  # 凸起敏感度阈值（可调参数）
                        start = tuple(cnt[s][0])
                        end = tuple(cnt[e][0])
                        far = tuple(cnt[f][0])
                        print(f"检测到凸起缺陷：{start} -> {end} (Far: {far}, Distance: {distance})")
                        defect_segments.append( (start, end, far) )

            # 6.5 无显著缺陷则跳过
            if not defect_segments:
                print("未检测到有效凸起缺陷")
                repaired_contours.append(cnt)
                continue

            # 6.6 轮廓修复处理
            contour_points = cnt.squeeze()
            mask = np.ones(len(contour_points), dtype=bool)

            # 创建坐标查找字典（加速匹配）
            coord_dict = {(x, y): idx for idx, (x, y) in enumerate(contour_points)}

            for s_p, e_p, f_p in defect_segments:
                # 使用容差匹配查找索引（±2像素范围）
                # start_index: 起始点索引，end_index: 结束点索引，分别表示凸起段的起始和结束
                start_idx = next( (coord_dict.get((x,y)) for x in range(s_p[0]-2, s_p[0]+3) 
                                 for y in range(s_p[1]-2, s_p[1]+3) if (x,y) in coord_dict), None)
                end_idx = next( (coord_dict.get((x,y)) for x in range(e_p[0]-2, e_p[0]+3) 
                               for y in range(e_p[1]-2, e_p[1]+3) if (x,y) in coord_dict), None)

                if start_idx is None or end_idx is None:
                    continue

                # 标记需要删除的点（保留端点）
                if start_idx < end_idx:
                    mask[start_idx+1:end_idx] = False
                else:
                    mask[start_idx+1:] = False
                    mask[:end_idx] = False

            # 保留有效点
            preserved_points = contour_points[mask]

            # 6.7 凸包段插入
            for s_p, e_p, _ in defect_segments:
                # 生成凸包连接段
                segment_points = np.array([s_p, e_p], dtype=np.int32)
                hull_segment = cv2.convexHull(segment_points).squeeze()

                # 查找插入位置
                try:
                    insert_pos = np.where((preserved_points == s_p).all(axis=1))[0][0]
                    # 插入凸包点（排除重复端点）
                    preserved_points = np.insert(
                        preserved_points,
                        insert_pos + 1,
                        hull_segment[1:-1],
                        axis=0
                    )
                except IndexError:
                    continue

            # 6.8 生成最终轮廓
            repaired_cnt = preserved_points.reshape(-1, 1, 2).astype(np.int32)
            repaired_contours.append(repaired_cnt)

        except Exception as e:
            print(f"轮廓处理异常: {str(e)}，保留原始轮廓")
            repaired_contours.append(cnt)

    # ========== 7. 主区域合并逻辑 ==========
    if repaired_contours:
        # 合并所有有效轮廓的坐标点
        merged_points = np.vstack(repaired_contours)
        x, y, w, h = cv2.boundingRect(merged_points)
    else:
        h, w = img.shape[:2]
        x, y = int(w*0.1), int(h*0.1)
        w, h = int(w*0.8), int(h*0.8)
    
    # 安全边界扩展
    x = max(0, x-10)
    y = max(0, y-10)
    w = min(img.shape[1]-x, w+20)
    h = min(img.shape[0]-y, h+20)

    # ========== 8. 最终裁剪与验证 ==========
    cropped = img[y:y+h, x:x+w]
    gray_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    black_ratio = np.sum(gray_cropped < 50) / gray_cropped.size
    
    if black_ratio > black_tolerance:
        print(f"警告：黑色区域占比 {black_ratio:.2%}，超过阈值 {black_tolerance}")
    
    return cropped

# =========================文字区域分块=========================
def smart_horizontal_split(img_path, min_gap=5):
    # 读取图像并获取尺寸信息
    img = cv2.imread(img_path)
    H, W = img.shape[:2]
    min_h = int(H * 0.25)  # 最小允许高度
    max_h = int(H * 0.45)  # 最大允许高度
    
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
        window_h = 1  # 将窗口高度设为1，逐行检测
        img_area = H * W
        white_pixel_ratio = np.sum(cleaned == 255) / img_area
        density_th = max(0.1, min(0.3, white_pixel_ratio * 0.5))  # 降低密度阈值
        
        text_mask = np.zeros(H, dtype=np.uint8)
        for y in range(H - window_h + 1):
            window = cleaned[y:y+window_h, :]
            if np.sum(window == 255) / (window_h * W) > density_th:
                text_mask[y:y+window_h] = 1  # 标记当前行
        
        # 区域合并（放宽合并条件）
        regions, start = [], None
        for y in range(H):
            if text_mask[y]:
                if start is None:
                    start = y
                end = y
            else:
                if start is not None:
                    if regions and start - regions[-1][1] <= 5:  # 合并间隔扩大至5行
                        regions[-1] = (regions[-1][0], end)
                    else:
                        regions.append((start, end))
                    start = None
        # 处理最后未闭合的区域
        if start is not None:
            if regions and start - regions[-1][1] <= 5:
                regions[-1] = (regions[-1][0], end)
            else:
                regions.append((start, end))
        
        return [r for r in regions if r[1]-r[0] >= 1]  # 允许最小高度为1

    white_regions = detect_text_regions(cleaned)

    # ========== 4. 间隙分析 ==========
    valid_gaps = []
    for i in range(1, len(white_regions)):
        gap_start = white_regions[i-1][1] + 1
        gap_end = white_regions[i][0] - 1
        if gap_end - gap_start >= min_gap:
            valid_gaps.append({
                'start': gap_start,
                'end': gap_end,
                'center': (gap_start + gap_end) // 2,
                'size': gap_end - gap_start
            })
    
    # # ==================================== 可视化调试 ====================================
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
            candidates.add(gap['start'])
            candidates.add(gap['end'])
        candidates = sorted(list(candidates))  # 转换为有序列表
        
        # 动态规划寻找最优路径
        dp = [{'score': -float('inf'), 'path': []} for _ in range(H+1)]
        dp[0] = {'score': 0, 'path': [0]}
        
        for y in range(1, H+1):
            for s in range(max(0, y-max_h), y-min_h+1):
                if dp[s]['score'] == -float('inf'):
                    continue
                    
                # 计算当前段得分
                current_score = dp[s]['score']
                # 间隙位置奖励（当s是某个间隙的结束，y是下个间隙的开始时获得奖励）
                if any(g['end'] == s for g in gaps) and any(g['start'] == y for g in gaps):
                    current_score += 15  # 连续间隙奖励
                elif s in candidates or y in candidates:
                    current_score += 10  # 候选点基础奖励
                
                # 更新最优路径
                if current_score > dp[y]['score']:
                    dp[y] = {
                        'score': current_score,
                        'path': dp[s]['path'] + [y]
                    }
        
        # 提取最佳路径
        best_splits = dp[H]['path'] if dp[H]['score'] != -float('inf') else None
        
        # 有效性验证
        def validate(splits):
            if not splits or splits[-1] != H:
                return False
            segments = [splits[i+1]-splits[i] for i in range(len(splits)-1)]
            return all(min_h <= h <= max_h for h in segments) and (3 <= len(segments) <=4)
        
        if validate(best_splits):
            return best_splits
        
        # 保底策略：全排列搜索
        for split_num in [3,4]:
            for points in combinations([c for c in candidates if 0 < c < H], split_num-1):
                trial = sorted([0] + list(points) + [H])
                if validate(trial):
                    return trial
        
        return None

    # ========== 5. 执行分割方案 ==========
    final_split = find_optimal_splits(valid_gaps)

    # ========== 6. 后处理验证 ==========
    segments = []
    for i in range(len(final_split)-1):
        y1, y2 = final_split[i], final_split[i+1]
        # 最终高度校验
        if (y2 - y1) >= min_h:
            segments.append(img[y1:y2])
    
    return segments

if __name__ == "__main__":
    for i in range(1, 4):
        origin_path = f'./text_classificate/content/images/{i}_.png'

        # 旋转校正
        print(f"开始旋转校正{i}_.png...")
        rotated = correct_book_rotation(origin_path)
        cv2.imwrite(f"./text_classificate/content/images/{i}_1_rotated.png", rotated)

        # 分页处理
        print(f"开始分页处理{i}_1_rotated.png...")
        left_page, right_page = find_book_corners_and_split(f"./text_classificate/content/images/{i}_1_rotated.png")
        cv2.imwrite(f"./text_classificate/content/images/{i}_2_left_page.png", left_page)
        cv2.imwrite(f"./text_classificate/content/images/{i}_2_right_page.png", right_page)

        # 书页矫正
        print(f"开始书页矫正{i}_2_left_page.png...")
        corrected_left = book_page_rectifier(f"./text_classificate/content/images/{i}_2_left_page.png")
        print(f"开始书页矫正{i}_2_right_page.png...")
        corrected_right = book_page_rectifier(f"./text_classificate/content/images/{i}_2_right_page.png")
        cv2.imwrite(f"./text_classificate/content/images/{i}_3_corrected_left.png", corrected_left)
        cv2.imwrite(f"./text_classificate/content/images/{i}_3_corrected_right.png", corrected_right)

        # 文字方向矫正
        print(f"开始文字方向矫正{i}_3_corrected_left.png...")
        text_corrected_left = rotate_text_image(f"./text_classificate/content/images/{i}_3_corrected_left.png")
        print(f"开始文字方向矫正{i}_3_corrected_right.png...")
        text_corrected_right = rotate_text_image(f"./text_classificate/content/images/{i}_3_corrected_right.png")
        cv2.imwrite(f"./text_classificate/content/images/{i}_4_text_corrected_left.png", text_corrected_left)
        cv2.imwrite(f"./text_classificate/content/images/{i}_4_text_corrected_right.png", text_corrected_right)

        # 文字区域切割
        print(f"开始文字区域切割{i}_4_text_corrected_left.png...")
        text_block_left = get_text_block(f"./text_classificate/content/images/{i}_4_text_corrected_left.png")
        print(f"开始文字区域切割{i}_4_text_corrected_right.png...")
        text_block_right = get_text_block(f"./text_classificate/content/images/{i}_4_text_corrected_right.png")
        cv2.imwrite(f"./text_classificate/content/images/{i}_5_text_block_left.png", text_block_left)
        cv2.imwrite(f"./text_classificate/content/images/{i}_5_text_block_right.png", text_block_right)

        # 文字区域分块
        print(f"开始文字区域分块{i}_5_text_block_left.png...")
        text_blocks_left = smart_horizontal_split(f"./text_classificate/content/images/{i}_5_text_block_left.png")
        print(f"开始文字区域分块{i}_5_text_block_right.png...")
        text_blocks_right = smart_horizontal_split(f"./text_classificate/content/images/{i}_5_text_block_right.png")
        print(f"左页分块数量：{len(text_blocks_left)}，右页分块数量：{len(text_blocks_right)}")
        for j, block in enumerate(text_blocks_left):
            block_process = process_before_OCR(block)
            cv2.imwrite(f"./text_classificate/content/images/{i}_6_text_block_left_{j}.png", block_process)
        for j, block in enumerate(text_blocks_right):
            block_process = process_before_OCR(block)
            cv2.imwrite(f"./text_classificate/content/images/{i}_6_text_block_right_{j}.png", block_process)

        # 识别文字
        left_text = ""
        print(f"开始文字识别{i}_6_text_block_left.png...")
        for j in range(len(text_blocks_left)):
            left_text += get_pic_text(f"./text_classificate/content/images/{i}_6_text_block_left_{j}.png")
        print(f"左页文字识别结果：{left_text}")

        right_text = ""
        print(f"开始文字识别{i}_6_text_block_right.png...")
        for j in range(len(text_blocks_right)):
            right_text += get_pic_text(f"./text_classificate/content/images/{i}_6_text_block_right_{j}.png")
        print(f"右页文字识别结果：{right_text}")
        