import cv2
from matplotlib import patches
import numpy as np

def xyxy_to_xywh(xyxy):
    """
    将边界框格式从 (x1, y1, x2, y2) 转换为 (center_x, center_y, width, height)。

    参数:
        xyxy (list): 包含边界框坐标 (x1, y1, x2, y2) 的列表。

    返回值:
        tuple: 边界框格式为 (center_x, center_y, width, height) 的元组。
    """
    center_x = (xyxy[0] + xyxy[2]) / 2
    center_y = (xyxy[1] + xyxy[3]) / 2
    w = xyxy[2] - xyxy[0]
    h = xyxy[3] - xyxy[1]
    return (center_x, center_y, w, h)



def plot_one_box(xyxy, img, color=(0, 200, 0), target=False):
    """
    在图像上绘制单个边界框。这个函数用于在图像上绘制边界框。参数xyxy是边界框的坐标（左上角和右下角），
    img是输入的图像，color是边界框的颜色，默认为绿色，target是一个标志，如果为True，则使用红色表示边界框，
    否则使用指定颜色。

    参数:
        xyxy (list): 边界框坐标 (x1, y1, x2, y2)。
        img (numpy.ndarray): 输入图像。
        color (tuple): 边界框颜色 (默认为绿色)。
        target (bool): 标志指示边界框是否表示目标 (默认为False)。
    """
    xy1 = (int(xyxy[0]), int(xyxy[1]))
    xy2 = (int(xyxy[2]), int(xyxy[3]))
    if target:
        color = (0, 0, 255)
    cv2.rectangle(img, xy1, xy2, color, 1, cv2.LINE_AA)  # filled

def plot_box_map(ax, box_coords):
    """
    在指定的画布上画出矩形框。

    Parameters:
        ax (matplotlib.axes._axes.Axes): Matplotlib的坐标轴对象
        box_coords (tuple): 矩形框的坐标，形式为 (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = box_coords

    # 绘制矩形框
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')

    # 将矩形框添加到坐标轴
    ax.add_patch(rect)


def updata_trace_list(box_center, trace_list, max_list_len=50):
    """
    更新轨迹列表，保持其长度不超过指定的最大长度。

    参数:
        box_center (tuple): 边界框中心坐标 (x, y)。
        trace_list (list): 轨迹列表。
        max_list_len (int): 最大轨迹列表长度 (默认为50)。

    返回值:
        list: 更新后的轨迹列表。
    """
    if len(trace_list) <= max_list_len:
        trace_list.append(box_center)
    else:
        trace_list.pop(0)
        trace_list.append(box_center)
    return trace_list



def draw_trace(img, trace_list):
    """
    更新trace_list,绘制trace
    :param trace_list:
    :param max_list_len:
    :return:
    """
    for i, item in enumerate(trace_list):

        if i < 1:
            continue
        cv2.line(img,
                 (trace_list[i][0], trace_list[i][1]), (trace_list[i - 1][0], trace_list[i - 1][1]),
                 (255, 255, 0), 3)


def cal_iou(box1, box2):
    """

    :param box1: xyxy 左上右下
    :param box2: xyxy
    :return:
    """
    x1min, y1min, x1max, y1max = box1[0], box1[1], box1[2], box1[3]
    x2min, y2min, x2max, y2max = box2[0], box2[1], box2[2], box2[3]
    # 计算两个框的面积
    s1 = (y1max - y1min + 1.) * (x1max - x1min + 1.)
    s2 = (y2max - y2min + 1.) * (x2max - x2min + 1.)

    # 计算相交部分的坐标
    xmin = max(x1min, x2min)
    ymin = max(y1min, y2min)
    xmax = min(x1max, x2max)
    ymax = min(y1max, y2max)

    inter_h = max(ymax - ymin + 1, 0)
    inter_w = max(xmax - xmin + 1, 0)

    intersection = inter_h * inter_w
    union = s1 + s2 - intersection

    # 计算iou
    iou = intersection / union
    return iou


def cal_distance(box1, box2):
    """
    计算两个box中心点的距离
    :param box1: xyxy 左上右下
    :param box2: xyxy
    :return:
    """
    center1 = ((box1[0] + box1[2]) // 2, (box1[1] + box1[3]) // 2)
    center2 = ((box2[0] + box2[2]) // 2, (box2[1] + box2[3]) // 2)
    dis = ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5

    return dis


def xywh_to_xyxy(xywh):
    x1 = xywh[0] - xywh[2] // 2
    y1 = xywh[1] - xywh[3] // 2
    x2 = xywh[0] + xywh[2] // 2
    y2 = xywh[1] + xywh[3] // 2

    return [x1, y1, x2, y2]


def coord_to_pixel(ax, coord, y_range=900):
    """
    将坐标中的点映射到画布中的像素点。

    Parameters:
        ax (matplotlib.axes._axes.Axes): Matplotlib的坐标轴对象
        coord (tuple): 坐标中的点，形式为 (x, y)
        y_range (int): 画布y轴的范围，默认为900像素

    Returns:
        tuple: 画布中的像素点，形式为 (pixel_x, pixel_y)
    """
    # 从坐标轴的数据坐标系转换为画布的像素坐标系
    x, y = coord
    pixel_x, pixel_y = ax.transData.transform_point((x, y))

    # 反转y轴，将坐标系原点从左上角移动到左下角，这是因为matpoltlib中的坐标原点在坐下而cv2中的坐标原点在左上
    pixel_y = y_range - pixel_y

    # 返回整数类型的像素坐标
    return int(pixel_x), int(pixel_y)



def plot_box_map(ax, box_coords):
    """
    在指定的画布上画出矩形框。

    Parameters:
        ax (matplotlib.axes._axes.Axes): Matplotlib的坐标轴对象
        box_coords (tuple): 矩形框的坐标，形式为 (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = box_coords

    # 绘制矩形框
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='g', facecolor='none')

    # 将矩形框添加到坐标轴
    ax.add_patch(rect)


def content2detections(content, ax):
    """
    从读取的文件中解析出检测到的目标信息
    :param content: readline返回的列表
    :return: [X1, X2, ...],X1.shape = (6,)
    """
    detections = []
    for i, detection in enumerate(content):
        data = detection.replace('\n', "").split(" ")
        # detect_xywh = np.array(data[1:5], dtype="float")
        detect_xywh = np.array(data, dtype="float")
        detect_xywh = np.delete(detect_xywh, -1)

        detect_xywh = coord_to_pixel(ax, detect_xywh)
        detections.append(detect_xywh)
    return detections


def vector_norm(vector):
    """
    计算向量的二范数（Euclidean norm）。

    Parameters:
        vector (list or numpy.ndarray): 输入的向量。

    Returns:
        float: 向量的二范数。
    """
    # 将输入的向量转换为NumPy数组
    vector = np.array(vector)

    # 计算向量的二范数
    norm = np.linalg.norm(vector, ord=2)

    return np.array(norm)


def vector_norm_ax1(vector):
    """
    计算向量的二范数（Euclidean norm）。

    Parameters:
        vector (list or numpy.ndarray): 输入的向量。

    Returns:
        float: 向量的二范数。
    """
    # 将输入的向量转换为NumPy数组
    vector = np.array(vector)

    # 计算向量的二范数
    norm = np.linalg.norm(vector, ord=2, axis=1)

    return np.array(norm)


def intersect(A, B, C, D):
    """
    判断两条线段是否相交。

    Parameters:
        A (tuple): 第一条线段的起点坐标 (x, y)
        B (tuple): 第一条线段的终点坐标 (x, y)
        C (tuple): 第二条线段的起点坐标 (x, y)
        D (tuple): 第二条线段的终点坐标 (x, y)

    Returns:
        bool: 如果两条线段相交则返回True，否则返回False
    """
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def ccw(A, B, C):
    """
    判断三个点的方向关系是否为逆时针（Counter-Clockwise）。

    Parameters:
        A (tuple): 第一个点的坐标 (x, y)
        B (tuple): 第二个点的坐标 (x, y)
        C (tuple): 第三个点的坐标 (x, y)

    Returns:
        bool: 如果方向为逆时针则返回True，否则返回False
    """
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])



def is_point_inside_rectangle(rectangle_left_top, rectangle_right_bottom, point):
    """
    判断点是否在矩形框内部。

    参数:
    - rectangle_left_top: 矩形框左上角坐标 (x1, y1)
    - rectangle_right_bottom: 矩形框右下角坐标 (x2, y2)
    - point: 要判断的点坐标 (x, y)

    返回值:
    - True 如果点在矩形框内，否则返回 False
    """
    x1, y1 = rectangle_left_top
    x2, y2 = rectangle_right_bottom
    x, y = point
    return x1 <= x <= x2 and y1 <= y <= y2

if __name__ == "__main__":
    box1 = [100, 100, 200, 200]
    box2 = [100, 100, 200, 300]
    iou = cal_iou(box1, box2)
    print(iou)
    box1.pop(0)
    box1.append(555)
    print(box1)
