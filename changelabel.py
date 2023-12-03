import os
import cv2
import numpy as np
from utils import plot_one_box, cal_iou, xyxy_to_xywh, xywh_to_xyxy, updata_trace_list, draw_trace
import datetime
from tracker import Tracks
from tracker1 import Tracker
from kalmanfilter import KalmanFilter
from scipy.optimize import linear_sum_assignment
from matplotlib import pyplot as plt

from matplotlib import patches
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def coord_to_pixel(ax, coord):
    """
    将坐标中的点映射到画布中的像素点。

    Parameters:
        ax (matplotlib.axes._axes.Axes): Matplotlib的坐标轴对象
        coord (tuple): 坐标中的点，形式为 (x, y)

    Returns:
        tuple: 画布中的像素点，形式为 (pixel_x, pixel_y)
    """
    x, y = coord
    pixel_x, pixel_y = ax.transData.transform_point((x, y))

    # # 反转y轴
    # pixel_y = 900 - pixel_y

    return int(pixel_x), int(pixel_y)


root = "example_5"
label_path = "example_5/saved_points"
file_name = 'world_coords'
save_txt = 'saved_txt'  # 转换后的label文件


# # 设置视频相关参数
# video_filename = 'output_video.mp4'
# frame_rate = 6
# # duration = 10  # 视频时长（秒）

def content2detections(content, ax):
    """
    从读取的文件中解析出检测到的目标信息
    :param content: readline返回的列表
    :param ax: 创建的画布，我们需要将地图坐标点转化为像素坐标
    :return: [X1, X2]
    """
    detections = []
    for i, detection in enumerate(content):
        data = detection.replace('\n', "").split(" ")
        # detect_xywh = np.array(data[1:5], dtype="float")
        detect_xywh = np.array(data, dtype="float")
        detect_xywh = np.delete(detect_xywh, -1)

        if detect_xywh[1] > -3.0 and detect_xywh[1] < 3.0:
            detect_xywh = coord_to_pixel(ax, detect_xywh)
            detections.append(detect_xywh)
    return detections


# 创建Matplotlib画布和坐标轴
fig, ax = plt.subplots(figsize=(14, 9))

ax.set_xlim([-15, 10])
ax.set_ylim([-10, 10])

if not os.path.exists(os.path.join(root, save_txt)):
    os.makedirs(os.path.join(root, save_txt))

filelist = os.listdir(label_path)
n = len(filelist)
for i in range(n):
    with open(os.path.join(label_path, file_name + '_' + str(i) + ".txt"), 'r') as f:
        content = f.readlines()
        detection = content2detections(content, ax)
    with open(os.path.join(root, save_txt) + f'/{i}.txt', 'w') as file:
        for item in detection:
            # 将元组转换为字符串，并写入文件
            file.write(f"{item[0]} {item[1]}\n")
        print(f'write file {i}')
