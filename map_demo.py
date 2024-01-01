import os
import cv2
from utils import plot_one_box, cal_iou, xyxy_to_xywh, xywh_to_xyxy, updata_trace_list, draw_trace, intersect, ccw
import datetime
from tracker import Tracks
from tracker1 import Tracker
from kalmanfilter import KalmanFilter
from matplotlib import patches
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from Map import Map


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

    # 反转y轴
    pixel_y = 900 - pixel_y
    return int(pixel_x), int(pixel_y)


# label_path = "exp-11-13/saved_points"
# file_name = 'world_coords'
# save_txt = 'save_txt'

label_path = "example_7_校正角度/saved_txt"
file_name = ''
save_txt = 'save_txt'

# 设置视频相关参数
SAVE_VIDEO = False
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
if not os.path.exists('video_out'):
    os.mkdir('video_out')
video_filename = os.path.join('video_out', 'exp' + current_time + '.mp4')
frame_rate = 6
filelist = os.listdir(label_path)
frame_number = len(filelist)


# duration = 10  # 视频时长（秒）

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


# 创建Matplotlib画布和坐标轴
fig, ax = plt.subplots(figsize=(14, 9))

ax.set_xlim([0, 25])
ax.set_ylim([-10, 10])


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


p1 = [-11+14.34, -2.4, -8.6+14.34, -7.4]
p2 = [-8.6+14.34, -2.4, -6.2+14.34, -7.4]
p3 = [-6.2+14.34, -2.4, -3.8+14.34, -7.4]

p4 = [-11+14.34, 2.4, -8.6+14.34, 7.4]
p5 = [-8.6+14.34, 2.4, -6.2+14.34, 7.4]
p6 = [-6.2+14.34, 2.4, -3.8+14.34, 7.4]

p7 = [-2.4+14.34, -2.4, 0+14.34, -7.4]
p8 = [0+14.34, -2.4, 2.4+14.34, -7.4]
p9 = [2.4+14.34, -2.4, 4.8+14.34, -7.4]

p10 = [-2.4+14.34, 2.4, 0+14.34, 7.4]
p11 = [0+14.34, 2.4, 2.4+14.34, 7.4]
p12 = [2.4+14.34, 2.4, 4.8+14.34, 7.4]

P = [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12]
for p in P:
    plot_box_map(ax, p)

with open(os.path.join(label_path, file_name + str(0) + ".txt"), 'r') as f:
    content = f.readlines()
    tracker = Tracker(content)

xlim = ax.get_xlim()
ylim = ax.get_ylim()
canvas = FigureCanvas(fig)

canvas.draw()
img_array = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
img_array = img_array.reshape(canvas.get_width_height()[::-1] + (3,))

frame = img_array

# 设置视频分辨率和大小
width, height = 640, 480
video_size = (width, height)

# 设置视频编解码器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

if SAVE_VIDEO:
    # 创建视频写入对象
    video_writer = cv2.VideoWriter(video_filename, fourcc, frame_rate, (1400, 900))

mat = tracker.iou_mat(content)

frame_counter = 1  # 这里由视频label文件由0还是1开始命名确定
count1 = 0
while (True):
    # fig, ax = plt.subplots(figsize=(14, 9))
    # canvas.draw()
    img_array = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(canvas.get_width_height()[::-1] + (3,))
    frame = img_array

    print(f"当前帧数：{frame_counter}")
    if frame_counter > frame_number:
        break
    # label_file_path = os.path.join(label_path, file_name + "_" + str(frame_counter) + ".txt")
    label_file_path = os.path.join(label_path, file_name + str(frame_counter) + ".txt")
    if not os.path.exists(label_file_path):
        with open(label_file_path, "w") as f:
            pass
    with open(label_file_path, "r") as f:
        content = f.readlines()
        # track.predict()
        tracker.update(content)
    tracker.draw_tracks(frame)

    ####################################################计数
    # 定义边界线
    line1 = [coord_to_pixel(ax, (-11+14.34, 2.4)), coord_to_pixel(ax, (-11+14.34, -2.4))]  # 最左边的
    line2 = [coord_to_pixel(ax, (4.8+14.34, 2.4)), coord_to_pixel(ax, (4.8+14.34, -2.4))]  # 最右边的
    line3 = [coord_to_pixel(ax, (-2.4+14.34, 2.4)), coord_to_pixel(ax, (-2.4+14.34, -2.4))]  # 中间的
    # cv2.line(frame, line1[0], line1[1], (0, 255, 255), 2)
    cv2.line(frame, line2[0], line2[1], (0, 255, 255), 2)
    cv2.line(frame, line3[0], line3[1], (0, 255, 255), 2)

    already_counted_1 = []  # 我们假设每个id只穿越一次每个line，将已经穿越line1的id记录在这个list中
    already_counted_2 = []  # 我们假设每个id只穿越一次每个line，将已经穿越line2的id记录在这个list中
    already_counted_3 = []
    for track in tracker.tracks:
        if len(track.trace_point_list) < 2:
            break
        point = track.trace_point_list[-1]
        previous_point = track.trace_point_list[-2]
        # print(point, previous_point)

        if intersect(point, previous_point, line3[0], line3[1]) and track.track_id not in already_counted_3:
            already_counted_2.append(track.track_id)
            cv2.line(frame, line3[0], line3[1], (0, 0, 255), 4)
            print("QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ")
            if point[0] > previous_point[0]:
                count1 = count1 + 1
            else:
                count1 = count1 - 1

        if intersect(point, previous_point, line2[0], line2[1]) and track.track_id not in already_counted_2:
            already_counted_2.append(track.track_id)
            cv2.line(frame, line2[0], line2[1], (0, 0, 255), 4)
            print("QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ")
            if point[0] > previous_point[0]:
                count1 = count1 - 1
            else:
                count1 = count1 + 1
        print(count1)

    cv2.putText(frame, "ALL BOXES(Green)", (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
    cv2.putText(frame, "TRACKED BOX(Red)", (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"count of area_1:    {count1}", (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255),
                2)

    cv2.imshow('track', frame)
    if SAVE_VIDEO:
        video_writer.write(frame)
    frame_counter = frame_counter + 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    plt.close()

video_writer.release()

"""
# 循环绘制图形并保存为视频
for i in range(318):
    # 清除之前的绘图
    ax.clear()

    # 在坐标轴上绘制新的图形（这里用一个简单的例子）
    x = np.linspace(0, 20 * np.pi, 100)
    y = np.sin(x + i * 0.1)
    ax.plot(x, y)

    # 将绘制的内容渲染到画布上
    canvas.draw()

    # 从画布获取图像并将其转换为OpenCV格式
    img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    img = img.reshape(canvas.get_width_height()[::-1] + (3,))

    # 将图像写入视频
    video_writer.write(img)

    cv2.imshow('Matplotlib Video', img)

    # 检查按键 'q' 是否被按下，如果是则退出循环
    if cv2.waitKey(1000 // frame_rate) & 0xFF == ord('q'):
        break

# 释放资源
video_writer.release()

# 显示Matplotlib图形（可选）
plt.show()
"""
