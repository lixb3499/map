from matplotlib import pyplot as plt
from matplotlib import patches
import utils
import cv2
import numpy as np

#
#
# fig, ax = plt.subplots()
# ax.set_xlim([-15, 10])
# ax.set_ylim([-10, 10])
#
# def plot_box_map(ax, box_coords):
#     """
#     在指定的画布上画出矩形框。
#
#     Parameters:
#         ax (matplotlib.axes._axes.Axes): Matplotlib的坐标轴对象
#         box_coords (tuple): 矩形框的坐标，形式为 (x1, y1, x2, y2)
#     """
#     x1, y1, x2, y2 = box_coords
#
#     # 绘制矩形框
#     rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='g', facecolor='none')
#
#     # 将矩形框添加到坐标轴
#     ax.add_patch(rect)
#
# p1 = [-11, -2.4, -8.6, -7.4]
# p2 = [-8.6, -2.4, -6.2, -7.4]
# p3 = [-6.2, -2.4, -3.8, -7.4]
#
# p4 = [-11, 2.4, -8.6, 7.4]
# p5 = [-8.6, 2.4, -6.2, 7.4]
# p6 = [-6.2, 2.4, -3.8, 7.4]
#
# p7 = [-2.4, -2.4, 0, -7.4]
# p8 = [0, -2.4, 2.4, -7.4]
# p9 = [2.4, -2.4, 4.8, -7.4]
#
# p10 = [-2.4, 2.4, 0, 7.4]
# p11 = [0, 2.4, 2.4, 7.4]
# p12 = [2.4, 2.4, 4.8, 7.4]
#
# P = [p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12]
# for p in P:
#     plot_box_map(ax, p)
#
# plt.show()

# import cv2
# import numpy as np
#
# def coord_to_pixel(ax, coord):
#     """
#     将坐标中的点映射到画布中的像素点。
#
#     Parameters:
#         ax (matplotlib.axes._axes.Axes): Matplotlib的坐标轴对象
#         coord (tuple): 坐标中的点，形式为 (x, y)
#
#     Returns:
#         tuple: 画布中的像素点，形式为 (pixel_x, pixel_y)
#     """
#     x, y = coord
#     pixel_x, pixel_y = ax.transData.transform_point((x, y))
#     return int(pixel_x), int(pixel_y)
#
# # 创建一个图形和坐标轴，并指定画布大小与坐标轴一致
# fig, ax = plt.subplots(figsize=(25, 20))  # 这里可以根据需要调整 figsize
#
# # 设置坐标轴范围
# ax.set_xlim([-15, 10])
# ax.set_ylim([-10, 10])
#
# # 在坐标轴上绘制一个点
# point_coord = (5, 2)
# ax.plot(*point_coord, 'ro')  # 在图上以红色圆点标记该点
# plt.show()
# # 获取画布中的像素点
# pixel_point = coord_to_pixel(ax, point_coord)
#
# # 创建一个黑色图像，大小与画布一致
# image = 255*np.ones((int(fig.get_figheight()), int(fig.get_figwidth()), 3), dtype=np.uint8)
#
# # 在图像上画出像素点
# cv2.circle(image, pixel_point, 5, (0, 0, 255), -1)  # 5是半径，(0, 0, 255)是颜色 (BGR格式)
#
# # 显示图像
# cv2.imshow('Pixel Point', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# plt.close()

# import cv2
# import numpy as np
#
# def coord_to_pixel(ax, coord):
#     """
#     将坐标中的点映射到画布中的像素点。
#
#     Parameters:
#         ax (matplotlib.axes._axes.Axes): Matplotlib的坐标轴对象
#         coord (tuple): 坐标中的点，形式为 (x, y)
#
#     Returns:
#         tuple: 画布中的像素点，形式为 (pixel_x, pixel_y)
#     """
#     x, y = coord
#     pixel_x, pixel_y = ax.transData.transform_point((x, y))
#     return int(pixel_x), int(pixel_y)
#
# # 创建一个图形和坐标轴，并指定画布大小与坐标轴一致
# fig, ax = plt.subplots(figsize=(8, 6))
#
# # 在坐标轴上绘制一个圆
# circle = plt.Circle((0, 0), 2, color='blue', fill=False)
# ax.add_patch(circle)
#
# # 显示图形
# plt.show()
#
# # 获取图形的坐标轴范围
# xlim = ax.get_xlim()
# ylim = ax.get_ylim()
#
# # 获取画布中的像素点
# pixel_center = coord_to_pixel(ax, ((xlim[0] + xlim[1]) / 2, (ylim[0] + ylim[1]) / 2))
#
# # 创建一个白色图像，大小与画布一致
# image = np.ones((int(fig.get_figheight()), int(fig.get_figwidth()), 3), dtype=np.uint8) * 255
#
# # 在图像上画出像素层面的圆
# cv2.circle(image, pixel_center, int(2 * fig.dpi), (0, 0, 255), 2)  # 2 * fig.dpi 是半径，(0, 0, 255)是颜色 (BGR格式)
#
# # 显示图像
# cv2.imshow('Pixel Circle', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# plt.close()

"'这个程序演示坐标系坐标与像素坐标转换以及作图" \
"fuckingcv2'"

import cv2
import numpy as np


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
    return int(pixel_x), int(pixel_y)


def distance_to_pixel(ax, distance):
    """
    将坐标中的距离（以点为单位）转换为画布中的像素距离。

    Parameters:
        ax (matplotlib.axes._axes.Axes): Matplotlib的坐标轴对象
        distance (float): 坐标中的距离，以点为单位

    Returns:
        float: 画布中的像素距离
    """
    # 获取坐标轴的变换
    trans = ax.transData

    # 计算距离对应的像素距离
    pixel_distance = trans.transform((distance, 0))[0] - trans.transform((0, 0))[0]

    return int(pixel_distance)


# 创建一个图形和坐标轴，并指定画布大小与坐标轴一致
fig, ax = plt.subplots(figsize=(14, 9))
ax.set_xlim([-15, 10])
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


p1 = [-11, -2.4, -8.6, -7.4]
p2 = [-8.6, -2.4, -6.2, -7.4]
p3 = [-6.2, -2.4, -3.8, -7.4]

p4 = [-11, 2.4, -8.6, 7.4]
p5 = [-8.6, 2.4, -6.2, 7.4]
p6 = [-6.2, 2.4, -3.8, 7.4]

p7 = [-2.4, -2.4, 0, -7.4]
p8 = [0, -2.4, 2.4, -7.4]
p9 = [2.4, -2.4, 4.8, -7.4]

p10 = [-2.4, 2.4, 0, 7.4]
p11 = [0, 2.4, 2.4, 7.4]
p12 = [2.4, 2.4, 4.8, 7.4]

P = [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12]
for p in P:
    plot_box_map(ax, p)

# 在坐标轴上绘制一个圆
circle = plt.Circle((0, 0), 2, color='blue', fill=False)
ax.add_patch(circle)

# plt.show()
# 获取图形的坐标轴范围
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# 获取画布中的像素点
# pixel_center = coord_to_pixel(ax, ((xlim[0] + xlim[1]) / 2, (ylim[0] + ylim[1]) / 2))
pixel_center = coord_to_pixel(ax, (0, 0))

# 获取图形的 Canvas 对象，并将其转换为像素数组
canvas = fig.canvas
canvas.draw()
img_array = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
img_array = img_array.reshape(canvas.get_width_height()[::-1] + (3,))

# 在图像数组上画出像素层面的圆
cv2.circle(img_array, pixel_center, distance_to_pixel(ax, 2), (0, 255, 255),
           2)  # 2 * fig.dpi 是半径，(0, 0, 255)是颜色 (BGR格式)

# 显示图像数组
# cv2.imshow('Pixel Circle on Matplotlib', img_array)
cv2.waitKey(0)
cv2.destroyAllWindows()
plt.close()


def unit_length_to_pixels(ax, unit_length_x, unit_length_y):
    """
    将坐标轴上的单位长度转换为画布上的像素点个数。

    Parameters:
        ax (matplotlib.axes._axes.Axes): Matplotlib的坐标轴对象
        unit_length_x (float): 横坐标轴上的单位长度
        unit_length_y (float): 纵坐标轴上的单位长度

    Returns:
        tuple: 横纵坐标轴上的单位长度对应的像素点个数，形式为 (pixels_x, pixels_y)
    """
    # 选择两个相邻的点在横纵坐标轴上的坐标，计算它们之间的像素距离
    point1 = (0, 0)
    point2 = (unit_length_x, unit_length_y)

    pixel1 = coord_to_pixel(ax, point1)
    pixel2 = coord_to_pixel(ax, point2)

    print(pixel1, pixel2)
    pixels_x = abs(pixel2[0] - pixel1[0])
    pixels_y = abs(pixel2[1] - pixel1[1])

    return pixels_x, pixels_y


# 创建 Matplotlib 图形和坐标轴
fig, ax = plt.subplots(figsize=(14, 9))
ax.set_xlim([-15, 10])
ax.set_ylim([-10, 10])

# 定义单位长度
unit_length_x = 1
unit_length_y = 1

# 将单位长度转换为像素点个数
pixels_x, pixels_y = unit_length_to_pixels(ax, unit_length_x, unit_length_y)

print(f"横坐标轴上 {unit_length_x} 单位长度对应 {pixels_x} 像素点")
print(f"纵坐标轴上 {unit_length_y} 单位长度对应 {pixels_y} 像素点")

print(utils.vector_norm([2, 2]))
