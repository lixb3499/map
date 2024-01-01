import os
import cv2
import numpy as np
from utils import plot_one_box, cal_iou, xyxy_to_xywh, xywh_to_xyxy, updata_trace_list, draw_trace, intersect, \
    is_point_inside_rectangle
import datetime
from tracker import Tracks
from tracker1 import Tracker
from kalmanfilter import KalmanFilter
from scipy.optimize import linear_sum_assignment


class Map(Tracker):
    '''
    class Map inherit from class Tracker
    '''

    def __init__(self, content, area_inf_list=[]):
        super(Map, self).__init__(content)
        '''
        parking_id由Map分配
        :param content: 第一帧检测到的信息，用于初始化跟踪器
        :param parking_file: 拿到的地图的位置信息的文件
        '''
        # self.parking_list = [Parking(1, [100, 100, 2, 2]), Parking(2, [200, 200, 2, 3])]
        self.areas = []
        for i, area_inf in enumerate(area_inf_list):
            self.areas.append(Area(i, area_inf[0], area_inf[1]))

    def intersect(self, point, previous_point):
        for area in self.areas:
            area.intersect(point, previous_point)

    def draw_area(self, frame):
        for area in self.areas:
            area.draw(frame)

    # def dis_matrix(self):


class Border(object):
    '''
    一个区域的边界，主要是两个点以及一个判断哪边是区域内部的函数
    '''

    def __init__(self, point1, point2):
        '''

        :param point1: 第一个点
        :param point2: 第二个点
        '''
        self.point = [point1, point2]
        self.color = (0, 255, 255)
        self.thickness = 2

    def __getitem__(self, index):
        return self.point[index]

    def intersect(self, point, previous_point):
        self.color = (0, 255, 255)
        self.thickness = 2
        isintersect = intersect(point, previous_point, self.point[0], self.point[1])
        if isintersect:
            self.color = (0, 0, 255)  # 如果相交则这条边界线画成红色
            self.thickness = 5
        return isintersect

    def draw_border(self, frame):
        cv2.line(frame, self.point[0], self.point[1], self.color, self.thickness)


class Area(object):

    def __init__(self, area_id, x1y1, x2y2, count_car=0):
        '''

        :param area_id: id
        :param count_car:此区域内现有的car的数量
        :param xyxy:此区域的坐标，我们使用左上右下两点的坐标表示矩形区域
        '''
        self.id, self.count_car = area_id, count_car
        self.rectangle_left_top = x1y1
        self.rectangle_right_bottom = x2y2
        self.border_lines = [Border(x1y1, (x1y1[0], x2y2[1])), Border(x1y1, (x2y2[0], x1y1[1])),
                             Border((x1y1[0], x2y2[1]), x2y2), Border((x2y2[0], x1y1[1]), x2y2)]

    def update(self):
        pass

    def isenter(self, point, previous_point):
        result = is_point_inside_rectangle(self.rectangle_left_top, self.rectangle_right_bottom, point)
        return result

    def intersect(self, point, previous_point):
        for border_line in self.border_lines:
            if border_line.intersect(point, previous_point):
                if self.isenter(point, previous_point):
                    self.count_car = self.count_car + 1
                else:
                    self.count_car = self.count_car - 1
                return border_line

    def draw(self, frame):
        for i in range(len(self.border_lines)):
            self.border_lines[i].draw_border(frame)


if __name__ == "__main__":
    video_path = "test100_6mm/connect.avi"
    label_path = "test100_6mm/point_center"
    file_name = ""  # label文件数字前的
    # cap = cv2.VideoCapture(video_path)
    # frame_number = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # print(f"Video FPS: {frame_number}")
    # SAVE_VIDEO = True  # True
    # current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # if not os.path.exists('video_out'):
    #     os.mkdir('video_out')
    with open(os.path.join(label_path, file_name + str(0) + ".txt"), 'r') as f:
        content = f.readlines()
        map = Map(content)
    a = map.iou_mat(content)
    print(a)
