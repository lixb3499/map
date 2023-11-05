import os
import cv2
import numpy as np
from utils import plot_one_box, cal_iou, xyxy_to_xywh, xywh_to_xyxy, updata_trace_list, draw_trace
import datetime
from tracker import Tracks
from tracker1 import Tracker
from kalmanfilter import KalmanFilter
from scipy.optimize import linear_sum_assignment

class Map(Tracker):
    '''
    class Map inherit from class Tracker
    '''
    def __init__(self, content, parking_file = ''):
        super(Map, self).__init__(content)
        '''
        parking_id由Map分配
        :param content: 第一帧检测到的信息，用于初始化跟踪器
        :param parking_file: 拿到的地图的位置信息的文件
        '''
        parking_list = []





class Parking:

    def __init__(self, parking_id, xywh, havecar=False):
        '''
        :param park_id:每个车位id不同,parking_id由Map分配
        :param xywh:车位的位置
        :param havecar:是否u车的标志
        '''
        self.id, self.xywh, self.havecar = parking_id, xywh, havecar

    def update(self):
        pass

if __name__ == "__main__":
    video_path = "test100_6mm/connect.avi"
    label_path = "test100_6mm/point_center"
    file_name = ""  #label文件数字前的
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


