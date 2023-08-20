import os
import cv2
import numpy as np
from utils import plot_one_box, cal_iou, xyxy_to_xywh, xywh_to_xyxy, updata_trace_list, draw_trace
import datetime
from tracker import Tracks

class Tracker:
    def Tracker(self, detection):
        """
        初始化时需要读入第一帧文件的信息。[X1, X2, ...],X1.shape = (6,)
        """



    def update(self):
        pass