import os
import cv2
import numpy as np
from utils import plot_one_box, cal_iou, xyxy_to_xywh, xywh_to_xyxy, updata_trace_list, draw_trace
# import datetime

class KalmanFilter(object):
    def __init__(self):
        # 状态转移矩阵，上一时刻的状态转移到当前时刻
        self.A = np.array([[1, 0, 0, 0, 1, 0],
                      [0, 1, 0, 0, 0, 1],
                      [0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 1]])

        # 状态观测矩阵
        self.H = np.eye(6)

        # 过程噪声协方差矩阵Q，p(w)~N(0,Q)，噪声来自真实世界中的不确定性,
        # 在跟踪任务当中，过程噪声来自于目标移动的不确定性（突然加速、减速、转弯等）
        self.Q = np.eye(6) * 0.1

        # 观测噪声协方差矩阵R，p(v)~N(0,R)
        # 观测噪声来自于检测框丢失、重叠等
        self.R = np.eye(6) * 1

        # 控制输入矩阵B
        self.B = None
        # 状态估计协方差矩阵P初始化
        self.P = np.eye(6)

    def predict(self, X, cov):
        X_predict = np.dot(self.A, X)
        cov1 = np.dot(self.A, cov)
        cov_predict = np.dot(cov1, self.A.T) + self.Q
        return X_predict, cov_predict
    def update(self, X_predict, cov_predict):
        pass