import os
import cv2
import numpy as np
from utils import plot_one_box, cal_iou, xyxy_to_xywh, xywh_to_xyxy, updata_trace_list, draw_trace
import datetime
from kalmanfilter import KalmanFilter

class  Tracks:
    def __init__(self, initial_state, initial_cov = np.eye(6), track_id=0):
        # 默认初始化状态为[x ,y, w, h, 0, 0]
        # initial_state =
        if len(initial_state) == 2:
            self.X = np.append(initial_state, [30, 30, 0, 0])
        self.X = np.append(initial_state, [0, 0])
        self.P =initial_cov
        self.IOU_Threshold = 0.02 # 匹配时的阈值
        self.target_box = xywh_to_xyxy(self.X[0:4]) #检测到匹配上的目标的框
        self.target_xywh = self.X[0:4]
        self.box_center = (int((self.target_box[0] + self.target_box[2]) // 2), int((self.target_box[1] + self.target_box[3]) // 2))
        self.KF = KalmanFilter()
        self.trace_point_list = [] #储存轨迹的历史点信息用于画轨迹
        self.trace_v_list = []
        self.max_trace_number = 50
        self.max_iou_matched = False
        self.Z = np.array(self.X) #初始化观测矩阵为X
        self.track_id = track_id
        self.lost_number = 0
        self.number_since_match = 0
        self.confirmflag = False  # 轨迹需要检测到三帧以上才能变为确定的一条轨迹
        self.parking_id = None # 该轨迹在地图中对应的车位

    # def predict(self):
    #     self.X, self.P = self.KF.predict(self.X, self.P)

    def iou_match(self, detect):
        self.max_iou_matched = False
        max_iou = self.IOU_Threshold
        self.target_box = xywh_to_xyxy(self.X[0:4]) #predict后更新一下目标框
        # for j, data_ in enumerate(detect):
        #     data = data_.replace('\n', "").split(" ")
        #     detect_xywh = np.array(data[1:5], dtype="float")
        #     detect_xyxy = xywh_to_xyxy(detect_xywh)
        #     # plot_one_box(xyxy, frame)
        #     iou = cal_iou(detect_xyxy, xywh_to_xyxy(self.X[0:4]))
        #     if iou > max_iou:
        #         self.target_box = detect_xyxy
        #         max_iou = iou
        #         self.max_iou_matched = True

        detect_xyxy = xywh_to_xyxy(detect)
        # plot_one_box(xyxy, frame)
        iou = cal_iou(detect_xyxy, xywh_to_xyxy(self.X[0:4]))
        if iou > max_iou:
            self.target_box = detect_xyxy
            max_iou = iou
            self.max_iou_matched = True

    def update(self):
        if self.max_iou_matched:
            # self.detect_xywh = xyxy_to_xywh(self.detect_xyxy)
            self.target_xywh = xyxy_to_xywh(self.target_box)
            self.dx = self.target_xywh[0] - self.X[0]
            self.dy = self.target_xywh[1] - self.X[1]

            self.Z[0:4] = np.array(self.target_xywh).T
            self.Z[4::] = np.array([self.dx, self.dy])
            self.X, self.P = self.KF.predict(self.X, self.P)
            self.X, self.P = self.KF.update(self.X, self.P, self.Z)

            self.X[2: 4] = [120, 120]

            self.number_since_match += 1
            self.lost_number = 0
        else:
            if len(self.trace_v_list) > 0:      #第一次检测到的时候也是匹配不到的
                self.X[-2::] = self.trace_v_list[0]
            self.X, self.P = self.KF.predict(self.X, self.P)

            self.X[2: 4] = [120, 120]

            self.number_since_match = 0
            self.lost_number += 1
        self.xywh = self.X[0:4]
        self.target_box = xywh_to_xyxy(self.X[0:4])
        self.box_center = (int((self.target_box[0] + self.target_box[2]) // 2), int((self.target_box[1] + self.target_box[3]) // 2))
        self.updata_trace_list(50)
        # self.updata_v_list(25)


    def updata_trace_list(self, max_trace_number=50):
        if len(self.trace_point_list) <= max_trace_number:
            self.trace_point_list.append(self.box_center)
        else:
            self.trace_point_list.pop(0)
            self.trace_point_list.append(self.box_center)

    def updata_v_list(self, max_v_number=5):
        if len(self.trace_v_list) <= max_v_number:
            if hasattr(self, 'dx'):
                self.trace_v_list.append([self.dx, self.dy])
        else:
            self.trace_v_list.pop(0)
            self.trace_v_list.append([self.dx, self.dy])

    def draw(self, img):
        draw_trace(img, self.trace_point_list)
        if self.max_iou_matched:
            cv2.putText(img, f"Tracking  ID={self.track_id}", (int(self.target_box[0]), int(self.target_box[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 0, 0), 2)

            # draw_trace(img, self.trace_point_list)
        else:
            pass
            cv2.putText(img, "Lost", (int(self.target_box[0]), int(self.target_box[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 0), 2)
        plot_one_box(self.target_box, img, color=(255, 255, 255), target=self.max_iou_matched)
