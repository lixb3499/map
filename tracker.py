import os
import cv2
import numpy as np
from utils import plot_one_box, cal_iou, xyxy_to_xywh, xywh_to_xyxy, updata_trace_list, draw_trace, vector_norm, \
    vector_norm_ax1
import datetime
from kalmanfilter import KalmanFilter


class Tracks:
    def __init__(self, initial_state, initial_cov=np.eye(6), track_id=0, frame_rate=6):
        # 默认初始化状态为[x ,y, w, h, 0, 0]
        # initial_state =
        self.h = 80  # 用中心点初始化时设置默认边框大小
        if len(initial_state) == 2:
            self.X = np.append(initial_state, [self.h, self.h, 0, 0])
        self.frame_rate = frame_rate
        self.X = np.append(initial_state, [0, 0])
        self.P = initial_cov
        self.IOU_Threshold = 0.1  # 匹配时的阈值
        self.target_box = xywh_to_xyxy(self.X[0:4])  # 检测到匹配上的目标的框
        self.target_xywh = self.X[0:4]
        self.box_center = (
            int((self.target_box[0] + self.target_box[2]) // 2), int((self.target_box[1] + self.target_box[3]) // 2))
        self.KF = KalmanFilter()
        self.trace_point_list = []  # 储存轨迹的历史点信息用于画轨迹
        self.trace_v_list = []
        self.trace_v_list_value = []  # 存放速度的大小，不是向量
        self.v_average = 0  # 这里计算的是trace_v_list里面的平均值
        self.max_trace_number = 50
        self.max_iou_matched = False
        self.Z = np.array(self.X)  # 初始化观测矩阵为X
        self.track_id = track_id
        self.lost_number = 0
        self.number_since_match = 0
        self.confirmflag = False  # 轨迹需要检测到三帧以上才能变为确定的一条轨迹
        self.parking_id = None  # 该轨迹在地图中对应的车位
        self.v_Threshold = 2
        self.stoptime = 0  # 滞留时间，判断为stop的帧数除以帧率

    # def predict(self):
    #     self.X, self.P = self.KF.predict(self.X, self.P)

    def iou_match(self, detect):
        self.max_iou_matched = False
        max_iou = self.IOU_Threshold
        self.target_box = xywh_to_xyxy(self.X[0:4])  # predict后更新一下目标框
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

    def ifstop(self, v_Threshold):
        if 21.6 * self.v_average < v_Threshold:
            return True
        else:
            return False

    def update(self):
        if self.max_iou_matched:
            # self.detect_xywh = xyxy_to_xywh(self.detect_xyxy)
            self.target_xywh = xyxy_to_xywh(self.target_box)
            dx = self.target_xywh[0] - self.X[0]
            dy = self.target_xywh[1] - self.X[1]

            self.Z[0:4] = np.array(self.target_xywh).T
            self.Z[4::] = np.array([dx, dy])
            self.X, self.P = self.KF.predict(self.X, self.P)
            self.X, self.P = self.KF.update(self.X, self.P, self.Z)

            self.X[2: 4] = [self.h, self.h]
            self.dx = self.X[4]
            self.dy = self.X[5]
            self.number_since_match += 1
            self.lost_number = 0
        else:
            if len(self.trace_v_list) > 0:  # 第一次检测到的时候也是匹配不到的
                self.X[-2::] = self.trace_v_list[0]
            self.X, self.P = self.KF.predict(self.X, self.P)
            # self.X[2: 4] = [120, 120]

            self.number_since_match = 0
            self.lost_number += 1
        self.xywh = self.X[0:4]
        self.target_box = xywh_to_xyxy(self.X[0:4])
        self.box_center = (
            int((self.target_box[0] + self.target_box[2]) // 2), int((self.target_box[1] + self.target_box[3]) // 2))
        self.update_trace_list(50)
        self.update_v_list(2)
        self.v_average = np.mean(self.trace_v_list_value)
        self.updatestoptime(self.frame_rate)

    def update_trace_list(self, max_trace_number=50):
        if len(self.trace_point_list) <= max_trace_number:
            self.trace_point_list.append(self.box_center)
        else:
            self.trace_point_list.pop(0)
            self.trace_point_list.append(self.box_center)

    def update_v_list(self, max_v_number=5):
        if len(self.trace_v_list) <= max_v_number:
            if hasattr(self, 'dx'):
                # 横坐标轴上1单位长度对应43像素点
                # 纵坐标轴上1单位长度对应35像素点
                # 这里要改
                self.trace_v_list.append([self.dx / 43, self.dy / 35])
                self.trace_v_list_value.append(vector_norm([self.dx / 43, self.dy / 35]))
        else:
            self.trace_v_list.pop(0)
            self.trace_v_list_value.pop(0)
            self.trace_v_list.append([self.dx / 43, self.dy / 35])
            self.trace_v_list_value.append(vector_norm([self.dx / 43, self.dy / 35]))

    def updatestoptime(self, frame_rate):
        self.stoptime = self.stoptime + 1 / self.frame_rate

    def draw(self, img):
        draw_trace(img, self.trace_point_list)
        if self.ifstop(self.v_Threshold):
            if self.max_iou_matched:
                # cv2.putText(img, f"Tracking  ID={self.track_id}, V={21.6*vector_norm(self.trace_v_list[
                # -1]):.2f}km/h", (int(self.target_box[0]-30), int(self.target_box[1] - 5)),
                # cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(img, f"Tracking  ID={self.track_id}, stop, t = {self.stoptime:.2f}s",
                            (int(self.target_box[0] - 30), int(self.target_box[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 0, 0), 2)

                # draw_trace(img, self.trace_point_list)
            else:
                # pass
                cv2.putText(img, f"Lost ID={self.track_id}, stop, t = {self.stoptime:.2f}s",
                            (int(self.target_box[0] - 30), int(self.target_box[1] - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 0), 2)
        else:
            if self.max_iou_matched:
                # cv2.putText(img, f"Tracking  ID={self.track_id}, V={21.6*vector_norm(self.trace_v_list[-1]):.2f}km/h", (int(self.target_box[0]-30), int(self.target_box[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                #             0.7, (255, 0, 0), 2)
                cv2.putText(img, f"Tracking  ID={self.track_id}, V={3.6 * self.frame_rate * self.v_average:.2f}km/h",
                            (int(self.target_box[0] - 30), int(self.target_box[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 0, 0), 2)

                # draw_trace(img, self.trace_point_list)
            else:
                # pass
                cv2.putText(img,
                            f"Lost ID={self.track_id}, V={3.6 * self.frame_rate * vector_norm(self.trace_v_list[-1]):.2f}km/h",
                            (int(self.target_box[0] - 30), int(self.target_box[1] - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 0), 2)
        plot_one_box(self.target_box, img, color=(255, 0, 255), target=self.max_iou_matched)
