import os
import cv2
import numpy as np
from utils import plot_one_box, cal_iou, xyxy_to_xywh, xywh_to_xyxy, updata_trace_list, draw_trace
import datetime
from tracker import Tracks
from kalmanfilter import KalmanFilter
from scipy.optimize import linear_sum_assignment


def iou_match(mat):
    tracks_indices, det_indices = linear_sum_assignment(-1 * mat)
    return tracks_indices, det_indices


def content2detections(content):
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
        if len(detect_xywh) == 2:
            detect_xywh = np.append(detect_xywh, [80, 80])
        if len(detect_xywh) == 3:
            detect_xywh = detect_xywh.pop()

        detect_xyxy = xywh_to_xyxy(detect_xywh)
        detections.append(detect_xywh)
    return detections


class Tracker:
    def __init__(self, content):
        """
        初始化时需要读入第一帧文件的信息。[X1, X2, ...],X1.shape = (6,)
        """
        self.tracks = []
        detections = content2detections(content)
        i = -1  # 这里先定义i的目的是防止第一帧里面没有目标导致下面的next_id出错
        for i, detection in enumerate(detections):
            self.tracks.append(Tracks(detection, track_id=i))
        for track in self.tracks:  # 第一次检测到的目标直接设置为确定态
            track.confirmflag = True
        self.next_id = i + 1
        self.max_lost_number = 10
        self.KF = KalmanFilter()
        self.confirm_frame = 3  # 设为确定态所需要连续匹配到的帧数

    def iou_mat(self, content):
        """
        计算IOU矩阵
        :param content: 读入文本信息
        :return:mat,(tracks, detections)
        """
        detections = content2detections(content)
        mat = np.zeros((len(self.tracks), len(detections)))
        for i, track in enumerate(self.tracks):
            for j, detection in enumerate(detections):
                detection = xywh_to_xyxy(detection)
                mat[i][j] = cal_iou(track.target_box, detection)
        # mat[mat <= track.IOU_Threshold] = 0
        return mat

    def update(self, content):
        mat = self.iou_mat(content)
        detections = content2detections(content)
        if np.all(mat == 0):
            track_indices, det_indices = ([], [])
        else:
            track_indices, det_indices = iou_match(mat)
        # track_indices, det_indices = self.iou_match(mat)
        for i, track in enumerate(self.tracks):  # 给匹配上的轨迹改变标记
            if i not in track_indices:
                track.max_iou_matched = False
                track.number_since_match = 0
            else:
                track.max_iou_matched = True
        for track_indice, det_indice in zip(track_indices, det_indices):  # 再验证是否满足IOU匹配
            self.tracks[track_indice].iou_match(detections[det_indice])
        for j, track in enumerate(self.tracks):  # 对每条轨迹进行更新
            track.update()  # 卡尔曼状态更新
            if track.lost_number > self.max_lost_number:  # 超过一段时间没有匹配上则直接删除
                self.tracks.remove(track)
            if (not track.confirmflag) and (track.number_since_match > self.confirm_frame):
                track.confirmflag = True
                # self.next_id +=1  #注释掉这行应该解决了给不同目标分配同一个id的问题
        for k, detections in enumerate(detections):  # 没有匹配上轨迹的检测目标创建一个新的轨迹
            if k not in det_indices:
                self.tracks.append(Tracks(detections, track_id=self.next_id))
                self.next_id += 1

    def draw_tracks(self, img):
        for track in self.tracks:
            if track.confirmflag:
                track.draw(img)
