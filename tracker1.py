import os
import cv2
import numpy as np
from utils import plot_one_box, cal_iou, xyxy_to_xywh, xywh_to_xyxy, updata_trace_list, draw_trace
import datetime
from tracker import Tracks
from kalmanfilter import KalmanFilter
from scipy.optimize import linear_sum_assignment

class Tracker:
    def __init__(self, detections):
        """
        初始化时需要读入第一帧文件的信息。[X1, X2, ...],X1.shape = (6,)
        """
        self.tracks = []
        detections = self.content2detections(detections)
        for i, detection in enumerate(detections):
            self.tracks.append(Tracks(detection, track_id=i))
        for track in self.tracks:   # 第一次检测到的目标直接设置为确定态
            track.confirmflag = True
        self.next_id = i+1
        self.max_lost_number = 100
        self.KF = KalmanFilter()

    def content2detections(self, content):
        """
        从读取的文件中解析出检测到的目标信息
        :param content: readline返回的列表
        :return: [X1, X2, ...],X1.shape = (6,)
        """
        detections = []
        for i, detection in enumerate(content):
            data = detection.replace('\n', "").split(" ")
            detect_xywh = np.array(data[1:5], dtype="float")
            detect_xyxy = xywh_to_xyxy(detect_xywh)
            detections.append(detect_xywh)
        return detections

    def iou_mat(self, content):
        """
        计算IOU矩阵
        :param content: 读入文本信息
        :return:mat,(tracks, detections)
        """
        detections = self.content2detections(content)
        mat = np.zeros((len(self.tracks), len(detections)))
        for i, track in enumerate(self.tracks):
            for j, detection in enumerate(detections):
                detection = xywh_to_xyxy(detection)
                mat[i][j] = cal_iou(track.target_box, detection)
        # mat[mat <= track.IOU_Threshold] = 0
        return mat

    def iou_match(self, mat):
        tracks_indices, det_indices = linear_sum_assignment(-1*mat)
        return tracks_indices, det_indices




    def update(self, content):
        mat = self.iou_mat(content)
        detections = self.content2detections(content)
        track_indices, det_indices = self.iou_match(mat)
        for i, track in enumerate(self.tracks):     #   给匹配上的轨迹改变标记
            if i not in track_indices:
                track.max_iou_matched = False
                track.number_since_match = 0
            else:
                track.max_iou_matched = True
        for track_indice, det_indice in zip(track_indices, det_indices):    #再验证是否满足IOU匹配
            self.tracks[track_indice].iou_match(detections[det_indice])
        for j, track in enumerate(self.tracks): #对每条轨迹进行更新
            track.update()  # 卡尔曼状态更新
            if track.lost_number>self.max_lost_number:  #超过一段时间没有匹配上则直接删除
                self.tracks.remove(track)
            if (not track.confirmflag) and (track.number_since_match>5):
                track.confirmflag = True
                self.next_id +=1
        for k, detections in enumerate(detections): #没有匹配上轨迹的检测目标创建一个新的轨迹
            if k not in det_indices:
                self.tracks.append(Tracks(detections, track_id=self.next_id))

    def draw_tracks(self, img):
        for track in self.tracks:
            if track.confirmflag:
                track.draw(img)



