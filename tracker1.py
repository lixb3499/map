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
    从文件读取的内容中提取检测到的目标信息。

    :param content: readline 返回的列表。
    :return: 包含每个检测对象信息的字典列表。
             字典键：'coordinate'、'id'、'licence'、'licence_cls'。
    """
    detections_list = []

    for i, detection in enumerate(content):
        data = detection.replace('\n', "").split(" ")
        detect_xywh = np.array(data[0:2], dtype="float")

        # 如果只提供 x 和 y 坐标，假设默认宽度和高度为 (80, 80)。
        if len(detect_xywh) == 2:
            detect_xywh = np.append(detect_xywh, [80, 80])
        # 如果提供 x、y 和cls，假设默认高度为 80。，这行现在实际没用了
        elif len(detect_xywh) == 3:
            detect_xywh = detect_xywh.pop()
            detect_xywh = np.append(detect_xywh, [80])

        detection_dict = {
            'coordinate': detect_xywh,
            'id': int(data[3]),
            'licence': data[4],
            'licence_cls': data[5]
        }

        detections_list.append(detection_dict)

    return detections_list


class Tracker:
    def __init__(self, content, frame_rate=6):
        """
        初始化时需要读入第一帧文件的信息
        """
        self.tracks = []
        detections_list = content2detections(content)
        i = -1  # 这里先定义i的目的是防止第一帧里面没有目标导致下面的next_id出错
        for i, detection in enumerate(detections_list):
            self.tracks.append(Tracks(detection['coordinate'], track_id=detection['id'], frame_rate=frame_rate, licence=detection['licence'], licence_cls=detection['licence_cls']))
        for track in self.tracks:  # 第一次检测到的目标直接设置为确定态
            track.confirmflag = True
        self.next_id = i + 1
        self.max_lost_number = 10
        self.KF = KalmanFilter()
        self.frame_rate = frame_rate
        self.confirm_frame = 3  # 设为确定态所需要连续匹配到的帧数

    def iou_mat(self, content):
        """
        计算IOU矩阵
        :param content: 读入文本信息
        :return:mat：tracks与detections之间的iou矩阵，(tracks, detections)
        """
        detections = content2detections(content)
        mat = np.zeros((len(self.tracks), len(detections)))
        for i, track in enumerate(self.tracks):
            for j, detection in enumerate(detections):
                detection = xywh_to_xyxy(detection)
                mat[i][j] = cal_iou(track.target_box, detection)
        # mat[mat <= track.IOU_Threshold] = 0
        return mat

    def update1(self, content):
        """
        更新轨迹状态和创建新轨迹的方法。这个函数用于没有yolov8的id（也就是自己分配id的时候）用的更新函数

        :param content: 读入的文本信息
        """
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
                self.tracks.append(Tracks(detections, track_id=self.next_id, frame_rate=self.frame_rate))
                self.next_id += 1

    def update(self, content):
        """
        更新轨迹状态和创建新轨迹的方法。当使用yolov8的id（即不用自己做跟踪的时候）用的更新函数

        :param content: 读入的文本信息
        """
        detection_list = content2detections(content)
        track_detection_id_list = [detection['id'] for detection in detection_list]  # 检测到的车的id
        tracker_id_list = [track.track_id for track in self.tracks]
        for track in self.tracks:
            if track.track_id not in track_detection_id_list:
                track.id_matched = False
                track.update()
            for detection in detection_list:
                if track.track_id == detection['id']:
                    track.id_matched = True
                    track.update(detection['coordinate'])
            if track.lost_number > self.max_lost_number:  # 超过一段时间没有匹配上则直接删除
                self.tracks.remove(track)
            if (not track.confirmflag) and (track.number_since_match > self.confirm_frame):
                track.confirmflag = True
        for detection in detection_list:
            if detection['id'] not in tracker_id_list:
                self.tracks.append(
                    Tracks(detection['coordinate'], track_id=detection['id'], frame_rate=self.frame_rate, licence=detection['licence'], licence_cls=detection['licence_cls']))

    def draw_tracks(self, img):
        for track in self.tracks:
            if track.confirmflag:
                track.draw(img)
