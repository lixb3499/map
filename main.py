import os
import cv2
import numpy as np
from utils import plot_one_box, cal_iou, xyxy_to_xywh, xywh_to_xyxy, updata_trace_list, draw_trace
import datetime
from tracker import Tracks

initial_stata = np.array([104.5, 546, 207, 232, 0, 0])
# video_path = "./data/testvideo1.mp4"
# label_path = "./data/labels"
# file_name = "testvideo1"
video_path = "exp-2023-08-20_18-27-05/connect.mp4"
label_path = "exp-2023-08-20_18-27-05/labels"
file_name = "connect"
cap = cv2.VideoCapture(video_path)
frame_number = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print(f"Video FPS: {frame_number}")
SAVE_VIDEO = True  # True
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
if not os.path.exists('video_out'):
    os.mkdir('video_out')
if SAVE_VIDEO:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f'video_out/{current_time}.mp4', fourcc, 20, (1920, 1080))

frame_counter = 1
track = Tracks(initial_stata)
while (True):
    if frame_counter > frame_number:
        break
    # Capture frame-by-frame
    ret, frame = cap.read()

    # plot_one_box(track.target_box, frame, color=(255, 255, 255), target=False)
    if not ret:
        break
    label_file_path = os.path.join(label_path, file_name + "_" + str(frame_counter) + ".txt")
    if not os.path.exists(label_file_path):
        with open(label_file_path, "w") as f:
            pass

    with open(label_file_path, "r") as f:
        content = f.readlines()
        # track.predict()
        track.iou_match(content)
        track.update()
        track.draw(frame)
    cv2.putText(frame, "ALL BOXES(Green)", (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
    cv2.putText(frame, "TRACKED BOX(Red)", (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, "Last frame best estimation(White)", (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),
                2)
    if track.max_iou_matched:
        cv2.putText(frame, "Tracking", (int(track.target_box[0]), int(track.target_box[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 0, 0), 2)
    else:
        cv2.putText(frame, "Lost", (int(track.target_box[0]), int(track.target_box[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 0), 2)
    plot_one_box(track.target_box, frame, color=(255, 255, 255), target=track.max_iou_matched)
    cv2.imshow('track', frame)
    if SAVE_VIDEO:
        out.write(frame)
    frame_counter = frame_counter + 1
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
