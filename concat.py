# import cv2
# import numpy as np
#
# video_file1 = "concat/point_center_pic.avi"
# video_file2 = "concat/exp2023-12-05_00-15-35.mp4"
# ccap = cv2.VideoCapture(video_file1)
# tcap = cv2.VideoCapture(video_file2)
# width = 2560 + 2560
# height = 1920
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('concat/out.avi', fourcc, 10, (int(width), int(height)))
#
# while ccap.isOpened():
#     ret, img = ccap.read()
#     tret, timg = tcap.read()
#     if ret == True and tret == True:
#         # timg = cv2.copyMakeBorder(timg, 300, 300, 0, 0, cv2.BORDER_CONSTANT, value=0)
#         vis = np.concatenate((img, timg), axis=1)
#         print('write video')
#         out.write(vis)
#     else:
#         break
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

import cv2
import numpy as np

video_file1 = "concat/example_7_校正角度.avi"
video_file2 = "concat/example_7_校正角度.mp4"

ccap = cv2.VideoCapture(video_file1)
tcap = cv2.VideoCapture(video_file2)

# 获取视频1的大小
width1 = int(ccap.get(cv2.CAP_PROP_FRAME_WIDTH))
height1 = int(ccap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 获取视频2的大小
width2 = int(tcap.get(cv2.CAP_PROP_FRAME_WIDTH))
height2 = int(tcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 确保视频2与视频1的高度相同，如果不同，进行填充
if height2 != height1:
    padding_height = height1 - height2
    padding_left = padding_right = padding_height // 2
else:
    padding_left = padding_right = 0

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('concat/example_7_校正角度out.avi', fourcc, 6, (width1 + width2, height1))

frame_count = 0
while ccap.isOpened():
    ret, img = ccap.read()
    tret, timg = tcap.read()

    if ret and tret:
        # 如果需要，调整 timg 的大小
        timg = cv2.copyMakeBorder(timg, padding_left, padding_right, 0, 0, cv2.BORDER_CONSTANT, value=0)

        # 水平拼接
        vis = np.concatenate((img, timg), axis=1)

        print(f'写入视频 第{frame_count}帧')
        out.write(vis)
    else:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    frame_count = frame_count+1
    # cv2.imshow('track', vis)

# 释放资源
ccap.release()
tcap.release()
out.release()
cv2.destroyAllWindows()
