### 地图上车辆跟踪与停车区域监测系统主程序说明文档

#### 概述

这是一个地图上进行车辆跟踪与停车区域监测的主程序，通过读取目标检测的标签信息，使用车辆跟踪器追踪车辆轨迹，并在地图上标记停车区域并监测车辆是否停入。

#### 主要函数和流程

1. **`content2detections(content, ax)` 函数**
   - 输入：读取的标签文件的内容列表 `content` 和 Matplotlib 坐标轴 `ax`。
   - 输出：解析出的检测到的目标信息列表 `detections`，每个元素是一个矩形框的坐标。
   - 功能：将标签文件中的每行数据解析为检测到的目标的坐标信息，转换为 Matplotlib 坐标系中的坐标。
2. **`plot_box_map(ax, box_coords)` 函数**
   - 输入：Matplotlib 坐标轴 `ax` 和矩形框的坐标 `box_coords`。
   - 功能：在指定的 Matplotlib 坐标轴上画出矩形框。
3. **`main(args)` 函数**
   - 输入：命令行参数 `args`。
   - 功能：
     - 初始化视频输出相关参数，如视频文件名、帧率等。
     - 创建 Matplotlib 画布和坐标轴，绘制停车区域的边界框。
     - 读取第一帧的标签文件，初始化车辆跟踪器 `tracker` 和地图 `Map`。
     - 循环读取每一帧的标签文件，更新车辆跟踪器和地图，并在地图上标记车辆轨迹和停车区域。
     - 实时显示图像，如果指定了保存视频，将图像写入视频文件。
   - 输出：实时显示视频窗口，保存视频文件（可选）。
4. **`parse_args()` 函数**
   - 功能：解析命令行参数。
   - 输出：命令行参数的命名空间。

#### 主程序流程

```python
def plot_box_map(ax, box_coords):
    """
    在指定的画布上画出矩形框。

    Parameters:
        ax (matplotlib.axes._axes.Axes): Matplotlib的坐标轴对象
        box_coords (tuple): 矩形框的坐标，形式为 (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = box_coords

    # 绘制矩形框
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='g', facecolor='none')

    # 将矩形框添加到坐标轴
    ax.add_patch(rect)
```

这个函数 (`plot_box_map`) 用于在指定的画布上绘制矩形框。接受 `matplotlib` 的坐标轴对象和矩形框的坐标信息，使用 `patches` 模块创建矩形框并添加到坐标轴。

```python
def main(args):
    label_path = args.label_path
    file_name = args.file_name
    save_txt = args.save_txt

    # 设置视频相关参数
    SAVE_VIDEO = args.save_video
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if not os.path.exists('video_out'):
        os.mkdir('video_out')
    video_filename = os.path.join('video_out', 'exp' + current_time + '.mp4')
    frame_rate = args.frame_rate
    filelist = os.listdir(label_path)
    frame_number = len(filelist)
```

`main` 函数是整个程序的入口。它从命令行参数中获取文件路径、文件名等信息，并设置视频保存相关的参数。然后，获取文件列表和帧数等信息。

```python
    # 创建Matplotlib画布和坐标轴
    fig, ax = plt.subplots(figsize=args.fig_size)

    ax.set_xlim(args.x_lim)
    ax.set_ylim(args.y_lim)
    ax.invert_yaxis()
```

在这里，创建了 `Matplotlib` 画布和坐标轴，设置了坐标轴的范围。

```python
    p1 = [-11 + 14.34, -2.4, -8.6 + 14.34, -7.4]
    p2 = [-8.6 + 14.34, -2.4, -6.2 + 14.34, -7.4]
    # ... (省略其他矩形框的坐标)
    P = [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12]
    for p in P:
        plot_box_map(ax, p)
```

这部分代码定义了一些矩形框的坐标，并使用之前定义的 `plot_box_map` 函数在坐标轴上画出这些矩形框。

```python
    area1 = [coord_to_pixel(ax, (-2.4 + 14.34, 2.4)), coord_to_pixel(ax, (4.8 + 14.34, -2.4))]  
    area2 = [coord_to_pixel(ax, (3, 7)), coord_to_pixel(ax, (11, 1.2))]
    areas_list = args.areas
    area_pixels = []
    for area in areas_list:
        area_pixels.append([coord_to_pixel(ax, (area[i], area[i + 1])) for i in range(0, len(area), 2)])
```

定义了一些区域的坐标，并将这些坐标通过 `coord_to_pixel` 函数转换为像素坐标。

```python
    with open(os.path.join(label_path, file_name + str(0) + ".txt"), 'r', encoding='utf-8') as f:
        content = f.readlines()
        tracker = Map(content, area_pixels, frame_rate)
```

从文件读取第一帧的目标信息，然后初始化地图类 (`Map`) 对象 `tracker`。

```python
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    canvas = FigureCanvas(fig)

    canvas.draw()
    img_array = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(canvas.get_width_height()[::-1] + (3,))

    frame = img_array

    # 设置视频分辨率和大小
    # width, height = 640, 480
    # video_size = (width, height)
```

获取坐标轴的范围，创建 `Matplotlib` 画布的 `FigureCanvas` 对象，并绘制图像。

```python
    # 设置视频编解码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    if SAVE_VIDEO:
        # 创建视频写入对象
        video_writer = cv2.VideoWriter(video_filename, fourcc, frame_rate,
                                       (100 * args.fig_size[0], 100 * args.fig_size[1]))
```

这部分代码设置视频编解码器和创建视频写入对象。

```python
    frame_counter = 1
    count1 = 0
    for frame_counter in range(1, frame_number):
        img_array = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(canvas.get_width_height()[::-1] + (3,))
        frame = img_array

        print(f"当前帧数：{frame_counter}")
        if frame_counter > frame_number:
            break

        label_file_path = os.path.join(label_path, file_name + str(frame_counter) + ".txt")
        if not os.path.exists(label_file_path):
            with open(label_file_path, "w") as f:
                pass
        with open(label_file_path, "r", encoding='utf-8') as f:
            content = f.readlines()
            tracker.update(content)
        tracker.draw_tracks(frame)

        for track in tracker.tracks:
            if len(track.trace_point_list) < 2:
                break
            point = track.trace_point_list[-1]
            previous_point = track.trace_point_list[-2]
            tracker.intersect(point, previous_point)
        tracker.draw_area(frame)

        cv2.putText(frame, "ALL BOXES(Green)", (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
        cv2.putText(frame, "TRACKED BOX(Red)", (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        for area in tracker.areas:
            cv2.putText(frame, f"count of area_{area.id}:    {area.count_car}", (25, 100 + 25 * area.id),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255),
                        2)

        cv2.imshow('track', frame)
        if SAVE_VIDEO:
            video_writer.write(frame)
        frame_counter = frame_counter + 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        plt.close()
    if SAVE_VIDEO:
        video_writer.release()
```

这个循环迭代每一帧的处理。首先，获取文件路径，读取文件内容并更新地图跟踪器。接着，绘制目标轨迹和区域信息，并在图像上添加文字标签。最后，通过 OpenCV 显示图像，如果指定了保存视频，则将帧写入视频文件。

```python
def parse_args():
    parser = argparse.ArgumentParser()

    # 添加参数
    parser.add_argument('--label_path', type=str, default='example_7_demo_0105/saved_txt',
                        help='Path to the label directory')
    parser.add_argument('--file_name', type=str, default='', help='Specify a file name')
    parser.add_argument('--save_txt', type=str, default='save_txt', help='Specify the save_txt directory')
    parser.add_argument('--save_video', action='store_true', help='Flag to save video', default=False)
    parser.add_argument('--frame_rate', type=int, default=6, help='Frame rate for video')
    parser.add_argument('--fig_size', nargs=2, type=float, default=[14, 9], help='Figure size as width height')
    parser.add_argument('--x_lim', nargs=2, type=float, default=[0, 25], help='X-axis limits')
    parser.add_argument('--y_lim', nargs=2, type=float, default=[-10, 10], help='Y-axis limits')

    parser.add_argument('--areas', nargs='+', type=float, default=[[11.94, -2.4, 19.14, 2.4], [3, -7, 11, -1.2]],
                        help='Define areas as left-top and right-bottom coordinates in the format x1 y1 x2 y2, '
                             'x1 y1 x2 y2, ...')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
```

最后，定义了命令行参数的解析函数 `parse_args` 和程序的主入口 `__main__`。 `parse_args` 函数使用 `argparse` 模块解析命令行参数，而 `__main__` 函数调用 `main` 函数，传递解析得到的参数并执行整个程序的流程。

#### 类间关系

- 主程序中通过调用 `Map` 类的方法实现车辆轨迹的跟踪和停车区域的监测。
- `Map` 类继承自 `Tracker` 类，扩展了车辆跟踪的功能，并添加了停车区域监测。

#### 使用示例

```bash
python main.py --label_path example_7_demo_0105/saved_txt --file_name video_label --save_txt save_txt --save_video --frame_rate 6 --fig_size 14 9 --x_lim 0 25 --y_lim -10 10 --areas 11.94 -2.4 19.14 2.4 3 -7 11 -1.2
```

