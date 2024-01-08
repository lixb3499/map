### 地图上车辆跟踪与停车区域监测代码详解

#### 概述

该代码实现了一个在地图上进行车辆跟踪以及监测车辆是否停入停车区域的功能。主要包含 `Map` 类、`Border` 类和 `Area` 类。

#### 类说明

1. **`Map` 类**
   - 继承自 `Tracker` 类，包含了车辆跟踪的功能，并额外增加了停车区域的监测。
   - 初始化时接收目标检测信息 `content`，并初始化停车区域列表。
   - `areas` 列表存储了各个停车区域的信息，每个区域由 `Area` 类表示。
   - `intersect(self, point, previous_point)`: 判断车辆轨迹是否与停车区域相交，更新区域状态。
   - `draw_area(self, frame)`: 在图像上绘制停车区域。
2. **`Border` 类**
   - 表示停车区域的边界线，由两个点定义。
   - `intersect(self, point, previous_point)`: 判断车辆轨迹是否与边界相交，返回相交的边界。
   - `draw_border(self, frame)`: 在图像上绘制边界线。
3. **`Area` 类**
   - 表示一个停车区域，包含一个矩形区域和边界线列表。
   - `update(self)`: 更新区域状态。
   - `isenter(self, point, previous_point)`: 判断点是否在区域内。
   - `intersect(self, point, previous_point)`: 判断车辆轨迹是否与边界相交，更新区域状态。
   - `draw(self, frame)`: 在图像上绘制停车区域的边界线。

#### 主要功能流程

1. **初始化地图和区域**
   - 创建 `Map` 对象，继承了车辆跟踪器，用于初始化车辆跟踪器和停车区域。
   - 通过 `area_inf_list` 参数初始化停车区域列表。
2. **车辆跟踪与停车区域监测**
   - 在每一帧图像上调用 `Map` 对象的 `update` 函数，更新车辆的轨迹信息。
   - 遍历停车区域列表，判断车辆轨迹是否与停车区域相交，更新区域内车辆数量。
   - 绘制车辆轨迹和停车区域。

#### 类间关系

- `Map` 类继承了 `Tracker` 类，扩展了车辆跟踪的功能，并添加了停车区域监测。
- `Area` 类包含 `Border` 类的实例，每个 `Area` 对象由多个 `Border` 边界线组成。