def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

# 定义测试用例
A = (0, 0)
B = (2, 2)
C = (0, 2)
D = (2, 0)

# 调用intersect函数进行测试
result = intersect(A, B, C, D)

# 输出结果
print(result)


# 定义测试用例
A = (0, 0)
B = (2, 2)
C = (3, 3)
D = (4, 4)

# 调用intersect函数进行测试
result = intersect(A, B, C, D)

# 输出结果
print(result)

def is_point_inside_rectangle(rectangle_left_top, rectangle_right_bottom, point):
    """
    判断点是否在矩形框内部。

    参数:
    - rectangle_left_top: 矩形框左上角坐标 (x1, y1)
    - rectangle_right_bottom: 矩形框右下角坐标 (x2, y2)
    - point: 要判断的点坐标 (x, y)

    返回值:
    - True 如果点在矩形框内，否则返回 False
    """
    x1, y1 = rectangle_left_top
    x2, y2 = rectangle_right_bottom
    x, y = point
    return x1 <= x <= x2 and y1 <= y <= y2

# 示例
rectangle_left_top = (0, 0)
rectangle_right_bottom = (5, 5)

point_inside = (3, 3)
point_outside = (6, 6)

# 测试点是否在矩形框内
print(is_point_inside_rectangle(rectangle_left_top, rectangle_right_bottom, point_inside))  # 输出: True
print(is_point_inside_rectangle(rectangle_left_top, rectangle_right_bottom, point_outside))  # 输出: False

for frame_counter in range(1, 11):  # 假设有10帧
    if frame_counter % 2 == 0:
        # 如果 frame_counter 是偶数，跳过当前循环
        continue

    # 这里是只在 frame_counter 是奇数时执行的代码
    print(f"处理奇数帧: {frame_counter}")
