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

