import numpy as np

# 常用变换矩阵

# 缩放矩阵
def scale(x, y, z):
    a = np.eye(4, dtype=np.float32)
    a[0, 0] = x
    a[1, 1] = y
    a[2, 2] = z
    return a

# 旋转矩阵
def rotate(r, axis: tuple):
    a = np.eye(4, dtype = np.float32)
    a[axis[0], axis[0]] = np.cos(r)
    a[axis[0], axis[1]] = np.sin(r)
    a[axis[1], axis[0]] = - np.sin(r)
    a[axis[1], axis[1]] = np.cos(r)
    return a

# 平移矩阵
def translate(x, y, z):
    a = np.eye(4, dtype = np.float32)
    a[3, 0] = x
    a[3, 1] = y
    a[3, 2] = z
    return a

# 透视矩阵
def perspective():
    a = np.eye(4, dtype = np.float32)
    a[2, 2] = 1 / 1000
    a[3, 2] = -0.0001
    a[2, 3] = 1
    a[3, 3] = 0
    return a

# 逆透视矩阵
def inperspective():
    a = np.eye(4, dtype = np.float32)
    a[2, 2] = 0
    a[3, 2] = 1
    a[2, 3] = -10000
    a[3, 3] = 10
    return a

# 平移后旋转后再反向平移
# 此函数用于绕指定轴旋转
def tran_and_rot(dx, dy, rot):
    view = perspective() @ translate(-dx, -dy, 0) @ inperspective()
    view = view @ translate(0, 0, -0.3)
    view = view @ rotate(rot[0], axis=(0, 2)) @ rotate(rot[1], axis=(2, 1)) @ rotate(rot[2], axis=(0, 1))
    view = view @ translate(0,0,0.3)
    view = view @ perspective() @ translate(dx, dy, 0) @ inperspective()
    return view

# 透视变换矩阵
def get_perspective_transform_matrix(src_points, dst_points):
    A = np.zeros((8, 8))
    b = np.zeros((8, 1))

    for i in range(4):
        A[i * 2] = [src_points[i][0], src_points[i][1], 1, 0, 0, 0, -src_points[i][0] * dst_points[i][0], -src_points[i][1] * dst_points[i][0]]
        A[i * 2 + 1] = [0, 0, 0, src_points[i][0], src_points[i][1], 1, -src_points[i][0] * dst_points[i][1], -src_points[i][1] * dst_points[i][1]]
        b[i * 2] = dst_points[i][0]
        b[i * 2 + 1] = dst_points[i][1]

    h = np.linalg.solve(A, b)
    h = np.append(h, [1])
    transform_matrix = h.reshape((3, 3))
    return transform_matrix

# Bezier曲线生成
def bezier_curve(control_points, n_points=100):
    n = len(control_points) - 1
    t = np.linspace(0, 1, n_points)
    curve = np.zeros((n_points, 2))
    # 预先计算并存储所有需要的二项式系数
    binomials = [np.math.factorial(n) // (np.math.factorial(i) * np.math.factorial(n - i)) for i in range(n + 1)]
    for i in range(n_points):
        for j in range(n + 1):
            curve[i] += binomials[j] * (1 - t[i]) ** (n - j) * t[i] ** j * control_points[j]
    return curve

