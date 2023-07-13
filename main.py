from itertools import starmap
import logging
import math
from os import rename
from re import I, M
import threading
import numpy as np
import glfw
import OpenGL
from OpenGL.GL import *
from OpenGL.GLU import *
import yaml
import time

from PIL import Image

import pdb

import win32api
import cv2
# ---------- 矩阵定义 ----------
def scale(x, y, z):
    a = np.eye(4, dtype=np.float32)
    a[0, 0] = x
    a[1, 1] = y
    a[2, 2] = z
    return a

def rotate(r, axis: tuple):
    a = np.eye(4, dtype = np.float32)
    a[axis[0], axis[0]] = np.cos(r)
    a[axis[0], axis[1]] = np.sin(r)
    a[axis[1], axis[0]] = - np.sin(r)
    a[axis[1], axis[1]] = np.cos(r)
    return a

def translate(x, y, z):
    a = np.eye(4, dtype = np.float32)
    a[3, 0] = x
    a[3, 1] = y
    a[3, 2] = z
    return a

def perspective():
    a = np.eye(4, dtype = np.float32)
    a[2, 2] = 1 / 1000
    a[3, 2] = -0.0001
    a[2, 3] = 1
    a[3, 3] = 0
    return a

def inperspective():
    a = np.eye(4, dtype = np.float32)
    a[2, 2] = 0
    a[3, 2] = 1
    a[2, 3] = -10000
    a[3, 3] = 10
    return a

def tran_and_rot(dx, dy, rot):
    view = perspective() @ translate(-dx, -dy, 0) @ inperspective()
    view = view @ translate(0, 0, -0.3)
    view = view @ rotate(rot[0], axis=(0, 2)) @ rotate(rot[1], axis=(2, 1)) @ rotate(rot[2], axis=(0, 1))
    view = view @ translate(0,0,0.3)
    view = view @ perspective() @ translate(dx, dy, 0) @ inperspective()
    return view        


# ---------- 图层类定义 ----------
class Layer:
    def __init__(self, name, bbox, npdata):
        self.name = name
        self.npdata = npdata
        self.texture_num, texture_pos = self.get_texture()

        q, w = texture_pos
        a, b, c, d = bbox

        p1 = np.array([a, b, 0, 1, 0, 0])
        p2 = np.array([a, d, 0, 1, w, 0])
        p3 = np.array([c, d, 0, 1, w, q])
        p4 = np.array([c, b, 0, 1, 0, q])

        self.vertex = [p1,p2,p3,p4]

    # 生成纹理
    def get_texture(self):
        w, h = self.npdata.shape[:2]
        d = 2**int(max(np.log2(w), np.log2(h)) + 1)
        texture = np.zeros([d, d, 4], dtype = self.npdata.dtype)
        texture[:, :, :3] = 255
        texture[:w, :h] = self.npdata
        width, height = texture.shape[:2]
        texture_num = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_num)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR) 
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_BGRA, GL_FLOAT, texture)
        glGenerateMipmap(GL_TEXTURE_2D)
        return texture_num, (w / d, h / d) # 返回纹理坐标和纹理编号

    def get_vertex(self):
        return self.vertex.copy()

def test(init_yaml,psd_size=(354,612)):
    global model
    global test_bezier_start
    global test_bezier_finish
    global test_bezier_control
    global test_constant
    global mouse_pos_dxy
    global test_point
    global key_texture
    global keymap_list
    window = init_window()
    with open(init_yaml, encoding="utf8") as f:
        init_inf = yaml.safe_load(f)
    Layers = []
    for l in init_inf:
        a,b,c,d = init_inf[l]["bbox"]
        npdate = np.load(init_inf[l]["path"])
        #pdb.set_trace()
        layer_class = Layer(name=l,bbox=(b,a,d,c),npdata=npdate)
        Layers.append(layer_class)
    texture_cls = glGenTextures(1)
    #load_key_texture(key_yaml="./keymap.yaml")
    model = scale(2/psd_size[0],2/psd_size[1],1) @ translate(-1,-1,0) @ rotate(-np.pi/2,axis=(0,1))
    while not glfw.window_should_close(window):
        #t_start = time.time()
        glClearColor(0,0,0,0)
        glClear(GL_COLOR_BUFFER_BIT)
        curve,mouse_pos_dxy = calc_bezier()
        for layer in Layers:
            glBindTexture(GL_TEXTURE_2D,layer.texture_num)
            glColor4f(1,1,1,1)
            glPolygonMode(GL_FRONT_AND_BACK,GL_FILL)
            glBegin(GL_QUADS)
            for p in layer.get_vertex():
                a = p[:4]
                if layer.name == "mouse":
                    dx,dy = mouse_pos_dxy
                    a = a @ translate(dx,dy,0)
                b = p[4:6]
                a = a @ model
                glTexCoord2f(*b)
                glVertex4f(*a)
            glEnd()
        draw_key()
        glBindTexture(GL_TEXTURE_2D,texture_cls)
        draw_bezier(curve)
        glfw.swap_buffers(window)
        glfw.poll_events()
        time.sleep(1/30)
        #dt = time.time() - t_start
        #if dt != 0:
        #    print(1/(time.time() - t_start))

import keyboard
with open("./keymap.yaml",encoding="utf8") as f:
    keymap_list = yaml.safe_load(f)

def callback(x):
    global keymap_list
    if x.name in keymap_list:
        if x.event_type == "down":
            keymap_list[x.name]["mode"] = 1
        elif x.event_type == "up":
            keymap_list[x.name]["mode"] = 0
keyboard.hook(callback)

key_texture = {}
def load_key_texture(key_yaml):
    global key_texture
    with open(key_yaml, encoding="utf8") as f:
        key_inf = yaml.safe_load(f)
    for key in key_inf:
        key_image = Image.open(key_inf[key]["path"])
        key_npdata = np.array(key_image).astype(np.float32)
        key_npdata = key_npdata / 255
        key_class = Layer(name=key,bbox=(0,0,354,612),npdata=key_npdata)
        key_texture[key] = key_class

def draw_key():
    global key_texture
    global keymap_list
    global model
    for key in keymap_list:
        if keymap_list[key]["mode"] == 1:
            key_image = Image.open(keymap_list[key]["path"])
            key_npdata = np.array(key_image).astype(np.float32)
            key_npdata = key_npdata / 255
            key_npdata[:,:,[0,2]] = key_npdata[:,:,[2,0]]
            #pdb.set_trace()
            layer = Layer(name=key,bbox=(0,0,354,612),npdata=key_npdata)
            glBindTexture(GL_TEXTURE_2D,layer.texture_num)
            glColor4f(1,1,1,1)
            glPolygonMode(GL_FRONT_AND_BACK,GL_FILL)
            glBegin(GL_QUADS)
            for p in layer.get_vertex():
                a = p[:4]
                b = p[4:6]
                a = a @ model
                glTexCoord2f(*b)
                glVertex4f(*a)
            glEnd()
            glDeleteTextures([layer.texture_num])

# ---------- 窗口生成 ----------
window = None
from win32api import SetWindowLong,RGB
from win32con import WS_EX_LAYERED,WS_EX_TRANSPARENT,GWL_EXSTYLE,LWA_ALPHA
from win32gui import GetWindowLong,GetForegroundWindow,SetLayeredWindowAttributes,FindWindow

def init_window(v_size=(306,177)):
    glfw.init()
    glfw.window_hint(glfw.DECORATED, False)
    glfw.window_hint(glfw.TRANSPARENT_FRAMEBUFFER, True)
    glfw.window_hint(glfw.FLOATING, True)
    glfw.window_hint(glfw.SAMPLES, 4)
    glfw.window_hint(glfw.FOCUSED, True)
    glfw.window_hint(glfw.RESIZABLE, False)
    window = glfw.create_window(*v_size, "V", None, None)
    glfw.make_context_current(window)
    monitor_size = glfw.get_video_mode(glfw.get_primary_monitor()).size
    glfw.set_window_pos(window, monitor_size.width - v_size[0], monitor_size.height - v_size[1] - 30)
    glViewport(0, 0, *v_size)
    glEnable(GL_BLEND)
    glEnable(GL_TEXTURE_2D)
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA)
    hWindow = FindWindow("GLFW30","V")
    exStyle = WS_EX_LAYERED | WS_EX_TRANSPARENT
    SetWindowLong(hWindow, GWL_EXSTYLE, exStyle)
    return window

def reload(path):
    try:
        with open(path, encoding="utf8") as f:
            return yaml.safe_load(f)
    except Exception as error:
       logging.exception(error)

def reload_thread():
    global test_point
    global test_line 
    global test_bezier_start
    global test_bezier_finish
    global test_bezier_control
    global test_constant
    logging.warning("Reloading...")
    while True:
        test_point = reload("test.yaml")["test_point"]
        test_line = reload("test.yaml")["test_line"]
        test_bezier_start = reload("test.yaml")["test_bezier_start"]
        test_bezier_finish = reload("test.yaml")["test_bezier_finish"]
        test_bezier_control = reload("test.yaml")["test_bezier_control"]
        test_constant = reload("test.yaml")["test_constant"]
        time.sleep(1)

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


def draw_point(x,y,color=(1,0,1)):
    global model
    p = np.array([x,y,0,1])
    p = p @ model
    glPointSize(5)
    glColor4f(*color,1.0)
    glBegin(GL_POINTS)
    glVertex4f(*p)
    glEnd()

def calc_bezier():
    global test_bezier_start
    global test_bezier_finish
    global test_bezier_control
    global test_constant
    global mouse_pos_dxy
    start_point_p1 = np.array(test_bezier_start[0:2])
    start_point_p2 = np.array(test_bezier_start[2:4])
    control_point = get_pos_from_custom()
    #control_point = np.array(test_bezier_control[0:2])
    finish_point_p1 = np.array(test_bezier_finish[0:2])
    finish_point_p2 = np.array(test_bezier_finish[2:4])
    control_point_d = test_bezier_control[3:5]
    
    dist = np.linalg.norm(control_point - start_point_p1)
    kc = np.array([0.69,-0.7237])
    center_left = start_point_p1 + 1 * kc * dist / 2
    p_1 = [start_point_p1,\
           center_left,\
           control_point]
    
    p_a = center_left[1] - control_point[1]
    p_b = control_point[0] - center_left[0]
    p_ab = np.array([p_a,p_b])
    le = np.linalg.norm(p_ab)
    p_ab = control_point + (45/le) * p_ab
    
    dist = np.linalg.norm(finish_point_p1 - p_ab)
    push = 20
    kc2 = np.array([0.8,-0.6])
    center_right = finish_point_p1 + 0.5 * kc2 * dist / 2
    
    p_st = control_point - center_right
    le = np.linalg.norm(p_st)
    p_st = p_st * (push/le)

    p_st2 = p_ab - center_right
    le = np.linalg.norm(p_st2)
    p_st2 = p_st2 * (push/le)

    p_2 = [control_point,\
           control_point + p_st,\
           p_ab + p_st2,\
           p_ab]

    p_3 = [p_ab,\
           center_right,\
           finish_point_p1]
    p_1 = np.vstack(p_1)
    p_2 = np.vstack(p_2)
    p_3 = np.vstack(p_3)
    
    cur_1 = bezier_curve(p_1,n_points=5)
    cur_2 = bezier_curve(p_2,n_points=5)
    cur_3 = bezier_curve(p_3,n_points=5)
    
    pss2 = np.vstack((cur_1,cur_2,cur_3))
    cur_pss = bezier_curve(pss2,n_points=30)
    mouse_pos_dxy = (control_point + p_ab)/2 + \
                     np.array([test_constant[0],test_constant[1]]) - \
                     np.array([124,203])
    return cur_pss, mouse_pos_dxy

def draw_bezier(curve):
    global model
    xx = curve[:,0]
    yy = curve[:,1]
    glBegin(GL_POLYGON)
    glColor4f(1,1,1,1)
    for i in range(xx.size):
        p = np.array([xx[i],yy[i],0,1])
        p = p @ model
        glVertex4f(*p)
    glEnd()
    glLineWidth(test_bezier_control[5])
    glBegin(GL_LINE_STRIP)
    glColor4f(0,0,0,1)
    for i in range(xx.size):
        p = np.array([xx[i],yy[i],0,1])
        p = p @ model
        glVertex4f(*p)
    glEnd()
    #draw_point(*center_left,color=(0,1,0))

test_point = [0,0,0,5]
test_line = [0,0,0,0,0,0,5]
test_bezier_start = [0,0,0,0]
test_bezier_finish = [0,0,0,0]
test_bezier_control = [0,0,0,0,20,5]
test_constant = [0] * 10

def get_custom_pos():
    x, y = win32api.GetCursorPos()
    return x, y

src_points = np.array([[0, 0], [1359, 0], [1359, 767], [0, 767]])
dst_points = np.array([[254, 135], [212, 70], [187, 111],[232, 192]])
src_points = np.float32(src_points)
dst_points = np.float32(dst_points)
M_custom = cv2.getPerspectiveTransform(src_points, dst_points)

def get_perspective_transform_matrix(src_points, dst_points):
    """
    计算透射变换矩阵
    :param src_points: 源图像中的四个点，形状为 4x2 的 numpy 数组
    :param dst_points: 目标图像中的四个点，形状为 4x2 的 numpy 数组
    :return: 3x3 的透射变换矩阵
    """
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

def get_pos_from_custom():
    global M_custom
    custom_x, custom_y = get_custom_pos()
    point = np.array([custom_x,custom_y, 1])
    new_point = np.dot(M_custom, point)
    X, Y, Z = new_point
    x_prime = X / Z
    y_prime = Y / Z
    return np.array([x_prime, y_prime])

t = threading.Thread(target=reload_thread)
t.setDaemon(True)
t.start()

if __name__ == '__main__':
    test(init_yaml="./Cat2/init.yaml")
