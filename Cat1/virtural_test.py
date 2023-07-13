import time
import logging

import numpy as np
# import numba as nb
import yaml

import glfw
import OpenGL
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.arrays import vbo

import win32api

# import facetracter
import threading

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

def preprocess(x):
    if x is None:
        return x
    if type(x) is str:
        return preprocess(eval(x))
    if type(x) in [int, float]:
        return np.array([[x, x], [x, x]])
    else:
        return np.array(x)

# ---------- 图层类定义 ----------
class layer:
    def __init__(self, name, bbox, z, npdata, visual):
        self.name = name
        self.npdata = npdata
        self.visual = visual
        self.texture_num, texture_pos = self.get_texture()

        q, w = texture_pos
        a, b, c, d = bbox
        if type(z) in [int, float]:
            depth = np.array([[z, z], [z, z]])
        else:
            depth = np.array(z)
        assert len(depth.shape) == 2
        self.shape = depth.shape

        [[p1, p2],
         [p4, p3]] = np.array([
             [[a, b, 0, 1, 0, 0, 0, 1], [a, d, 0, 1, w, 0, 0, 1]],
             [[c, b, 0, 1, 0, q, 0, 1], [c, d, 0, 1, w, q, 0, 1]],
         ])
        x, y = self.shape
        self.vertex = np.zeros(shape=[x, y, 8])
        for i in range(x):
            for j in range(y):
                self.vertex[i, j] = p1 + (p4-p1)*i/(x-1) + (p2-p1)*j/(y-1)
                self.vertex[i, j, 2] = depth[i, j]

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
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_BGRA, GL_FLOAT, texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glGenerateMipmap(GL_TEXTURE_2D)

        return texture_num, (w / d, h / d)

    def get_vertex(self):
        return self.vertex.copy()

class Virtural:
    def __init__(self,init_yaml,change_yaml,size=(512,512),pic_size=(1024,1024)):
        with open(init_yaml, encoding="utf8") as f:
            init_inf = yaml.safe_load(f)
        with open(change_yaml, encoding="utf8") as f:
            change_inf = yaml.safe_load(f)
        
        self.Layers = []
        self.psd_size = pic_size
        self.change_inf = change_inf

        for l in init_inf:
            a, b, c, d = init_inf[l]["bbox"]
            z_inf = preprocess(init_inf[l]['depth'])

            self.Layers.append(layer(
                name=l,
                z=z_inf,
                bbox=(b, a, d, c),
                npdata=np.load(init_inf[l]["path"]),
                visual=init_inf[l]["visual"]
            ))

        def reload(path):
            try:
                with open(path, encoding="utf8") as f:
                    return yaml.safe_load(f)
            except Exception as error:
                logging.exception(error)

        def reload_thread():
            logging.warning("Reload......")
            while True:
                self.change_inf = reload(change_yaml)
                time.sleep(1)
        
        t = threading.Thread(target = reload_thread)
        t.setDaemon(True)
        t.start()

    # 缩放
    def add_cut(self, a):
        model_g = \
                scale(2 / self.psd_size[0], 2 / self.psd_size[1], 1) @ \
                translate(-1, -1, 0) @ \
                rotate(- np.pi / 2, axis=(0, 1))
        return a @ model_g
    
    # 叠加变形
    def add_changes(self, Changes, layer_name, a):
        for change_name, intensity in Changes:
            change = self.change_inf[change_name]
            if layer_name in change:
                if "pos" in change[layer_name]:
                    d = np.array(change[layer_name]["pos"])
                    a[:, :2] += d.reshape(a.shape[0], 2) * intensity
        return a
    
    def draw_bezier(self,\
                    start_point,\
                    control_point,\
                    finish_point):
        p_1 = [start_point,\
               start_point + np.array([-1,-2]),\
               control_point + np.array([-2,0]),\
               control_point]

        p_2 = [control_point,\
               control_point + np.array([2,0]),\
               finish_point + np.array([-1,-2]),
               finish_point]
        x = np.linspace(0, 1, 50)
        xx_1, yy_1 = get_newxy(p_1, x)
        xx_2, yy_2 = get_newxy(p_2, x)
        xx = np.hstack((xx_1,xx_2))
        yy = np.hstack((yy_1,yy_2))
        glColor4ub(255, 0, 0, 255)
        glLineWidth(2.0)
        glBegin(GL_LINE_LOOP)
        for i in range(xx.size):
            glVertex4f(xx[i],yy[i],0.22,0)
        glEnd()

    def draw(self, layer):
        vertex = layer.get_vertex()
        x, y, _ = vertex.shape
        ps = vertex.reshape(x*y, 8)
        a, b = ps[:, :4], ps[:, 4:]
        a = self.add_cut(a)

        z = a[:, 2:3]
        z -= 0.1
        a[:, :2] *= z

        a = self.add_changes([
            ["left", 1]
            ],layer.name, a)
        a = a @ perspective()

        b *= z
        ps[:, :4], ps[:, 4:] = a, b
        ps = ps.reshape([x, y, 8])

        glBegin(GL_QUADS)
        for i in range(x-1):
            for j in range(y-1):
                for p in [ps[i, j], ps[i, j+1], ps[i+1, j+1], ps[i+1, j]]:
                    glTexCoord4f(*p[4:])
                    glVertex4f(*p[:4])
        glEnd()

    def draw_loop(self, window):
        while not glfw.window_should_close(window):
            glfw.poll_events()
            glClearColor(0,0,0,0)
            glClear(GL_COLOR_BUFFER_BIT)
            for layer in self.Layers:
                if layer.visual == True:
                    glEnable(GL_TEXTURE_2D)
                    glBindTexture(GL_TEXTURE_2D, layer.texture_num)
                    glColor4f(1, 1, 1, 1)
                    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
                    self.draw(layer)

                    # 框图
                    glDisable(GL_TEXTURE_2D)
                    glColor4f(0, 0, 0, 1)
                    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
                    self.draw(layer)
            self.draw_bezier(np.array([-10,0]),np.array([0,-1]),np.array([1,0]))
            glfw.swap_buffers(window)

# ---------- 窗口生成 ----------
monitor_size = None
window_pos = None
window = None
def init_window(v_size=(512,512)):
    global monitor_size
    global window_pos
    glfw.init()
    #glfw.window_hint(glfw.DECORATED, False)
    glfw.window_hint(glfw.TRANSPARENT_FRAMEBUFFER, True)
    glfw.window_hint(glfw.FLOATING, True)
    glfw.window_hint(glfw.SAMPLES, 4)
    glfw.window_hint(glfw.FOCUSED, True)
    glfw.window_hint(glfw.RESIZABLE, False)
    window = glfw.create_window(*v_size, "V", None, None)
    glfw.make_context_current(window)
    monitor_size = glfw.get_video_mode(glfw.get_primary_monitor()).size
    glfw.set_window_pos(window, monitor_size.width - v_size[0], monitor_size.height - v_size[1] - 30)
    # glfw.set_window_pos_callback(window, window_pos_callback)
    # glfw.set_key_callback(window, key_callback)
    window_pos = np.array([monitor_size.width - v_size[0], monitor_size.height - v_size[1] - 30])
    glViewport(0, 0, *v_size)
    glEnable(GL_TEXTURE_2D)
    glEnable(GL_BLEND)
    glEnable(GL_MULTISAMPLE)
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA)
    return window

class Bezier:
    def __init__(self,control_points):
        self.control_points = control_points

    def update_control_points(new_control_points):
        self.control_points = new_control_points

    def B_nx(n, i, x):
        if i > n:
            return 0
        elif i == 0:
            return (1-x)**n
        elif i == 1:
            return n*x*((1-x)**(n-1))
        return B_nx(n-1, i, x)*(1-x)+B_nx(n-1, i-1, x)*x

    def get_value(p, canshu):
        sumx = 0.
        sumy = 0.
        length = len(p)-1
        for i in range(0, len(p)):
            sumx += (B_nx(length, i, canshu) * p[i][0])
            sumy += (B_nx(length, i, canshu) * p[i][1])
        return sumx, sumy

    def get_points(x):
        xx = [0] * len(x)
        yy = [0] * len(x)
        for i in range(0, len(x)):
            a, b = get_value(control_points, x[i])
            xx[i] = a
            yy[i] = b
        return xx, yy

def B_nx(n, i, x):
    if i > n:
        return 0
    elif i == 0:
        return (1-x)**n
    elif i == 1:
        return n*x*((1-x)**(n-1))
    return B_nx(n-1, i, x)*(1-x)+B_nx(n-1, i-1, x)*x

def get_value(p, canshu):
    sumx = 0.
    sumy = 0.
    length = len(p)-1
    for i in range(0, len(p)):
        sumx += (B_nx(length, i, canshu) * p[i][0])
        sumy += (B_nx(length, i, canshu) * p[i][1])
    return sumx, sumy

def get_newxy(p,x):
    xx = [0] * len(x)
    yy = [0] * len(x)
    for i in range(0, len(x)):
        a, b = get_value(p, x[i])
        xx[i] = a
        yy[i] = b
    return xx, yy

if __name__ == "__main__":
    window = init_window()
    V = Virtural(init_yaml="./Cat1/init.yaml",change_yaml="./Cat1/change.yaml")
    V.draw_loop(window)
