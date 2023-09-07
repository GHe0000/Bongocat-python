import logging
import threading
import numpy as np
import glfw
import OpenGL
from OpenGL.GL import *
from OpenGL.GLU import *
import yaml
import time

import pdb
import win32api
import keyboard

from win32api import SetWindowLong,RGB
from win32con import WS_EX_LAYERED,WS_EX_TRANSPARENT,GWL_EXSTYLE,LWA_ALPHA
from win32gui import GetWindowLong,GetForegroundWindow,SetLayeredWindowAttributes,FindWindow

# math_calc定义了常用的数学计算
from math_calc import *

# ---------- 图层类定义 ----------
class Layer:
    def __init__(self, name, bbox, npdata, model=None):
        self.name = name
        self.npdata = npdata
        self.texture_num, texture_pos = self.get_texture()

        q, w = texture_pos
        a, b, c, d = bbox

        if model != None:
            p1 = np.array([a, b, 0, 1, 0, 0]) @ model
            p2 = np.array([a, d, 0, 1, w, 0]) @ model
            p3 = np.array([c, d, 0, 1, w, q]) @ model
            p4 = np.array([c, b, 0, 1, 0, q]) @ model
        else:
            p1 = np.array([a, b, 0, 1, 0, 0])
            p2 = np.array([a, d, 0, 1, w, 0])
            p3 = np.array([c, d, 0, 1, w, q])
            p4 = np.array([c, b, 0, 1, 0, q])

        self.vertex = np.vstack((p1,p2,p3,p4))

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

def rander(init_yaml, key_yaml, conf_inf, psd_size=(354,612)):
    global model
    #global test_point
    global key_inf

    t = threading.Thread(target=reload_thread,args=(conf_inf,))
    #t.setDaemon(True) 旧写法，已弃用
    t.daemon = True
    t.start()

    model = scale(2/psd_size[0],2/psd_size[1],1) @ translate(-1,-1,0) @ rotate(-np.pi/2,axis=(0,1))
    window = init_window()

    with open(init_yaml, encoding="utf8") as f:
        init_inf = yaml.safe_load(f)
    Layers = []
    for l in init_inf:
        a,b,c,d = init_inf[l]["bbox"]
        npdate = np.load(init_inf[l]["path"])
        layer_class = Layer(name=l,bbox=(a,b,c,d),npdata=npdate)
        Layers.append(layer_class)

    with open(key_yaml,encoding="utf8") as f:
        key_inf = yaml.safe_load(f)
    keyboard.hook(key_callback)

    texture_cls = glGenTextures(1)

    while not glfw.window_should_close(window):
        glClearColor(0,0,0,0)
        glClear(GL_COLOR_BUFFER_BIT)
        curve,mouse_pos_dxy = calc_bezier()
        for layer in Layers:
            glBindTexture(GL_TEXTURE_2D,layer.texture_num)
            glColor4f(1,1,1,1)
            glPolygonMode(GL_FRONT_AND_BACK,GL_FILL)
            p = layer.get_vertex()
            a, b = p[:,:4],p[:,4:]
            if layer.name == "mouse":
                dx,dy = mouse_pos_dxy
                a = a @ translate(dx,dy,0)
            a = a @ model
            glBegin(GL_QUADS)
            for i in range(4):
                glTexCoord2f(*b[i])
                glVertex4f(*a[i])
            glEnd()
        draw_key()
        glBindTexture(GL_TEXTURE_2D,texture_cls)
        draw_bezier(curve)
        glfw.swap_buffers(window)
        glfw.poll_events()
        time.sleep(1/30)

def key_callback(x):
    global key_inf
    if x.name in key_inf:
        if x.event_type == "down":
            key_inf[x.name]["mode"] = 1
        elif x.event_type == "up":
            key_inf[x.name]["mode"] = 0

def draw_key():
    global key_inf
    global model
    for key in key_inf:
        if key_inf[key]["mode"] == 1:
            key_npdata = np.load(key_inf[key]["path"])
            a,b,c,d = key_inf[key]["bbox"]
            layer = Layer(name=key,bbox=(a,b,c,d),npdata=key_npdata)
            glBindTexture(GL_TEXTURE_2D,layer.texture_num)
            glColor4f(1,1,1,1)
            glPolygonMode(GL_FRONT_AND_BACK,GL_FILL)
            p = layer.get_vertex()
            a, b = p[:,:4],p[:,4:]
            a = a @ model
            glBegin(GL_QUADS)
            for i in range(4):
                glTexCoord2f(*b[i])
                glVertex4f(*a[i])
            glEnd()
            glDeleteTextures([layer.texture_num])

# ---------- 窗口生成 ----------
window = None
hWindow = FindWindow("GLFW30","V")
exStyle = WS_EX_LAYERED | WS_EX_TRANSPARENT

def init_window(v_size=(612,354)):
    glfw.init()
    glfw.window_hint(glfw.DECORATED, False)
    glfw.window_hint(glfw.TRANSPARENT_FRAMEBUFFER, True)
    glfw.window_hint(glfw.FLOATING, True)
    glfw.window_hint(glfw.SAMPLES, 4)
    #glfw.window_hint(glfw.FOCUS_ON_SHOW, False)
    #glfw.window_hint(glfw.FOCUSED, False)
    glfw.window_hint(glfw.RESIZABLE, False)
    window = glfw.create_window(*v_size, "V", None, None)
    glfw.make_context_current(window)
    monitor_size = glfw.get_video_mode(glfw.get_primary_monitor()).size
    glfw.set_window_pos(window, monitor_size.width - v_size[0], monitor_size.height - v_size[1] - 95)
    glViewport(0, 0, *v_size)
    glEnable(GL_BLEND)
    glEnable(GL_TEXTURE_2D)
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA)
    hWindow = FindWindow("GLFW30","V")
    exStyle = WS_EX_LAYERED | WS_EX_TRANSPARENT
    SetWindowLong(hWindow, GWL_EXSTYLE, exStyle)
    return window

def reload_thread(path):
    global bezier_start
    global bezier_finish
    global draw_constant
    global test_point
    logging.warning("Reloading...")
    while True:
        try:
            with open(path, encoding="utf8") as f:
                conf_inf = yaml.safe_load(f)
                bezier_start = conf_inf["bezier_start"]
                bezier_finish = conf_inf["bezier_finish"]
                draw_constant = conf_inf["draw_constant"]
                test_point = conf_inf["test_point"]
        except Exception as error:
            logging.exception(error)
        time.sleep(1)

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
    global bezier_start
    global bezier_finish
    global draw_constant
    start_point = np.array(bezier_start[0:2])
    control_point = get_pos_from_custom()
    finish_point = np.array(bezier_finish[0:2])
    
    dist = np.linalg.norm(control_point - start_point)
    kc = np.array([0.69,-0.7237])
    center_left = start_point + 1 * kc * dist / 2
    p_1 = [start_point,\
           center_left,\
           control_point]
    
    p_a = center_left[1] - control_point[1]
    p_b = control_point[0] - center_left[0]
    p_ab = np.array([p_a,p_b])
    le = np.linalg.norm(p_ab)
    p_ab = control_point + (45/le) * p_ab
    
    dist = np.linalg.norm(finish_point - p_ab)
    push = 20
    kc2 = np.array([0.8,-0.6])
    center_right = finish_point + 0.5 * kc2 * dist / 2
    
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
           finish_point]
    p_1 = np.vstack(p_1)
    p_2 = np.vstack(p_2)
    p_3 = np.vstack(p_3)
    
    cur_1 = bezier_curve(p_1,n_points=5)
    cur_2 = bezier_curve(p_2,n_points=5)
    cur_3 = bezier_curve(p_3,n_points=5)
    
    pss2 = np.vstack((cur_1,cur_2,cur_3))
    cur_pss = bezier_curve(pss2,n_points=30)
    mouse_pos_dxy = (control_point + p_ab)/2 + \
                     np.array([draw_constant[0], draw_constant[1]]) - \
                     np.array([124,203])
    return cur_pss, mouse_pos_dxy

def draw_bezier(curve):
    global model
    global draw_constant
    xx = curve[:,0]
    yy = curve[:,1]
    glBegin(GL_POLYGON)
    glColor4f(1,1,1,1)
    for i in range(xx.size):
        p = np.array([xx[i],yy[i],0,1])
        p = p @ model
        glVertex4f(*p)
    glEnd()
    glLineWidth(draw_constant[2])
    glBegin(GL_LINE_STRIP)
    glColor4f(0,0,0,1)
    for i in range(xx.size):
        p = np.array([xx[i],yy[i],0,1])
        p = p @ model
        glVertex4f(*p)
    glEnd()

src_points = np.array([[0, 0], [2880, 0], [2880, 1800], [0, 1800]])
dst_points = np.array([[254, 135], [212, 70], [187, 111],[232, 192]])
M_custom = get_perspective_transform_matrix(src_points, dst_points)

translucent = 0
def get_pos_from_custom():
    global M_custom
    global translucent
    global hWindow
    hWindow = FindWindow("GLFW30","V")
    custom_x, custom_y = win32api.GetCursorPos()
    if custom_x >= (2880 - 612) and custom_x <= 2880 and (1800 - 354 -50) <= custom_y and custom_y <= (1800 - 50) and translucent == 0:
        translucent = 1
        SetLayeredWindowAttributes(hWindow,RGB(0,0,0),int(0.5*255),LWA_ALPHA)
    elif (custom_x < (2880 - 612) or custom_x > 2880 or (1800 - 354 -50) > custom_y or custom_y > (1800 - 50)) and translucent == 1:
        translucent = 0
        SetLayeredWindowAttributes(hWindow,RGB(0,0,0),255,LWA_ALPHA)
    point = np.array([custom_x,custom_y, 1])
    new_point = np.dot(M_custom, point)
    X, Y, Z = new_point
    x_prime = X / Z
    y_prime = Y / Z
    return np.array([x_prime, y_prime])

if __name__ == '__main__':
    rander(init_yaml="./Cat2/init.yaml",\
           key_yaml="./Cat2/keyinf.yaml",\
           conf_inf="./conf.yaml")
