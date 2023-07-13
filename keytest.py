import keyboard
import yaml
import time

with open("./keymap.yaml",encoding="utf8") as f:
    keymap_list = yaml.safe_load(f)

def callback(x):
    if x.name in keymap_list:
        if x.event_type == "down":
            print("1")
            keymap_list[x.name]["mode"] = 1
        elif x.event_type == "up":
            print("0")
            keymap_list[x.name]["mode"] = 0
keyboard.hook(callback)
keyboard.wait()
