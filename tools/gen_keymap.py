import os

l = os.listdir("./Cat2/keyboard")
l.sort(key = lambda x:-int(float(x[:2])), reverse = True)
with open("./keymap.yaml",encoding="utf8",mode="a") as f:
    for i in l:
        f.write(":\n")
        f.write("    path:\n")
        f.write("        ")
        f.write(i)
        f.write("\n")
        f.write("    mode:\n")
        f.write("        1")
        f.write("\n")
        f.write("\n")
