import numpy as np
import yaml

def find_nonzero_boundary(arr):
    # Find the indices of non-zero elements
    row, col = np.nonzero(arr)
    # Find the minimum and maximum row and column indices
    row_min, row_max = np.min(row), np.max(row)
    col_min, col_max = np.min(col), np.max(col)
    return row_min, row_max, col_min, col_max

with open("./keymap.yaml",encoding="utf8") as f:
    key_inf = yaml.safe_load(f)

with open("./keyinf.yaml",encoding="utf8",mode="a") as f:
    for key in key_inf:
        key_npdata = np.load(key_inf[key]["path"])
        a,b,c,d = find_nonzero_boundary(key_npdata[:,:,3])
        save_npdata = key_npdata[a:b+1,c:d+1,:]
        np.save(key_inf[key]["path"],save_npdata)
        f.write("\"" + key + "\"" + ":\n")
        f.write("    path:\n")
        f.write("        " + key_inf[key]["path"] + "\n")
        f.write("    bbox:\n")
        f.write("        " + "- " + str(a) + "\n")
        f.write("        " + "- " + str(c) + "\n")
        f.write("        " + "- " + str(b+1) + "\n")
        f.write("        " + "- " + str(d+1) + "\n")
        f.write("    mode:\n")
        f.write("        0" + "\n")
        f.write("\n")
