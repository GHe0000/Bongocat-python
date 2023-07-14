import numpy as np
import yaml

def find_nonzero_boundary(arr):
    # Find the indices of non-zero elements
    row, col = np.nonzero(arr)
    # Find the minimum and maximum row and column indices
    row_min, row_max = np.min(row), np.max(row)
    col_min, col_max = np.min(col), np.max(col)
    return row_min, row_max, col_min, col_max

with open("./keymap.yaml", encoding="utf8") as f:
    key_inf = yaml.safe_load(f)

for key in key_inf:
    path = key_inf[key]["path"]
    key_npdata = np.load(path)
    #save_path = path.replace(".npy",".npy")
    np.save(save_path, texture)
