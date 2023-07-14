from PIL import Image
import numpy as np
import yaml

with open("./keymap.yaml", encoding="utf8") as f:
    key_inf = yaml.safe_load(f)

for key in key_inf:
    path = key_inf[key]["path"]
    key_image = Image.open(path)
    key_npdata = np.array(key_image).astype(np.float32)
    key_npdata = key_npdata / 255
    key_npdata[:,:,[0,2]] = key_npdata[:,:,[2,0]]
    save_path = path.replace(".png",".npy")
    np.save(save_path,key_npdata)
    key_inf[key]["path"] = save_path

with open('./key.yaml', 'w', encoding='utf-8') as f:
   yaml.dump(data=key_inf, stream=f, allow_unicode=True)
