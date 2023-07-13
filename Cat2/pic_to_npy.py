import numpy as np
import psd_tools

psd_path = 'mousebg.psd'
psd = psd_tools.PSDImage.open(psd_path)
# print(psd)
for l in psd:
    print(l.name)
    print(np.array(l.bbox))
    a, b, c, d = l.bbox
    save_path = './texture/' + l.name + '.npy'
    npdata = l.numpy()
    np.save(save_path, npdata)
