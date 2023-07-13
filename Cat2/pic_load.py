import numpy as np
import matplotlib.pyplot as plt
import yaml
from PIL import Image

img = np.load("texture\hand.npy")

cov = (img*255).astype(np.uint8)

out = Image.fromarray(img)

out.show()

plt.imshow(img)

plt.show()
