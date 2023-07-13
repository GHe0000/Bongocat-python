import numpy as np
import matplotlib.pyplot as plt
import yaml

nx = 10
ny = 15

img = np.load("texture\hand.npy")
a, b, _ = img.shape

dx = b/nx
dy = a/ny

plt.imshow(img)

for i in range(ny):
    for j in range(nx):
        plt.gca().add_patch(plt.Rectangle((j*dx,i*dy),dx,dy,\
                                          linewidth=1,\
                                          edgecolor='r',\
                                          facecolor='none'))

plt.show()
