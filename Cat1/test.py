import numpy as np  # version 1.17.2
from scipy.spatial import Delaunay  # version 1.4.1
import matplotlib.pyplot as plt  # version 3.1.2
import random


def create_data():
    x = [166.6,117.4,74,25.9,0,0,30.6,79.8]
    y = [127.7,186.8,213.4,212.8,185,137,67.5,0]
    points = [(i, j) for i, j in zip(x, y)]
    points = np.array(points)
    return points


def create_delauney(points):
    # create a Delauney object using (x, y)
    tri = Delaunay(points)

    # paint a triangle
    plt.imshow(np.load("texture\hand.npy"))
    plt.triplot(points[:, 0], points[:, 1], tri.simplices.copy(), c='black')
    plt.plot(points[:, 0], points[:, 1], 'o', c='green')
    plt.show()


point = create_data()
    create_delauney(point)


