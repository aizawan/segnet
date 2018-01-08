""" utils.py
"""

import os
import math
import numpy as np
from PIL import Image


def make_dirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def save_img(img, dir_path, filename):
    make_dirs(dir_path)
    img = np.squeeze(img).astype(np.uint8)
    img = Image.fromarray(img)
    _res_path = os.path.join(dir_path, filename)
    img.save(_res_path)
    return _res_path


def vis_semseg(y, cmap, dir_path=None, filename=None):
    y = np.squeeze(y)
    r = y.copy()
    g = y.copy()
    b = y.copy()
    for l in range(0, len(cmap)):
        r[y == l] = cmap[l, 0]
        g[y == l] = cmap[l, 1]
        b[y == l] = cmap[l, 2]
    rgb = np.zeros((y.shape[0], y.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0

    if dir_path is None and filename is None:
        return (rgb * 255).astype(np.uint8)
    else:
        return save_img(rgb * 255, dir_path, filename)
