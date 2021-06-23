#
# MIT License
#
# Copyright (c) 2018 Matteo Poggi m.poggi@unibo.it
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from matplotlib import cm
import cv2

# Colormap wrapper
def applyColorMap(img, cmap):
    colormap = cm.get_cmap(cmap)
    colored = colormap(img)
    return np.float32(cv2.cvtColor(np.uint8(colored*255),cv2.COLOR_RGBA2BGR))/255.


# 2D convolution wrapper
def count_text_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)


# Reconstruct 3D shape from depth
def reconstruct_3d(depth, fov):

    height = depth.shape[0]
    width = depth.shape[1]
    aspect_ratio = width / height
    fy = 0.5 / np.tan(fov * 0.5)
    fx = fy / aspect_ratio

    # by construction depth is always greater than 0
    # however where helps to change the shape of a depthmap
    # to construct the related 3D point cloud
    mask = np.where(depth > 0)

    row = mask[0]
    col = mask[1]

    normalized_x = (col.astype(np.float32) - width * 0.5) / width
    normalized_y = (row.astype(np.float32) - height * 0.5) / height

    world_x = normalized_x * depth[row, col] / fx
    world_y = normalized_y * depth[row, col] / fy
    world_z = depth[row, col]
    pcd = np.vstack((world_x, world_y, world_z)).T
    return pcd
