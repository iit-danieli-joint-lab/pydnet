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


import tensorflow as tf
import imageio
import os
import argparse
import time
import shutil
# examples/Python/Basic/working_with_numpy.py

import copy
import numpy as np
import open3d as o3d

import datetime
from utils import *
from pydnet import *

# forces tensorflow to run on CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

parser = argparse.ArgumentParser(description='Argument parser')

""" Arguments related to network architecture"""
parser.add_argument('--width', dest='width', type=int, default=512, help='width of input images')
parser.add_argument('--height', dest='height', type=int, default=256, help='height of input images')
parser.add_argument('--resolution', dest='resolution', type=int, default=1, help='resolution [1:H, 2:Q, 3:E]')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', type=str, default='checkpoint/IROS18/pydnet', help='checkpoint directory')
parser.add_argument('--path', dest='path', type=str, default=".", help='image path')

args = parser.parse_args()

def main(_):

    pathname = "./scene_pcd"
    if os.path.isdir(pathname):
        shutil.rmtree(pathname, ignore_errors=False, onerror=None)
    os.mkdir(pathname)

    with tf.Graph().as_default():
        height = args.height
        width = args.width
        img_path = args.path
        placeholders = {'im0':tf.compat.v1.placeholder(tf.float32,[None, None, None, 3], name='im0')}

        with tf.compat.v1.variable_scope("model") as scope:
            model = pydnet(placeholders)

        init = tf.group(tf.compat.v1.global_variables_initializer(),
                        tf.compat.v1.local_variables_initializer())

        loader = tf.compat.v1.train.Saver()

        with tf.compat.v1.Session() as sess:
            sess.run(init)
            loader.restore(sess, args.checkpoint_dir)

            img = cv2.imread(img_path)
            img = cv2.resize(img, (width, height)).astype(np.float32) / 255.
            img = np.expand_dims(img, 0)
            start = time.time()
            disparity = sess.run(model.results[args.resolution-1], feed_dict={placeholders['im0']: img})
            end = time.time()

            fov = 90*np.pi/180  # in radians
            disparity_temp = disparity[0, :, :, 0]
            pcd = reconstruct_3d(disparity_temp, fov)

            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(pcd)
            o3d.io.write_point_cloud(pathname+"/scene.pcd", pc)

            scaling = 20  # scaling depth factor
            disp_color = applyColorMap(disparity_temp*scaling, 'plasma')
            imageio.imwrite(pathname+"/depth.jpg", disp_color)

            print("Time: " + str(end - start))


if __name__ == '__main__':
    tf.compat.v1.app.run()
