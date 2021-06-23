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
import sys
import os
import argparse
import time
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

args = parser.parse_args()

def main(_):

    pathname = "scene_pcd"
    if os.path.isfile(pathname):

        os.mkdir(pathname)

    with tf.Graph().as_default():
        height = args.height
        width = args.width
        placeholders = {'im0':tf.compat.v1.placeholder(tf.float32,[None, None, None, 3], name='im0')}

        with tf.compat.v1.variable_scope("model") as scope:
            model = pydnet(placeholders)

        init = tf.group(tf.compat.v1.global_variables_initializer(),
                        tf.compat.v1.local_variables_initializer())

        loader = tf.compat.v1.train.Saver()
        saver = tf.compat.v1.train.Saver()
        cam = cv2.VideoCapture(0)

        with tf.compat.v1.Session() as sess:
            sess.run(init)
            loader.restore(sess, args.checkpoint_dir)

            pcd_n=0
            while True:
                pcd_n = pcd_n + 1
                ret_val, img = cam.read()
                img = cv2.resize(img, (width, height)).astype(np.float32) / 255.
                img = np.expand_dims(img, 0)
                start = time.time()
                disp = sess.run(model.results[args.resolution - 1], feed_dict={placeholders['im0']: img})
                end = time.time()

                # scaling depth factor
                scaling_depth_map = 20
                fov = 120 * np.pi / 180
                disp_temp = disp[0, :, :, 0] * scaling_depth_map
                pcd = reconstruct_3d(disp_temp, fov)

                filename = pathname + "/scene_" + str(pcd_n) + ".pcd"

                pc = o3d.geometry.PointCloud()
                pc.points = o3d.utility.Vector3dVector(pcd)
                o3d.io.write_point_cloud(filename, pc)

                disp_color = applyColorMap(disp_temp, 'plasma')
                toShow = (np.concatenate((img[0], disp_color), 0) * 255.).astype(np.uint8)

                cv2.imshow('pydnet', toShow)
                k = cv2.waitKey(1)
                if k == 1048603 or k == 27:
                    break  # esc to quit
                if k == 1048688:
                    cv2.waitKey(0) # 'p' to pause

                print("Time: " + str(end - start))
                del img
                del disp
                del toShow

            cam.release()

if __name__ == '__main__':
    tf.compat.v1.app.run()
