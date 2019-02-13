# Software License Agreement (BSD License)
#
# Copyright (c) 2019, Zerong Zheng (zzr18@mails.tsinghua.edu.cn)
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of the <organization> nor the
#  names of its contributors may be used to endorse or promote products
#  derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from __future__ import division, print_function

import os
import sys
from absl import flags
import numpy as np
import skimage.io as io
import tensorflow as tf

from src.util import image as img_util
import src.config
from src.RunModel import RunModel

flags.DEFINE_string('img_path', 'data/im1963.jpg', 'Image to run')
flags.DEFINE_string('out_dir', './data', 'Output folder')
flags.DEFINE_string(
    'json_path', None,
    'If specified, uses the openpose output to crop the image.')


def preprocess_image(img_path, img_size=224):
    img = io.imread(img_path)
    if img.shape[2] == 4:
        img = img[:, :, :3]

    if np.max(img.shape[:2]) != img_size:
        print('Resizing so the max image size is %d..' % img_size)
        scale = (float(img_size) / np.max(img.shape[:2]))
    else:
        scale = 1.
    center = np.round(np.array(img.shape[:2]) / 2).astype(int)
    # image center in (x,y)
    center = center[::-1]

    crop, proc_param = img_util.scale_and_crop(img, scale, center,
                                               img_size)
    # Normalize image to [-1, 1]
    crop = 2 * ((crop / 255.) - 0.5)
    return crop, proc_param, img


def main(img_fname, out_dir):
    sess = tf.Session()
    model = RunModel(config, sess=sess)
    input_img, proc_param, img = preprocess_image(img_fname)
    input_img = np.expand_dims(input_img, 0)

    joints, verts, cams, joints3d, theta \
        = model.predict(input_img, get_theta=True)
    cam_s = cams[0][0]
    cam_t = cams[0][1:]
    theta = theta[0]
    verts = verts[0]

    img_dir, img_name = os.path.split(img_fname)
    with open(os.path.join(out_dir, img_name + '.smpl_vertex.txt'), 'w') as fp:
        for p in verts:
            fp.write('%f %f %f\n' % ((p[0] + cam_t[0]) * cam_s,
                                     (p[1] + cam_t[1]) * cam_s, p[2] * cam_s))
    with open(os.path.join(out_dir, img_name + '.cam_param.txt'), 'w') as fp:
        fp.write('%f %f %f' % (cam_s, cam_t[0], cam_t[1]))
    with open(os.path.join(out_dir, img_name + '.smpl_param.txt'), 'w') as fp:
        for p in theta[3:75]:
            fp.write('%f ' % p)
        fp.write('\n')
        for p in theta[75:]:
            fp.write('%f ' % p)


if __name__ == '__main__':
    config = flags.FLAGS
    config(sys.argv)
    config.load_path = src.config.PRETRAINED_MODEL
    config.batch_size = 1
    main(config.img_path, config.out_dir)
