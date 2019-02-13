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


# Faster R-CNN for Human Detection
# Writen by Zerong Zheng, based on the code from
# https://github.com/MVIG-SJTU/AlphaPose/blob/master/human-detection/tools/demo-alpha-pose.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test import im_detect, im_detect_fast
from newnms.nms import  soft_nms
import tensorflow as tf
import numpy as np
import os
import cv2
import argparse
from tqdm import tqdm

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
import h5py


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_file', required=True, help='path to image file')
    parser.add_argument('--out_dir', required=True, help='output directory')
    return parser.parse_args()


def extract_bbox(dets, thres=0.5):
    """extracts bounding box information from detection results"""
    xminarr, yminarr, xmaxarr, ymaxarr = [], [], [], []
    inds = np.where(dets[:, -1] >= thres)[0]
    if len(inds) == 0:
        print("Warning: No bounding box has a score above the threshold. ")
    else:
        for i in inds:
            bbox = dets[i, :4]
            xminarr.append(int(round(bbox[0])))
            yminarr.append(int(round(bbox[1])))
            xmaxarr.append(int(round(bbox[2])))
            ymaxarr.append(int(round(bbox[3])))
    return xminarr, yminarr, xmaxarr, ymaxarr


def detect(sess, net, imagedir, mode):
    """Detects object classes in an image using pre-computed object proposals."""
    # Load the demo image
    im = cv2.imread(imagedir)

    # Detect all object classes and regress object bounds
    if mode == 'fast':
        scores, boxes = im_detect_fast(sess, net, im)
    else:
        scores, boxes = im_detect(sess, net, im)

    # boxes for people
    cls_ind = 1
    cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
    cls_scores = scores[:, cls_ind]
    dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
    dets = soft_nms(dets,method=2)
    xminarr, yminarr, xmaxarr, ymaxarr = extract_bbox(dets)
    return xminarr, yminarr, xmaxarr, ymaxarr


def main(img_fname, out_dir, visualized=False):
    tfmodel = os.path.join('../output/res152/',
                           'coco_2014_train+coco_2014_valminusminival',
                           'default', 'res152.ckpt')
    if not os.path.exists(img_fname):
        raise IOError('Failed to load' + img_fname)
    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\n Please download from '
                       'Alphapose\'s Project Page').format(tfmodel + '.meta'))

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    img_dir, img_name = os.path.split(img_fname)

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True
    sess = tf.Session(config=tfconfig)

    net = resnetv1(num_layers=152)
    net.create_architecture("TEST", 81,
                          tag='default', anchor_scales=[2,4,8, 16, 32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)
    print('Loaded network {:s}'.format(tfmodel))

    xminarr, yminarr, xmaxarr, ymaxarr = detect(sess, net, img_fname, 'normal')
    print('Found %d bounding box for class human' % len(xminarr))
    with open(os.path.join(out_dir, img_name + '.bbox.txt'), 'w') as fp:
        for xmin, ymin, xmax, ymax in zip(xminarr, yminarr, xmaxarr, ymaxarr):
            fp.write('%d %d %d %d\n' % (xmin, ymin, xmax, ymax))

    if visualized:
        img = cv2.imread(img_fname)
        for xmin, ymin, xmax, ymax in zip(xminarr, yminarr, xmaxarr, ymaxarr):
            img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0))
        cv2.imwrite(os.path.join(out_dir, img_name + '.bbox.png'), img)


if __name__=='__main__':
    args = parse_args()
    img_fname = args.img_file
    out_dir = args.out_dir
    visualize = True
    main(img_fname, out_dir, visualize)
