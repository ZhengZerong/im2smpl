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


from __future__ import print_function, absolute_import, division

import os
import numpy as np
import cv2 as cv
import argparse
import random
import string
import shutil
from subprocess import call
import time
import pynvml

pynvml.nvmlInit()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_file', type=str, required=True, help='path to image file')
    parser.add_argument('--out_dir', type=str, required=True, help='output directory')
    return parser.parse_args()


def waitgpu(empty_thres_duration=10):
    empty_flag = 0
    while True:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        usage_percent = float(meminfo.used)/float(meminfo.total)
        if usage_percent < 0.1:
            if empty_flag >= empty_thres_duration:   # empty for 5 second
                break
            empty_flag += 1
            time.sleep(1)
            continue
        empty_flag = 0
        print('GPU is busy right now....waiting....')
        print('meminfo.used/meminfo.total = %f' % usage_percent)
        time.sleep(np.random.randint(5, 15))


def detect_human(fname, out_dir):
    """ obtains bounding box of the subject in the input image"""
    waitgpu()
    print('\n\nStep 1. Human Detection RCNN')
    # generate a temporal script to call RCNN
    shutil.copy('./detect_human.py', './AlphaPose/human-detection/tools/')
    temp_shname = ''.join(random.sample(string.ascii_letters + string.digits, 8)) + '.sh'
    temp_shname = os.path.join('./', temp_shname)
    with open(temp_shname, 'w') as fp:
        fp.write('#!/usr/local/bin/bash\n')
        fp.write('cd ./AlphaPose/human-detection/tools\n')
        fp.write('python2 detect_human.py --img_file %s --out_dir %s\n'
                 % (fname, out_dir))
        fp.write('cd ../../../\n')
    call(['sh', temp_shname])
    os.remove(temp_shname)
    # os.remove('./AlphaPose/human-detection/tools/detect_human.py')


def crop_or_pad_img(fname, out_dir):
    """ crops or pads the original image to make the subject located at the center
        of the image and occupy 90% of the image
    """
    print('\n\nStep 2. Image cropping or padding')
    img_dir, img_name = os.path.split(img_fname)
    with open(os.path.join(out_dir, img_name + '.bbox.txt'), 'r') as fp:
        lines = fp.readlines()
        if lines[-1] == '':
            lines = lines[:-1]
        if len(lines) > 1:
            print('Warning: More than 1 bounding boxes are detected. '
                  'Only the first one is used.')
        entries = lines[0].split(' ')
    xmin, ymin = int(entries[0]), int(entries[1])
    xmax, ymax = int(entries[2]), int(entries[3])
    x_center = int((xmin+xmax)/2)
    y_center = int((ymin+ymax)/2)
    edge_len = int(max(xmax-xmin, ymax-ymin) * 1.2)
    edge_len_half = int(edge_len/2)

    img = cv.imread(fname)
    cv.imwrite(os.path.join(out_dir, img_name[:-4]+'_orig.png'), img)
    if len(img.shape) == 2:
        img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    h, w = img.shape[0], img.shape[1]
    img_pad = np.zeros((3*h, 3*w, 3), dtype=np.uint8)
    img_pad[h:(h*2), w:(w*2), :] = img
    crop_tl = (h+y_center-edge_len_half, w+x_center-edge_len_half)
    crop_dr = (h+y_center+edge_len_half, w+x_center+edge_len_half)
    img_crop = img_pad[crop_tl[0]:crop_dr[0], crop_tl[1]:crop_dr[1], :]
    cv.imwrite(os.path.join(out_dir, img_name), img_crop)
    cv.imwrite(os.path.join(out_dir, img_name), img_crop)


def infer_smpl_and_pose(fname, out_dir):
    waitgpu()
    print('\n\nStep 3a Body model estimation using HMR. ')
    shutil.copy('./infer_smpl.py', './hmr/')
    temp_shname = ''.join(random.sample(string.ascii_letters + string.digits, 8)) + '.sh'
    temp_shname = os.path.join('./', temp_shname)
    with open(temp_shname, 'w') as fp:
        fp.write('#!/usr/local/bin/bash\n')
        fp.write('cd ./hmr/\n')
        fp.write('python2 infer_smpl.py --img_path %s --out_dir %s\n' % (fname, out_dir))
        fp.write('cd ../\n')
    call(['sh', temp_shname])
    os.remove(temp_shname)
    # os.remove('./hmr/infer_smpl.py')

    print('\n\nStep 3b Pose estimation using AlphaPose')
    img_dir, img_name = os.path.split(img_fname)
    tmp_folder = ''.join(random.sample(string.ascii_letters + string.digits, 8))
    os.mkdir(os.path.join('./AlphaPose/examples', tmp_folder))
    os.mkdir(os.path.join('./AlphaPose/examples', tmp_folder, 'demo'))
    os.mkdir(os.path.join('./AlphaPose/examples', tmp_folder, 'results'))
    call(['cp', os.path.join(out_dir, img_name),
          os.path.join('./AlphaPose/examples', tmp_folder, 'demo/1.jpg')])
    call(['./AlphaPose/run.sh', '--indir', os.path.join('./examples', tmp_folder, 'demo'),
          '--outdir', os.path.join('./examples', tmp_folder, 'results'), '--vis'])
    call(['mv', os.path.join('./AlphaPose/examples', tmp_folder, 'results/POSE/pred.txt'),
          os.path.join(out_dir, img_name+'.joints.txt')])
    call(['mv', os.path.join('./AlphaPose/examples', tmp_folder, 'results/POSE/scores.txt'),
          os.path.join(out_dir, img_name+'.joint_scores.txt')])
    call(['rm', '-r', os.path.join('./AlphaPose/examples', tmp_folder)])

    print('\n\nStep 3c Image segmentation')
    shutil.copy('./segment_by_parsing.py', './LIP_JPPNet/')
    temp_shname = ''.join(random.sample(string.ascii_letters + string.digits, 8)) + '.sh'
    temp_shname = os.path.join('./', temp_shname)
    with open(temp_shname, 'w') as fp:
        fp.write('#!/usr/local/bin/bash\n')
        fp.write('cd ./LIP_JPPNet/\n')
        fp.write('python2 segment_by_parsing.py --img_file %s --out_dir %s\n' % (fname, out_dir))
        fp.write('cd ../\n')
    call(['sh', temp_shname])
    os.remove(temp_shname)
    # os.remove('./LIP_JPPNet/segment_by_parsing.py')


def optimize_smpl(fname, out_dir):
    print('\n\nStep 4 SMPL model optimization')
    shutil.copy('./fit_3d_accurate.py', './smplify_public/code/')
    temp_shname = ''.join(random.sample(string.ascii_letters + string.digits, 8)) + '.sh'
    temp_shname = os.path.join('./', temp_shname)
    with open(temp_shname, 'w') as fp:
        fp.write('#!/usr/local/bin/bash\n')
        fp.write('cd ./smplify_public/code\n')
        fp.write('python2 fit_3d_accurate.py --img_file %s --out_dir %s\n' % (fname, out_dir))
        fp.write('cd ../../\n')
    call(['sh', temp_shname])
    os.remove(temp_shname)
    # os.remove('smplify_public/code/fit_3d_accurate.py')


def main(img_fname, out_dir):
    print('image file: ' + img_fname)
    print('output directory: ' + out_dir)
    if not os.path.isfile(img_fname):
        raise IOError('Image file does not exist!')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    infer_smpl_and_pose(img_fname, out_dir)
    optimize_smpl(img_fname, out_dir)


if __name__ == '__main__':
    args = parse_args()
    img_fname = args.img_file
    out_dir = args.out_dir
    img_fname = os.path.abspath(img_fname)
    out_dir = os.path.abspath(out_dir)
    main(img_fname, out_dir)
