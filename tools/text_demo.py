from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.test import test_net, im_detect
from model.config import cfg, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb
import argparse
import pprint
import time, os, sys
import os.path as osp
import cv2
import numpy as np
from utils.cython_nms import nms

import tensorflow as tf
from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1

NUM_CLASSES = 2

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Test a Faster R-CNN network')
  parser.add_argument('--model', dest='model',
            help='model to test',
            default='/home/hezheqi/Project/dev/frame_regression/output/res101/icdar_train_17/baseline_0.62/res101_faster_rcnn_iter_100000.ckpt', type=str)
  parser.add_argument('--gpu', dest='gpu')
  parser.add_argument('--num_dets', dest='max_per_image',
            help='max number of detections per image',
            default=100, type=int)
  parser.add_argument('--tag', dest='tag',
                        help='tag of the model',
                        default='', type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16 or res101',
                      default='res101', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)

  args = parser.parse_args()
  return args

def draw_polygon(ori_img, pts, is_copy=True):
  if is_copy:
    img = ori_img.copy()
  else:
    img = ori_img
  if pts == None or len(pts) == 0:
    return img
  pts = pts.reshape((-1, 1, 2))
  # print('pts', pts)
  cv2.polylines(img, [pts], True, (255, 0, 0), thickness=8)
  return img

def draw_one_page(img, frames, pos_list=None):
  for i, frame in enumerate(frames):
    if len(frame) != 0:
      if pos_list:
        frame[::2] += pos_list[i][0]
        frame[1::2] += pos_list[i][1]
      draw_polygon(img, frame, is_copy=False)

def vis_detections(im, dets, thresh=0.5):
  """Draw detected bounding boxes."""
  inds = np.where(dets[:, -1] >= thresh)[0]
  if len(inds) == 0:
    return
  for i in inds:
    frame = dets[i, 4:12]
    frame = frame.astype(int)
    # score = dets[i, -1]
    draw_polygon(im, frame, is_copy=False)

def detect_one(name, thresh=0.75):
  im = cv2.imread(osp.join('/home/hezheqi/data/tmp/p2', name))
  scores, polys = im_detect(sess, net, im)
  print(scores)
  boxes = np.zeros((polys.shape[0], 8), dtype=polys.dtype)
  boxes[:, 0] = np.min(polys[:, 0:8:2], axis=1)
  boxes[:, 1] = np.min(polys[:, 1:8:2], axis=1)
  boxes[:, 2] = np.max(polys[:, 0:8:2], axis=1)
  boxes[:, 3] = np.max(polys[:, 1:8:2], axis=1)
  boxes[:, 4] = np.min(polys[:, 8::2], axis=1)
  boxes[:, 5] = np.min(polys[:, 9::2], axis=1)
  boxes[:, 6] = np.max(polys[:, 8::2], axis=1)
  boxes[:, 7] = np.max(polys[:, 9::2], axis=1)
  for j in range(1, NUM_CLASSES):
    inds = np.where(scores[:, j] > thresh)[0]
    cls_scores = scores[inds, j]
    cls_boxes = boxes[inds, j * 4:(j + 1) * 4]
    cls_polys = polys[inds, j * 8:(j + 1) * 8]
    cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
      .astype(np.float32, copy=False)
    cls_dets_poly = cls_polys.astype(np.float32, copy=False)
    keep = nms(cls_dets, cfg.TEST.NMS)
    # cls_dets = cls_dets[keep, :]
    cls_dets = cls_boxes[keep, :]
    cls_dets_poly = cls_dets_poly[keep, :]
    cls_scores = cls_scores[:, np.newaxis]
    cls_scores = cls_scores[keep, :]
    cls_dets = np.hstack((cls_dets, cls_dets_poly, cls_scores))
    print(cls_dets)
    vis_detections(im, cls_dets)
    cv2.imwrite(osp.join(out_dir, name), im)
    fout = open(osp.join(out_dir, 'txt', name[:-4]+'.txt'), 'w')
    for det in cls_dets:
      fout.write('{}\n'.format(' '.join(str(int(d)) for d in det[4:12])))
	



if __name__ == '__main__':
  args = parse_args()
  print('Called with args:')
  print(args)

  # if has model, get the name from it
  # if does not, then just use the inialization weights
  if args.model:
    filename = os.path.splitext(os.path.basename(args.model))[0]
  else:
    filename = os.path.splitext(os.path.basename(args.weight))[0]
  os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
  tag = args.tag
  tag = tag if tag else 'default'
  filename = tag + '/' + filename

  tfconfig = tf.ConfigProto(allow_soft_placement=True)
  tfconfig.gpu_options.allow_growth = True

  # init session
  sess = tf.Session(config=tfconfig)
  # load network
  if args.net == 'vgg16':
    net = vgg16(batch_size=1)
  else:
    net = resnetv1(batch_size=1, num_layers=101)
  # load model
  anchors = [4, 8, 16, 32]
  ratios = [0.25,0.5,1,2,4]
  net.create_architecture(sess, "TEST", NUM_CLASSES,
                          tag='default', anchor_scales=anchors, anchor_ratios=ratios)

  if args.model:
    print(('Loading model check point from {:s}').format(args.model))
    saver = tf.train.Saver()
    saver.restore(sess, args.model)
    print('Loaded.')
  else:
    print(('Loading initial weights from {:s}').format(args.weight))
    sess.run(tf.global_variables_initializer())
    print('Loaded.')

  out_dir = osp.join(cfg.ROOT_DIR, 'output', 'demo')
  if not osp.isdir(out_dir):
    os.makedirs(out_dir)
  img_dir = '/home/hezheqi/data/tmp'
  names = open(osp.join(img_dir, 'img_list.txt')).read().strip().split('\n')
  # name = ['1.jpg']
  for name in names:
    detect_one(name)


