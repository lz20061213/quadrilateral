# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
import numpy as np

try:
  import cPickle as pickle
except ImportError:
  import pickle
import os
import math

from utils.timer import Timer
from utils.cython_nms import nms, nms_new
from utils.boxes_grid import get_boxes_grid
from utils.blob import im_list_to_blob

from model.config import cfg, get_output_dir
from model.bbox_transform import bbox_transform_inv, bbox_transform_inv_reg


def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)


def _get_blobs(im):
  """Convert an image and RoIs within that image into network inputs."""
  blobs = {}
  blobs['data'], im_scale_factors = _get_image_blob(im)

  return blobs, im_scale_factors


def _clip_boxes(boxes, im_shape):
  """Clip boxes to image boundaries."""
  # x1 >= 0
  boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
  # y1 >= 0
  boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
  # x2 < im_shape[1]
  boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
  # y2 < im_shape[0]
  boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
  return boxes


def _clip_boxes_reg(boxes, im_shape):
  """Clip boxes to image boundaries."""
  for i in range(0, 8, 2):
    boxes[:, i::8] = np.maximum(boxes[:, i::8], 0)
    boxes[:, i::8] = np.minimum(boxes[:, i::8], im_shape[1] - 1)
    boxes[:, i + 1::8] = np.maximum(boxes[:, i + 1::8], 0)
    boxes[:, i + 1::8] = np.minimum(boxes[:, i + 1::8], im_shape[0] - 1)
  return boxes


def _rescale_boxes(boxes, inds, scales):
  """Rescale boxes according to image rescaling."""
  for i in range(boxes.shape[0]):
    boxes[i, :] = boxes[i, :] / scales[int(inds[i])]

  return boxes


def im_detect(sess, net, im):
  blobs, im_scales = _get_blobs(im)
  assert len(im_scales) == 1, "Only single-image batch implemented"

  im_blob = blobs['data']
  # seems to have height, width, and image scales
  # still not sure about the scale, maybe full image it is 1.
  blobs['im_info'] = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

  _, scores, poly_pred, rois = net.test_image(sess, blobs['data'], blobs['im_info'])

  boxes = rois[:, 1:5] / im_scales[0]
  # print(scores.shape, bbox_pred.shape, rois.shape, boxes.shape)
  scores = np.reshape(scores, [scores.shape[0], -1])
  poly_pred = np.reshape(poly_pred, [poly_pred.shape[0], -1])
  if cfg.TEST.BBOX_REG:
    # Apply bounding-box regression deltas
    poly_deltas = poly_pred
    # print(poly_deltas[0])
    pred_poly = bbox_transform_inv_reg(boxes, poly_deltas)
    pred_poly = _clip_boxes_reg(pred_poly, im.shape)

  else:
    # Simply repeat the boxes, once for each class
    pred_poly = np.array([])

  return scores, pred_poly


def apply_nms(all_boxes, thresh):
  """Apply non-maximum suppression to all predicted boxes output by the
  test_net method.
  """
  num_classes = len(all_boxes)
  num_images = len(all_boxes[0])
  nms_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
  for cls_ind in range(num_classes):
    for im_ind in range(num_images):
      dets = all_boxes[cls_ind][im_ind]
      if dets == []:
        continue

      x1 = dets[:, 0]
      y1 = dets[:, 1]
      x2 = dets[:, 2]
      y2 = dets[:, 3]
      scores = dets[:, 4]
      inds = np.where((x2 > x1) & (y2 > y1) & (scores > cfg.TEST.DET_THRESHOLD))[0]
      dets = dets[inds, :]
      if dets == []:
        continue

      keep = nms(dets, thresh)
      if len(keep) == 0:
        continue
      nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
  return nms_boxes


def test_net(sess, net, imdb, weights_filename, max_per_image=100, thresh=0.05):
  np.random.seed(cfg.RNG_SEED)
  """Test a Fast R-CNN network on an image database."""
  num_images = len(imdb.image_index)
  # all detections are collected into:
  #  all_boxes[cls][image] = N x 5 array of detections in
  #  (x1, y1, x2, y2, score)
  all_boxes = [[[] for _ in range(num_images)]
               for _ in range(imdb.num_classes)]

  output_dir = get_output_dir(imdb, weights_filename)
  # timers
  _t = {'im_detect': Timer(), 'misc': Timer()}

  for i in range(num_images):
    im = cv2.imread(imdb.image_path_at(i))
    cls_dets_list = [[] for _ in range(imdb.num_classes)]
    cls_dets_list_box = [[] for _ in range(imdb.num_classes)]
    _t['im_detect'].tic()
    for scale in [480, 576, 688, 864, 1200]:
    # for scale in [864]:
      cfg.TEST.SCALES = [scale]
      scores, polys = im_detect(sess, net, im)

      _t['misc'].tic()
      boxes = np.zeros((polys.shape[0], 8), dtype=polys.dtype)
      boxes[:, 0] = np.min(polys[:, 0:8:2], axis=1)
      boxes[:, 1] = np.min(polys[:, 1:8:2], axis=1)
      boxes[:, 2] = np.max(polys[:, 0:8:2], axis=1)
      boxes[:, 3] = np.max(polys[:, 1:8:2], axis=1)
      boxes[:, 4] = np.min(polys[:, 8::2], axis=1)
      boxes[:, 5] = np.min(polys[:, 9::2], axis=1)
      boxes[:, 6] = np.max(polys[:, 8::2], axis=1)
      boxes[:, 7] = np.max(polys[:, 9::2], axis=1)

      for j in range(1, imdb.num_classes):
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
        # print(cls_dets)
        cls_dets_list[j].append(cls_dets)
        cls_dets_list_box[j].append(np.hstack((cls_dets, cls_scores)))
        # all_boxes[j][i] = cls_dets
    for j in range(1, imdb.num_classes):
      cls_dets_boxes = np.vstack(cls_dets_list_box[j])
      cls_dets = np.vstack(cls_dets_list[j])
      # print(len(cls_dets))
      keep = nms(cls_dets_boxes, cfg.TEST.NMS)
      cls_dets = cls_dets[keep, :]
      all_boxes[j][i] = cls_dets
      
    # Limit to max_per_image detections *over all classes*
    if max_per_image > 0:
      image_scores = np.hstack([all_boxes[j][i][:, -1]
                                for j in range(1, imdb.num_classes)])
      if len(image_scores) > max_per_image:
        image_thresh = np.sort(image_scores)[-max_per_image]
        for j in range(1, imdb.num_classes):
          keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
          all_boxes[j][i] = all_boxes[j][i][keep, :]
    _t['misc'].toc()

    _t['im_detect'].toc()
    print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
          .format(i + 1, num_images, _t['im_detect'].average_time,
                  _t['misc'].average_time))

  det_file = os.path.join(output_dir, 'detections.pkl')
  with open(det_file, 'wb') as f:
    pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

  print('Evaluating detections')
  imdb.evaluate_detections(all_boxes, output_dir)


