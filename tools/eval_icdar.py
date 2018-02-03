from __future__ import print_function, division
import _init_paths
import math
import os.path as osp
from shapely.geometry import Polygon
from gen_data import get_cent
from bbox_util import is_rect
import argparse
import sys
import numpy as np
from model.config import cfg

box_score_all = []
box_right_all = []
box_wrong_all = []

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Generate txt result file')
  parser.add_argument('--dir', dest='base_dir',
                      help='result base dir',
                      default='/home/hezheqi/data/icdar/result', type=str)
  parser.add_argument('--gt', dest='gt_dir',
                      help='gt base dir',
                      default='/home/hezheqi/data/icdar17/valid/gt', type=str) 
  parser.add_argument('--name', dest='name',
                      help='out name', default=None, type=str)
  parser.add_argument('--list', dest='img_list_dir',
                      help='image list', default='/home/hezheqi/data/icdar17/valid/img_list.txt', type=str)

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)

  args = parser.parse_args()
  return args

def read_txt(name, use_bound=False, rect_label=None, is_gt=False):
  ret = []
  if not osp.exists(name):
    print(name)
    return ret
  with open(name) as fin:
    for line in fin:
      if is_gt:
        info = line.strip().split(',')[:8]
        score = 1
      else:
        info = line.strip().split()
        score = float(info[-1])
        info = info[:-1]
      if len(info) <= 1:
        continue
      info = list(map(int, info))
      # for i in range(len(info)):
      #     info[i] = max(0, info[i])
      if rect_label != None:  # only use rectangle gt
        rect_label.append(is_rect(info[1:]))
      if not is_gt:
        pts = [(info[i], info[i + 1]) for i in range(1, len(info), 2)]
        cx, cy = get_cent(info[1:])
      else:
        pts = [(info[i], info[i + 1]) for i in range(0, len(info), 2)]
        cx, cy = get_cent(info)
      pts.sort(key=lambda a: math.atan2(a[1] - cy, a[0] - cx))
      # if is_gt:
      #   print(pts)
      frame = Polygon(pts)
      if use_bound:
        x1, y1, x2, y2 = frame.bounds
        # print(x1, y1, x2, y2)
        frame = Polygon([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
      if not frame.is_valid:
        print(info)
        # x1, y1, x2, y2 = frame.bounds
        # frame = Polygon([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        continue
        # frame = frame.convex_hull
      ret.append([frame, score])
  return ret


def calculate_iou(p1, p2):
  a1 = p1.area
  a2 = p2.area
  # print(a1, a2)
  # print(p1.is_valid, p2.is_valid)
  intersection = p1.intersection(p2).area
  return intersection / (a1 + a2 - intersection)


def verify_point_distance(poly1, poly2):
  pts1 = list(poly1.exterior.coords)
  pts2 = list(poly2.exterior.coords)
  for p1, p2 in zip(pts1, pts2):
    dis = math.pow(p1[0] - p2[0], 2) + math.pow(p1[1] - p2[1], 2)
    if dis > 2500:
      return False
  return True

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
                print(t, p)
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]
        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def cal_mAP(npos):
    confidence = np.array(box_score_all)
    tp = np.array(box_right_all)
    fp = np.array(box_wrong_all)
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    tp = tp[sorted_ind]
    fp = fp[sorted_ind]
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    # print(rec)
    # print(prec)
    ap = voc_ap(rec, prec, False)
    return ap

def eval_one(results, gts, point_dis=False, rect_label=None):
  '''
  :param results:
  :param gts:
  :param point_dis:
  :param rect_label: use rectangle or not
  :return right_num, error_num, mid_num
  '''
  m = len(gts)
  is_used = [False] * m
  right_num = 0
  err_num = 0
  mid_num = 0
  for r in results:
    res = r[0]
    box_score_all.append(r[1])
    if not point_dis:
      max_iou = -1
      max_index = -1
      for j, gt in enumerate(gts):
        gt = gt[0]
        if is_used[j]:
          continue
        iou = calculate_iou(res, gt)
        if max_iou < iou:
          max_iou = iou
          max_index = j
      if max_iou > th:
        is_used[max_index] = True
        if rect_label == None:
          right_num += 1
          box_right_all.append(1)
          box_wrong_all.append(0)
        elif rect_label[max_index]:
          right_num += 1
        elif not rect_label[max_index]:
          mid_num += 1
      else:
        err_num += 1
        box_right_all.append(0)
        box_wrong_all.append(1)
    else:
      flag = False
      for j, gt in enumerate(gts):
        if is_used[j]:
          continue
        if verify_point_distance(res, gt):
          is_used[j] = True
          right_num += 1
          flag = True
          break
      if not flag:
        err_num += 1

  assert (right_num <= m)
  assert (err_num <= len(results))
  return right_num, err_num, mid_num


def evaluate(mean_f=True, point_dis=False, rect_flag=False):
  name_list = open(name_list_dir).read().strip().split('\n')
  fout = open(osp.join(cfg.DATA_DIR, 'wrong.txt'), 'w')
  precision, recall, page_correct = 0, 0, 0
  right_all, error_all, gt_all, res_all = 0, 0, 0, 0
  for name in name_list:
    results = read_txt(osp.join(res_base_dir, name + '.txt'), use_bound=False)
    if rect_flag:
      rect_label = []
    else:
      rect_label = None
    gts = read_txt(osp.join(gt_base_dir, name + '.txt'), rect_label=rect_label,
                   is_gt=True, use_bound=False)
    right_num, error_num, mid_num = eval_one(results, gts, rect_label=rect_label, point_dis=point_dis)
    # right_num, error_num, mid_num = eval_one(results, gts)
    right_all += right_num
    error_all += error_num
    gt_all += len(gts) - mid_num
    res_all += len(results) - mid_num
    if len(results) - mid_num > 0:
      precision += right_num / (len(results) - mid_num)
    if len(gts) - mid_num > 0:
      recall += right_num / (len(gts) - mid_num)
    if right_num == len(gts) and error_num == 0:
    # if right_num == len(gts):
      page_correct += 1
    else:
      fout.write('{}\n'.format(name))

  n = len(name_list)
  precision /= n
  recall /= n
  page_correct /= n
  f1 = 2 * precision * recall / (precision + recall)
  print('{} {:.5f} {:.5f} {:.5f} {:.5f}'.format(th, precision, recall, f1, page_correct))
  if not mean_f:
    precision = right_all / res_all
    recall = right_all / gt_all
  f1 = 2 * precision * recall / (precision + recall)
  # print(th, precision, recall, f1, page_correct)
  print('{} {:.5f} {:.5f} {:.5f} {:.5f}'.format(th, precision, recall, f1, page_correct))
  print('map:', cal_mAP(gt_all))


if __name__ == '__main__':
  # gt_base_dir = '/data/datasets/frame/test_2000/gt'

  # res_base_dir = '/data/datasets/frame/result/result_all_0.8_th0.75'
  # res_base_dir = '/data3/dt'
  # res_base_dir = '/data/datasets/frame/result/result_ssd_th0.75'
  # res_base_dir = '/home/hezheqi/data/frame/result/faster_reg2_poly'
  # res_base_dir = '/home/hezheqi/Project/dpreg/net/results/pages_mult/txt'
  # res_base_dir = '/home/cpdp/Documents/yf-workspace/data/2000_res_txt'
  # res_base_dir = '/data3/20w_results/ly_crf_new'
  # res_base_dir = '/data3/20w_results/dt'
  # res_base_dir = '/home/cpdp/Documents/yf-workspace/data/29845_LD_DRR'
  # res_base_dir = '/data/datasets/frame/result/result_2000_0.8_th0.75'
  # name_list_dir = '/data/datasets/frame/test_2000/img_list.txt'

  args = parse_args()
  gt_base_dir = args.gt_dir
  res_base_dir = osp.join(args.base_dir, args.name)
  th = 0.5
  name_list_dir = args.img_list_dir

  evaluate(mean_f=False, point_dis=False)
  # evaluate(False, True)
