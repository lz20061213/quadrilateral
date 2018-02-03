from __future__ import print_function, division
import os
import math
# import cv2
import os.path as osp
import numpy as np
import pickle
from bbox_util import rect_area, rect_intersect, check_rect, rect_union, quadrangle2bbox
from scipy import misc

PAD_PERCENT = 0.1
IOU_TH = 0.8


def add_pad(x, w, dw, tp):
  pad = dw * PAD_PERCENT
  x += pad * tp
  x = max(0, x)
  x = min(x, w)
  return int(x)


def get_cent(poly):
  x = sum(poly[::2]) / 4
  y = sum(poly[1::2]) / 4
  return x, y


def gen_gt(bbox, gts):
  gt_id = -1
  pre_iou = 0
  for i, gt_quad in enumerate(gts):
    gt = quadrangle2bbox(gt_quad)
    intersect = rect_intersect(bbox, gt)
    if check_rect(intersect):
      inter_area = rect_area(intersect)
      union_arda = rect_area(rect_union(bbox, gt))
      if inter_area / rect_area(gt) > IOU_TH and inter_area / union_arda > pre_iou:
        pre_iou = inter_area / union_arda
        gt_id = i
  if gt_id == -1:
    return False, None
  x, y = bbox[:2]
  ret = gts[gt_id].copy()
  ret[:, 0] -= x
  ret[:, 1] -= y
  return True, ret


def draw_result(img, name, pts):
  out_dir = osp.join('/home/hezheqi/Project/dpreg/net/results/img')
  pts = pts.reshape((-1, 1, 2))
  cv2.polylines(img, [pts], True, (255, 0, 0), thickness=2)
  cv2.imwrite(osp.join(out_dir, name), img)


def crop_image(save_img=False, method=''):
  if method != '':
    method = '_' + method
  input_base_dir = '/data/datasets/frame/ssd'
  # input_base_dir = '/data/datasets/frame/test_2000'
  out_base_dir = '/data/datasets/frame/cut'
  pre_res_dir = '/data/datasets/frame/result/result_all_0.8_th0.75'
  # pre_res_dir = '/data/datasets/frame/result/result_2000_0.8_th0.75'
  img_list_name = '/data/datasets/frame/train_all/img_list.txt'
  # img_list_name = '/data/datasets/frame/test_2000/img_list.txt'

  img_dir = osp.join(input_base_dir, 'img')
  gt_dir = osp.join(input_base_dir, 'gt')

  img_out_dir = osp.join(out_base_dir, 'img{}'.format(method))
  gt_out_dir = osp.join(out_base_dir, 'gt{}'.format(method))
  gt_out = open(osp.join(gt_out_dir, 'new_gt.txt'), 'w')

  img_list = list(open(img_list_name).read().strip().split('\n'))
  cnt = 0
  if not osp.isdir(img_out_dir):
    os.mkdir(img_out_dir)
  for img_name in img_list:
    img = cv2.imread(osp.join(img_dir, img_name + '.jpg'))
    height, width = img.shape[:2]
    gt_arr = []
    with open(osp.join(gt_dir, img_name + '.txt')) as f_gt:
      for line in f_gt:
        points = list(map(int, line.strip().split()))
        if len(points) != 9:
          continue
        pts = [(points[i], points[i + 1]) for i in range(1, 9, 2)]
        cx, cy = get_cent(points[1:])
        pts.sort(key=lambda a: math.atan2(a[1] - cy, a[0] - cx))
        gt_arr.append(np.array(pts))

    with open(osp.join(pre_res_dir, img_name + '.txt')) as f:
      for i, line in enumerate(f):
        points = list(map(int, line.strip().split()))[1:]
        if len(points) != 8:
          continue
        x1 = min(points[0::2])
        x2 = max(points[0::2])
        y1 = min(points[1::2])
        y2 = max(points[1::2])
        dx = x2 - x1
        dy = y2 - y1
        # print(points)
        # print(x1, y1, dx, dy)

        crop_pts = [0, 0, 0, 0]
        crop_pts[0] = add_pad(x1, width, dx, -1)
        crop_pts[1] = add_pad(x2, width, dx, 1)
        crop_pts[2] = add_pad(y1, height, dy, -1)
        crop_pts[3] = add_pad(y2, height, dy, 1)
        new_img = img[crop_pts[2]:crop_pts[3], crop_pts[0]:crop_pts[1]]
        if save_img:
          cv2.imwrite(osp.join(img_out_dir, '{}_{}.jpg'.format(img_name, i)), new_img)
        # flag, new_gt = gen_gt([x1, y1, x2 - x1, y2 - y1], gt_arr) # bug!
        flag, new_gt = gen_gt([crop_pts[0], crop_pts[2], crop_pts[1] - crop_pts[0], crop_pts[3] - crop_pts[2]], gt_arr)
        gt_out.write('{}_{}.jpg'.format(img_name, i))
        if flag:
          gt_out.write(' 1')
          for g in new_gt:
            gt_out.write(' {} {}'.format(g[0], g[1]))
          if cnt < 100:
            draw_result(new_img, '{}_{}.jpg'.format(img_name, i), new_gt)
            cnt += 1
        else:
          gt_out.write(' 0')
        gt_out.write('\n')


def gen_compare_txt(tp, method):
  if method != '':
    method = '_' + method

  input_base_dir = '/data/datasets/frame/ssd'
  out_base_dir = '/data/datasets/frame/cut'
  pre_res_dir = '/data/datasets/frame/result/result_all_0.8_th0.75'

  # input_base_dir = '/data/datasets/frame/test_2000'
  # out_base_dir = '/data/datasets/frame/cut'
  # pre_res_dir = '/data/datasets/frame/result/result_2000_0.8_th0.75'

  img_dir = osp.join(input_base_dir, 'img')
  gt_dir = osp.join(input_base_dir, 'gt')

  # base_dir = '/data/datasets/frame/cut'
  # img_dir = '/data/datasets/frame/ssd/img'
  # gt_dir = '/data/datasets/frame/ssd/gt'
  # pre_res_dir = '/data/datasets/frame/result/result_all_0.8_th0.75'
  # gt_out_dir = '/data/datasets/frame/cut'
  pkl_out = open(osp.join(out_base_dir, 'compare{}.pkl'.format(method)), 'w')

  img_list = open(osp.join(out_base_dir, tp + '_list.txt'))
  # img_list = open('/data/datasets/frame/test_2000/img_list.txt')

  out_name_list = []
  out_tp_list = []
  out_gt_list = []
  out_pre_list = []
  out_pos_list = []  # position in origin image
  for img_name in img_list:
    img_name = img_name.strip()
    img = cv2.imread(osp.join(img_dir, img_name + '.jpg'))
    height, width = img.shape[:2]
    gt_arr = []
    with open(osp.join(gt_dir, img_name + '.txt')) as f_gt:
      for line in f_gt:
        points = list(map(int, line.strip().split()))
        if len(points) != 9:
          continue
        pts = [(points[i], points[i + 1]) for i in range(1, 9, 2)]
        cx, cy = get_cent(points[1:])
        pts.sort(key=lambda a: math.atan2(a[1] - cy, a[0] - cx))
        gt_arr.append(np.array(pts))

    with open(osp.join(pre_res_dir, img_name + '.txt')) as f:
      for i, line in enumerate(f):
        points = list(map(int, line.strip().split()))[1:]
        if len(points) != 8:
          continue
        x1 = min(points[0::2])
        x2 = max(points[0::2])
        y1 = min(points[1::2])
        y2 = max(points[1::2])
        dx = x2 - x1
        dy = y2 - y1
        # print(points)
        # print(x1, y1, dx, dy)

        crop_pts = [0, 0, 0, 0]
        crop_pts[0] = add_pad(x1, width, dx, -1)
        crop_pts[1] = add_pad(x2, width, dx, 1)
        crop_pts[2] = add_pad(y1, height, dy, -1)
        crop_pts[3] = add_pad(y2, height, dy, 1)

        flag, new_gt = gen_gt([crop_pts[0], crop_pts[2], crop_pts[1] - crop_pts[0], crop_pts[3] - crop_pts[2]], gt_arr)
        ###########
        # if flag == False:
        #     continue
        pts = [(points[_i], points[_i + 1]) for _i in range(0, 8, 2)]
        cx, cy = get_cent(points)
        pts.sort(key=lambda a: math.atan2(a[1] - cy, a[0] - cx))
        pts = np.array(pts)
        # print(pts)
        # print(crop_pts)
        pts[:, 0] -= crop_pts[0]
        pts[:, 1] -= crop_pts[2]
        # print(pts, '\n')
        out_name_list.append('{}_{}.jpg'.format(img_name, i))
        out_tp_list.append(flag)
        out_gt_list.append(new_gt)
        out_pre_list.append(pts)
        out_pos_list.append([crop_pts[0], crop_pts[2]])
        #  break
  pickle.dump([out_name_list, out_tp_list, out_gt_list, out_pre_list, out_pos_list], pkl_out)


def gt_split(tp, method=''):
  if method != '':
    method = '_' + method
  base_dir = '/data/datasets/frame/cut'
  img_list = open(osp.join(base_dir, tp + '_list.txt'))
  st = {}
  for line in img_list:
    st[line.strip()] = True
  print(len(st))
  fout = open(osp.join(base_dir, 'gt{}'.format(method), tp + '_gt.txt'), 'w')
  with open(osp.join(base_dir, 'gt{}'.format(method), 'new_gt.txt')) as fgt:
    for line in fgt:
      _name = line.split('_')
      name = _name[0]
      for i in range(1, len(_name)):
        if '.jpg' not in _name[i]:
          name += '_' + _name[i]
      print(name)
      if name in st:
        fout.write(line)
  img_list.close()
  fout.close()


def sample_validation_from_test(method):
  if method != '':
    method = '_' + method
  base_dir = '/data/datasets/frame/cut'
  fout = open(osp.join(base_dir, 'gt{}'.format(method), 'valid_gt.txt'), 'w')
  with open(osp.join(base_dir, 'gt{}'.format(method), 'test_gt.txt')) as fgt:
    for i, line in enumerate(fgt):
      if i % 10 == 0:
        fout.write(line)
  fout.close()


def gen_cut_name_list(tp):
  base_dir = '/data/datasets/frame/cut'
  fout = open(osp.join(base_dir, '{}_list_cut.txt'.format(tp)), 'w')
  with open(osp.join(base_dir, 'gt', '{}_gt.txt'.format(tp))) as fgt:
    for line in fgt:
      fout.write(line.split()[0] + '\n')


def normalize_training_data(method):
  if method != '':
    method = '_' + method
  base_dir = '/data/datasets/frame/cut'
  fout = open(osp.join(base_dir, 'gt{}'.format(method), 'train_gt_norm.txt'), 'w')
  with open(osp.join(base_dir, 'gt{}'.format(method), 'train_gt.txt')) as fgt:
    for cnt, line in enumerate(fgt):
      info = line.strip().split()
      img_name = osp.join(base_dir, 'img', info[0])
      label = np.array([int(info[1])])
      img = misc.imread(img_name)
      oh, ow = img.shape[:2]
      img = misc.imresize(img, (224, 224))
      xs = list(map(int, info[2::2]))
      ys = list(map(int, info[3::2]))
      bbox = []
      assert (len(xs) == len(ys))
      cx = ow / 2
      cy = oh / 2
      for i in range(len(xs)):
        bbox.append((xs[i] - cx) / ow)
        bbox.append((ys[i] - cy) / oh)
      bbox = np.array(bbox)
      if len(xs) == 0:
        bbox = -1.0 + np.random.rand(8) * 0.001
      img_name = img_name.split('/')[-1]
      fout.write('{} {}'.format(img_name, label[0]))
      for i in range(8):
        fout.write(' {}'.format(bbox[i]))
      fout.write('\n')
      misc.imsave(osp.join(base_dir, 'img_resize', img_name), img)
      if cnt % 10 == 0:
        print(cnt)

if __name__ == '__main__':
  # crop_image(False, 'th0.75')
  # gen_compare_txt('train', 'new')
  # gen_compare_txt('test', 'new')
  # gt_split('train', 'new')
  # gt_split('test', 'new')
  # gt_split('valid', 'new')
  # sample_validation_from_test('new')
  # gen_cut_name_list('test')
  normalize_training_data('new')
