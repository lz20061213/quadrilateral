# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
# import sys
# sys.path.append('../')
import datasets
import os
import os.path as osp
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import pickle
import subprocess
import uuid
from .voc_eval import voc_eval
from model.config import cfg
import PIL
import math


class Frame(imdb):
    def __init__(self, classname, image_set):
        imdb.__init__(self, 'frame_' + image_set)
        self._image_set = image_set
        self._data_path = osp.join(cfg.DATA_DIR, 'frame', classname, image_set)
        self._classes = ('__background__',  # always index 0
                         'quadrilateral')
        self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))
        self._image_ext = '.jpg'
        if 'dragonball' in image_set:
            self._image_ext = '.jpg'
        self._image_index = self._load_image_name()
        # Default to roidb handler
        self._roidb_handler = self.selective_search_roidb
        self._salt = str(uuid.uuid4())

        # PASCAL specific config options
        self.config = {'cleanup': True,
                       'use_salt': True,
                       'use_diff': False,
                       'matlab_eval': False,
                       'rpn_file': None,
                       'min_size': 2}

        assert os.path.exists(self._data_path), \
            'Path does not exist: {}'.format(self._data_path)

    def _load_image_name(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._data_path, 'img_list.txt')
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_path = [x.strip() for x in f.readlines()]
        return image_path

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path,
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} ss roidb loaded from {}'.format(self.name, cache_file))
            return roidb
        if self._image_set[:4] != 'test':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        with open(cache_file, 'wb') as fid:
            pickle.dump(roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote ss roidb to {}'.format(cache_file))

        return roidb

    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(self._data_path,
                                                self.name + '.pkl'))
        assert os.path.exists(filename), \
            'Selective search data not found at: {}'.format(filename)
        # raw_data = sio.loadmat(filename)['boxes'].ravel()
        # img_ids = sio.loadmat(filename)['images'].ravel()
        raw_data = pickle.load(open(filename))
        print((type(raw_data)))
        box_list = []
        for i in range(raw_data.shape[0]):
            # box_list.append(raw_data[i][:, (1, 0, 3, 2)] - 1)
            box_list.append(raw_data[i])
            # print(box_list[i], "########")
            if not (box_list[i][:, 0] <= box_list[i][:, 2]).all():
                print(1)
            if not (box_list[i][:, 1] <= box_list[i][:, 3]).all():
                print(2)
            if not (box_list[i][:, :] >= 0).all():
                print(3)
            assert (box_list[i][:, 0] <= box_list[i][:, 2]).all(), i
            assert (box_list[i][:, 1] <= box_list[i][:, 3]).all(), i
            assert (box_list[i][:, :] >= 0).all(), i
        # assert (boxes[:, 2] >= boxes[:, 0]).all()
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [self._load_frame_rec_annotation(img_name)
                    for img_name in self.image_index]
        print((len(gt_roidb)))
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    def rpn_roidb(self):
        if self._image_set[:4] != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print('loading {}'.format(filename))
        assert os.path.exists(filename), \
            'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = pickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def get_cent(self, poly):
        x = sum(poly[::2]) / 4
        y = sum(poly[1::2]) / 4
        return x, y

    def _load_frame_rec_annotation(self, img_name):
        gt_name = img_name + '.txt'
        #filename = os.path.join(self._data_path, 'gt', gt_name)
        filename = os.path.join(self._data_path, gt_name)
        if cfg.NEW_GT:
            #filename = os.path.join(self._data_path, 'new_gt', gt_name)
            filename = os.path.join(self._data_path, gt_name)
        print ('the filename is ' + str(filename))
        with open(filename) as f:
            num_panels = int(f.readline().strip())
            _img = PIL.Image.open(self.image_path_from_index(img_name))
            width = _img.size[0]
            height = _img.size[1]
            print num_panels, width, height
            if cfg.FRAME_REG:
                boxes = np.zeros((num_panels, 12), dtype=np.uint16)
            else:
                boxes = np.zeros((num_panels, 4), dtype=np.uint16)
            is_rec = np.zeros((num_panels, 1), dtype=np.uint16)
            gt_classes = np.zeros((num_panels), dtype=np.int32)
            overlaps = np.zeros((num_panels, self.num_classes), dtype=np.float32)
            cur = 0
            print ('the init value of cur is ' + str(cur))
            for panel_info in f.readlines():
                panel_info = panel_info.strip()
                print ('panel_info is ' + str(panel_info))
                print ('here,cur is ' + str(cur))
                if panel_info == "":
                    continue
                coordinates = list(map(float, panel_info.split()))
                #coordinates = re.split(' ', panel_info.split())
                if cfg.NEW_GT:
                    shape = panel_info.split()[0].strip()
                    if shape == 'rect':
                        is_rec[cur] = 1
                xs = coordinates[::2]
                ys = coordinates[1::2]
                x1 = int(max(min(xs), 0))
                x2 = int(min(max(xs), width - 1))
                y1 = int(max(min(ys), 0))
                y2 = int(min(max(ys), height - 1))
                cls = self._class_to_ind['quadrilateral']
                assert x2 >= x1, filename
                assert y2 >= y1, filename
                assert x2 < width, filename
                assert y2 < height, filename
                if cfg.FRAME_REG:
                    for i in range(0, 8, 2):
                        coordinates[i] = int(max(coordinates[i], 0))
                        coordinates[i] = int(min(coordinates[i], width - 1))
                        coordinates[i + 1] = int(max(coordinates[i + 1], 0))
                        coordinates[i + 1] = int(min(coordinates[i + 1], height - 1))
                    cx, cy = self.get_cent(coordinates)
                    pts = [(coordinates[i], coordinates[i + 1]) for i in range(0, 8, 2)]
                    pts.sort(key=lambda a: math.atan2(a[1] - cy, a[0] - cx))
                    coordinates = []
                    for p in pts:
                        coordinates.append(p[0])
                        coordinates.append(p[1])
                    print ('num_panels is ' + str(num_panels))
                    print ('cur is ' + str(cur))
                    boxes[cur, :] = [x1, y1, x2, y2] + coordinates
                    print ('boxes is ' + str(boxes[cur, :]))
                else:
                    boxes[cur, :] = [x1, y1, x2, y2]

                # rects[cur, :] = _polar_sort([x1, y1, x1, y2, x2, y1, x2, y2])
                gt_classes[cur] = cls
                overlaps[cur, cls] = 1.0
                cur += 1

        overlaps = scipy.sparse.csr_matrix(overlaps)
        print('function end~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'flipped': False,
                'is_rec': is_rec}

    def evaluate_detections(self, all_boxes, output_dir):
        print("to be update")

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True


if __name__ == '__main__':
    from datasets.pascal_voc import pascal_voc

    d = pascal_voc('trainval', '2007')
    res = d.roidb
    from IPython import embed;

    embed()
