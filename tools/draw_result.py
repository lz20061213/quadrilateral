import cv2
import os
import numpy as np
import os.path as osp


def draw_polygon(ori_img, pts, is_copy=True):
    if is_copy:
        img = ori_img.copy()
    else:
        img = ori_img
    if pts == None or len(pts) == 0:
        return img
    pts = np.array(pts)
    pts = pts.reshape((-1, 1, 2))
    # print('pts', pts)
    cv2.polylines(img, [pts], True, (255, 0, 0), thickness=2)
    return img


def get_compare_result(img, gt, pre_res, res):
    gt_img = draw_polygon(img, gt)
    pre_img = draw_polygon(img, pre_res)
    res_img = draw_polygon(img, res)
    top = np.column_stack((img, gt_img))
    bottom = np.column_stack((pre_img, res_img))
    return np.row_stack((top, bottom))


def draw_one_page(img, frames, pos_list=None):
    for i, frame in enumerate(frames):
        if len(frame) != 0:
            if pos_list:
                frame[::2] += pos_list[i][0]
                frame[1::2] += pos_list[i][1]
            draw_polygon(img, frame, is_copy=False)

if __name__ == '__main__':
    names = open('/home/hezheqi/Project/dev/frame_regression/data/icdar/test_detext/img_list.txt').read().strip().split('\n')
    base_dir = '/home/hezheqi/data/icdar15/test_detext'
    res_dir = '/home/hezheqi/data/detext/result/test_2000_poly'
    for id, name in enumerate(names):
        frames = []
        img = cv2.imread(osp.join(base_dir, 'img', name + '.jpg'))
        with open(osp.join(res_dir, name + '.txt')) as fin:
            for line in fin:
                if len(line.strip().split()) < 5: continue
                frame = list(map(int, line.strip().split()[1:9]))
                score = float(line.strip().split()[9])
                # if score < 0.7:
                #     continue
                for i, f in enumerate(frame):
                    frame[i] = max(0, f)
                frames.append(frame)
        print(name)
        draw_one_page(img, np.array(frames))
        if not osp.isdir(osp.join(base_dir, 'draw')):
            os.mkdir(osp.join(base_dir, 'draw'))
        cv2.imwrite(osp.join(base_dir, 'draw', 'res_' + name + '.jpg'), img)

