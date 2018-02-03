from shapely.geometry import Polygon
import numpy as np
import math

def poly_overlaps(gts, preds):
  gt_polys = []
  gt_areas = []
  for gt in gts:
    pts = [(gt[i], gt[i + 1]) for i in range(0, len(gt), 2)]
    gt_polys.append(Polygon(pts))
    gt_areas.append(gt_polys[-1].area)
  pred_polys = []
  pred_areas = []
  for pred in preds:
    pts = [(pred[i], pred[i + 1]) for i in range(0, len(pred) -1, 2)]
    pred_polys.append(Polygon(pts))
    pred_areas.append(pred_polys[-1].area)
  K = len(gt_polys)
  N = len(pred_polys)
  overlaps = np.zeros((N, K), dtype=np.float)
  for k in range(K):
    gt_area = gt_areas[k]
    for n in range(N):
      pred_area = pred_areas[n]
      try:
        intersection = gt_polys[k].intersection(pred_polys[n]).area
        # print(intersection)
      except:
        intersection = 0
      iou = intersection / (gt_area + pred_area - intersection)
      if math.isnan(iou):
        iou = 0
      overlaps[n, k] = iou
  return overlaps
