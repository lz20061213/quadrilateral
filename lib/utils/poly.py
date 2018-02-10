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
        pts = [(pred[i], pred[i + 1]) for i in range(0, len(pred) - 1, 2)]
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

def ploy_overlap(gt, pred):
    gt_pts = [(gt[i], gt[i + 1]) for i in range(0, len(gt), 2)]
    gt_poly = Polygon(gt_pts)
    gt_area = gt_poly.area

    pred_pts = [(pred[i], pred[i + 1]) for i in range(0, len(pred), 2)]
    pred_poly = Polygon(pred_pts)
    pred_area = pred_poly.area

    intersection = gt_poly.intersection(pred_poly).area
    iou = intersection / (gt_area + pred_area - intersection)
    if math.isnan(iou):
        iou = 0

    return iou

def pointCmp(pointA, pointB, center):
    if pointA[0] >= 0 and pointB[0] <= 0:
        return True
    if pointA[0] == 0 and pointB[0] == 0:
        return pointA[1] > pointB[1]
    # cross product of OA and OB, O presents the center
    # det = ax*by-ay*bx = |a|*|b|*sin(theta)
    det = (pointA[0] - center[0]) * (pointB[1] - center[1]) - (pointA[1] - center[1]) * (pointB[0] - center[0])
    if det < 0:
        return True
    if det > 0:
        return False
    # process if det=0, we use the distance of OA, OB
    dOA = (pointA[0] - center[0]) * (pointA[0] - center[0]) + (pointA[1] - center[1]) * (pointA[1] - center[1])
    dOB = (pointB[0] - center[0]) * (pointB[0] - center[0]) + (pointB[1] - center[1]) * (pointB[1] - center[1])
    return dOA > dOB

def poly_reorder(points):
    # here we use clockwise to sort the points
    center = [sum([p[0] for p in points]) / len(points), sum([p[1] for p in points]) / len(points)]
    #print(center)
    for i in range(len(points)):
        for j in range(len(points) - i - 1):
            #print((i, j + i + 1))
            if pointCmp(points[i], points[j + i + 1], center):
                temp = points[i]
                points[i] = points[j + i + 1]
                points[j + i + 1] = temp
    return points

# covert polygon points to circumscribed retangle
def poly2circu_rect(points, isNormlised=False):
    minx = min(points[0::2])
    maxx = max(points[0::2])
    miny = min(points[1::2])
    maxy = max(points[1::2])
    cx = (minx+maxx)/2
    cy = (miny+maxy)/2
    pw1 = points[0] - cx
    ph1 = points[1] - cy
    pw2 = points[2] - cx
    ph2 = points[3] - cy
    pw3 = points[4] - cx
    ph3 = points[5] - cy
    pw4 = points[6] - cx
    ph4 = points[7] - cy
    circu_rect = [cx, cy, pw1, ph1, pw2, ph2, pw3, ph3, pw4, ph4]
    if isNormlised:
        w = maxx - minx
        h = maxy - miny
        circu_rect[0::2] /= w
        circu_rect[1::2] /= h
    return circu_rect

if __name__ == '__main__':
    '''
    gts = [[0, 1, 2, 2, 3, 1, 2, 0]]
    preds = [[1, 0.5, 1, 1, 3, 1, 3, -0.5]]
    overlaps = poly_overlaps(gts, preds)
    print overlaps
    '''
    gt = [0, 1, 2, 2, 3, 1, 2, 0]
    pred = [1, 0.5, 1, 1, 3, 1, 3, -0.5]
