import numpy as np
import math


def cal_angle(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    lv1 = np.sqrt(v1.dot(v1))
    lv2 = np.sqrt(v2.dot(v2))
    cos_angle = v1.dot(v2) / (lv1 * lv2)
    return math.acos(cos_angle) / math.pi * 180


def check_rect(rect):
    """
    Check valid Rectangle
    """
    x, y, w, h = rect
    return x >= 0 and y >= 0 and w > 0 and h > 0


def is_rect(points):
    assert len(points) == 8
    for i in range(0, 8, 2):
        p1 = np.array([points[i], points[i + 1]])
        p2 = np.array([points[(i + 2) % 8], points[(i + 3) % 8]])
        p3 = np.array([points[(i + 4) % 8], points[(i + 5) % 8]])
        ang = cal_angle(p1, p2, p3)
        if ang < 85 or ang > 95:
            return False
    return True


def rect_intersect(r1, r2):
    """
    Intersect two rectangle
    """
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    if not check_rect((x1, y1, w1, h1)):
        return -1, -1, -1, -1
    if not check_rect((x2, y2, w2, h2)):
        return -1, -1, -1, -1
    x1_ = x1 + w1
    y1_ = y1 + h1
    x2_ = x2 + w2
    y2_ = y2 + h2
    x3 = max(x1, x2)
    y3 = max(y1, y2)
    x3_ = min(x1_, x2_)
    y3_ = min(y1_, y2_)
    if x3_ <= x3 or y3_ <= y3:
        return -1, -1, -1, -1
    return x3, y3, x3_ - x3, y3_ - y3


def rect_union(r1, r2):
    """
    Union two rectangle
    """
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    if not check_rect((x1, y1, w1, h1)):
        return x2, y2, w2, h2
    if not check_rect((x2, y2, w2, h2)):
        return x1, y1, w1, h1
    x1_ = x1 + w1
    y1_ = y1 + h1
    x2_ = x2 + w2
    y2_ = y2 + h2
    x3 = min(x1, x2)
    y3 = min(y1, y2)
    x3_ = max(x1_, x2_)
    y3_ = max(y1_, y2_)
    return x3, y3, x3_ - x3, y3_ - y3


def rect_area(r):
    return r[2] * r[3]


def quadrangle2bbox(quad):  # quad is numpy
    xmin = min(quad[:, 0])
    xmax = max(quad[:, 0])
    ymin = min(quad[:, 1])
    ymax = max(quad[:, 1])
    return xmin, ymin, xmax - xmin, ymax - ymin
