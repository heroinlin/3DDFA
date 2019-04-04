# -*- coding: utf-8 -*-

import numpy as np


def change_box_order(boxes, order):
    """
    change box order between (xmin, ymin, xmax, ymax) and (center_x, center_y, width, height)

    Args:
        boxes: (numpy array), bounding boxes, size [N, 4]
        order: (str), 'xyxy2xywh' or 'xywh2xyxy'
    Returns:
        converted bounding boxes, size [N, 4]
    """
    assert order in ['xyxy2xywh', 'xywh2xyxy']
    a = boxes[:, :2]
    b = boxes[:, 2:]
    if order == 'xyxy2xywh':
        return np.hstack([(a + b) / 2, b - a])
    else:
        return np.hstack([a - b / 2, a + b / 2])


def box_intersection(boxes_a, boxes_b):
    assert isinstance(boxes_a, np.ndarray)
    assert isinstance(boxes_b, np.ndarray)

    size_a = boxes_a.shape[0]  # M
    size_b = boxes_b.shape[0]  # N
    if size_a == 0:
        return np.zeros(shape=[1, size_b], dtype=np.float32)
    if size_b == 0:
        return np.zeros(shape=[size_a, 1], dtype=np.float32)
    boxes_a = np.expand_dims(boxes_a, axis=1)
    boxes_a = np.repeat(boxes_a, repeats=size_b, axis=1)  # convert boxes_a shape to [M, N, 4]
    boxes_b = np.expand_dims(boxes_b, axis=0)
    boxes_b = np.repeat(boxes_b, repeats=size_a, axis=0)  # convert boxes_b shape to [M, N, 4]

    lt = np.maximum(boxes_a[:, :, :2], boxes_b[:, :, :2])
    br = np.minimum(boxes_a[:, :, 2:], boxes_b[:, :, 2:])
    inter = np.clip(br - lt, a_min=0, a_max=np.inf)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(boxes_a, boxes_b):
    assert isinstance(boxes_a, np.ndarray)
    assert isinstance(boxes_b, np.ndarray)

    size_a = boxes_a.shape[0]  # M
    size_b = boxes_b.shape[0]  # N

    inter = box_intersection(boxes_a, boxes_b)
    if size_a == 0 or size_b == 0:
        return inter
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_a = np.repeat(np.expand_dims(area_a, axis=1), repeats=size_b, axis=1)
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    area_b = np.repeat(np.expand_dims(area_b, axis=0), repeats=size_a, axis=0)
    union = area_a + area_b - inter
    return inter / union


def encode(default_boxes_target, default_boxes):
    """
    default_boxes_target and default_boxes are all [cx, cy, w, h] format
    """
    assert isinstance(default_boxes_target, np.ndarray)
    assert isinstance(default_boxes, np.ndarray)

    g_cxcy = (default_boxes_target[:, :2] - default_boxes[:, :2]) / default_boxes[:, 2:]
    g_wh = np.log(default_boxes_target[:, 2:] / default_boxes[:, 2:])
    return np.hstack([g_cxcy, g_wh])


def decode(predict_values, default_boxes):
    assert isinstance(predict_values, np.ndarray)
    assert isinstance(default_boxes, np.ndarray)

    predict_cx_cy = predict_values[:, :2] * default_boxes[:, 2:] + default_boxes[:, :2]
    predict_w_h = np.exp(predict_values[:, 2:]) * default_boxes[:, 2:]
    predict_tl = predict_cx_cy - predict_w_h / 2
    predict_br = predict_cx_cy + predict_w_h / 2
    return np.hstack([predict_tl, predict_br])


def main():
    pass


# if __name__ == "__main__":
#     main()
