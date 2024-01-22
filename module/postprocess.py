# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import softmax as np_softmax

__all__ = ["detr_postprocess"]


def box_cxcywh_to_xyxy(x):
    cxcy, wh = np.split(x, 2, axis=-1)
    return np.concatenate([cxcy - 0.5 * wh, cxcy + 0.5 * wh], axis=-1)

def detr_postprocess(pred_logits, pred_boxes, ori_shape):
    scores = np_softmax(pred_logits, axis=-1)[:, :, :-1]
    scores, labels = scores.max(axis=-1), scores.argmax(axis=-1)
    boxes = box_cxcywh_to_xyxy(pred_boxes)
    img_h, img_w = ori_shape
    scale_fct = np.array([img_w, img_h, img_w, img_h], dtype=np.float32)
    boxes = boxes * scale_fct[None, None]

    return scores, labels, boxes
