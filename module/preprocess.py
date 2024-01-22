# -*- coding: utf-8 -*-

from numbers import Integral

import cv2
import numpy as np

__all__ = ["detr_preprocess"]


def detr_preprocess(input,
                    target_size=[800, 1333],
                    interp=cv2.INTER_LINEAR,
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]):
    assert 2 <= input.ndim <= 3, ValueError(input.shape)

    if input.ndim == 2:
        input = np.tile(input[:, :, None], [1, 1, 3])

    if isinstance(target_size, Integral):
        target_size = [target_size, target_size]

    im_shape = input.shape

    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    target_size_min = np.min(target_size)
    target_size_max = np.max(target_size)

    im_scale = min(target_size_min / im_size_min,
                   target_size_max / im_size_max)

    resize_h = int(im_scale * float(im_shape[0]) + 0.5)
    resize_w = int(im_scale * float(im_shape[1]) + 0.5)

    im_scale_y = resize_h / im_shape[0]
    im_scale_x = resize_w / im_shape[1]

    if input.dtype != np.uint8:
        input = input.astype(np.uint8)

    input = cv2.resize(
        input,
        None,
        None,
        fx=im_scale_x,
        fy=im_scale_y,
        interpolation=interp)

    input = input.astype(np.float32)

    input /= 255.
    mean = np.array(mean)[None, None, :]
    std = np.array(std)[None, None, :]
    input -= mean
    input /= std

    if target_size_min < resize_h:
        out_shape = [target_size_max, target_size_min]
    else:
        out_shape = [target_size_min, target_size_max]

    padded_input = np.zeros([*out_shape, 3], dtype=np.float32)
    padded_mask = np.zeros(out_shape, dtype=np.float32)
    if (resize_h < out_shape[0]) and (resize_w == out_shape[1]):
        padded_input[:resize_h, :, :] = input
        padded_mask[resize_h:, :] = 1
    if (resize_w < out_shape[1]) and (resize_h == out_shape[0]):
        padded_input[:, :resize_w, :] = input
        padded_mask[:, resize_w:] = 1

    padded_input = padded_input.transpose([2, 0, 1])

    return padded_input, padded_mask, im_shape[0:2]


