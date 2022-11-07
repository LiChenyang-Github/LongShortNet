#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
"""
Data augmentation functionality. Passed as callable transformations to
Dataset classes.

The data augmentation procedures were interpreted from @weiliu89's SSD paper
http://arxiv.org/abs/1512.02325
"""

import cv2
import numpy as np
from yolox.utils import xyxy2cxcywh

import math
import random

def augment_hsv(img, hgain=0.015, sgain=0.7, vgain=0.4):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge(
        (cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))
    ).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed


def _mirror(image, boxes, mirror=False):
    _, width, _ = image.shape
    # if random.randrange(2):
    if mirror:
        image = image[:, ::-1]
        boxes = boxes.copy()
        boxes[:, 0::2] = width - boxes[:, 2::-2]
    return image, boxes


def preproc(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


class TrainTransform:
    def __init__(self, max_labels=50, hsv=True, flip=True):
        self.max_labels = max_labels
        self.hsv = hsv
        self.flip = flip

    def __call__(self, image, targets, input_dim, mirror=False):
        boxes = targets[:, :4].copy()
        labels = targets[:, 4].copy()
        if len(boxes) == 0:
            targets = np.zeros((self.max_labels, 5), dtype=np.float32)
            ### TODOlcy 这里有个小bug
            ### 当图片不存在gt的时候，跳过了mirror，造成t-1和t的mirror操作不一致，尽管这种case在数据集中出现的很少
            image, r_o = preproc(image, input_dim)
            return image, targets

        image_o = image.copy()
        targets_o = targets.copy()
        height_o, width_o, _ = image_o.shape
        boxes_o = targets_o[:, :4]
        labels_o = targets_o[:, 4]
        # bbox_o: [xyxy] to [c_x,c_y,w,h]
        boxes_o = xyxy2cxcywh(boxes_o)

        if self.hsv:
            augment_hsv(image)
        if self.flip:
            image_t, boxes = _mirror(image, boxes, mirror=mirror)
        else:
            image_t = image
        height, width, _ = image_t.shape
        image_t, r_ = preproc(image_t, input_dim)
        # boxes [xyxy] 2 [cx,cy,w,h]
        boxes = xyxy2cxcywh(boxes)
        boxes *= r_

        mask_b = np.minimum(boxes[:, 2], boxes[:, 3]) > 1
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b]

        if len(boxes_t) == 0:
            image_t, r_o = preproc(image_o, input_dim)
            boxes_o *= r_o
            boxes_t = boxes_o
            labels_t = labels_o

        labels_t = np.expand_dims(labels_t, 1)

        targets_t = np.hstack((labels_t, boxes_t))
        padded_labels = np.zeros((self.max_labels, 5))
        padded_labels[range(len(targets_t))[: self.max_labels]] = targets_t[
            : self.max_labels
        ]
        padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)
        return image_t, padded_labels


class DoubleTrainTransform:
    def __init__(self, max_labels=50, hsv=True, flip=True):
        self.max_labels = max_labels
        self.trasform1 = TrainTransform(max_labels=max_labels, hsv=hsv, flip=flip)
        self.trasform2 = TrainTransform(max_labels=max_labels, hsv=hsv, flip=flip)

    def __call__(self, image, targets, input_dim):
        a = random.randrange(2)
        img1, label1 = self.trasform1(image[0], targets[0], input_dim, mirror=a)
        img2, label2 = self.trasform2(image[1], targets[1], input_dim, mirror=a)
        return img1, img2, label1, label2


class TripleTrainTransform:
    def __init__(self, max_labels=50, hsv=True, flip=True):
        self.max_labels = max_labels
        self.trasform1 = TrainTransform(max_labels=max_labels, hsv=hsv, flip=flip)
        self.trasform2 = TrainTransform(max_labels=max_labels, hsv=hsv, flip=flip)
        self.trasform3 = TrainTransform(max_labels=max_labels, hsv=hsv, flip=flip)

    def __call__(self, image, targets, input_dim):
        a = random.randrange(2)
        img1, label1 = self.trasform1(image[0], targets[0], input_dim, mirror=a)
        img2, label2 = self.trasform2(image[1], targets[1], input_dim, mirror=a)
        # t-2没有对应的targets需要处理，这里使用targets[1]，仅为了方便调用self.trasform3函数
        img3, label3 = self.trasform3(image[2], targets[1], input_dim, mirror=a)
        return img1, img2, img3, label1, label2


class LongShortTrainTransform:
    def __init__(self, max_labels=50, hsv=True, flip=True, short_frame_num=2, long_frame_num=2):
        self.max_labels = max_labels
        self.short_frame_num = short_frame_num
        self.long_frame_num = long_frame_num
        self.short_transforms = [TrainTransform(max_labels=max_labels, hsv=hsv, flip=flip) for _ in range(short_frame_num)]
        self.long_transforms = [TrainTransform(max_labels=max_labels, hsv=hsv, flip=flip) for _ in range(long_frame_num)]

    def __call__(self, short_images, long_images, targets, input_dim):
        a = random.randrange(2)
        short_res = []
        long_res = []

        for i, trans_func in enumerate(self.short_transforms):
            cur_res = trans_func(short_images[i], 
                                 targets[min(i, 1)], 
                                 input_dim, 
                                 mirror=a)
            short_res.append(cur_res)

        if self.short_frame_num == 1:
            support_res = trans_func(short_images[0].copy(), targets[1], input_dim, mirror=a)

        for i, trans_func in enumerate(self.long_transforms):
            cur_res = trans_func(long_images[i], targets[1], input_dim, mirror=a)
            long_res.append(cur_res)

        short_imgs = [x[0] for x in short_res]
        long_imgs = [x[0] for x in long_res]

        return short_imgs, long_imgs, short_res[0][1], short_res[1][1] if self.short_frame_num > 1 else support_res[1]


class ValTransform:
    """
    Defines the transformations that should be applied to test PIL image
    for input into the network

    dimension -> tensorize -> color adj

    Arguments:
        resize (int): input dimension to SSD
        rgb_means ((int,int,int)): average RGB of the dataset
            (104,117,123)
        swap ((int,int,int)): final order of channels

    Returns:
        transform (transform) : callable transform to be applied to test/val
        data
    """

    def __init__(self, swap=(2, 0, 1)):
        self.swap = swap

    # assume input is cv2 img for now
    def __call__(self, img, res, input_size):
        img, _ = preproc(img, input_size, self.swap)
        return img, np.zeros((1, 5))



class DoubleValTransform:
    def __init__(self, swap=(2, 0, 1)):
        self.trasform1 = ValTransform(swap=swap)
        self.trasform2 = ValTransform(swap=swap)

    def __call__(self, img, res, input_size):
        img1, label1 = self.trasform1(img[0], res[0], input_size)
        img2, label2 = self.trasform2(img[1], res[1], input_size)
        return img1, img2, label1, label2


class TripleValTransform:
    def __init__(self, swap=(2, 0, 1)):
        self.trasform1 = ValTransform(swap=swap)
        self.trasform2 = ValTransform(swap=swap)
        self.trasform3 = ValTransform(swap=swap)

    def __call__(self, img, res, input_size):
        img1, label1 = self.trasform1(img[0], res[0], input_size)
        img2, label2 = self.trasform2(img[1], res[1], input_size)
        img3, label3 = self.trasform3(img[2], res[1], input_size)

        return img1, img2, img3, label1, label2


class LongShortValTransform:
    def __init__(self, swap=(2, 0, 1), short_frame_num=2, long_frame_num=2):
        self.short_frame_num = short_frame_num
        self.long_frame_num = long_frame_num
        self.short_transforms = [ValTransform(swap=swap) for _ in range(short_frame_num)]
        self.long_transforms = [ValTransform(swap=swap) for _ in range(long_frame_num)]

    def __call__(self, short_images, long_images, res, input_size):

        short_res = []
        long_res = []

        for i, trans_func in enumerate(self.short_transforms):
            cur_res = trans_func(short_images[i], res[min(i, 1)], input_size)
            short_res.append(cur_res)

        if self.short_frame_num == 1:
            support_res = trans_func(short_images[0].copy(), res[1], input_size)

        for i, trans_func in enumerate(self.long_transforms):
            cur_res = trans_func(long_images[i], res[1], input_size)
            long_res.append(cur_res)

        short_imgs = [x[0] for x in short_res]
        long_imgs = [x[0] for x in long_res]

        return short_imgs, long_imgs, short_res[0][1], short_res[1][1] if self.short_frame_num > 1 else support_res[1]