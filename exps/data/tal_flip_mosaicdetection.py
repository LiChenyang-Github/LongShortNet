#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
# Copyright (c) DAMO Academy, Alibaba Group and its affiliates.

import cv2
import random

import numpy as np

from yolox.utils import adjust_box_anns
from yolox.data.datasets.datasets_wrapper import Dataset


def get_mosaic_coordinate(mosaic_image, mosaic_index, xc, yc, w, h, input_h, input_w):
    # TODO update doc
    # index0 to top left part of image
    if mosaic_index == 0:
        x1, y1, x2, y2 = max(xc - w, 0), max(yc - h, 0), xc, yc
        small_coord = w - (x2 - x1), h - (y2 - y1), w, h
    # index1 to top right part of image
    elif mosaic_index == 1:
        x1, y1, x2, y2 = xc, max(yc - h, 0), min(xc + w, input_w * 2), yc
        small_coord = 0, h - (y2 - y1), min(w, x2 - x1), h
    # index2 to bottom left part of image
    elif mosaic_index == 2:
        x1, y1, x2, y2 = max(xc - w, 0), yc, xc, min(input_h * 2, yc + h)
        small_coord = w - (x2 - x1), 0, w, min(y2 - y1, h)
    # index2 to bottom right part of image
    elif mosaic_index == 3:
        x1, y1, x2, y2 = xc, yc, min(xc + w, input_w * 2), min(input_h * 2, yc + h)  # noqa
        small_coord = 0, 0, min(w, x2 - x1), min(y2 - y1, h)
    return (x1, y1, x2, y2), small_coord


class MosaicDetection(Dataset):
    """Detection dataset wrapper that performs mixup for normal dataset."""

    def __init__(
        self, dataset, img_size, mosaic=True, preproc=None,
        degrees=10.0, translate=0.1, scale=(0.5, 1.5), mscale=(0.5, 1.5),
        shear=2.0, perspective=0.0, enable_mixup=True,
        mosaic_prob=1.0, mixup_prob=1.0, *args
    ):
        """

        Args:
            dataset(Dataset) : Pytorch dataset object.
            img_size (tuple):
            mosaic (bool): enable mosaic augmentation or not.
            preproc (func):
            degrees (float):
            translate (float):
            scale (tuple):
            mscale (tuple):
            shear (float):
            perspective (float):
            enable_mixup (bool):
            *args(tuple) : Additional arguments for mixup random sampler.
        """
        super().__init__(img_size, mosaic=mosaic)
        self._dataset = dataset
        self.preproc = preproc
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.mixup_scale = mscale
        self.enable_mosaic = mosaic
        self.enable_mixup = enable_mixup
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob

    def __len__(self):
        return len(self._dataset)

    @Dataset.mosaic_getitem
    def __getitem__(self, idx):
        if self.enable_mosaic and random.random() < self.mosaic_prob:
            raise Exception("Not implemented.")
        else:
            self._dataset._input_dim = self.input_dim
            img, support_img, label, support_label, img_info, id_ = self._dataset.pull_item(idx)
            img, support_img, label, support_label = self.preproc((img, support_img), (label, support_label), self.input_dim)
            return np.concatenate((img, support_img), axis=0), (label, support_label), img_info, id_


class TripleMosaicDetection(Dataset):
    """Detection dataset wrapper that performs mixup for normal dataset."""

    def __init__(
        self, dataset, img_size, mosaic=True, preproc=None,
        degrees=10.0, translate=0.1, scale=(0.5, 1.5), mscale=(0.5, 1.5),
        shear=2.0, perspective=0.0, enable_mixup=True,
        mosaic_prob=1.0, mixup_prob=1.0, *args
    ):
        """

        Args:
            dataset(Dataset) : Pytorch dataset object.
            img_size (tuple):
            mosaic (bool): enable mosaic augmentation or not.
            preproc (func):
            degrees (float):
            translate (float):
            scale (tuple):
            mscale (tuple):
            shear (float):
            perspective (float):
            enable_mixup (bool):
            *args(tuple) : Additional arguments for mixup random sampler.
        """
        super().__init__(img_size, mosaic=mosaic)
        self._dataset = dataset
        self.preproc = preproc
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.mixup_scale = mscale
        self.enable_mosaic = mosaic
        self.enable_mixup = enable_mixup
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob

    def __len__(self):
        return len(self._dataset)

    @Dataset.mosaic_getitem
    def __getitem__(self, idx):
        if self.enable_mosaic and random.random() < self.mosaic_prob:
            raise Exception("Not implemented.")
        else:
            self._dataset._input_dim = self.input_dim
            img, support_img, support_neg_2_img, label, support_label, img_info, id_ = self._dataset.pull_item(idx)
            # ######################## visualization
            # import os
            # vis_dir = f"/root/StreamYOLO/data/vis/{id_[0]}"
            # if not os.path.isdir(vis_dir):
            #     os.makedirs(vis_dir)
            # img_path = os.path.join(vis_dir, "img.jpg")
            # support_img_path = os.path.join(vis_dir, "support_img.jpg")
            # support_neg_2_img_path = os.path.join(vis_dir, "support_neg_2_img.jpg")
            # for i in range(len(support_label)):
            #     x1, y1, x2, y2 = support_label[i, :4].astype(np.int32)
            #     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
            # cv2.imwrite(img_path, img)
            # cv2.imwrite(support_img_path, support_img)
            # cv2.imwrite(support_neg_2_img_path, support_neg_2_img)
            # ########################
            img, support_img, support_neg_2_img, label, support_label = self.preproc((img, support_img, support_neg_2_img), (label, support_label), self.input_dim)
            return np.concatenate((img, support_img, support_neg_2_img), axis=0), (label, support_label), img_info, id_


class LongShortMosaicDetection(Dataset):
    """Detection dataset wrapper that performs mixup for normal dataset."""

    def __init__(
        self, dataset, img_size, mosaic=True, preproc=None,
        degrees=10.0, translate=0.1, scale=(0.5, 1.5), mscale=(0.5, 1.5),
        shear=2.0, perspective=0.0, enable_mixup=True,
        mosaic_prob=1.0, mixup_prob=1.0, *args
    ):
        """

        Args:
            dataset(Dataset) : Pytorch dataset object.
            img_size (tuple):
            mosaic (bool): enable mosaic augmentation or not.
            preproc (func):
            degrees (float):
            translate (float):
            scale (tuple):
            mscale (tuple):
            shear (float):
            perspective (float):
            enable_mixup (bool):
            *args(tuple) : Additional arguments for mixup random sampler.
        """
        super().__init__(img_size, mosaic=mosaic)
        self._dataset = dataset
        self.preproc = preproc
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.mixup_scale = mscale
        self.enable_mosaic = mosaic
        self.enable_mixup = enable_mixup
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob

    def __len__(self):
        return len(self._dataset)

    @Dataset.mosaic_getitem
    def __getitem__(self, idx):
        if self.enable_mosaic and random.random() < self.mosaic_prob:
            raise Exception("Not implemented.")
        else:
            self._dataset._input_dim = self.input_dim
            short_imgs, long_imgs, label, support_label, img_info, id_ = self._dataset.pull_item(idx)
            short_imgs, long_imgs, label, support_label = self.preproc(short_imgs, long_imgs, (label, support_label), self.input_dim)
            # return np.concatenate((img, support_img), axis=0), (label, support_label), img_info, id_
            if len(long_imgs) > 0:
                return np.concatenate(short_imgs, axis=0), np.concatenate(long_imgs, axis=0), (label, support_label), img_info, id_
            else: # 不使用long支路的情况
                return np.concatenate(short_imgs, axis=0), np.zeros((0, )), (label, support_label), img_info, id_
            # return np.zeros(np.concatenate(short_imgs, axis=0).shape, dtype=np.float32), np.ones(np.concatenate(long_imgs, axis=0).shape, dtype=np.float32), (label, support_label), img_info, id_