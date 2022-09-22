#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2022 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

from exps.model.tal_head import TALHead
from exps.model.dfp_pafpn_long import DFPPAFPNLONG
from exps.model.dfp_pafpn_short import DFPPAFPNSHORT

from yolox.models.network_blocks import BaseConv


class YOLOXLONGSHORT(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(
        self, 
        long_backbone=None, 
        short_backbone=None, 
        head=None, 
        merge_form="add", 
        in_channels=[256, 512, 1024], 
        width=1.0, 
        act="silu"
    ):
        """Summary
        
        Args:
            long_backbone (None, optional): Description
            short_backbone (None, optional): Description
            head (None, optional): Description
            merge_form (str, optional): "add" or "concat"
            in_channels (list, optional): Description
        """
        super().__init__()
        if short_backbone is None:
            short_backbone = DFPPAFPNSHORT()
        if head is None:
            head = TALHead(20)

        self.long_backbone = long_backbone
        self.short_backbone = short_backbone
        self.head = head
        self.merge_form = merge_form
        self.in_channels = in_channels
        if merge_form == "concat":
            self.jian2 = BaseConv(
                        in_channels=int(in_channels[0] * width),
                        out_channels=int(in_channels[0] * width) // 2,
                        ksize=1,
                        stride=1,
                        act=act,
                    )

            self.jian1 = BaseConv(
                        in_channels=int(in_channels[1] * width),
                        out_channels=int(in_channels[1] * width) // 2,
                        ksize=1,
                        stride=1,
                        act=act,
                    )

            self.jian0 = BaseConv(
                        in_channels=int(in_channels[2] * width),
                        out_channels=int(in_channels[2] * width) // 2,
                        ksize=1,
                        stride=1,
                        act=act,
                    )

    def forward(self, x, targets=None, buffer=None, mode='off_pipe'):
        # fpn output content features of [dark3, dark4, dark5]
        assert mode in ['off_pipe', 'on_pipe']

        if mode == 'off_pipe':
            short_fpn_outs = self.short_backbone(x[0], buffer=buffer, mode='off_pipe')
            long_fpn_outs = self.long_backbone(x[1], buffer=buffer, mode='off_pipe') if self.long_backbone is not None else None
            if self.long_backbone is None:
                fpn_outs = short_fpn_outs
            else:
                if self.merge_form == "add":
                    fpn_outs = [x + y for x, y in zip(short_fpn_outs, long_fpn_outs)]
                elif self.merge_form == "concat":
                    fpn_outs_2 = torch.cat([self.jian2(short_fpn_outs[0]), self.jian2(long_fpn_outs[0])], dim=1)
                    fpn_outs_1 = torch.cat([self.jian1(short_fpn_outs[1]), self.jian1(long_fpn_outs[1])], dim=1)
                    fpn_outs_0 = torch.cat([self.jian0(short_fpn_outs[2]), self.jian0(long_fpn_outs[2])], dim=1)
                    fpn_outs = (fpn_outs_2, fpn_outs_1, fpn_outs_0)
                else:
                    raise Exception(f'merge_form must be in ["add", "concat"]')

            if self.training:
                assert targets is not None
                loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                    fpn_outs, targets, x
                )
                outputs = {
                    "total_loss": loss,
                    "iou_loss": iou_loss,
                    "l1_loss": l1_loss,
                    "conf_loss": conf_loss,
                    "cls_loss": cls_loss,
                    "num_fg": num_fg,
                }
            else:
                outputs = self.head(fpn_outs)

            return outputs
        elif mode == 'on_pipe':
            fpn_outs, buffer_ = self.backbone(x,  buffer=buffer, mode='on_pipe')
            outputs = self.head(fpn_outs)
            
            return outputs, buffer_




