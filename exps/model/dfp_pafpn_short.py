#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
# Copyright (c) DAMO Academy, Alibaba Group and its affiliates.

import torch
import torch.nn as nn
import torch.nn.functional as F

from exps.model.darknet import CSPDarknet
from yolox.models.network_blocks import BaseConv, CSPLayer, DWConv


class DFPPAFPNSHORT(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("dark3", "dark4", "dark5"),
        in_channels=[256, 512, 1024],
        depthwise=False,
        act="silu",
        frame_num=2,
        with_short_cut=True,
        avg_channel=True, # 是否所有通道平均，False当前帧占一半通道
        dynamic_fusion=False, # 对除了当前帧的其它帧进一步融合，如果为True，则不使用aux layers
    ):
        super().__init__()
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features
        self.in_channels = in_channels
        self.frame_num = frame_num
        self.with_short_cut = with_short_cut
        self.avg_channel = avg_channel
        self.dynamic_fusion = dynamic_fusion
        if self.dynamic_fusion:
            assert frame_num > 1
        self.need_aux_layers = ([not (x * width % frame_num == 0) for x in in_channels]
                                if self.avg_channel else
                                [not (x // 2 * width % (frame_num - 1) == 0) for x in in_channels])
        Conv = DWConv if depthwise else BaseConv

        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        if self.avg_channel:
            self.jian2 = Conv(
                        in_channels=int(in_channels[0] * width),
                        out_channels=int(in_channels[0] * width) // frame_num,
                        ksize=1,
                        stride=1,
                        act=act,
                    )

            self.jian1 = Conv(
                        in_channels=int(in_channels[1] * width),
                        out_channels=int(in_channels[1] * width) // frame_num,
                        ksize=1,
                        stride=1,
                        act=act,
                    )

            self.jian0 = Conv(
                        in_channels=int(in_channels[2] * width),
                        out_channels=int(in_channels[2] * width) // frame_num,
                        ksize=1,
                        stride=1,
                        act=act,
                    )

            if not self.dynamic_fusion and self.need_aux_layers[0]:
                self.jian2_aux = Conv(
                            in_channels=int(in_channels[0] * width),
                            out_channels=int(in_channels[0] * width) // frame_num + int(in_channels[0] * width) % frame_num,
                            ksize=1,
                            stride=1,
                            act=act,
                        )
        
            if not self.dynamic_fusion and self.need_aux_layers[1]:
                self.jian1_aux = Conv(
                            in_channels=int(in_channels[1] * width),
                            out_channels=int(in_channels[1] * width) // frame_num + int(in_channels[1] * width) % frame_num,
                            ksize=1,
                            stride=1,
                            act=act,
                        )

            if not self.dynamic_fusion and self.need_aux_layers[2]:
                self.jian0_aux = Conv(
                            in_channels=int(in_channels[2] * width),
                            out_channels=int(in_channels[2] * width) // frame_num + int(in_channels[2] * width) % frame_num,
                            ksize=1,
                            stride=1,
                            act=act,
                        )

            if self.dynamic_fusion:
                self.jian2_dynamic_fusion = Conv(
                            in_channels=int(in_channels[0] * width) // frame_num * (frame_num - 1),
                            out_channels=int(in_channels[0] * width) - (int(in_channels[0] * width) // frame_num),
                            ksize=1,
                            stride=1,
                            act=act,
                        )

                self.jian1_dynamic_fusion = Conv(
                            in_channels=int(in_channels[1] * width) // frame_num * (frame_num - 1),
                            out_channels=int(in_channels[1] * width) - (int(in_channels[1] * width) // frame_num),
                            ksize=1,
                            stride=1,
                            act=act,
                        )

                self.jian0_dynamic_fusion = Conv(
                            in_channels=int(in_channels[2] * width) // frame_num * (frame_num - 1),
                            out_channels=int(in_channels[2] * width) - (int(in_channels[2] * width) // frame_num),
                            ksize=1,
                            stride=1,
                            act=act,
                        )
        else:
            self.jian2_static = Conv(
                        in_channels=int(in_channels[0] * width),
                        out_channels=int(in_channels[0] * width) // 2,
                        ksize=1,
                        stride=1,
                        act=act,
                    )

            self.jian1_static = Conv(
                        in_channels=int(in_channels[1] * width),
                        out_channels=int(in_channels[1] * width) // 2,
                        ksize=1,
                        stride=1,
                        act=act,
                    )

            self.jian0_static = Conv(
                        in_channels=int(in_channels[2] * width),
                        out_channels=int(in_channels[2] * width) // 2,
                        ksize=1,
                        stride=1,
                        act=act,
                    )

            self.jian2_dynamic = Conv(
                        in_channels=int(in_channels[0] * width),
                        out_channels=int(in_channels[0] * width) // 2 // (frame_num - 1),
                        ksize=1,
                        stride=1,
                        act=act,
                    )

            self.jian1_dynamic = Conv(
                        in_channels=int(in_channels[1] * width),
                        out_channels=int(in_channels[1] * width) // 2 // (frame_num - 1),
                        ksize=1,
                        stride=1,
                        act=act,
                    )

            self.jian0_dynamic = Conv(
                        in_channels=int(in_channels[2] * width),
                        out_channels=int(in_channels[2] * width) // 2 // (frame_num - 1),
                        ksize=1,
                        stride=1,
                        act=act,
                    )

            if not self.dynamic_fusion and self.need_aux_layers[0]:
                self.jian2_dynamic_aux = Conv(
                            in_channels=int(in_channels[0] * width),
                            out_channels=int(in_channels[0] * width) // 2 // (frame_num - 1) + int(in_channels[0] * width) // 2 % (frame_num - 1),
                            ksize=1,
                            stride=1,
                            act=act,
                        )
        
            if not self.dynamic_fusion and self.need_aux_layers[1]:
                self.jian1_dynamic_aux = Conv(
                            in_channels=int(in_channels[1] * width),
                            out_channels=int(in_channels[1] * width) // 2 // (frame_num - 1) + int(in_channels[1] * width) // 2 % (frame_num - 1),
                            ksize=1,
                            stride=1,
                            act=act,
                        )

            if not self.dynamic_fusion and self.need_aux_layers[2]:
                self.jian0_dynamic_aux = Conv(
                            in_channels=int(in_channels[2] * width),
                            out_channels=int(in_channels[2] * width) // 2 // (frame_num - 1) + int(in_channels[2] * width) // 2 % (frame_num - 1),
                            ksize=1,
                            stride=1,
                            act=act,
                        )

            if self.dynamic_fusion:
                self.jian2_dynamic_fusion = Conv(
                            in_channels=int(in_channels[0] * width) // 2 // (frame_num - 1) * (frame_num - 1),
                            out_channels=int(in_channels[0] * width) // 2,
                            ksize=1,
                            stride=1,
                            act=act,
                        )

                self.jian1_dynamic_fusion = Conv(
                            in_channels=int(in_channels[1] * width) // 2 // (frame_num - 1) * (frame_num - 1),
                            out_channels=int(in_channels[1] * width) // 2,
                            ksize=1,
                            stride=1,
                            act=act,
                        )

                self.jian0_dynamic_fusion = Conv(
                            in_channels=int(in_channels[2] * width) // 2 // (frame_num - 1) * (frame_num - 1),
                            out_channels=int(in_channels[2] * width) // 2,
                            ksize=1,
                            stride=1,
                            act=act,
                        )

    def off_forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """


        #  backbone
        rurrent_out_features = self.backbone(torch.split(input, 3, dim=1)[0])
        rurrent_features = [rurrent_out_features[f] for f in self.in_features]
        [rurrent_x2, rurrent_x1, rurrent_x0] = rurrent_features

        rurrent_fpn_out0 = self.lateral_conv0(rurrent_x0)  # 1024->512/32
        rurrent_f_out0 = F.interpolate(rurrent_fpn_out0, size=rurrent_x1.shape[2:4], mode='nearest')  # 512/16
        rurrent_f_out0 = torch.cat([rurrent_f_out0, rurrent_x1], 1)  # 512->1024/16
        rurrent_f_out0 = self.C3_p4(rurrent_f_out0)  # 1024->512/16

        rurrent_fpn_out1 = self.reduce_conv1(rurrent_f_out0)  # 512->256/16
        rurrent_f_out1 = F.interpolate(rurrent_fpn_out1, size=rurrent_x2.shape[2:4], mode='nearest')  # 256/8
        rurrent_f_out1 = torch.cat([rurrent_f_out1, rurrent_x2], 1)  # 256->512/8
        rurrent_pan_out2 = self.C3_p3(rurrent_f_out1)  # 512->256/8

        rurrent_p_out1 = self.bu_conv2(rurrent_pan_out2)  # 256->256/16
        rurrent_p_out1 = torch.cat([rurrent_p_out1, rurrent_fpn_out1], 1)  # 256->512/16
        rurrent_pan_out1 = self.C3_n3(rurrent_p_out1)  # 512->512/16

        rurrent_p_out0 = self.bu_conv1(rurrent_pan_out1)  # 512->512/32
        rurrent_p_out0 = torch.cat([rurrent_p_out0, rurrent_fpn_out0], 1)  # 512->1024/32
        rurrent_pan_out0 = self.C3_n4(rurrent_p_out0)  # 1024->1024/32

        #####

        support_pan_out2s = []
        support_pan_out1s = []
        support_pan_out0s = []
        for i in range(self.frame_num - 1):

            support_out_features = self.backbone(torch.split(input, 3, dim=1)[i+1])
            support_features = [support_out_features[f] for f in self.in_features]
            [support_x2, support_x1, support_x0] = support_features

            support_fpn_out0 = self.lateral_conv0(support_x0)  # 1024->512/32
            support_f_out0 = F.interpolate(support_fpn_out0, size=support_x1.shape[2:4], mode='nearest')  # 512/16
            support_f_out0 = torch.cat([support_f_out0, support_x1], 1)  # 512->1024/16
            support_f_out0 = self.C3_p4(support_f_out0)  # 1024->512/16

            support_fpn_out1 = self.reduce_conv1(support_f_out0)  # 512->256/16
            support_f_out1 = F.interpolate(support_fpn_out1, size=support_x2.shape[2:4], mode='nearest')  # 256/8
            support_f_out1 = torch.cat([support_f_out1, support_x2], 1)  # 256->512/8
            support_pan_out2 = self.C3_p3(support_f_out1)  # 512->256/8

            support_p_out1 = self.bu_conv2(support_pan_out2)  # 256->256/16
            support_p_out1 = torch.cat([support_p_out1, support_fpn_out1], 1)  # 256->512/16
            support_pan_out1 = self.C3_n3(support_p_out1)  # 512->512/16

            support_p_out0 = self.bu_conv1(support_pan_out1)  # 512->512/32
            support_p_out0 = torch.cat([support_p_out0, support_fpn_out0], 1)  # 512->1024/32
            support_pan_out0 = self.C3_n4(support_p_out0)  # 1024->1024/32

            support_pan_out2s.append(support_pan_out2)
            support_pan_out1s.append(support_pan_out1)
            support_pan_out0s.append(support_pan_out0)

        if not self.dynamic_fusion:
            if self.avg_channel:
                if self.with_short_cut:
                    pan_out2 = (torch.cat([self.jian2(rurrent_pan_out2), *[self.jian2(x) for x in support_pan_out2s]], dim=1) + rurrent_pan_out2 if not self.need_aux_layers[0] else 
                                torch.cat([self.jian2_aux(rurrent_pan_out2), *[self.jian2(x) for x in support_pan_out2s]], dim=1) + rurrent_pan_out2)
                    pan_out1 = (torch.cat([self.jian1(rurrent_pan_out1), *[self.jian1(x) for x in support_pan_out1s]], dim=1) + rurrent_pan_out1 if not self.need_aux_layers[1] else 
                                torch.cat([self.jian1_aux(rurrent_pan_out1), *[self.jian1(x) for x in support_pan_out1s]], dim=1) + rurrent_pan_out1)
                    pan_out0 = (torch.cat([self.jian0(rurrent_pan_out0), *[self.jian0(x) for x in support_pan_out0s]], dim=1) + rurrent_pan_out0 if not self.need_aux_layers[2] else 
                                torch.cat([self.jian0_aux(rurrent_pan_out0), *[self.jian0(x) for x in support_pan_out0s]], dim=1) + rurrent_pan_out0)
                else:
                    pan_out2 = (torch.cat([self.jian2(rurrent_pan_out2), *[self.jian2(x) for x in support_pan_out2s]], dim=1) if not self.need_aux_layers[0] else 
                                torch.cat([self.jian2_aux(rurrent_pan_out2), *[self.jian2(x) for x in support_pan_out2s]], dim=1))
                    pan_out1 = (torch.cat([self.jian1(rurrent_pan_out1), *[self.jian1(x) for x in support_pan_out1s]], dim=1) if not self.need_aux_layers[1] else 
                                torch.cat([self.jian1_aux(rurrent_pan_out1), *[self.jian1(x) for x in support_pan_out1s]], dim=1))
                    pan_out0 = (torch.cat([self.jian0(rurrent_pan_out0), *[self.jian0(x) for x in support_pan_out0s]], dim=1) if not self.need_aux_layers[2] else 
                                torch.cat([self.jian0_aux(rurrent_pan_out0), *[self.jian0(x) for x in support_pan_out0s]], dim=1))
            else:
                if self.with_short_cut:
                    pan_out2 = (torch.cat([self.jian2_static(rurrent_pan_out2), *[self.jian2_dynamic(x) for x in support_pan_out2s]], dim=1) + rurrent_pan_out2 if not self.need_aux_layers[0] else 
                                torch.cat([self.jian2_static(rurrent_pan_out2), self.jian2_dynamic_aux(support_pan_out2s[0]), *[self.jian2_dynamic(x) for x in support_pan_out2s[1:]]], dim=1) + rurrent_pan_out2)
                    pan_out1 = (torch.cat([self.jian1_static(rurrent_pan_out1), *[self.jian1_dynamic(x) for x in support_pan_out1s]], dim=1) + rurrent_pan_out1 if not self.need_aux_layers[1] else 
                                torch.cat([self.jian1_static(rurrent_pan_out1), self.jian1_dynamic_aux(support_pan_out1s[0]), *[self.jian1_dynamic(x) for x in support_pan_out1s[1:]]], dim=1) + rurrent_pan_out1)
                    pan_out0 = (torch.cat([self.jian0_static(rurrent_pan_out0), *[self.jian0_dynamic(x) for x in support_pan_out0s]], dim=1) + rurrent_pan_out0 if not self.need_aux_layers[2] else 
                                torch.cat([self.jian0_static(rurrent_pan_out0), self.jian0_dynamic_aux(support_pan_out0s[0]), *[self.jian0_dynamic(x) for x in support_pan_out0s[1:]]], dim=1) + rurrent_pan_out0)
                else:
                    pan_out2 = (torch.cat([self.jian2_static(rurrent_pan_out2), *[self.jian2_dynamic(x) for x in support_pan_out2s]], dim=1) if not self.need_aux_layers[0] else 
                                torch.cat([self.jian2_static(rurrent_pan_out2), self.jian2_dynamic_aux(support_pan_out2s[0]), *[self.jian2_dynamic(x) for x in support_pan_out2s[1:]]], dim=1))
                    pan_out1 = (torch.cat([self.jian1_static(rurrent_pan_out1), *[self.jian1_dynamic(x) for x in support_pan_out1s]], dim=1) if not self.need_aux_layers[1] else 
                                torch.cat([self.jian1_static(rurrent_pan_out1), self.jian1_dynamic_aux(support_pan_out1s[0]), *[self.jian1_dynamic(x) for x in support_pan_out1s[1:]]], dim=1))
                    pan_out0 = (torch.cat([self.jian0_static(rurrent_pan_out0), *[self.jian0_dynamic(x) for x in support_pan_out0s]], dim=1) if not self.need_aux_layers[2] else 
                                torch.cat([self.jian0_static(rurrent_pan_out0), self.jian0_dynamic_aux(support_pan_out0s[0]), *[self.jian0_dynamic(x) for x in support_pan_out0s[1:]]], dim=1))
        else: # dynamic_fusion=True
            if self.avg_channel:
                if self.with_short_cut:
                    pan_out2 = torch.cat([self.jian2(rurrent_pan_out2), self.jian2_dynamic_fusion(torch.cat([self.jian2(x) for x in support_pan_out2s], dim=1))], dim=1) + rurrent_pan_out2
                    pan_out1 = torch.cat([self.jian1(rurrent_pan_out1), self.jian1_dynamic_fusion(torch.cat([self.jian1(x) for x in support_pan_out1s], dim=1))], dim=1) + rurrent_pan_out1
                    pan_out0 = torch.cat([self.jian0(rurrent_pan_out0), self.jian0_dynamic_fusion(torch.cat([self.jian0(x) for x in support_pan_out0s], dim=1))], dim=1) + rurrent_pan_out0
                else:
                    pan_out2 = torch.cat([self.jian2(rurrent_pan_out2), self.jian2_dynamic_fusion(torch.cat([self.jian2(x) for x in support_pan_out2s], dim=1))], dim=1)
                    pan_out1 = torch.cat([self.jian1(rurrent_pan_out1), self.jian1_dynamic_fusion(torch.cat([self.jian1(x) for x in support_pan_out1s], dim=1))], dim=1)
                    pan_out0 = torch.cat([self.jian0(rurrent_pan_out0), self.jian0_dynamic_fusion(torch.cat([self.jian0(x) for x in support_pan_out0s], dim=1))], dim=1)
            else:
                if self.with_short_cut:
                    pan_out2 = torch.cat([self.jian2_static(rurrent_pan_out2), self.jian2_dynamic_fusion(torch.cat([self.jian2_dynamic(x) for x in support_pan_out2s], dim=1))], dim=1) + rurrent_pan_out2
                    pan_out1 = torch.cat([self.jian1_static(rurrent_pan_out1), self.jian1_dynamic_fusion(torch.cat([self.jian1_dynamic(x) for x in support_pan_out1s], dim=1))], dim=1) + rurrent_pan_out1
                    pan_out0 = torch.cat([self.jian0_static(rurrent_pan_out0), self.jian0_dynamic_fusion(torch.cat([self.jian0_dynamic(x) for x in support_pan_out0s], dim=1))], dim=1) + rurrent_pan_out0
                else:
                    pan_out2 = torch.cat([self.jian2_static(rurrent_pan_out2), self.jian2_dynamic_fusion(torch.cat([self.jian2_dynamic(x) for x in support_pan_out2s], dim=1))], dim=1)
                    pan_out1 = torch.cat([self.jian1_static(rurrent_pan_out1), self.jian1_dynamic_fusion(torch.cat([self.jian1_dynamic(x) for x in support_pan_out1s], dim=1))], dim=1)
                    pan_out0 = torch.cat([self.jian0_static(rurrent_pan_out0), self.jian0_dynamic_fusion(torch.cat([self.jian0_dynamic(x) for x in support_pan_out0s], dim=1))], dim=1)
        outputs = (pan_out2, pan_out1, pan_out0)
        rurrent_pan_outs = (rurrent_pan_out2, rurrent_pan_out1, rurrent_pan_out0)

        return outputs, rurrent_pan_outs

    def online_forward(self, input, buffer=None, node='star'):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """


        #  backbone
        rurrent_out_features = self.backbone(input)
        rurrent_features = [rurrent_out_features[f] for f in self.in_features]
        [rurrent_x2, rurrent_x1, rurrent_x0] = rurrent_features

        rurrent_fpn_out0 = self.lateral_conv0(rurrent_x0)  # 1024->512/32
        rurrent_f_out0 = F.interpolate(rurrent_fpn_out0, size=rurrent_x1.shape[2:4], mode='nearest')  # 512/16
        rurrent_f_out0 = torch.cat([rurrent_f_out0, rurrent_x1], 1)  # 512->1024/16
        rurrent_f_out0 = self.C3_p4(rurrent_f_out0)  # 1024->512/16

        rurrent_fpn_out1 = self.reduce_conv1(rurrent_f_out0)  # 512->256/16
        rurrent_f_out1 = F.interpolate(rurrent_fpn_out1, size=rurrent_x2.shape[2:4], mode='nearest')  # 256/8
        rurrent_f_out1 = torch.cat([rurrent_f_out1, rurrent_x2], 1)  # 256->512/8
        rurrent_pan_out2 = self.C3_p3(rurrent_f_out1)  # 512->256/8

        rurrent_p_out1 = self.bu_conv2(rurrent_pan_out2)  # 256->256/16
        rurrent_p_out1 = torch.cat([rurrent_p_out1, rurrent_fpn_out1], 1)  # 256->512/16
        rurrent_pan_out1 = self.C3_n3(rurrent_p_out1)  # 512->512/16

        rurrent_p_out0 = self.bu_conv1(rurrent_pan_out1)  # 512->512/32
        rurrent_p_out0 = torch.cat([rurrent_p_out0, rurrent_fpn_out0], 1)  # 512->1024/32
        rurrent_pan_out0 = self.C3_n4(rurrent_p_out0)  # 1024->1024/32

        #####
        if node=='star':
            pan_out2 = torch.cat([self.jian2(rurrent_pan_out2), self.jian2(rurrent_pan_out2)], dim=1) + rurrent_pan_out2
            pan_out1 = torch.cat([self.jian1(rurrent_pan_out1), self.jian1(rurrent_pan_out1)], dim=1) + rurrent_pan_out1
            pan_out0 = torch.cat([self.jian0(rurrent_pan_out0), self.jian0(rurrent_pan_out0)], dim=1) + rurrent_pan_out0
        elif node=='buffer':

            [support_pan_out2, support_pan_out1, support_pan_out0] = buffer

            pan_out2 = torch.cat([self.jian2(rurrent_pan_out2), self.jian2(support_pan_out2)], dim=1) + rurrent_pan_out2
            pan_out1 = torch.cat([self.jian1(rurrent_pan_out1), self.jian1(support_pan_out1)], dim=1) + rurrent_pan_out1
            pan_out0 = torch.cat([self.jian0(rurrent_pan_out0), self.jian0(support_pan_out0)], dim=1) + rurrent_pan_out0


        outputs = (pan_out2, pan_out1, pan_out0)

        buffer_ = (rurrent_pan_out2,rurrent_pan_out1,rurrent_pan_out0)

        return outputs, buffer_
    


    def forward(self, input, buffer=None, mode='off_pipe'):

        if mode=='off_pipe':
            # Glops caculate mode
            if input.size()[1] == 3:
                input = torch.cat([input, input], dim=1)
                output = self.off_forward(input)
            # offline train mode
            else:
                output = self.off_forward(input)
            
            return output
        
        elif mode=='on_pipe':
            # online star state
            if buffer == None:
                output, buffer_ = self.online_forward(input, node='star')
            # online inference
            else:
                assert len(buffer) == 3
                assert input.size()[1] == 3
                output, buffer_ = self.online_forward(input, buffer=buffer, node='buffer')
            
            return output, buffer_




