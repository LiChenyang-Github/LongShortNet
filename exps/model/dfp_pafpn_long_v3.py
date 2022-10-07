#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import defaultdict

from exps.model.darknet import CSPDarknet
from yolox.models.network_blocks import BaseConv, CSPLayer, DWConv


class DFPPAFPNLONGV3(nn.Module):
    """
    相比DFPPAFPNLONG，直接指定输出的通道数
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
        # dynamic_fusion=False,
        out_channels=[((64, 128, 256), 1), ] # [((c11, c12, c13), n1), ((c21, c22, c23), n2), ...] c表示各层输出的通道数，n表示作用于几张frame，即不同的frame可能对应不同的conv

    ):
        super().__init__()
        self.in_features = in_features
        self.in_channels = in_channels
        self.frame_num = frame_num
        self.with_short_cut = with_short_cut
        self.out_channels = out_channels
        self.conv_group_num = len(out_channels)
        self.conv_group_dict = defaultdict(dict)
        assert self.frame_num == sum([x[1] for x in out_channels])
        Conv = DWConv if depthwise else BaseConv

        for i in range(self.conv_group_num):
            setattr(self,
                    f"group_{i}_jian2",
                    Conv(
                        in_channels=int(in_channels[0] * width),
                        out_channels=self.out_channels[i][0][0],
                        ksize=1,
                        stride=1,
                        act=act,
                    )
                )

            setattr(self,
                    f"group_{i}_jian1",
                    Conv(
                        in_channels=int(in_channels[1] * width),
                        out_channels=self.out_channels[i][0][1],
                        ksize=1,
                        stride=1,
                        act=act,
                    )
                )

            setattr(self,
                    f"group_{i}_jian0",
                    Conv(
                        in_channels=int(in_channels[2] * width),
                        out_channels=self.out_channels[i][0][2],
                        ksize=1,
                        stride=1,
                        act=act,
                    )
                )

    def off_forward(self, input, backbone_neck):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """


        # backbone
        rurrent_pan_out2, rurrent_pan_out1, rurrent_pan_out0 = backbone_neck(torch.split(input, 3, dim=1)[0])

        #####

        support_pan_out2s = []
        support_pan_out1s = []
        support_pan_out0s = []
        for i in range(self.frame_num - 1):

            support_pan_out2, support_pan_out1, support_pan_out0 = backbone_neck(torch.split(input, 3, dim=1)[i+1])

            support_pan_out2s.append(support_pan_out2)
            support_pan_out1s.append(support_pan_out1)
            support_pan_out0s.append(support_pan_out0)

        all_pan_out2s = [rurrent_pan_out2] + support_pan_out2s
        all_pan_out1s = [rurrent_pan_out1] + support_pan_out1s
        all_pan_out0s = [rurrent_pan_out0] + support_pan_out0s
        pan_out2s = []
        pan_out1s = []
        pan_out0s = []

        frame_start_id = 0
        for i in range(self.conv_group_num):
            group_frame_num = self.out_channels[i][1]
            for j in range(group_frame_num):
                frame_id = frame_start_id + j
                pan_out2s.append(getattr(self, f"group_{i}_jian2")(all_pan_out2s[frame_id]))
                pan_out1s.append(getattr(self, f"group_{i}_jian1")(all_pan_out1s[frame_id]))
                pan_out0s.append(getattr(self, f"group_{i}_jian0")(all_pan_out0s[frame_id]))
            frame_start_id += group_frame_num

        if self.with_short_cut:
            pan_out2 = torch.cat(pan_out2s, dim=1) + rurrent_pan_out2
            pan_out1 = torch.cat(pan_out1s, dim=1) + rurrent_pan_out1
            pan_out0 = torch.cat(pan_out0s, dim=1) + rurrent_pan_out0
        else:
            pan_out2 = torch.cat(pan_out2s, dim=1)
            pan_out1 = torch.cat(pan_out1s, dim=1)
            pan_out0 = torch.cat(pan_out0s, dim=1)

        outputs = (pan_out2, pan_out1, pan_out0)

        return outputs

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
    


    def forward(self, input, buffer=None, mode='off_pipe', backbone_neck=None):

        if mode=='off_pipe':
            # Glops caculate mode
            if input.size()[1] == 3:
                input = torch.cat([input, input], dim=1)
                output = self.off_forward(input, backbone_neck)
            # offline train mode
            else:
                output = self.off_forward(input, backbone_neck)
            
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




