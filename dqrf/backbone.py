# ------------------------------------------------------------------------
# Modified by Matthieu Lin
# Modified from Deformable-DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
from detectron2.modeling import build_backbone
from detectron2.modeling import BACKBONE_REGISTRY
import detectron2.utils.comm as comm
from dqrf.utils.utils import NestedTensor

from dqrf.positional_encoding import build_position_encoding


__all__ = ["build_deformable_detr_backbone"]

class MaskedBackbone(nn.Module):
    def __init__(self, cfg):
        super(MaskedBackbone, self).__init__()
        self.backbone = build_backbone(cfg)
        self.strides = [self.backbone.output_shape()[key].stride for key in self.backbone.output_shape().keys()]
        self.num_channels = self.backbone.output_shape()[cfg.MODEL.RESNETS.OUT_FEATURES[-1]].channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.backbone(tensor_list.tensors)
        out: Dict[str, NestedTensor] = []
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out.insert(0, NestedTensor(x, mask))
        return out


class Conv2d_GN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=32, bias=False):
        super(Conv2d_GN, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        self.norm = nn.GroupNorm(
            num_groups=groups,
            num_channels=out_channels
        )
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)

        return x


class Neck(nn.Module):
    def __init__(self, in_channels=(512, 1024, 2048), feature_size=256):
        super(Neck, self).__init__()
        """
        in_channels: output channel size of each channel
        feature_size: projected channel size
        """
        self.feature_size = feature_size
        self.num_outs = 4
        self.n_layers = len(in_channels)
        # add lateral layers
        self.lateral_convs = self._makes_conv(in_channels, kernel_size=1, stride=1, padding=0)

        # self.fpn_convs = nn.Conv2d(in_channels[-1], feature_size, kernel_size=3, stride=2, padding=1)
        self.fpn_convs = Conv2d_GN(in_channels[-1], feature_size, kernel_size=3, stride=2, padding=1)

        self._reset_parameters()
    def _reset_parameters(self):
        for p in self.parameters():
            if isinstance(p, nn.Conv2d):
            # if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def _makes_conv(self, in_channels, kernel_size, stride, padding):
        layer = []
        for i in range(len(in_channels)):
            layer.insert(0, Conv2d_GN(in_channels[i], self.feature_size, kernel_size, stride, padding))
            # layer.insert(0, nn.Conv2d(in_channels[i], self.feature_size, kernel_size, stride, padding))

        # [2048, 1024, 512]
        return nn.ModuleList(layer)

    def forward(self, x):
        """
        :param x: list of output from backbone starting from last layer
        :return: list of NestedTensor, output values at each of the layers in descending order (first elt is biggest feature map)
        """
        p_results = []

        for i in range(0, self.n_layers):

            #retrieve lateral input
            x_lateral = self.lateral_convs[i](x[i].tensors)

            p_results.insert(0, NestedTensor(x_lateral, x[i].mask))

        p6 = self.fpn_convs(x[0].tensors)
        m = F.interpolate(x[0].mask[None].float(), size=p6.shape[-2:]).to(torch.bool)[0]
        p_results.append(NestedTensor(p6, m))

        return p_results

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding, neck=None):
        super().__init__(backbone, position_embedding)
        self.neck = neck
        self.strides = backbone.strides
        if neck is not None:
            for _ in range(neck.num_outs - len(self.strides)):
                self.strides.append(self.strides[-1] * 2)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)  # Backbone: return list of NestedTensor for each layer
        if self.neck is not None:
            xs = self.neck(xs)

        out: List[NestedTensor] = []
        pos = []
        for x in xs:  # for all output of backbone
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))  # perform forward pass of positional encoder

        return out, pos




def build_deformable_detr_backbone(cfg):
    position_embedding = build_position_encoding(cfg)
    neck = Neck()
    backbone = MaskedBackbone(cfg)
    model = Joiner(backbone, position_embedding, neck)
    model.num_channels = backbone.num_channels if neck is None else 256
    return model



