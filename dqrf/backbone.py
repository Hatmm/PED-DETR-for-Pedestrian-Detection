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
            # p_results.append(NestedTensor(x_lateral, x[i].mask))

        p6 = self.fpn_convs(x[0].tensors)
        m = F.interpolate(x[0].mask[None].float(), size=p6.shape[-2:]).to(torch.bool)[0]
        p_results.append(NestedTensor(p6, m))
        # p_results.insert(0, NestedTensor(p6, m))

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
        # if self[2] is not None:
        #     xs = self[2](xs)  # FPN: return list of NestedTensor from biggest map to smallest
        out: List[NestedTensor] = []
        pos = []
        for x in xs:  # for all output of backbone
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))  # perform forward pass of positional encoder

        return out, pos

# class BackboneBase(nn.Module):
#
#     def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
#         super().__init__()
#         for name, parameter in backbone.named_parameters():
#             # We follow standard practice in Faster R-CNN models, and keep the first resnet block frozen, and fine-tune the other 3 blocks
#             if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
#                 parameter.requires_grad_(False)
#         if return_interm_layers:
#             # return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
#             return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
#             self.strides = [8, 16, 32]
#         else:
#             return_layers = {'layer4': "0"}
#             self.strides = [32]
#         self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
#         self.num_channels = num_channels
#
#     def forward(self, tensor_list: NestedTensor):
#         xs = self.body(tensor_list.tensors)
#         out: Dict[str, NestedTensor] = []#{}
#         for name, x in xs.items():
#             m = tensor_list.mask
#             assert m is not None
#             mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
#             out.insert(0, NestedTensor(x, mask))
#             # out[name] = NestedTensor(x, mask)
#         return out
#
#
#
# class Backbone(BackboneBase):
#     """ResNet backbone with frozen BatchNorm."""
#     def __init__(self, name: str,
#                  train_backbone: bool,
#                  return_interm_layers: bool,
#                  dilation: bool,
#                  norm):
#         backbone = getattr(torchvision.models, name)(
#             replace_stride_with_dilation=[False, False, dilation],
#             pretrained=comm.is_main_process(), norm_layer=norm)
#         num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
#         super().__init__(backbone, train_backbone, num_channels, return_interm_layers)
#
# class Conv2d_GN(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=32, bias=False):
#         super(Conv2d_GN, self).__init__()
#         self.conv = nn.Conv2d(
#             in_channels=in_channels,
#             out_channels=out_channels,
#             kernel_size=kernel_size,
#             stride=stride,
#             padding=padding,
#             bias=bias
#         )
#         self.norm = nn.GroupNorm(
#             num_groups=groups,
#             num_channels=out_channels
#         )
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.norm(x)
#
#         return x
#
#
# class Neck(nn.Module):
#     def __init__(self, in_channels=(512, 1024, 2048), feature_size=256):
#         super(Neck, self).__init__()
#         """
#         in_channels: output channel size of each channel
#         feature_size: projected channel size
#         """
#         self.feature_size = feature_size
#         self.num_outs = 4
#         self.n_layers = len(in_channels)
#         # add lateral layers
#         self.lateral_convs = self._makes_conv(in_channels, kernel_size=1, stride=1, padding=0)
#
#         # self.fpn_convs = nn.Conv2d(in_channels[-1], feature_size, kernel_size=3, stride=2, padding=1)
#         self.fpn_convs = Conv2d_GN(in_channels[-1], feature_size, kernel_size=3, stride=2, padding=1)
#
#         self._reset_parameters()
#     def _reset_parameters(self):
#         for p in self.parameters():
#             if isinstance(p, nn.Conv2d):
#             # if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)
#
#
#     def _makes_conv(self, in_channels, kernel_size, stride, padding):
#         layer = []
#         for i in range(len(in_channels)):
#             layer.insert(0, Conv2d_GN(in_channels[i], self.feature_size, kernel_size, stride, padding))
#             # layer.insert(0, nn.Conv2d(in_channels[i], self.feature_size, kernel_size, stride, padding))
#
#         # [2048, 1024, 512]
#         return nn.ModuleList(layer)
#
#     def forward(self, x):
#         """
#         :param x: list of output from backbone starting from last layer
#         :return: list of NestedTensor, output values at each of the layers in descending order (first elt is biggest feature map)
#         """
#         p_results = []
#
#         for i in range(0, self.n_layers):
#
#             #retrieve lateral input
#             x_lateral = self.lateral_convs[i](x[i].tensors)
#
#             p_results.insert(0, NestedTensor(x_lateral, x[i].mask))
#             # p_results.append(NestedTensor(x_lateral, x[i].mask))
#
#         p6 = self.fpn_convs(x[0].tensors)
#         m = F.interpolate(x[0].mask[None].float(), size=p6.shape[-2:]).to(torch.bool)[0]
#         p_results.append(NestedTensor(p6, m))
#         # p_results.insert(0, NestedTensor(p6, m))
#
#         return p_results
#
# class Joiner(nn.Sequential):
#     def __init__(self, backbone, position_embedding, neck=None):
#         super().__init__(backbone, position_embedding)
#         self.neck = neck
#         self.strides = backbone.strides
#         if neck is not None:
#             for _ in range(neck.num_outs - len(self.strides)):
#                 self.strides.append(self.strides[-1] * 2)
#
#     def forward(self, tensor_list: NestedTensor):
#         xs = self[0](tensor_list)  # Backbone: return list of NestedTensor for each layer
#         if self.neck is not None:
#             xs = self.neck(xs)
#         # if self[2] is not None:
#         #     xs = self[2](xs)  # FPN: return list of NestedTensor from biggest map to smallest
#         out: List[NestedTensor] = []
#         pos = []
#         for x in xs:  # for all output of backbone
#             out.append(x)
#             # position encoding
#             pos.append(self[1](x).to(x.tensors.dtype))  # perform forward pass of positional encoder
#
#         return out, pos
# class FrozenBatchNorm2d(torch.nn.Module):
#     """
#     BatchNorm2d where the batch statistics and the affine parameters are fixed.
#
#     Copy-paste from torchvision.misc.ops with added eps before rqsrt,
#     without which any other models_FPN than torchvision.models_FPN.resnet[18,34,50,101]
#     produce nans.
#     """
#
#     def __init__(self, n):
#         super(FrozenBatchNorm2d, self).__init__()
#         self.register_buffer("weight", torch.ones(n))
#         self.register_buffer("bias", torch.zeros(n))
#         self.register_buffer("running_mean", torch.zeros(n))
#         self.register_buffer("running_var", torch.ones(n))
#
#     def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
#                               missing_keys, unexpected_keys, error_msgs):
#         num_batches_tracked_key = prefix + 'num_batches_tracked'
#         if num_batches_tracked_key in state_dict:
#             del state_dict[num_batches_tracked_key]
#
#         super(FrozenBatchNorm2d, self)._load_from_state_dict(
#             state_dict, prefix, local_metadata, strict,
#             missing_keys, unexpected_keys, error_msgs)
#
#     def forward(self, x):
#         # move reshapes to the beginning
#         # to make it fuser-friendly
#         w = self.weight.reshape(1, -1, 1, 1)
#         b = self.bias.reshape(1, -1, 1, 1)
#         rv = self.running_var.reshape(1, -1, 1, 1)
#         rm = self.running_mean.reshape(1, -1, 1, 1)
#         eps = 1e-5
#         scale = w * (rv + eps).rsqrt()
#         bias = b - rm * scale
#         return x * scale + bias


def build_deformable_detr_backbone(cfg):
    position_embedding = build_position_encoding(cfg)
    neck = Neck()
    backbone = MaskedBackbone(cfg)
    # backbone = Backbone("resnet50", True, True, False,
    #                     FrozenBatchNorm2d)
    model = Joiner(backbone, position_embedding, neck)
    model.num_channels = backbone.num_channels if neck is None else 256
    return model



