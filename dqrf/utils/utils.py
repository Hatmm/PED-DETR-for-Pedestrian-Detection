# ------------------------------------------------------------------------
# Modified by Matthieu Lin & Li Chuming
# Modified from Deformable-DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

from torch import Tensor
from typing import Optional, List
import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
import copy
from collections import deque
import numpy as np
from abc import ABC, abstractmethod
from PIL import Image
import os

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True



class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def _get_activation_fn(activation):
    """Return an activation functions given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class ImageMeta(object):
    """
    A class that provides a simple internface to encode and decode dict to save memory,
    by encoding instances into a string
    """
    @classmethod
    def encode(cls, dict_obj):
        """
        Arguments:
            dict_obj: single original data read from json

        Returns:
            list_data: [filename, [instances]]
        """

        instances = []
        for item in dict_obj.get('annotations', []):
            instance = item['bbox'] + item['vbbox'] + [item.get('category_id', None), item.get('is_ignored', False), item['area']]#, item['bbox_mode']]
            instances.append(instance)
        list_obj = [
            dict_obj['file_name'], instances, dict_obj['image_id']#, dict_obj['height'], dict_obj['width'])
        ]
        return list_obj

    @classmethod
    def decode(cls, list_obj):
        filename, instances, buckey = list_obj
        decoded_instances = []
        for ins in instances:
            ins = {
                'bbox': ins[:4],
                'vbbox': ins[4:8],
                'category_id': ins[8],
                'is_ignored': ins[9],
                'area': ins[10],
                # 'bbox_mode': ins[11]
            }
            decoded_instances.append(ins)
        dict_obj = {
            'file_name': filename,
            'image_id': buckey,
            # 'height': buckey[1],
            'annotations': decoded_instances,
            # 'width': buckey[2],
        }
        return dict_obj

class ImageReader(ABC):
    def __init__(self):
        super(ImageReader, self).__init__()

    @abstractmethod
    def __call__(self, image_id, image_dir_idx=0):
        raise NotImplementedError


def get_cur_image_dir(image_dir, idx):
    if isinstance(image_dir, list) or isinstance(image_dir, tuple):
        assert idx < len(image_dir)
        return image_dir[idx]
    return image_dir

class FileSystemPILReader(ImageReader):
    def __init__(self, image_dir='/', color_mode='RGB'):
        super(FileSystemPILReader, self).__init__()
        # self.image_dir = image_dir
        self.color_mode = color_mode
        assert color_mode == 'RGB', 'only RGB mode supported for pillow for now'

    # def image_directory(self):
    #     return self.image_dir

    def image_color(self):
        return self.color_mode

    def fake_image(self, *size):
        if len(size) == 0:
            size = (512, 512, 3)
        return Image.new(self.color_mode, size)

    def __call__(self, filename, image_dir_idx=0):
        # image_dir = get_cur_image_dir(self.image_dir, image_dir_idx)
        # filename = os.path.join(image_dir, filename)
        assert os.path.exists(filename), filename
        img = Image.open(filename).convert(self.color_mode)
        return img
