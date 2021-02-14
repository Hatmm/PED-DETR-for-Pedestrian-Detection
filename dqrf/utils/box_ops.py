# ------------------------------------------------------------------------
# Modified by Matthieu Lin
# Modified from Deformable-DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import torch
from torchvision.ops.boxes import box_area
import numpy as np

def xyxy2xywh(bbox):
    """
    bbox [#box, 4]
    """
    x1, y1, x2, y2 = torch.chunk(bbox, 4, dim=1)  # [#box, 1]
    w = x2 - x1
    h = y2 - y1
    return torch.cat((x1, y1, w, h), dim=-1)

def xywh2xyxy(bbox):
    x1, y1, w, h = torch.chunk(bbox, 4, dim=1) # [#box, 1]
    x2 = x1 + w
    y2 = y1 + h
    return torch.cat((x1, y1, x2, y2), dim=-1)


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def regression(src_bbox, loc):
    """
    Apply regression, by applying the offset coefficient
    Input:
        srx_bbox [N, R, 4]: coordinates of bounding boxes in XYXY.REL format
        loc [N, R, 4]: offset coefficient t_x, t_y, t_w, t_h in CxCyWH.REL format
    Return:
        dst_bbox[N, R, 4]: regressed anchor CXCYWH

    """
    # convert XYXY -> XYWH
    src_h = src_bbox[:, :, 3] - src_bbox[:, :, 1]
    src_w = src_bbox[:, :, 2] - src_bbox[:, :, 0]
    src_ctr_x = src_bbox[:, :, 0] + 0.5 * src_h
    src_ctr_y = src_bbox[:, :, 1] + 0.5 * src_w

    # unwrap cache
    dx = loc[:, :, 0]
    dy = loc[:, :, 1]
    dw = loc[:, :, 2]
    dh = loc[:, :, 3]

    # compute XYWH for dst_bbox
    ctr_x = dx * src_w + src_ctr_x
    ctr_y = dy * src_h + src_ctr_y
    w = torch.exp(dw) * src_w
    h = torch.exp(dh) * src_h

    # convert XYWH -> XYXY CXCYWH
    dst_bbox = torch.zeros(loc.shape, dtype=loc.dtype, device=src_bbox.device)
    dst_bbox[:, :, 0] = ctr_x#ctr_x - 0.5 * w
    dst_bbox[:, :, 1] = ctr_y #ctr_y - 0.5 * h
    dst_bbox[:, :, 2] = w #ctr_x + 0.5 * w
    dst_bbox[:, :, 3] = h #ctr_y + 0.5 * h

    return dst_bbox

def box_iof(boxes1, boxes2):
    """
    difference with iou is we divide total area of boxes1 by its intersection with boxes2
    """
    area1 = box_area(boxes1) #[N, ]
    area2 = box_area(boxes2) #[M, ]

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    # union = area1[:, None] + area2 - inter
    # import pdb
    # pdb.set_trace()
    # [N, M] / [N, ]
    iou = inter / area1.unsqueeze(-1).expand_as(inter)
    return iou

def calIof(b, g):
    """
    :param b: [N, 4]
    :param g: [M, 4]
    :return: [N, M]
    """
    area1 = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    inter_xmin = np.maximum(b[:, 0].reshape(-1, 1), g[:, 0].reshape(1, -1))
    inter_ymin = np.maximum(b[:, 1].reshape(-1, 1), g[:, 1].reshape(1, -1))
    inter_xmax = np.minimum(b[:, 2].reshape(-1, 1), g[:, 2].reshape(1, -1))
    inter_ymax = np.minimum(b[:, 3].reshape(-1, 1), g[:, 3].reshape(1, -1))
    inter_h = np.maximum(inter_xmax - inter_xmin, 0)
    inter_w = np.maximum(inter_ymax - inter_ymin, 0)
    inter_area = inter_h * inter_w
    return inter_area / np.maximum(area1, 1)

def calIoU(b1, b2):
    area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
    inter_xmin = np.maximum(b1[:, 0].reshape(-1, 1), b2[:, 0].reshape(1, -1))
    inter_ymin = np.maximum(b1[:, 1].reshape(-1, 1), b2[:, 1].reshape(1, -1))
    inter_xmax = np.minimum(b1[:, 2].reshape(-1, 1), b2[:, 2].reshape(1, -1))
    inter_ymax = np.minimum(b1[:, 3].reshape(-1, 1), b2[:, 3].reshape(1, -1))
    inter_h = np.maximum(inter_xmax - inter_xmin, 0)
    inter_w = np.maximum(inter_ymax - inter_ymin, 0)
    inter_area = inter_h * inter_w
    union_area1 = area1.reshape(-1, 1) + area2.reshape(1, -1)
    union_area2 = (union_area1 - inter_area)
    return inter_area / np.maximum(union_area2, 1)

def generalized_box_iou_(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area, union

def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)
