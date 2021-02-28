# ------------------------------------------------------------------------
# Modified by Matthieu Lin & Li Chuming
# Modified from Deformable-DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from dqrf.utils.box_ops import box_cxcywh_to_xyxy, box_iof, generalized_box_iou_, generalized_box_iou

class IgnoreMatcher_vbox(nn.Module):

    def __init__(self, cost_class=1, cost_bbox=1, cost_giou=1, ignore_iou_thresh=1.1):
        """Creates the matcher

        Params:
            cost_class: This is the indicesrelative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.ignore_iou_thresh = ignore_iou_thresh
        self.NEGATIVE_TARGET = -1
        self.IGNORE_TARGET = -2
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching
        Params:
                 "preds_prob": Tensor of dim [num_queries, num_classes] with the classification logits
                 "preds_boxes": Tensor of dim [num_queries, 4] with the predicted box coordinates
                 "gt_bboxes": Tensor of dim [num_target_boxes, 5] [x1, y1, x2, y2, label]
        Returns:
            target_gt
            overlaps
        """
        result = []

        for preds_prob, preds_boxes, t in zip(outputs['pred_logits'], outputs['pred_boxes'], targets):
            preds_prob = preds_prob.sigmoid()
            gt_bboxes = torch.cat((t['vboxes'], t['labels'].unsqueeze(-1).float()),dim=-1)
            ig_bboxes = t['iboxes']

            K = preds_prob.shape[0]

            target_gt = gt_bboxes.new_full((K,), self.NEGATIVE_TARGET, dtype=torch.int64)
            target_gt_iou = gt_bboxes.new_full((K,), 0)
            pos_mask = gt_bboxes.new_zeros((K,), dtype=torch.bool)

            if gt_bboxes.numel() > 0:
                tgt_ids = gt_bboxes[:, 4].long()
                tgt_bbox = gt_bboxes[:, :4]


                alpha = 0.25
                gamma = 2.0
                neg_cost_class = (1 - alpha) * (preds_prob ** gamma) * (-(1 - preds_prob + 1e-8).log())
                pos_cost_class = alpha * ((1 - preds_prob) ** gamma) * (-(preds_prob + 1e-8).log())
                cost_class = pos_cost_class - neg_cost_class


                cost_bbox = torch.cdist(preds_boxes, tgt_bbox, p=1)

                cost_giou, overlaps = generalized_box_iou_(box_cxcywh_to_xyxy(preds_boxes), box_cxcywh_to_xyxy(tgt_bbox))
                cost_giou = -cost_giou

                C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
                C = C.cpu()

                src_idx, tgt_idx = linear_sum_assignment(C)


                src_idx = torch.from_numpy(src_idx).to(device=gt_bboxes.device, dtype=torch.int64)
                tgt_idx = torch.from_numpy(tgt_idx).to(device=gt_bboxes.device, dtype=torch.int64)
                target_gt[src_idx] = tgt_idx
                target_gt_iou[src_idx] = overlaps[src_idx, tgt_idx]
                pos_mask[src_idx] = True

            if ig_bboxes.numel() > 0:
                ign_bbox = ig_bboxes[:, :4]

                overlaps = box_iof(box_cxcywh_to_xyxy(preds_boxes), box_cxcywh_to_xyxy(ign_bbox))
                dt_to_ig_max, _ = overlaps.max(dim=1)
                ignored_dt_mask = dt_to_ig_max >= self.ignore_iou_thresh
                ignored_dt_mask = (ignored_dt_mask ^ (ignored_dt_mask & pos_mask))
                target_gt[ignored_dt_mask] = self.IGNORE_TARGET
            result.append(target_gt)
        return result

class IgnoreMatcher(nn.Module):

    def __init__(self, cost_class=1, cost_bbox=1, cost_giou=1, ignore_iou_thresh=1.1, vbox=False, fast=False):
        """Creates the matcher

        Params:
            cost_class: This is the indicesrelative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.ignore_iou_thresh = ignore_iou_thresh
        self.vbox = vbox
        self.NEGATIVE_TARGET = -1
        self.IGNORE_TARGET = -2
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching
        Params:
                 "preds_prob": Tensor of dim [num_queries, num_classes] with the classification logits
                 "preds_boxes": Tensor of dim [num_queries, 4] with the predicted box coordinates
                 "gt_bboxes": Tensor of dim [num_target_boxes, 5] [x1, y1, x2, y2, label]
        Returns:
            list of tensor of dim [num_queries] with idx of corresponding GT, -1 for background, -2 for ignore
        """
        result = []

        for preds_prob, preds_boxes, t in zip(outputs['pred_logits'], outputs['pred_boxes'], targets):
            preds_prob = preds_prob.sigmoid()
            gt_bboxes = torch.cat((t['boxes'], t['labels'].unsqueeze(-1).float()),dim=-1)
            ig_bboxes = t['iboxes']

            K = preds_prob.shape[0]

            target_gt = gt_bboxes.new_full((K,), self.NEGATIVE_TARGET, dtype=torch.int64)
            target_gt_iou = gt_bboxes.new_full((K,), 0)
            pos_mask = gt_bboxes.new_zeros((K,), dtype=torch.bool)

            if gt_bboxes.numel() > 0:
                tgt_ids = gt_bboxes[:, 4].long()
                tgt_bbox = gt_bboxes[:, :4]


                alpha = 0.25
                gamma = 2.0
                neg_cost_class = (1 - alpha) * (preds_prob ** gamma) * (-(1 - preds_prob + 1e-8).log())
                pos_cost_class = alpha * ((1 - preds_prob) ** gamma) * (-(preds_prob + 1e-8).log())
                cost_class = pos_cost_class - neg_cost_class


                cost_bbox = torch.cdist(preds_boxes, tgt_bbox, p=1)

                cost_giou, overlaps = generalized_box_iou_(box_cxcywh_to_xyxy(preds_boxes), box_cxcywh_to_xyxy(tgt_bbox))
                cost_giou = -cost_giou

                C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
                C = C.cpu()

                src_idx, tgt_idx = linear_sum_assignment(C)

                src_idx = torch.from_numpy(src_idx).to(device=gt_bboxes.device, dtype=torch.int64)
                tgt_idx = torch.from_numpy(tgt_idx).to(device=gt_bboxes.device, dtype=torch.int64)
                target_gt[src_idx] = tgt_idx
                target_gt_iou[src_idx] = overlaps[src_idx, tgt_idx]
                pos_mask[src_idx] = True

            if ig_bboxes.numel() > 0:
                ign_bbox = ig_bboxes[:, :4]
                overlaps = box_iof(box_cxcywh_to_xyxy(preds_boxes), box_cxcywh_to_xyxy(ign_bbox))
                dt_to_ig_max, _ = overlaps.max(dim=1)
                ignored_dt_mask = dt_to_ig_max >= self.ignore_iou_thresh
                ignored_dt_mask = (ignored_dt_mask ^ (ignored_dt_mask & pos_mask))
                target_gt[ignored_dt_mask] = self.IGNORE_TARGET

            result.append(target_gt)
        return result

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]

        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]


        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))


        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

def build_matcher(cfg):
    matcher = {
        'vbox': IgnoreMatcher_vbox(cost_class=cfg.MODEL.DQRF_DETR.COST_CLASS, cost_bbox=cfg.MODEL.DQRF_DETR.COST_BBOX,
                              cost_giou=cfg.MODEL.DQRF_DETR.COST_GIOU,  ignore_iou_thresh=cfg.MODEL.DQRF_DETR.IGNORE_IOU_THRESHOLD),
        'fbox': IgnoreMatcher(cost_class=cfg.MODEL.DQRF_DETR.COST_CLASS, cost_bbox=cfg.MODEL.DQRF_DETR.COST_BBOX,
                              cost_giou=cfg.MODEL.DQRF_DETR.COST_GIOU, ignore_iou_thresh=cfg.MODEL.DQRF_DETR.IGNORE_IOU_THRESHOLD)
    }

    return matcher

def build_vanilla_matcher(cfg):
    return HungarianMatcher(cost_class=cfg.MODEL.DQRF_DETR.COST_CLASS, cost_bbox=cfg.MODEL.DQRF_DETR.COST_BBOX,
                            cost_giou=cfg.MODEL.DQRF_DETR.COST_GIOU)

