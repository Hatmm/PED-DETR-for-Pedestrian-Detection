# ------------------------------------------------------------------------
# Modified by Matthieu Lin
# Modified from Deformable-DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

from torch import nn
import torch
import copy
import torch.nn.functional as F
from dqrf.utils.box_ops import generalized_box_iou, box_cxcywh_to_xyxy
from dqrf.utils.utils import is_dist_avail_and_initialized
import detectron2.utils.comm as comm

def sigmoid_focal_loss(inputs, targets, num_boxes,ignore, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        ignore: tuple (batch_idx, corresponding ignore tensor idx)
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        We per- form the normalization by the number of assigned anchors,
        not total anchors, since the vast majority of anchors are easy negatives and receive negligible loss values under the focal loss.
        hence after mean(1) multiply again by number of anchors
        Loss tensor
    """
    prob = inputs.sigmoid()
    # [bs, nquery, nclass]
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    ignore_factor = torch.ones_like(loss)
    if len(ignore[0]) > 0:
        ignore_factor[ignore] = 0

    # [bs, nquery, #class]
    loss *= ignore_factor
    return loss.mean(1).sum() / num_boxes

class SetCriterion(nn.Module):
    """
    only changed cost class and cls loss coeff to 2
    """

    def __init__(self, cfg, matcher):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = cfg.MODEL.DQRF_DETR.NUM_CLASSES
        self.matcher = matcher
        self.weight_dict = {'loss_ce': cfg.MODEL.DQRF_DETR.COST_CLASS,
                            'loss_bbox': cfg.MODEL.DQRF_DETR.COST_BBOX,
                            'loss_giou': cfg.MODEL.DQRF_DETR.COST_GIOU,
                            # 'loss_contrastive': 1
                            }
        # TODO this is a hack
        if cfg.MODEL.DQRF_DETR.AUX_LOSS:
            aux_weight_dict = {}
            for i in range(cfg.MODEL.DQRF_DETR.NUM_DECODER_LAYERS - 1):
                aux_weight_dict.update({k + f'_{i}': v for k, v in self.weight_dict.items()})
            self.weight_dict.update(aux_weight_dict)

        self.losses = ['labels', 'boxes']#, 'contrastive']
        self.focal_alpha = cfg.MODEL.DQRF_DETR.FOCAL_ALPHA
        self.gammma = cfg.MODEL.DQRF_DETR.GAMMA
        self.v_match = cfg.MODEL.DQRF_DETR.V_MATCH


    def loss_labels(self, outputs, targets, indices, num_boxes):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        indices is list len batch size containing tensor [#query] containing the ground truth idx for each query
        """

        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        # gt idx for each query matched , len is batch size
        tgt_idx = [i[torch.where(i >= 0)] for i in indices]
        target_classes_o = torch.cat([t['labels'][j] for t, j in zip(targets, tgt_idx)])
        # [bs, query]
        target_classes = torch.zeros(src_logits.shape[:2], dtype=torch.int64, device=src_logits.device)

        target_classes[idx] = target_classes_o
        ignore_idx = self._get_tgt_permutation_idx(indices)

        loss_ce = sigmoid_focal_loss(src_logits, target_classes.unsqueeze(-1).float(), num_boxes, alpha=self.focal_alpha, gamma=self.gammma, ignore=ignore_idx) * src_logits.shape[1]

        losses = {'loss_ce': loss_ce}

        return losses

    # def loss_contrastive(self, outputs, targets, indices, num_boxes):
    #     """
    #     :param outputs: N, L, E
    #     :param targets: a dict containing labels and boxes
    #     :param indices: list of len batch size [L, ]
    #     :param num_boxes:
    #     :return: contrastive loss for all positives
    #     """
    #     src_logits = outputs['pred_logits']
    #     src_queries_emb = outputs['queries']
    #     idx = self._get_src_permutation_idx(indices) # (batch_idx, query_idx)
    #     # gt idx for each query matched , len is batch size
    #     tgt_idx = [i[torch.where(i >= 0)] for i in indices]
    #     target_classes_o = torch.cat([t['labels'][j] for t, j in zip(targets, tgt_idx)])
    #     # [bs, query]
    #     target_classes = torch.zeros(src_logits.shape[:2], dtype=torch.int64, device=src_logits.device)
    #
    #     target_classes[idx] = target_classes_o
    #
    #     # [N, L, L]
    #     sim_score = src_queries_emb @ src_queries_emb.transpose(1,2)
    #
    #     # pos_count = src_queries_emb[idx].size(0)
    #     # diag_mask = torch.ones((pos_count,pos_count)) - torch.eye(pos)
    #     #
    #     # sim_score = torch.div(
    #     #     torch.matmul(src_queries_emb[idx], src_queries_emb[idx].T), 0.07
    #     # )
    #     # sim_score_max = torch.max(sim_score, dim=1, keepdim=True)[0]
    #     # sim_score -= sim_score_max.detach()
    #
    #     return 0

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.

        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        tgt_idx = [i[torch.where(i >= 0)[0]] for i in indices]
        target_boxes = torch.cat([t['boxes'][i] for t, i in zip(targets, tgt_idx)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')


        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses


    def loss_boxes_v(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """

        assert 'pred_boxes' in outputs

        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        tgt_idx = [i[torch.where(i >= 0)[0]] for i in indices]
        target_boxes = torch.cat([t['vboxes'][i] for t, i in zip(targets, tgt_idx)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        # indices is list len batch size containing tensor [#query] containing the target idx for each query
        # batch idx for each query selected, and idx of each matched query
        batch_idx = torch.cat([torch.full_like(src[torch.where(src>=0)[0]] , i) for i, src in enumerate(indices)])
        src_idx = torch.cat([torch.where(src>=0)[0] for src in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # return ignore idx a tuple (batch idx, idx query to ignore)
        batch_idx = torch.cat([torch.full_like(src[torch.where(src==-2)[0]] , i) for i, src in enumerate(indices)])
        tgt_idx = torch.cat([torch.where(src==-2)[0] for src in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'vboxes': self.loss_boxes_v,
            'contrastive': self.loss_contrastive
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes)


    def forward(self, outputs, targets):
        """
        This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
            match vbox with vbox and match vbox.detach() + offset with fbox
        """

        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        indices = self.matcher['fbox'](outputs_without_aux, targets)


        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / comm.get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:

            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            # 0 , 1, 2, 3, 4
            for i, aux_outputs in enumerate(outputs['aux_outputs']):


                if i <= self.v_match:
                    indices = self.matcher['vbox'](aux_outputs, targets)
                else:
                    indices = self.matcher['fbox'](aux_outputs, targets)

                for loss in self.losses:
                    if i <= self.v_match and loss == 'boxes':
                        loss = 'vboxes'

                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)


        return losses



