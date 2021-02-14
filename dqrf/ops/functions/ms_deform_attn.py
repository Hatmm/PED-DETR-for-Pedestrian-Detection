# ------------------------------------------------------------------------
# Modified by Matthieu Lin
# Modified from Deformable-DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

from torch import nn
import torch
import torch.nn.functional as F
import copy
from torch.nn.init import xavier_uniform_, constant_
import dqrf.utils.box_ops as box_ops

def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class SamplingEncAttention(nn.Module):
    """encoder attn"""
    def __init__(self, d_model=512, dec_sampling_heads=8, dec_sampling_points=1, feature_levels=4):
        super(SamplingEncAttention, self).__init__()
        self.d_model = d_model
        self.dec_sampling_heads = dec_sampling_heads
        self.dec_sampling_points = dec_sampling_points
        self.feature_levels = feature_levels


        self.sampling_locs = nn.Conv2d(in_channels=d_model,
                                       out_channels=dec_sampling_heads * dec_sampling_points * feature_levels * 2,
                                       kernel_size=1)
        self.sampling_weight = nn.Conv2d(in_channels=d_model,
                                         out_channels=dec_sampling_heads * dec_sampling_points * feature_levels,
                                         kernel_size=1)

        self.value_conv = nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=1)
        self.output_conv = nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=1)

        self.scale = nn.Parameter(torch.Tensor(d_model))

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.scale.data, 1.)

        constant_(self.sampling_locs.weight.data, 0.)

        grid_init = torch.tensor([-1, -1, -1, 0, -1, 1, 0, -1, 0, 1, 1, -1, 1, 0, 1, 1], dtype=torch.float32) \
            .view(self.dec_sampling_heads, 1, 1, 2).repeat(1, self.feature_levels, self.dec_sampling_points, 1)

        for i in range(self.dec_sampling_points):
            grid_init[:, :, i, :] *= i + 1

        with torch.no_grad():
            self.sampling_locs.bias = nn.Parameter(grid_init.view(-1))

        constant_(self.sampling_weight.weight.data, 0.)
        constant_(self.sampling_weight.bias.data, 0.)

        xavier_uniform_(self.value_conv.weight.data)
        constant_(self.value_conv.bias.data, 0.)
        xavier_uniform_(self.output_conv.weight.data)
        constant_(self.output_conv.bias.data, 0.)

    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos

    def get_pre_grid(self, device, H_, W_):
        grid_y, grid_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                        torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
        pre_grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1) #[H_, W_, 2] each coordinate
        return pre_grid

    def get_valid_size(self, key_padding_mask, N_, H_, W_):

        valid_H = torch.sum(~key_padding_mask[:, 0, :, 0], 1)
        valid_W = torch.sum(~key_padding_mask[:, 0, 0, :], 1)
        valid_size = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 2)
        size = torch.tensor([W_, H_], dtype=torch.float32, device=key_padding_mask.device)
        return valid_size, size

    def forward(self, x, pos=None, key_padding_mask=None, spatial_shape=None):

        # x, pos: N, C, H, W
        M_ = self.dec_sampling_heads
        P_ = self.dec_sampling_points
        F_ = self.feature_levels
        N_, C_, S_ = x.shape

        spatial_shape_, valid_sizes, valid_scales = spatial_shape

        valid_sizes = valid_sizes.view(N_, F_, 1, 1, 2).repeat_interleave(M_, 0) # [bs * nhead, #level, 1, 1, 2]
        valid_scales = 2 * valid_scales.view(N_, F_, 1, 1, 2).repeat_interleave(M_, 0) # [bs * nhead, #level, 1, 1, 2]

        x_pos = self.with_pos_embed(x, pos)
        value = self.value_conv(x.unsqueeze(-1)).squeeze(-1)
        x_pos = x_pos.masked_fill(key_padding_mask.view(N_, 1, S_), float(0))
        value = value.masked_fill(key_padding_mask.view(N_, 1, S_), float(0))

        spatial_splits = [H_ * W_ for H_, W_ in spatial_shape_]
        values = torch.split(value, spatial_splits, dim=-1)
        values = [value_.view(N_ * M_, C_ // M_, H_, W_) for value_, (H_, W_) in zip(values, spatial_shape_)]

        outs = []
        _cur = 0
        strides = [8, 16, 32, 64]
        for lvl_, (s, (H_, W_)) in enumerate(zip(strides, spatial_shape_)):
            x_pos_ = x_pos[..., _cur:(_cur + H_ * W_)].view(N_, C_, H_, W_) # [N, C, L * #level]
            _cur += (H_ * W_)
            # [N, C, L * #level] @ [C, nhead * #key * #level *2]-> [bs * nhead, #level, #key, H*W, 2]
            offsets = self.sampling_locs(x_pos_).view(N_ * M_, F_, P_, 2, H_ * W_).transpose(3, 4)
            # [N, C, L * #level] @ [C, nhead * #key * #level] -> [bs * nhead, 1, #level * #key, H*W]
            weights = self.sampling_weight(x_pos_).view(N_ * M_, 1, F_ * P_, H_ * W_).softmax(2)


            pre_grid = self.get_pre_grid(x.device, H_, W_)

            grid_scale = valid_scales / valid_sizes[:, lvl_].view(N_ * M_, 1, 1, 1, 2)

            offset_scale = valid_scales / valid_sizes[:, lvl_].view(N_ * M_, 1, 1, 1, 2)

            grids = offsets * offset_scale + (pre_grid.view(1, 1, 1, H_ * W_, 2) * grid_scale - 1)
            grids = grids.transpose(0, 1)

            #list of [bs *nhead, C// nhead, #key, H*W]
            samples_values = [F.grid_sample(value, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
                              for value, grid in zip(values, grids)]

            output = torch.sum(torch.cat(samples_values, 2) * weights, 2).view(N_, C_, H_, W_) # [N, C, H, W]

            output = self.output_conv(output)
            # list of tensors [N, C, H*W]
            outs.append(output.flatten(2))
        return torch.cat(outs, -1) * self.scale.view(1, -1, 1)


class SamplingAttention_RA(nn.Module):
    """decoder attn"""
    def __init__(self, d_model=512, share_centers=True,dec_sampling_heads=8, dec_sampling_points=4., num_feature_levels=4):
        super(SamplingAttention_RA, self).__init__()
        self.d_model = d_model
        self.dec_sampling_heads = dec_sampling_heads
        self.feature_levels = num_feature_levels
        assert (share_centers == True)

        self.pool_resolution = (1., dec_sampling_points)  # width height
        self.dec_sampling_points = int(self.pool_resolution[0] * self.pool_resolution[1])


        self.value_conv = nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=1)
        self.output_proj = nn.Linear(d_model, d_model)

        self.sampling_locs = nn.Linear(d_model, dec_sampling_heads * self.dec_sampling_points * self.feature_levels * 2)
        self.sampling_weight = nn.Linear(d_model , dec_sampling_heads * self.feature_levels * self.dec_sampling_points)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_locs.weight.data, 0.)
        constant_(self.sampling_locs.bias.data, 0.)

        constant_(self.sampling_weight.weight.data, 0.)
        constant_(self.sampling_weight.bias.data, 0.)

        xavier_uniform_(self.value_conv.weight.data)
        constant_(self.value_conv.bias.data, 0.)

        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)


    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos


    def forward(self, q, q_pos, k, k_pos, key_padding_mask=None, pos_centers=None, spatial_shape=None):

        M_ = self.dec_sampling_heads
        P_ = self.dec_sampling_points
        F_ = self.feature_levels
        N_, C_, S_ = k.shape  # memoy of encoder
        L_ = q.shape[0]

        spatial_shape_, valid_sizes, valid_scales = spatial_shape
        # [bs, #level, 2] -> [1, nhead*bs, 1, #level, 2]
        valid_sizes = valid_sizes.view(1, N_, 1, F_, 2).repeat_interleave(M_, 1)
        valid_scales = 2 * valid_scales.view(1, N_, 1, F_, 2).repeat_interleave(M_, 1)

        value = self.value_conv(k.unsqueeze(-1)).squeeze(-1)
        value = value.masked_fill(key_padding_mask.view(N_, 1, S_), float(0))

        spatial_splits = [H_ * W_ for H_, W_ in spatial_shape_]
        value_list = torch.split(value, spatial_splits, dim=-1)
        value_list = [value_.view(N_ * M_, C_ // M_, H_, W_) for value_, (H_, W_) in zip(value_list, spatial_shape_)]

        weights = self.sampling_weight(q).view(L_, N_ * M_, 1, F_ * P_).softmax(3)
        # [L, bs, C] -> [L, nhead*bs, #key, #level, 2]
        grids = self.sampling_locs(q).view(L_, N_ * M_, P_, F_, 2)

        # [N * nhead, L, 4]
        pos_centers = pos_centers.permute(1,0,2).sigmoid().repeat_interleave(M_, 0)


        ##
        # [bs * nhead, L, 2 (wh)] -> [L, bs * nhead, 1, 1, 2]
        wh = pos_centers[:, :, 2:].permute(1, 0, 2).view(L_, N_ * M_, 1, 1, 2)
        # [L, nhead*bs, #key, #level, 2]
        grid_pts = torch.zeros((L_, M_, P_, F_, 2), dtype=weights.dtype, device=weights.device)

        for h_i in range(M_):
            for i in range(self.dec_sampling_points):
                grid_pts[:, h_i, i, :, 0] = ((i % int(self.pool_resolution[1])) + 0.5) / self.pool_resolution[1]
                grid_pts[:, h_i, i, :, 1] = (h_i  + 0.5 ) / M_

        grid_pts = grid_pts.repeat(1, N_, 1, 1, 1)
        grid_pts *= wh

        # [N * nhead, L, 4] -> [L, bs*nhead, 1, 1, 2]
        boxes_xy = box_ops.box_cxcywh_to_xyxy(pos_centers)[:, :, :2].permute(1, 0, 2).view(L_, N_ * M_, 1, 1, -1)

        grids = ( (grids * wh / P_) + boxes_xy + grid_pts) * valid_scales - 1

        # [L, bs*nhead, #key, #level, 2] -> [#level, bs*nhead, L, #key, 2]
        grids = grids.permute(3, 1, 0, 2, 4)

        samples_value_list = [F.grid_sample(value, grids, mode='bilinear', padding_mode='zeros', align_corners=False)
                              for value, grids in zip(value_list, grids)]

        # [bs*nhead, C / nhead, L, #key*#level]
        samples_value = torch.cat(samples_value_list, -1)
        # [bs*nhead, 1, L, #level*key]
        weights = weights.permute(1, 2, 0, 3)

        # sum all keys on all level [bs*nhead, C / nhead, L] -> [L, N, C]
        output = torch.sum(samples_value * weights, -1).permute(2, 0, 1).view(L_, N_, C_)
        output = self.output_proj(output)

        # [#level, bs*nhead, #key, 2]  -> [#level, bs, nhead, #level, #key, 2] -> [bs, L, #level, nhead, #key, 2]
        output_sample_pts = ((grids + 1.0) / 2.0).view(F_, N_, M_, L_, P_, 2).permute(1, 3, 0, 2, 4, 5)
        # [bs*nhead, 1, L, #level*key] -> [bs, #level, #level, nhead, #key]
        output_sample_weights = weights.view(N_, M_, L_, F_, P_).permute(0, 2, 3, 1, 4)
        # concat weight to sampled weight on last dim, last dim contains cx cy weight
        output_sample_attn = torch.cat((output_sample_pts, output_sample_weights[..., None]), -1)

        return output, output_sample_attn
