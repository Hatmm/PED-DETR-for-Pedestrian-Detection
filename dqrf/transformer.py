# ------------------------------------------------------------------------
# Modified by Matthieu Lin
# Modified from Deformable-DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
from typing import Optional
from dqrf.ops.functions.local_attn import MultiHeadAttention
from dqrf.ops.functions.ms_deform_attn import SamplingAttention_RA, SamplingEncAttention, SamplingAttention_dec
import torch
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, normal_
from dqrf.utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from dqrf.utils.utils import _get_clones, _get_activation_fn



class Transformer(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        d_model = cfg.MODEL.DQRF_DETR.HIDDEN_DIM
        nhead = cfg.MODEL.DQRF_DETR.NHEAD
        num_decoder_layers = cfg.MODEL.DQRF_DETR.NUM_DECODER_LAYERS
        num_encoder_layers = cfg.MODEL.DQRF_DETR.NUM_ENCODER_LAYERS
        dim_feedforward = cfg.MODEL.DQRF_DETR.DIM_FEEDFORWARD
        activation = cfg.MODEL.DQRF_DETR.ACTIVATION
        dropout = cfg.MODEL.DQRF_DETR.DROPOUT
        return_intermediate_dec = cfg.MODEL.DQRF_DETR.AUX_LOSS
        self.return_intermediate_dec = return_intermediate_dec
        num_feature_levels = cfg.MODEL.DQRF_DETR.NUM_FEATURE_LEVELS
        enc_sampling_points = cfg.MODEL.DQRF_DETR.ENC_SAMPLING_POINTS
        dec_sampling_points = cfg.MODEL.DQRF_DETR.DEC_SAMPLING_POINTS
        dense_query = cfg.MODEL.DQRF_DETR.DENSE_QUERY
        rectified_attention = cfg.MODEL.DQRF_DETR.RECTIFIED_ATTENTION
        self.zero_tgt = False

        checkpoint_enc_ffn = True
        checkpoint_dec_ffn = True

        self.d_model = d_model
        self.num_decoder_layers = num_decoder_layers

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation,
                                                enc_sampling_points=enc_sampling_points,
                                                checkpoint_ffn=checkpoint_enc_ffn,
                                                num_feature_levels=num_feature_levels)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,dropout, activation,
                                                num_feature_levels=num_feature_levels, dec_sampling_points=dec_sampling_points,
                                                checkpoint_ffn=checkpoint_dec_ffn, rectified_attention=rectified_attention)



        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, dense_query=dense_query, norm=decoder_norm, return_intermediate=return_intermediate_dec)

        self._reset_parameters()


        self.sampling_cens = nn.Linear(d_model, 4)
        constant_(self.sampling_cens.weight.data, 0.)
        constant_(self.sampling_cens.bias.data, 0.)

        xavier_uniform_(self.sampling_cens.weight.data[:2], gain=1.0)

        constant_(self.sampling_cens.bias.data[2:3], -2.0)
        constant_(self.sampling_cens.bias.data[3:4], -1.5)

        # constant_(self.sampling_cens.bias.data[2:], -2.0)


        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        normal_(self.level_embed)


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, (SamplingAttention_RA, SamplingEncAttention, SamplingAttention_dec)):
                m._reset_parameters()


    def get_valid_size(self, key_padding_mask, N_, H_, W_):
        valid_H = torch.sum(~key_padding_mask[:, :, 0], 1)
        valid_W = torch.sum(~key_padding_mask[:, 0, :], 1)
        valid_size = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 2)
        size = torch.tensor([W_, H_], dtype=torch.float32, device=key_padding_mask.device)
        valid_scale = valid_size / size.view(1, 2)
        return valid_size, valid_scale

    def forward(self, srcs, masks, query_embed, pos_embeds):

        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        valid_size_list = []
        valid_scale_list = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2)

            valid_size, valid_scale = self.get_valid_size(mask.view(bs, h, w), bs, h, w)
            valid_size_list.append(valid_size.view(bs, 1, 2))
            valid_scale_list.append(valid_scale.view(bs, 1, 2))

            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, -1, 1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)

        src_flatten = torch.cat(src_flatten, -1)
        mask_flatten = torch.cat(mask_flatten, -1)
        valid_sizes = torch.cat(valid_size_list, 1)
        valid_scales = torch.cat(valid_scale_list, 1)

        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, -1)

        spatial_shapes = (spatial_shapes, valid_sizes, valid_scales)

        memory = self.encoder(src_flatten, src_key_padding_mask=mask_flatten, pos=lvl_pos_embed_flatten,
                              spatial_shape=spatial_shapes)

        bs, c = memory.shape[:2]
        # L, E
        if not self.zero_tgt:
            query_embed, tgt = torch.split(query_embed, c, dim=1)
            query_embed = query_embed.unsqueeze(1).expand(-1, bs, -1)
            tgt = tgt.unsqueeze(1).expand(-1, bs, -1)
        else:
            tgt = torch.zeros_like(query_embed)
            query_embed = query_embed.unsqueeze(1).expand(-1, bs, -1)
        pos_centers = self.sampling_cens(tgt)
        # [#dec, #query, bs, dim] or [#query, bs, dim]
        hs, inter_ps, dec_attns = self.decoder(tgt, memory, memory_key_padding_mask=mask_flatten,
                                               pos=lvl_pos_embed_flatten, query_pos=query_embed,
                                               pos_centers=pos_centers, spatial_shape=spatial_shapes,
                                               )
        if self.return_intermediate_dec:
            return hs.transpose(1, 2), memory, inter_ps.transpose(1, 2), dec_attns
        else:
            return hs.transpose(0, 1), memory, inter_ps.transpose(0, 1), dec_attns


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src, src_key_padding_mask, pos, spatial_shape):
        output = src

        for layer in self.layers:
            output = layer(output, src_key_padding_mask=src_key_padding_mask, pos=pos, spatial_shape=spatial_shape)

        return output



class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, dense_query, norm=None, return_intermediate=False):
        super().__init__()

        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.bbox_embed = None
        self.dense_query = dense_query


    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                pos_centers=None, spatial_shape=None):

        output = tgt

        intermediate = []
        intermediate_centers = []
        intermediate_dec_attns = []
        # intermediate_tgt = []

        for lvl, layer in enumerate(self.layers):

            # intermediate_tgt.append(output)

            if self.dense_query is True: # work around for dense query implementation

                outputs_coord = pos_centers.permute(1, 0, 2).sigmoid()
                nquery = outputs_coord.size(1)
                tgt_masks = []
                for pred in outputs_coord:

                    tgt_masks_ = torch.zeros((nquery, nquery), device=pos_centers.device)
                    boxes = box_cxcywh_to_xyxy(pred)
                    giou_score = 1 - generalized_box_iou( boxes, boxes)
                    score = giou_score
                    top_idx = torch.sort(score, dim=-1)[1][:, :100] # returns a longtensor
                    # _, top_idx = torch.topk(score, k=100, largest=False, sorted=True,dim=-1)#[nquery, topk] #torch.sort is faster on GPU
                    tgt_masks_.scatter_(1, top_idx, 1.)
                    tgt_masks.append(tgt_masks_)

                tgt_mask = torch.stack(tgt_masks, dim=0).repeat_interleave(8, 0)

            output, dec_attn = layer(output, memory, tgt_mask=tgt_mask,
                                     memory_mask=memory_mask,
                                     tgt_key_padding_mask=tgt_key_padding_mask,
                                     memory_key_padding_mask=memory_key_padding_mask,
                                     pos=pos, query_pos=query_pos,
                                     pos_centers=pos_centers,
                                     spatial_shape=spatial_shape)

            if self.return_intermediate:
                intermediate.append(self.norm(output))
                intermediate_centers.append(pos_centers)
                intermediate_dec_attns.append(dec_attn)

            if self.bbox_embed is not None:

                tmp = self.bbox_embed[lvl](self.norm(output))
                new_pos_centers = tmp + pos_centers
                pos_centers = new_pos_centers.detach()

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()

                intermediate.append(output)

        if self.return_intermediate:

            return torch.stack(intermediate), torch.stack(intermediate_centers), torch.stack(intermediate_dec_attns)#torch.stack(intermediate_tgt)

        return output, pos_centers, dec_attn

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", enc_sampling_points=1, checkpoint_ffn=False,
                 num_feature_levels=4):
        super().__init__()

        self.self_attn = SamplingEncAttention(d_model, dec_sampling_heads=nhead, dec_sampling_points=enc_sampling_points, feature_levels=num_feature_levels)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)


        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.checkpoint_ffn = checkpoint_ffn

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, src, src_key_padding_mask, pos, spatial_shape):
        src2 = self.self_attn(src, pos=pos, key_padding_mask=src_key_padding_mask, spatial_shape=spatial_shape)
        src = src + self.dropout1(src2)
        def wrap1(src):

            N_, C_, S_ = src.shape
            src = src.permute(2, 0, 1)
            src = self.norm1(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)
            src = src.view(S_, N_, C_).permute(1, 2, 0)

            return src
        if self.checkpoint_ffn and self.train() == True:
            #does not store intermediate activation and instead recompute then in backward pass, trades computatin for memory
            src = torch.utils.checkpoint.checkpoint(wrap1, src)
        else:
            src = wrap1(src)
        return src



class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", dec_sampling_points=8, num_feature_levels=4, checkpoint_ffn=False,
                 rectified_attention=False
                 ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        if rectified_attention:
            self.multihead_attn = SamplingAttention_RA(d_model, dec_sampling_points=dec_sampling_points,
                                                    num_feature_levels=num_feature_levels)
        else:
            self.multihead_attn = SamplingAttention_dec(d_model, dec_sampling_points=dec_sampling_points,
                                                   num_feature_levels=num_feature_levels)



        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.checkpoint_ffn = checkpoint_ffn

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                pos_centers=None, spatial_shape=None):

        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        # tgt2 = self.self_attn(q, k, tgt, attn_mask=tgt_mask, mask=tgt_key_padding_mask)[0]

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2, dec_attn = self.multihead_attn(tgt, query_pos,
                                                 memory, pos,
                                                 key_padding_mask=memory_key_padding_mask,
                                                 pos_centers=pos_centers, spatial_shape=spatial_shape)

        tgt = tgt + self.dropout2(tgt2)

        def wrap2(tgt):
            tgt = self.norm2(tgt)
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
            tgt = tgt + self.dropout3(tgt2)
            tgt = self.norm3(tgt)
            return tgt

        if self.checkpoint_ffn and self.train() == True:
            tgt = torch.utils.checkpoint.checkpoint(wrap2, tgt)
        else:
            tgt = wrap2(tgt)
        return tgt, dec_attn


def build_transformer(cfg):

    return Transformer(cfg)



