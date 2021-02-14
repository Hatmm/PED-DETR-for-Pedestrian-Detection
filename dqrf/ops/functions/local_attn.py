#
# Modified by Matthieu Lin
# Contact: linmatthieu@gmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Same as MultiHeadAttention on pytorch adapted for binary masking on attention weights
"""
from torch import nn
import torch
import torch.nn.functional as F
from torch.nn.init import constant_
from dqrf.ops.src.cpu import local_dot_product, local_weighted_average

class DenseQueryAttention(nn.Module):
    def __init__(self, d_model, h, dropout=0.1, kdim=None, vdim=None):
        "Take in model size and number of heads."
        super(DenseQueryAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.scaling = float(self.d_k) ** -0.5
        self.h = h

        self.kdim = kdim if kdim is not None else d_model
        self.vdim = vdim if vdim is not None else d_model
        self._qkv_same_embed_dim = self.kdim == d_model and self.vdim == d_model

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = nn.Parameter(torch.Tensor(d_model, d_model))
            self.k_proj_weight = nn.Parameter(torch.Tensor(d_model, self.kdim))
            self.v_proj_weight = nn.Parameter(torch.Tensor(d_model, self.vdim))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = nn.Parameter(torch.empty(3 * d_model, d_model))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        self.in_proj_bias = nn.Parameter(torch.empty(3 * d_model))
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim is False:
            nn.init.xavier_uniform_(self.q_proj_weight)
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        else:
            nn.init.xavier_uniform_(self.in_proj_weight)
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, query, key, value, attn_mask=None, mask=None):
        # [L, N, E]
        tgt_len, nbatches, embed_dim = query.size()
        topk = attn_mask.size(-1)
        src_len, _, _ = key.size()

        # 1) Do all the linear projections in batch from d_model => h x d_k
        # [L, N, E]
        if torch.equal(query, key) and torch.equal(key, value):
            # self-attention
            q, k, v = F.linear(query, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)

        elif torch.equal(key, value):
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = self.in_proj_bias
            _start = 0
            _end = embed_dim
            _w = self.in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)

            if key is None:
                assert value is None
                k = None
                v = None
            else:

                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = self.in_proj_bias
                _start = embed_dim
                _end = None
                _w = self.in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = F.linear(key, _w, _b).chunk(2, dim=-1)
        # query, key, value = [l(x) for l, x in zip(self.linears, (query, key ,value))]
        else:
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = self.in_proj_bias
            _start = 0
            _end = embed_dim
            _w = self.in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = self.in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = self.in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = F.linear(key, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = self.in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = self.in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = F.linear(value, _w, _b)

        # [L, N, h, E/h] -> [N, h, L, E/h]
        q = q.contiguous().view(tgt_len, nbatches, self.h, self.d_k).permute(1, 2, 0, 3)
        k = k.contiguous().view(src_len, nbatches, self.h, self.d_k).permute(1, 2, 0, 3)
        q = q * self.scaling

        v = v.contiguous().view(tgt_len, nbatches, self.h, self.d_k).permute(1, 2, 0, 3)

        #[bs, nhead, #query, topk]
        attn_output_weights = local_dot_product(q, k, attn_mask)

        if mask is not None:
            #[N, 1, 1, topk]
            attn_output_weights = attn_output_weights.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))


        # [N, h, L, topk]
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = self.dropout(attn_output_weights)

        #[bs, nhead, #query, dim]
        attn_output = local_weighted_average(attn_output_weights, v, attn_mask)
        #*** RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
        attn_output = attn_output.permute(2, 0, 1, 3).reshape(tgt_len, nbatches, embed_dim)
        #L, N, E
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_output_weights.sum(dim=1) / self.h





class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h, dropout=0.1, kdim=None, vdim=None):
        "Take in model size and number of heads."
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.scaling = float(self.d_k) ** -0.5
        self.h = h

        self.kdim = kdim if kdim is not None else d_model
        self.vdim = vdim if vdim is not None else d_model
        self._qkv_same_embed_dim = self.kdim == d_model and self.vdim == d_model

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = nn.Parameter(torch.Tensor(d_model, d_model))
            self.k_proj_weight = nn.Parameter(torch.Tensor(d_model, self.kdim))
            self.v_proj_weight = nn.Parameter(torch.Tensor(d_model, self.vdim))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = nn.Parameter(torch.empty(3 * d_model, d_model))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        self.in_proj_bias = nn.Parameter(torch.empty(3 * d_model))
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim is False:
            nn.init.xavier_uniform_(self.q_proj_weight)
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)
            constant_(self.in_proj_bias, 0.) # forgot this
            constant_(self.out_proj.bias, 0.)
        else:
            nn.init.xavier_uniform_(self.in_proj_weight)
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, query, key, value, attn_mask=None, mask=None, rois=None):
        # [L, N, E]
        tgt_len, nbatches, embed_dim = query.size()
        src_len, _, _ = key.size()

        # 1) Do all the linear projections in batch from d_model => h x d_k
        # pdb.set_trace()
        # [L, N, E]
        if torch.equal(query, key) and torch.equal(key, value):
            # self-attention
            q, k, v = F.linear(query, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)

        elif torch.equal(key, value):
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = self.in_proj_bias
            _start = 0
            _end = embed_dim
            _w = self.in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)

            if key is None:
                assert value is None
                k = None
                v = None
            else:

                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = self.in_proj_bias
                _start = embed_dim
                _end = None
                _w = self.in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = F.linear(key, _w, _b).chunk(2, dim=-1)
        # query, key, value = [l(x) for l, x in zip(self.linears, (query, key ,value))]
        else:
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = self.in_proj_bias
            _start = 0
            _end = embed_dim
            _w = self.in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = self.in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = self.in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = F.linear(key, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = self.in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = self.in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = F.linear(value, _w, _b)

        # [L, N * h, E/h] -> [N * h, L, E/h]
        q = q.contiguous().view(tgt_len, nbatches * self.h, self.d_k).transpose(0, 1)
        k = k.contiguous().view(src_len, nbatches * self.h, self.d_k).transpose(0, 1)
        q = q * self.scaling

        v = v.contiguous().view(src_len, nbatches * self.h, self.d_k).transpose(0, 1)

        attn_output_weights = torch.bmm(q, k.transpose(1, 2))

        if attn_mask is not None:
            attn_output_weights *= attn_mask



        if mask is not None:
            #[N, h, L, L]
            attn_output_weights = attn_output_weights.view(nbatches, self.h, tgt_len, src_len)
            #[N, 1, 1, L]
            attn_output_weights = attn_output_weights.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            #[N*h, L, L]
            attn_output_weights = attn_output_weights.view(nbatches * self.h, tgt_len, src_len)

        # [N*h, L, L]
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        # attn_output_weights /= src_len
        attn_output_weights = self.dropout(attn_output_weights)


        # [N*h, L, L] @ [N*h, L, E/h]
        attn_ouput = torch.bmm(attn_output_weights, v)

        #[L, N*h, E/h] -> [L, N, E]
        attn_ouput = attn_ouput.transpose(0, 1).contiguous().view(tgt_len, nbatches, embed_dim)
        attn_ouput = self.out_proj(attn_ouput)

        attn_output_weights = attn_output_weights.view(nbatches, self.h, tgt_len, src_len)
        return attn_ouput, attn_output_weights.sum(dim=1) / self.h
