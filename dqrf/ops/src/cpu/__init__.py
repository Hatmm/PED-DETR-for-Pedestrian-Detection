# Modified by Matthieu Lin
# Contact linmatthieu@gmail.com
# modified from https://github.com/idiap/fast-transformers/
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/

import torch
from torch.autograd.function import once_differentiable
from DenseQuerySelfAttention import \
    local_dot_product as local_dot_product_cpu, \
    local_dot_backward as local_dot_backward_cpu, \
    local_weighted_average as local_weighted_average_cpu, \
    local_weighted_average_backward as local_weighted_average_backward_cpu


class LocalDotProduct(torch.autograd.Function):
    """
    Copute the dot product of queries and keys but only consider nearest one
    based on the computed score (GIOU for PED)
    """

    dot = {
        "cuda": local_dot_product_cpu,
        "cpu": local_dot_product_cpu,

    }
    dot_backward = {
        "cuda": local_dot_backward_cpu,
        "cpu": local_dot_backward_cpu
    }
    @staticmethod
    def forward(ctx, queries, keys, attn_mask):
        ctx.save_for_backward(queries, keys, attn_mask)


        return LocalDotProduct.dot[queries.device.type](
            queries,
            keys,
            attn_mask
        )

    @staticmethod
    @once_differentiable
    def backward(ctx, upstream_gradient):
        queries, keys, attn_mask = ctx.saved_tensors
        grad_queries, grad_keys = LocalDotProduct.dot_backward[queries.device.type](
            queries,
            keys,
            attn_mask,
            upstream_gradient
        )
        #None for attn_mask
        return grad_queries, grad_keys, None

class LocalWeightedAverage(torch.autograd.Function):

    avg = {
        "cuda": local_weighted_average_cpu,
        "cpu": local_weighted_average_cpu
    }
    avg_backward = {
        "cuda": local_weighted_average_backward_cpu,
        "cpu": local_weighted_average_cpu
    }

    @staticmethod
    def forward(ctx, A, V, attn_mask):
        ctx.save_for_backward(A, V, attn_mask)

        return LocalWeightedAverage.avg[A.device.type](A, V ,attn_mask)

    @staticmethod
    @once_differentiable
    def backward(ctx, upstream_grad):
        A, V, attn_mask = ctx.saved_tensors
        grad_attention, grad_values =  LocalWeightedAverage.avg_backward[A.device.type](
            A, V, attn_mask, upstream_grad
        )
        return grad_attention, grad_values, None

# Alias the autograd functions to python style snake case naming
local_dot_product = LocalDotProduct.apply
local_weighted_average = LocalWeightedAverage.apply
