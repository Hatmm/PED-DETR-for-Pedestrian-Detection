# Modified by Matthieu Lin
# Contact linmatthieu@gmail.com
# modified from https://github.com/idiap/fast-transformers/
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
import torch
import time
import os
import unittest
import numpy as np
from scipy import sparse
import torch.nn.functional as F
import random
from torch.autograd import gradcheck
from dqrf.ops.src.cpu import local_dot_product, local_weighted_average
from dqrf.ops.functions.local_attn import DenseQueryAttention, MultiHeadAttention
from DenseQuerySelfAttention import local_dot_product, \
    local_dot_backward, local_weighted_average, local_weighted_average_backward

class TestLocalProductCPU(unittest.TestCase):

    kernels = {
        "dot": local_dot_product,
        "dot_backward": local_dot_backward,
        "wa": local_weighted_average,
        "wa_backward": local_weighted_average_backward
    }

    def test_result_forward(self):
        for t in range(10):
            N = 4
            L = 100
            H = 8
            E = 32
            Q = torch.rand(N, H, L, E)
            K = torch.rand(N, H, L, E)

            topk = np.random.randint(8, 24)

            ## prepare topk attn_mask
            attn_mask = []
            for i in range(N):

                scores = torch.randn((L, L))
                top_idx = torch.sort(scores, dim=-1)[1][:, :topk]
                attn_mask.append(top_idx)

            attn_mask = torch.stack(attn_mask, dim=0).repeat_interleave(8, 0)  # [bs*nhead, num_q, topk]
            attn_mask = attn_mask.contiguous().view(N, H, L, topk)
            ##

            out = self.kernels["dot"](Q, K, attn_mask)

            ground_truth = torch.zeros(N, H, L, topk)
            for n in range(N):
                for h in range(H):
                    for i in range(L):
                        idx = attn_mask[n, h, i, :]

                        ground_truth[n, h, i, :] = Q[n, h, i, :] @ K[n, h, idx, :].T

            self.assertTrue(torch.allclose(ground_truth, out, atol=1e-5, rtol=1e-5))

    def test_sum_forward(self):
        for t in range(10):
            N = 4
            L = 100
            H = 8
            E = 32
            Q = torch.rand(N, H, L, E)
            K = torch.rand(N, H, L, E)

            topk = np.random.randint(8, 24)

            ## prepare binary attn mask and topk attn mask, binary attn is multiplied with vanilla
            ## attention score this is another hack to achieve topk results but is computational more expensive
            ## than vanilla attention
            attn_mask = []
            tgt_masks = []
            for i in range(N):
                tgt_masks_ = torch.zeros((L, L))
                scores = torch.randn((L, L))
                top_idx = torch.sort(scores, dim=-1)[1][:, :topk]
                attn_mask.append(top_idx)
                tgt_masks_.scatter_(1, top_idx, 1.)
                tgt_masks.append(tgt_masks_)
            tgt_mask = torch.stack(tgt_masks, dim=0).repeat_interleave(8, 0)  # [bs*nhead, num_q, num_q]
            attn_mask = torch.stack(attn_mask, dim=0).repeat_interleave(8, 0)  # [bs*nhead, num_q, topk]
            attn_mask = attn_mask.contiguous().view(N, H, L, topk)
            out = self.kernels["dot"](Q, K, attn_mask)


            Q = Q.contiguous().permute(2, 0, 1, 3).view(L, N * H, E).transpose(0, 1)
            K = K.contiguous().permute(2, 0, 1, 3).view(L, N * H, E).transpose(0, 1)
            ground_truth = torch.bmm(Q, K.transpose(1, 2))
            ground_truth *= tgt_mask #[N*H, L, L]

            ground_truth = ground_truth[ground_truth.nonzero(as_tuple=True)]


            self.assertTrue(torch.allclose(ground_truth.sum(), out.sum(), atol=1e-5, rtol=1e-5),
                            msg='{}, {}'.format(ground_truth.sum(), out.sum()))

    def test_result_backward(self):
        for t in range(10):
            N = 4
            L = 100
            H = 8
            E = 32
            Q = torch.rand(N, H, L, E)
            K = torch.rand(N, H, L, E)

            Q = Q.requires_grad_(True)
            K = K.requires_grad_(True)

            topk = np.random.randint(8, 24)

            upstream_grad = torch.ones(N, H, L, topk, dtype=torch.float32)

            attn_mask = []

            for i in range(N):
                scores = torch.randn((L, L))
                top_idx = torch.sort(scores, dim=-1)[1][:, :topk]
                attn_mask.append(top_idx)

            attn_mask = torch.stack(attn_mask, dim=0).repeat_interleave(8, 0)  # [bs*nhead, num_q, topk]
            attn_mask = attn_mask.contiguous().view(N, H, L, topk)

            GQ, GK = self.kernels["dot_backward"](Q, K, attn_mask, upstream_grad)

            ground_truth = torch.zeros(N, H, L, topk)
            for n in range(N):
                for h in range(H):
                    for i in range(L):
                        idx = attn_mask[n, h, i, :]

                        ground_truth[n, h, i, :] = Q[n, h, i, :] @ K[n, h, idx, :].T

            ground_truth.sum().backward()
            self.assertTrue(torch.allclose(Q.grad, GQ, atol=1e-5, rtol=1e-5))
            self.assertTrue(torch.allclose(K.grad, GK, atol=1e-5, rtol=1e-5))

    def test_result_weighted_average(self):
        for t in range(10):
            N = 10
            L = 100
            H = 8
            E = np.random.randint(32, 256)
            topk = np.random.randint(8, 24)
            A = torch.softmax(torch.randn(N, H, L, topk), dim=-1) # [bs, nhead, num_q, topk]
            V = torch.rand(N, H, L, E) # [bs, nhead, num_q, E]
            ## prepare topk attn_mask
            attn_mask = []
            for i in range(N):
                scores = torch.randn((L, L))
                top_idx = torch.sort(scores, dim=-1)[1][:, :topk]
                attn_mask.append(top_idx)

            attn_mask = torch.stack(attn_mask, dim=0).repeat_interleave(8, 0)  # [bs*nhead, num_q, topk]
            attn_mask = attn_mask.contiguous().view(N, H, L, topk)# [bs, nhead, num_q, topk]
            ##
            out = self.kernels["wa"](A, V, attn_mask)

            ground_truth = torch.zeros(N, H, L, E)
            for n in range(N):
                for h in range(H):
                    for i in range(L):
                        idx = attn_mask[n, h, i, :]
                        # [bs, nhead, num_q, topk] @ [bs, nhead, topk, E]
                        ground_truth[n, h, i, :] = A[n, h, i, :] @ V[n, h, idx, :]

            
            self.assertTrue(torch.allclose(ground_truth, out, atol=1e-5, rtol=1e-5))

    def test_result_weighted_average_backward(self):
        for t in range(10):
            N = 10
            L = 100
            H = 8
            E = np.random.randint(32, 256)
            topk = np.random.randint(8, 24)
            A = torch.softmax(torch.randn(N, H, L, topk), dim=-1) # [bs, nhead, num_q, topk]
            V = torch.rand(N, H, L, E) # [bs, nhead, num_q, E]

            A = A.requires_grad_(True)
            V = V.requires_grad_(True)
            ## prepare topk attn_mask
            attn_mask = []
            for i in range(N):
                scores = torch.randn((L, L))
                top_idx = torch.sort(scores, dim=-1)[1][:, :topk]
                attn_mask.append(top_idx)

            attn_mask = torch.stack(attn_mask, dim=0).repeat_interleave(8, 0)  # [bs*nhead, num_q, topk]
            attn_mask = attn_mask.contiguous().view(N, H, L, topk)# [bs, nhead, num_q, topk]
            ##

            upstream_grad = torch.ones(N, H, L, E)
            GA, GV = self.kernels["wa_backward"](A, V, attn_mask, upstream_grad)


            ground_truth = torch.zeros(N, H, L, E)
            for n in range(N):
                for h in range(H):
                    for i in range(L):
                        idx = attn_mask[n, h, i, :]
                        # [bs, nhead, num_q, topk] @ [bs, nhead, topk, E]
                        ground_truth[n, h, i, :] = A[n, h, i, :] @ V[n, h, idx, :]


            ground_truth.sum().backward()

            self.assertTrue(torch.allclose(A.grad, GA, atol=1e-5, rtol=1e-5))
            self.assertTrue(torch.allclose(V.grad, GV, atol=1e-5, rtol=1e-5))

    def test_dq_module_forward(self):

        L = 1000
        H = 8
        N = 4
        E = 256

        topk = 100
        v = torch.rand(L, N, E)#.to(device)
        q = k = v

        attn_mask = []
        tgt_masks = []
        for i in range(N):
            tgt_masks_ = torch.zeros((L, L))
            scores = torch.randn((L, L))
            top_idx = torch.sort(scores, dim=-1)[1][:, :topk]
            attn_mask.append(top_idx)
            tgt_masks_.scatter_(1, top_idx, 1.)
            tgt_masks.append(tgt_masks_)
        tgt_mask = torch.stack(tgt_masks, dim=0).repeat_interleave(8, 0)  # [bs*nhead, num_q, num_q]
        attn_mask = torch.stack(attn_mask, dim=0).repeat_interleave(8, 0)  # [bs*nhead, num_q, topk]
        attn_mask = attn_mask.contiguous().view(N, H, L, topk)

        q_out = q.contiguous().view(L, N, H, E//H).permute(1, 2, 0, 3)
        k_out = k.contiguous().view(L, N, H, E//H).permute(1, 2, 0, 3)
        v_out = v.contiguous().view(L, N, H, E//H).permute(1, 2, 0, 3)

        attn_output_weights = local_dot_product(q_out, k_out, attn_mask)
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output = local_weighted_average(attn_output_weights, v_out, attn_mask)
        out = attn_output.permute(2, 0, 1, 3).reshape(L, N, E)



        q = q.contiguous().view(L, N * H, E//H).transpose(0, 1)
        k = k.contiguous().view(L, N * H, E//H).transpose(0, 1)
        v = v.contiguous().view(L, N * H, E//H).transpose(0, 1)
        attn_output_weights_ = torch.bmm(q, k.transpose(1, 2))
        attn_output_weights_ *= tgt_mask
        attn_output_weights_ = F.softmax(attn_output_weights_, dim=-1)
        attn_ouput = torch.bmm(attn_output_weights_, v)
        gt = attn_ouput.transpose(0, 1).contiguous().view(L, N, E)

        self.assertTrue(torch.allclose(attn_output_weights_.sum(), attn_output_weights.sum(), atol=1e-5, rtol=1e-5))
        # when using softmax for element zero-ed outwe get very small numbers ~ exp(0) / large number
        # hence the difference might be more noisy on the final result
        self.assertTrue(torch.allclose(gt, out, atol=1e-2, rtol=1e-2))

    def test_benchmark_speed(self):
        L = 1000
        H = 8
        N = 4
        E = 256

        topk = 100
        v = torch.rand(L, N, E)  # .to(device)
        q = k = v

        attn_mask = []
        tgt_masks = []
        for i in range(N):
            tgt_masks_ = torch.zeros((L, L))
            scores = torch.randn((L, L))
            top_idx = torch.sort(scores, dim=-1)[1][:, :topk]
            attn_mask.append(top_idx)
            tgt_masks_.scatter_(1, top_idx, 1.)
            tgt_masks.append(tgt_masks_)
        tgt_mask = torch.stack(tgt_masks, dim=0).repeat_interleave(8, 0)  # [bs*nhead, num_q, num_q]
        attn_mask = torch.stack(attn_mask, dim=0).repeat_interleave(8, 0)  # [bs*nhead, num_q, topk]
        attn_mask = attn_mask.contiguous().view(N, H, L, topk)

        multihead_optim = DenseQueryAttention(E, H, 0.1)
        multihead = MultiHeadAttention(E, H, 0.1)

        #warmup the cache
        for i in range(10):
            multihead_optim(q, k, v, attn_mask)


        start = time.time()
        multihead_optim(q, k, v, attn_mask)
        end = time.time()
        print("c++ time in ms: ", (end-start)*1000)

        start = time.time()
        multihead(q, k, v, tgt_mask)
        end = time.time()
        print("multihead time in ms: ", (end - start) * 1000)

        start = time.time()
        multihead(q, k, v)
        end = time.time()
        print("baseline time in ms: ", (end - start) * 1000)


if __name__ == "__main__":
    unittest.main()

