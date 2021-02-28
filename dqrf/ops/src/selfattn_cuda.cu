//
// Modified by Matthieu Lin from https://github.com/idiap/fast-transformers/blob/master/fast_transformers
//
#include <cooperative_groups.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
using namespace cooperative_groups;

//#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
//#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
//#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

typedef torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> float_accessor_4d;
typedef torch::PackedTensorAccessor32<int64_t, 4, torch::RestrictPtrTraits> int64_accessor_4d;

template <typename scalar_t>
inline __device__ float dot(const scalar_t *a, const scalar_t *b, int n) {
    scalar_t s = 0;
    for (int i=0; i<n; i++) {
        s += (*a) * (*b);
        a++;
        b++;
    }
    return s;
}


template <typename scalar_t>
__global__ void sparse_dot_product_kernel(
        const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> queries,
        const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> keys,
        const int64_accessor_4d topk,
        torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> products,
        int q_load
) {
    const int N = queries.size(0);
    const int H = queries.size(1);
    const int L = queries.size(2);
    const int E = queries.size(3);
    const int S = keys.size(2);
    const int hl = H*L;
    extern __shared__ float shared_qs[];

    int full_indx = q_load*blockIdx.x + threadIdx.x;
    int n = full_indx / (hl);
    int h = (full_indx - n*hl) / L;
    int l = (full_indx - n*hl) % L;
    if ((threadIdx.x < q_load) && ((q_load*blockIdx.x + threadIdx.x) < (N*L*H))) {
        int q_indx = threadIdx.x;
        float *s_ptr = shared_qs + q_indx;
        //#pragma unroll
        for (int e=0; e<E; e++) {
            *s_ptr = queries[n][h][l][e];
            s_ptr += q_load;
        }
    }
    __syncthreads();

    int q_indx = threadIdx.x % q_load;
    int topk_idx = threadIdx.x / q_load;
    int q_processed = (blockIdx.x*q_load) + q_indx;
    int seq_idx = q_processed / (hl);
    int h_idx = (q_processed - seq_idx*hl)/L;
    int l_idx = (q_processed - seq_idx*hl)%L;

    if ((seq_idx >= N) || (l_idx >= L) || (h_idx >= H)) {
        return;
    }

    float s = 0;
    const float *q_cur = shared_qs + q_indx;
    int k_idx = topk[seq_idx][h_idx][l_idx][topk_idx];

    //#pragma unroll
    for (int e=0; e<E; e++) {
        s += (*q_cur) * keys[seq_idx][h_idx][k_idx][e];
        q_cur += q_load;
    }
    products[seq_idx][h_idx][l_idx][topk_idx] = s;
}


at::Tensor sparse_dot_product(
        const torch::Tensor& Q,
        const torch::Tensor& K,
        const torch::Tensor& topk
) {
    int N = Q.size(0);
    int H = Q.size(1);
    int L = Q.size(2);
    int E = Q.size(3);
    int k = topk.size(3);
    int S = K.size(2);

    auto output = torch::zeros({N, H, L, k}, Q.options());//Q.new_full({N, H, L, k}, 0.0);

    int max_threads = 1024;
    int q_max = (48 * max_threads)/(4*E) < L ? (48 * max_threads)/(4*E):L;

    int q_load = (max_threads/k) < q_max ? (max_threads/k):q_max;
    int threads = q_load * k;

    const int shared_mem_queries = q_load * E * sizeof(float);
    int total_products = L*N*H*k;
    int blocks = ceil(float(total_products)/(q_load * k));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(Q.scalar_type(), "sparse_dot_product", [&] {
        sparse_dot_product_kernel<scalar_t><<<blocks, threads, shared_mem_queries, stream>>>(
                Q.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                K.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                topk.packed_accessor32<int64_t, 4, torch::RestrictPtrTraits>(),
                output.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                q_load);
    });


    return output;
}

template <typename scalar_t>
__global__ void sparse_weighted_average_kernel(
        const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> weights,
        const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> values,
        const int64_accessor_4d topk,
        torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> output,
        int N,
        int H,
        int L,
        int E,
        int k,
        int n_dim_per_thread
) {
    extern __shared__ float shared_mem[];
    int block_idx = blockIdx.x;
    if ((block_idx > N*H*L)){
        return;
    }

    int n = (block_idx) / (H*L);
    int h = (block_idx - n*H*L) / (L);
    int l = block_idx  % L;


    if ((threadIdx.x < k)) {
        shared_mem[k*E + threadIdx.x] = weights[n][h][l][threadIdx.x];
        shared_mem[(k*(E+1)) +  threadIdx.x] = topk[n][h][l][threadIdx.x];
    }

    __syncthreads();

    if (threadIdx.x < k) {
        int n_threads_per_key  = E / n_dim_per_thread;
        int j = threadIdx.x / n_threads_per_key ;
        int d_start = (threadIdx.x - j*n_threads_per_key) * n_dim_per_thread;

        int key_idx = int(shared_mem[(k*(E+1)) + j]);
        const float s = shared_mem[k*E + j];

        for(int i=0; i<n_dim_per_thread; i++) {
            int cur_d = d_start + i;
            float v = values[n][h][key_idx][cur_d];
            shared_mem[j + (cur_d * k)] =  v * s;
        }
    }
    __syncthreads();

    if ((threadIdx.x < E)) {
        float sum = 0;
        int start = threadIdx.x*k;
        for (int i=start; i<start+k; i++) {
            sum = sum + shared_mem[i];
        }
        output[n][h][l][threadIdx.x] = sum;
    }
}


torch::Tensor sparse_weighted_average(
        const torch::Tensor& weights,
        const torch::Tensor& values,
        const torch::Tensor& topk
) {
    int N = weights.size(0);
    int H = weights.size(1);
    int L = weights.size(2);
    int k = weights.size(3);
    int E = values.size(3);

    auto output = torch::zeros({N, H, L, E}, values.options());


    int n_dim_per_thread = E;
    // We need at least E threads for the final reduction
    int threads = ceil((E * k)/n_dim_per_thread) > E ? ceil((E * k)/n_dim_per_thread):E;
    int total_products = L*N*H*k;
    int blocks = ceil(float(total_products)/(k));
    const int shared_mem = (((k * E) + 2*k)* sizeof(float));

    AT_DISPATCH_FLOATING_TYPES(values.scalar_type(), "sparse_weighted_average", [&] {
        sparse_weighted_average_kernel<scalar_t><<<blocks, threads, shared_mem>>>(
            weights.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            values.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            topk.packed_accessor32<int64_t, 4, torch::RestrictPtrTraits>(),
            output.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            N,
            H,
            L,
            E,
            k,
            n_dim_per_thread);
    });

    return output;
}

template <typename scalar_t>
__global__ void sparse_dot_backward_kernel(
        const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> queries,
        const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> keys,
        const int64_accessor_4d topk,
        const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad_out,
        torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad_q,
        torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad_k
) {
    const int N = queries.size(0);
    const int H = queries.size(1);
    const int L = queries.size(2);
    const int E = queries.size(3);
    const int S = keys.size(2);
    const int k = topk.size(3);

    int full_index = blockIdx.x * blockDim.x + threadIdx.x;
    int n = full_index / (H*L*k);
    int h = (full_index - n*H*L*k) / (L*k);
    int l = (full_index - n*H*L*k - h*L*k) / k;
    int j = full_index % k;

    if (n >= N) {
        return;
    }

    const int key_index = topk[n][h][l][j];
    const float grad = grad_out[n][h][l][j];
    for (int e=0; e<E; e++) {
        atomicAdd(&grad_q[n][h][l][e], grad * keys[n][h][key_index][e]);
    }
    for (int e=0; e<E; e++) {
        atomicAdd(&grad_k[n][h][key_index][e], grad * queries[n][h][l][e]);
    }
}

std::tuple<torch::Tensor, torch::Tensor> sparse_dot_backward(
        const torch::Tensor& Q,
        const torch::Tensor& K,
        const torch::Tensor& topk,
        const torch::Tensor& grad_out
) {
    int N = Q.size(0);
    int H = Q.size(1);
    int L = Q.size(2);
    int E = Q.size(3);
    int k = topk.size(3);
    int S = K.size(2);

    auto grad_Q = torch::zeros_like(Q);
    auto grad_K = torch::zeros_like(K);

    int threads = 1024;
    int blocks = (N*H*L*k + threads - 1) / threads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(Q.scalar_type(), "sparse_dot_backward", [&] {
        sparse_dot_backward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                Q.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                K.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                topk.packed_accessor32<int64_t, 4, torch::RestrictPtrTraits>(),
                grad_out.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                grad_Q.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                grad_K.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>());
    });
    return std::make_tuple(grad_Q, grad_K);
}



template <typename scalar_t>
__global__ void sparse_weighted_average_backward_kernel(
        const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> weights,
        const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> values,
        const torch::PackedTensorAccessor32<int64_t, 4, torch::RestrictPtrTraits> topk,
        const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad_out,
        torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad_weights,
        torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad_values,
        int N,
        int H,
        int L,
        int E,
        int k,
        int dim_per_thread
) {
    int full_index = blockIdx.x * blockDim.x + threadIdx.x;
    int n = full_index / (H*L*k);
    int h = (full_index - n*H*L*k) / (L*k);
    int l = (full_index - n*H*L*k - h*L*k) / k;
    int j = full_index % k;

    if (n >= N) {
        return;
    }
    int key_idx = topk[n][h][l][j];
    int start_dim = threadIdx.y * dim_per_thread;
    int end_dim = start_dim + dim_per_thread;
    if (threadIdx.y == 0) {
        grad_weights[n][h][l][j] = dot(
                &values[n][h][key_idx][0],
                &grad_out[n][h][l][0],
                E
        );
    }
    float weight = weights[n][h][l][j];
    for (int e=start_dim; e<end_dim; e++) {
        atomicAdd(
                &grad_values[n][h][key_idx][e],
                weight * grad_out[n][h][l][e]
        );
    }
}


std::tuple<torch::Tensor, torch::Tensor> sparse_weighted_average_backward(
        const torch::Tensor& weights,
        const torch::Tensor& values,
        const torch::Tensor& topk,
        const torch::Tensor& grad_out
) {
    int N = weights.size(0);
    int H = weights.size(1);
    int L = weights.size(2);
    int k = weights.size(3);
    int E = values.size(3);

    auto grad_weights = torch::zeros_like(weights);
    auto grad_values = torch::zeros_like(values);


    int threads_x = 256;
    int threads_y = 4;
    int dim_per_thread = E / threads_y;
    dim3 threads(threads_x, threads_y);
    int blocks = (N*H*L*k + threads_x - 1)/threads_x;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(grad_out.scalar_type(), "sparse_weighted_average_backward", [&] {
        sparse_weighted_average_backward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                weights.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                values.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                topk.packed_accessor32<int64_t, 4, torch::RestrictPtrTraits>(),
                grad_out.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                grad_weights.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                grad_values.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                N,
                H,
                L,
                E,
                k,
                dim_per_thread);
    });
    return std::make_tuple(grad_weights, grad_values);
}


