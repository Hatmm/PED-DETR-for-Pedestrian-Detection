//
// Modified by Matthieu Lin from https://github.com/idiap/fast-transformers/blob/master/fast_transformers
//

#include <limits>
#include <torch/extension.h>


inline float dot(const float *A, const float *B, int D) {
    // perform dot product of two vector
    float s = 0;
    for (int i=0; i<D; i++) {
        s += (*A) * (*B);
        A++;
        B++;

    }
    return s;
}

inline void scaled_copy_add(const float *src, float *dst, float scale, int D) {
    for (int i=0; i<D; i++) {
        *dst += (*src) * scale;
        dst++;
        src++;
    }
}

torch::Tensor local_dot_product(
        const torch::Tensor& queries,
        const torch::Tensor& keys,
        const torch::Tensor& attn_mask
        )
/* This functions takes queries and keys and output multi-head attention weights
 * queries' shape: [bs, nhead, #query, dim]
 * keys' shape: [bs, nhead, #query, dim]
 * attn_mask: [bs, nhead, #query, topk] contains index for topk queries to attend
 * return: attn scores: [bs, nhead, #query, topk] */
        {
    const int batch_size = queries.size(0);
    const int nhead = queries.size(1);
    const int nquery = queries.size(2);
    const int dim = queries.size(3);

    const int topk = attn_mask.size(3);

    // Allocate space for the output
    auto output = queries.new_full({batch_size, nhead, nquery, topk}, 0.0);


    // Note that this kind of accessor is not compatible with CUDA tensors inside kernel functions
    // accessor are efficient for efficient element-wise access https://pytorch.org/cppdocs/notes/tensor_basics.html because of the cost of dynamic dispatch
    // https://eli.thegreenplace.net/2013/12/05/the-cost-of-dynamic-virtual-calls-vs-static-crtp-dispatch-in-c
    auto qa = queries.accessor<float, 4>();
    auto ka = keys.accessor<float, 4>();
    auto oa = output.accessor<float, 4>();
    auto ama = attn_mask.accessor<int64_t , 4>();
    auto idx = 0;
    #pragma omp parallel for collapse(2)
    for (int n=0; n<batch_size; n++) {
        for (int h=0; h<nhead; h++){
            for (int l=0; l< nquery; l++){
                for (int k=0; k<topk; k++){
                    idx = ama[n][h][l][k];
                    oa[n][h][l][k] = dot(
                            &qa[n][h][l][0],
                            &ka[n][h][idx][0], //random memorry access make this algorithm slower
                            dim
                            );
                }
            }

        }

    }
    return output;
}


std::tuple<torch::Tensor, torch::Tensor> local_dot_backward(
        const torch::Tensor& queries,
        const torch::Tensor& keys,
        const torch::Tensor& attn_mask,
        const torch::Tensor& upstream_grad
        )
/* This functions takes queries and keys and output multi-head attention weights
* queries' shape: [bs, nhead, #query, dim]
* keys' shape: [bs, nhead, #query, dim]
* attn_mask: [bs, nhead, #query, topk] contains index for topk queries to attend
* upstream_grad: [bs, nhead, #query, topk] contains upstream gradient dattn
* return: gradient for query and keys*/
        {

    const int batch_size = queries.size(0);
    const int nhead = queries.size(1);
    const int nquery = queries.size(2);
    const int dim = queries.size(3);
    const int topk = attn_mask.size(3);


    // Allocate space for the output
    auto grad_queries = torch::zeros_like(queries);
    auto grad_keys = torch::zeros_like(keys);

    // Create accessors for all arguments
    auto qa = queries.accessor<float, 4>();
    auto ka = keys.accessor<float, 4>();
    auto ama = attn_mask.accessor<int64_t , 4>();
    auto gqa = grad_queries.accessor<float, 4>();
    auto gka = grad_keys.accessor<float, 4>();
    auto ga = upstream_grad.accessor<float, 4>();
    auto idx = 0;

    #pragma omp parallel for collapse(2)
    for (int n=0; n<batch_size; n++){
        for (int h=0; h<nhead; h++){
            for (int l=0; l<nquery; l++){
                for (int k=0; k<topk; k++){
                    idx = ama[n][h][l][k];
                    scaled_copy_add(
                            &ka[n][h][idx][0],
                            &gqa[n][h][l][0],
                            ga[n][h][l][k],
                            dim
                            );
                    scaled_copy_add(
                            &qa[n][h][l][0],
                            &gka[n][h][idx][0],
                            ga[n][h][l][k],
                            dim
                            );
                }
            }
        }
    }
    return std::make_tuple(grad_queries, grad_keys);

}

torch::Tensor local_weighted_average(
        const torch::Tensor& attention,
        const torch::Tensor& values,
        const torch::Tensor& attn_mask
        ){
    /* attention's weight: [bs, nhead, #query, topk]
     * attn_mask: [bs, nhead, #query, topk] contains index for topk queries to attend
     * values' shape: [bs, nhead, #query, dim]
     * return output: [bs, nhead, #query, dim]
     * */

    const int batch_size = attention.size(0);
    const int nhead = attention.size(1);
    const int nquery = attention.size(2);
    const int dim = values.size(3);

    const int topk = attn_mask.size(3);

    // TensorOptions configures the data type, device and layout and other properties of the resulting tensor.
    auto output = torch::zeros({batch_size, nhead, nquery, dim}, values.options());

    auto aa = attention.accessor<float, 4>();
    auto va = values.accessor<float, 4>();
    auto oa = output.accessor<float, 4>();
    auto ama = attn_mask.accessor<int64_t, 4>();
    auto idx = 0;

    #pragma omp parallel for collapse(2)
    for (int n=0; n<batch_size; n++){
        for (int h=0; h<nhead; h++){
            for (int l=0; l<nquery; l++){
                for (int k=0; k<topk; k++){
                    idx = ama[n][h][l][k];
                    scaled_copy_add(
                            &va[n][h][idx][0],
                            &oa[n][h][l][0],
                            aa[n][h][l][k],
                            dim
                            );

                }
            }
        }
    }
    return output;

}

std::tuple<torch::Tensor, torch::Tensor> local_weighted_average_backward(
        const torch::Tensor& attention,
        const torch::Tensor& values,
        const torch::Tensor& attn_mask,
        const torch::Tensor& upstream_grad
        )
/* attention's weight: [bs, nhead, #query, topk]
* values' shape: [bs, nhead, #query, dim]
* attn_mask: [bs, nhead, #query, topk] contains index for topk queries to attend
* upstream_grad: [bs, nhead, #query, dim] upstream gradient corresponding to dtgt in DETR
* */
        {
    const int batch_size = attention.size(0);
    const int nhead = attention.size(1);
    const int nquery = attention.size(2);
    const int dim = values.size(3); // debugged this for hour realized put attention instead of values :)))))

    const int topk = attn_mask.size(3);

    auto grad_attention = torch::zeros_like(attention);
    auto grad_values = torch::zeros_like(values);

    auto aa = attention.accessor<float, 4>();
    auto va = values.accessor<float, 4>();
    auto ga = upstream_grad.accessor<float, 4>();
    auto gaa = grad_attention.accessor<float, 4>();
    auto gva = grad_values.accessor<float, 4>();
    auto ama = attn_mask.accessor<int64_t, 4>();
    auto idx = 0;

    #pragma omp parallel for collapse(2)
    for (int n=0; n<batch_size; n++){
        for (int h=0; h<nhead; h++){
            for (int l=0; l<nquery; l++){
                for (int k=0; k<topk; k++){
                    idx = ama[n][h][l][k];
                    scaled_copy_add(
                            &ga[n][h][l][0],
                            &gva[n][h][idx][0],
                            aa[n][h][l][k],
                            dim
                    );
                    gaa[n][h][l][k] = dot(
                            &ga[n][h][l][0],
                            &va[n][h][idx][0],
                            dim
                            );

                }
            }
        }
    }
    return std::make_tuple(grad_attention, grad_values);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    m.def("local_dot_product", &local_dot_product, "Compute the product of Q and K for k nearest Keys");
    m.def("local_dot_backward", &local_dot_backward, "Backward pass");
    m.def("local_weighted_average", &local_weighted_average, "Compute attn @ V for k nearest Keys");
    m.def("local_weighted_average_backward", &local_weighted_average_backward, "COmpute gradient of local weighted average");

}