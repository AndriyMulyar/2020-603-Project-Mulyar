//
// A faster sliding window attention variant.
// Andriy Mulyar <contact@andriymulyar.com>
//
// Adapted from the skeleton of the sliding window implementation of:
// Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
// Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
//

#include <limits>
#include <functional>

#include <torch/extension.h>


typedef torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> float4_accessor;
typedef torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> float3_accessor;
typedef torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> float2_accessor;
typedef torch::PackedTensorAccessor32<long, 1, torch::RestrictPtrTraits> long_accessor;


inline int ceildiv(int a, int b) {
    return (a + b - 1)/b;
}


/**
 * This copy implementation simply copies the appropriate values from the
 * buffer and adds the corresponding value from the attention mask.
 */
struct masked_lp_copy
{
    float2_accessor attn_mask;
    long_accessor key_lengths;
    int n_heads;

    masked_lp_copy(float2_accessor _attn_mask, long_accessor _key_lengths,
                   int _n_heads) :
        attn_mask(_attn_mask), key_lengths(_key_lengths), n_heads(_n_heads)
        {}

    __device__ void operator()(
        float3_accessor buffer,
        float3_accessor output,
        int local_context,
        int l_start,
        int s_start,
        int buffer_dim12,
        int buffer_dim2
    ) {
        int idx = blockIdx.x*blockDim.x + threadIdx.x;
        int n = idx / buffer_dim12; //the nth sequence in the batch, divide by the ~ num_queries*block_a (slightly larger due to context)
        idx = idx - n*buffer_dim12; //set idx to the n'th sequence
        int l_offset = idx / buffer_dim2; //the l'th query in the n'th sequence
        idx = idx - l_offset*buffer_dim2; //set idx to the l'th query in the sequence
        int s_offset = idx; //

        if (n >= buffer.size(0)) {
            return;
        }

        int l = l_start + l_offset;
        int s = s_start + s_offset;
        int k = s - l + local_context/2;

        if (k < 0 || k >= local_context) {
            return;
        }

        if (s >= key_lengths[n / n_heads]) {
            return;
        }

        output[n][l][k] = buffer[n][l_offset][s_offset] + attn_mask[l][s];
    }

    static masked_lp_copy factory(
        torch::Tensor attn_mask,
        torch::Tensor key_lengths,
        int n_heads
    ) {
        return masked_lp_copy(
            attn_mask.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            key_lengths.packed_accessor32<long, 1, torch::RestrictPtrTraits>(),
            n_heads
        );
    }
};


/**
 * The simplest copy implementation just copies the values if they are within
 * bounds.
 */
struct lp_copy
{
    __device__ void operator()(
        float3_accessor buffer,
        float3_accessor output,
        int local_context,
        int l_start,
        int s_start,
        int buffer_dim12,
        int buffer_dim2
    ) {
        int idx = blockIdx.x*blockDim.x + threadIdx.x;
        int n = idx / buffer_dim12;
        idx = idx - n*buffer_dim12;
        int l_offset = idx / buffer_dim2;
        idx = idx - l_offset*buffer_dim2;
        int s_offset = idx;

        if (n >= buffer.size(0)) {
            return;
        }

        int l = l_start + l_offset;
        int s = s_start + s_offset;
        int k = s - l + local_context/2;

        if (k < 0 || k >= local_context) {
            return;
        }

        output[n][l][k] = buffer[n][l_offset][s_offset];
    }
};


/**
 * A kernel that just delegates the copying to the passed in copy
 * implementation.
 */
template <typename copy_implementation>
__global__ void sliding_dot_copy_kernel(
    copy_implementation copy,
    float3_accessor buffer,
    float3_accessor output,
    int local_context,
    int l_start,
    int s_start,
    int buffer_dim12,
    int buffer_dim2
) {
    copy(
        buffer,
        output,
        local_context,
        l_start,
        s_start,
        buffer_dim12,
        buffer_dim2
    );
}

/**
* A kernel that dots the queries with a local context window keys. Each thread performs a dot product.
* Computes a single element of out (N, L, local_context)
*
* Assumes N*H*L number of thread blocks each containing context_window threads.
* Arguments
* ---------
*     copy_implementation: The kernel implementation that selects the results.
*     XQ: Tensor of shape (N*H, L, E)
*     XV: Tensor of shape (N*H, L, E)
*     out: Tensor of shape (N*H, L, local_context)
*/
__global__ void local_dot_product_kernel(float3_accessor XQ, float3_accessor XK,
                                         float3_accessor out, int local_context, int L, int E){
    int batch_index = blockIdx.x; //the key,context_window pair we are computing the dot products for
    int query = blockIdx.x / L; // the query this thread is dotting with

    // The value this threads query is dotted with.
    // Each thread in block handles one element of the window.
    int value = query - local_context/2 + threadIdx.x;
    //ignore context window elements that cross the boundaries of the input X
    if(value < 0 || value >= L){
        return;
    }
    for(int i = 0; i < E; i++){
        out[batch_index][query][threadIdx.x] += XQ[batch_index][query][i]*XK[batch_index][value][i];
    }
}

/**
 * Multiply every A_i with every B_j iff |i-j| < local_context/2.
 *
 * This operation is delegated to a custom kernel that takes as input XQ and XK
 * matrices and perform a dot product in a local context window.
 *
 * Arguments
 * ---------
 *     copy_implementation: The kernel implementation that selects the results.
 *     A: Tensor of shape (N, L, E)
 *     B: Tensor of shape (N, L, E)
 *     out: Tensor of shape (N, L, local_context)
 */
template <int a_blocks=64, typename CopyImplementation>
void sliding_dot(
    CopyImplementation copy_implementation,
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor out,
    int local_context
) {
    int N = A.size(0); // batch size (not the number of instances, but number of instances times number of attention heads).
    int L = A.size(1); // number of queries and keys

    dim3 gridDim(N * L);// a thread block for each query, in each instance
    dim3 blockDim(local_context); //a thread for each value in the context window of the query.

    local_dot_product_kernel<<<gridDim, blockDim>>>(
        A.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        B.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        out.packed_accessor32<float, 3, torch::RestrictPtrTraits>(), // dim are (N, L, context size)
        local_context,
        L,
        A.size(2)
    );

    }



template <
    int sequence_blocksize=32,
    int feature_blocksize=32,
    int k_blocksize=32
>
struct ForwardValueIndexing
{
    int L;
    int E;
    int local_context;

    ForwardValueIndexing(int _L, int _E, int _local_context) :
        L(_L), E(_E), local_context(_local_context)
        {}

    template <typename accessor>
    inline __device__
    void load_factors(
        accessor factors,
        accessor values,
        float * s_factors,
        int sblock,
        int eblock,
        int s_local,
        int e_local,
        int k_start
    ) {
        int k = k_start + e_local;
        int l = (
            sblock * sequence_blocksize + s_local - (local_context-1)/2 + k
        );

        if (l >= 0 && l < L && k < local_context) {
            s_factors[s_local * k_blocksize + e_local] = factors[l][k];
        } else {
            s_factors[s_local * k_blocksize + e_local] = 0;
        }
    }

    template <typename accessor>
    inline __device__
    void load_values(
        accessor factors,
        accessor values,
        float * s_values,
        int sblock,
        int eblock,
        int s_local,
        int e_local,
        int k_start
    ) {
        int l = (
            sblock * sequence_blocksize + s_local - (local_context-1)/2 + k_start
        );
        int e = eblock * feature_blocksize + e_local;

        if (e < E && l >= 0 && l < L) {
            s_values[s_local*feature_blocksize + e_local] = values[l][e];
        } else {
            s_values[s_local*feature_blocksize + e_local] = 0;
        }

        l += sequence_blocksize;
        if (e < E && l >= 0 && l < L) {
            s_values[(s_local + feature_blocksize)*feature_blocksize + e_local] = values[l][e];
        } else {
            s_values[(s_local + feature_blocksize)*feature_blocksize + e_local] = 0;
        }
    }
};


template <
    int sequence_blocksize=32,
    int feature_blocksize=32,
    int k_blocksize=32
>
struct ReverseValueIndexing : ForwardValueIndexing<sequence_blocksize,
                                                   feature_blocksize,
                                                   k_blocksize>
{
    typedef ForwardValueIndexing<sequence_blocksize, feature_blocksize, k_blocksize> parent;

    ReverseValueIndexing(int _L, int _E, int _local_context) :
        parent(_L, _E, _local_context)
        {}

    template <typename accessor>
    inline __device__
    void load_factors(
        accessor factors,
        accessor values,
        float * s_factors,
        int sblock,
        int eblock,
        int s_local,
        int e_local,
        int k_start
    ) {
        int k = k_start + e_local;
        int l = (
            sblock * sequence_blocksize + s_local - (parent::local_context-1)/2 + k
        );

        if (l >= 0 && l < parent::L && k < parent::local_context) {
            s_factors[s_local * k_blocksize + e_local] = factors[l][parent::local_context-k-1];
        } else {
            s_factors[s_local * k_blocksize + e_local] = 0;
        }
    }
};


template <
    int sequence_blocksize=32,
    int feature_blocksize=32,
    int k_blocksize=32
>
struct ForwardFactorIndexing
{
    int L;
    int E;
    int local_context;

    ForwardFactorIndexing(int _L, int _E, int _local_context) :
        L(_L), E(_E), local_context(_local_context)
        {}

    template <typename accessor>
    inline __device__
    void load_factors(
        accessor factors,
        accessor values,
        float * s_factors,
        int sblock,
        int eblock,
        int s_local,
        int e_local,
        int k_start
    ) {
        int l = sblock*sequence_blocksize + s_local;
        int k = k_start + e_local;
        int s = l - local_context/2 + k;

        if (l < L && k < local_context && s >= 0 && s < L) {
            s_factors[s_local*k_blocksize + e_local] = factors[l][k];
        } else {
            s_factors[s_local*k_blocksize + e_local] = 0;
        }
    }

    template <typename accessor>
    inline __device__
    void load_values(
        accessor factors,
        accessor values,
        float * s_values,
        int sblock,
        int eblock,
        int s_local,
        int e_local,
        int k_start
    ) {
        int l = sblock*sequence_blocksize + s_local;
        int e = eblock * feature_blocksize + e_local;
        int s = l - local_context/2 + k_start;

        if (e < E && s >= 0 && s < L) {
            s_values[s_local*feature_blocksize + e_local] = values[s][e];
        } else {
            s_values[s_local*feature_blocksize + e_local] = 0;
        }

        s += sequence_blocksize;
        if (e < E && s >= 0 && s < L) {
            s_values[(s_local + feature_blocksize)*feature_blocksize + e_local] = values[s][e];
        } else {
            s_values[(s_local + feature_blocksize)*feature_blocksize + e_local] = 0;
        }
    }
};


/**
 * This kernel performs the dot product of factors with sliding chunks of
 * values and saves the result in output.
 *
 * The shapes of the arguments are NHLK and NHLE.
 */
template <
    typename IndexPolicy,
    int sequence_blocksize=32,
    int feature_blocksize=32,
    int k_blocksize=32
>
__global__ void sliding_weighted_average(
    IndexPolicy indexing,
    float4_accessor factors,
    float4_accessor values,
    float4_accessor output,
    dim3 strides
) {
    int idx = blockIdx.x;
    int n = idx / strides.x;
    idx -= n*strides.x;
    int h = idx / strides.y;
    idx -= h*strides.y;
    int sblock = idx / strides.z;
    idx -= sblock*strides.z;
    int eblock = idx;

    int local_context = factors.size(3);

    int s_local = threadIdx.x / feature_blocksize;
    int e_local = threadIdx.x - s_local*feature_blocksize;
    int s = sblock * sequence_blocksize + s_local;
    int e = eblock * feature_blocksize + e_local;

    if (n > factors.size(0)) {
        return;
    }

    // Declare the shared memory to load the values and factors.
    extern __shared__ float shared_mem[];
    float * s_factors = shared_mem;
    float * s_values = s_factors + sequence_blocksize * k_blocksize;

    // Main dot product loop
    for (int k=0; k<local_context; k+=k_blocksize) {
        // Load the factors in shared memory
        indexing.load_factors(
            factors[n][h],
            values[n][h],
            s_factors,
            sblock,
            eblock,
            s_local,
            e_local,
            k
        );

        // Load the values in shared memory
        indexing.load_values(
            factors[n][h],
            values[n][h],
            s_values,
            sblock,
            eblock,
            s_local,
            e_local,
            k
        );

        __syncthreads();

        // Compute the dot product
        float result = 0;
        #pragma unroll
        for (int k_local=0; k_local<k_blocksize; k_local++) {
            result += (
                s_factors[s_local*k_blocksize + k_local] *
                s_values[(s_local + k_local)*feature_blocksize + e_local]
            );
        }
        if (s < output.size(2) && e < output.size(3)) {
            output[n][h][s][e] += result;
        }
        __syncthreads();
    }
}


torch::Tensor local_dot_product(
    const torch::Tensor queries,
    const torch::Tensor keys,
    const torch::Tensor attn_mask,
    const torch::Tensor key_lengths,
    int local_context
) {
    // Extract some shapes
    int N = queries.size(0);
    int H = queries.size(1);
    int L = queries.size(2);
    int S = keys.size(2);
    int E = queries.size(3);

    // Allocate space for the output
    auto output = queries.new_full({N, H, L, local_context}, -1e24);

    sliding_dot(
        masked_lp_copy::factory(attn_mask, key_lengths, H),
        queries.reshape({N*H, L, E}),
        keys.reshape({N*H, L, E}),
        output.view({N*H, L, local_context}),
        local_context
    );

    return output;
}


std::tuple<torch::Tensor, torch::Tensor> local_dot_backward(
    const torch::Tensor queries,
    const torch::Tensor keys,
    const torch::Tensor key_lengths,
    const torch::Tensor grad,
    int local_context
) {
    // Extract some shapes
    int N = grad.size(0);
    int H = grad.size(1);
    int L = grad.size(2);
    int K = grad.size(3);
    int E = keys.size(3);

    // Allocate space for the output
    auto grad_queries = torch::zeros_like(queries);
    auto grad_keys = torch::zeros_like(keys);

    const int threads = 32*32;
    int lblocks = ceildiv(L, 32);
    int eblocks = ceildiv(E, 32);
    int blocks = N * H * lblocks * eblocks;
    int shared_mem = 32*32 * 3 * sizeof(float);
    dim3 strides(
        H*lblocks*eblocks,
        lblocks*eblocks,
        eblocks
    );

    sliding_weighted_average<<<blocks, threads, shared_mem>>>(
        ForwardFactorIndexing<32, 32, 32>(L, E, K),
        grad.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        keys.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        grad_queries.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        strides
    );
    sliding_weighted_average<<<blocks, threads, shared_mem>>>(
        ForwardValueIndexing<32, 32, 32>(L, E, K),
        grad.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        queries.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        grad_keys.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        strides
    );

    return std::make_tuple(grad_queries, grad_keys);
}


torch::Tensor local_weighted_average(
    const torch::Tensor attention,
    const torch::Tensor values
) {
    // Extract some shapes
    int N = attention.size(0);
    int H = attention.size(1);
    int L = attention.size(2);
    int K = attention.size(3);
    int E = values.size(3);

    // Allocate space for the output
    auto output = torch::zeros_like(values);

    const int threads = 32*32;
    int lblocks = ceildiv(L, 32);
    int eblocks = ceildiv(E, 32);
    int blocks = N * H * lblocks * eblocks;
    int shared_mem = 32*32 * 3 * sizeof(float);
    dim3 strides(
        H*lblocks*eblocks,
        lblocks*eblocks,
        eblocks
    );

    sliding_weighted_average<<<blocks, threads, shared_mem>>>(
        ForwardFactorIndexing<32, 32, 32>(L, E, K),
        attention.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        values.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        output.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        strides
    );

    return output;
}


std::tuple<torch::Tensor, torch::Tensor> local_weighted_average_backward(
    const torch::Tensor attention,
    const torch::Tensor values,
    const torch::Tensor grad
) {
    // Extract some shapes
    int N = attention.size(0);
    int H = attention.size(1);
    int L = attention.size(2);
    int local_context = attention.size(3);
    int S = values.size(2);
    int E = values.size(3);

    // Allocate space for the output
    auto grad_attention = torch::zeros_like(attention);
    auto grad_values = torch::zeros_like(values);

    // Compute the gradient wrt to the values
    {
        const int threads = 32*32;
        int lblocks = ceildiv(L, 32);
        int eblocks = ceildiv(E, 32);
        int blocks = N * H * lblocks * eblocks;
        int shared_mem = 32*32 * 3 * sizeof(float);
        dim3 strides(
            H*lblocks*eblocks,
            lblocks*eblocks,
            eblocks
        );

        sliding_weighted_average<<<blocks, threads, shared_mem>>>(
            ReverseValueIndexing<32, 32, 32>(L, E, local_context),
            attention.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            grad.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            grad_values.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            strides
        );
    }

    // Compute the gradient wrt to the attention
    sliding_dot(
        lp_copy(),
        grad.reshape({N*H, L, E}),
        values.reshape({N*H, L, E}),
        grad_attention.view({N*H, L, local_context}),
        local_context
    );

    return std::make_tuple(grad_attention, grad_values);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "local_dot_product",
        &local_dot_product,
        "Compute the product of Q and K for a small context around each Q"
    );
    m.def(
        "local_dot_backward",
        &local_dot_backward,
        "Compute the gradient of local_dot_product"
    );
    m.def(
        "local_weighted_average",
        &local_weighted_average,
        "Perform the weighted average of V for a small context around each Q"
    );
    m.def(
        "local_weighted_average_backward",
        &local_weighted_average_backward,
        "Compute the gradient of the local weighted average"
    );
}