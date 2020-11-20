#include <limits>
#include <functional>

#include <torch/extension.h>


typedef torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> float4_accessor;
typedef torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> float3_accessor;
typedef torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> float2_accessor;
typedef torch::PackedTensorAccessor32<long, 1, torch::RestrictPtrTraits> long_accessor;

template <int a_blocks=64, typename CopyImplementation>
void window_attention(
    CopyImplementation copy_implementation,
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor out,
    int local_context
) {
    int N = A.size(0);
    int L = A.size(1);

    // Save the intermediate results in here
    auto buffer = torch::zeros(
        {N, a_blocks, a_blocks+local_context},
        A.options()
    );

    for (int l=0; l<L; l+=a_blocks) {
        // Compute the sizes of the sub problems to be computed in this
        // block iteration
        int s_start = std::max(0, l-local_context/2);
        int s_end = std::min(L, l-local_context/2+local_context+a_blocks);
        int n_b = s_end-s_start;
        int n_a = std::min(L-l, a_blocks);

        // Compute the dot products
        auto buff = buffer.narrow(1, 0, n_a).narrow(2, 0, n_b);
        at::matmul_out(
            buff,
            A.narrow(1, l, n_a),
            B.narrow(1, s_start, n_b).transpose(1, 2)
        );

        // Select the correct results from the buffer
        const int threads = 1024;
        int blocks = ceildiv(buff.numel(), threads);
        sliding_dot_copy_kernel<<<blocks, threads>>>(
            copy_implementation,
            buff.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            out.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            local_context,
            l,
            s_start,
            buff.size(1)*buff.size(2),
            buff.size(2)
        );
    }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "window_attention_forward",
        &window_attention_forward,
        "Window attention forward pass"
    );
}