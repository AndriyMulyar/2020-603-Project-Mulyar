# 2020-603-Project-Mulyar
Benchmarking of Transformer Attention Kernel CUDA Implementations

### Organization.

This repository is organized as follows.
- [My benchmarkable implementation of sliding window attention.](https://github.com/AndriyMulyar/fast-transformers/blob/master/fast_transformers/local_product/local_product_cuda.cu#L168)
    The kernels at line 168
- [Debugging version of custom attention CUDA kernel](sliding_window_attention/local_product)
### Installation
Insure the machine you are on has the CUDA toolkit installed (and aliased with nvcc to the binary)
The build of this package relies on the presence of python3-dev headers. You can find these in your package manager if they are not present.

**Maple does not have the required C++ headers installed. I had to run these experiments on another machine with GPUs**.

1. Clone the repository and run `make all`.
    - This will compile all of the relevant C++ and CUDA extensions and link them for use via python calls.



### Benchmarking
Once installed run `python benchmarks/attention_benchmarks` to replicate the results to generate the table the write-up.

### Background
A transformer is a deep neural network adept for capturing correlations between sets of object representations.
Since 2017, they have achieved state ot the art performance on tasks such as language modeling and generation, image
representation learning and generation, speech recognition and synthesis and music generation.

Despite these impressive results, transformers are resource intensive to train and serve making practical application
in low latency environments difficult. The main performance bottleneck resides in the quadratic *self-attention* operation that
transformers execute on the input set of object representations.

Recent work () has attempted to mitigate this bottleneck by proposing several classes of techniques for relaxing
the computational requirements for achieving the performance gains of transformers. These mainly revolve around
customized self-attention operations that have sub-quadratic time and memory complexity with respect to the input
 representation set cardinality.
 
This project analyzes the theory and GPU implementation behind two such techniques: cluster attention and sliding window attention.


### Implementation Reference Resources

- [Pytorch's C++/CUDA extension documentation](https://pytorch.org/tutorials/advanced/cpp_extension.html?highlight=backward)