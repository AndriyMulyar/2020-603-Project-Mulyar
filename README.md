# 2020-603-Project-Mulyar
Benchmarking of Transformer Attention Kernel CUDA Implementations

### Installation
1. Clone the repository and run `make all`.

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
 
This project analyzes the theory and GPU implementation behind two such techniques: cluster attention and linear auto-regressive decoding attention.