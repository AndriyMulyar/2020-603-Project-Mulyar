# 2020-603-Project-Mulyar
Benchmarking of Transformer Attention Kernel CUDA Implementations

A transformer is a deep neural network adept for capturing correlations between sets of object representations.
Since 2017, they have achieved state ot the art performance on tasks such as language modeling and generation, image
representation learning and generation, speech recognition and synthesis and music generation.

Despite these impressive results, transformers are resource intensive to train and serve making practical application
in low latency environments difficult. The main performance bottleneck resides in the quadratic *multi-head attention* operation that
transformers execute on the input set of object representations.