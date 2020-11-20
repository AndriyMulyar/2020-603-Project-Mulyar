from torch.utils.cpp_extension import load
local_product_cuda = load(
    'local_product_cuda', ['local_product_cuda.cu'], verbose=True)
help(local_product_cuda)