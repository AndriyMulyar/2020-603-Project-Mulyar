from torch.utils.cpp_extension import load
import torch
local_product_cuda = load(
    'local_product_cuda', ['local_product_cuda_no_mm.cu'], verbose=True)

help(local_product_cuda)