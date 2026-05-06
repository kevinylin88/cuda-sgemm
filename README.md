# cuda-sgemm
CUDA SGEMM optimization in C++: naive → coalesced → shared memory tiling → register tiling → vectorized float4, reaching 80%+ cuBLAS performance. Integrated as PyTorch custom op.
