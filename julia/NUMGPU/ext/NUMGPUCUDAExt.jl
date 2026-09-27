# Loaded when CUDA.jl is present. Use `adapt(CuArray, s)` / `adapt(CuArray, g)` (not `cu`,
# which converts to Float32; the model needs Float64 for eps = 1e-200).
# Verified on A100 and H100 GPUs against the CPU path (scripts/gpu_check.jl).
module NUMGPUCUDAExt

using NUMGPU, CUDA, CUDA.CUSPARSE, SparseArrays

NUMGPU.sparse_to_device(::Type{<:CuArray}, A::SparseMatrixCSC) = CuSparseMatrixCSR(A)

end
