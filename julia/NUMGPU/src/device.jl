# Moving the global model to a device: `adapt(CuArray, g)` (the biology constants move with
# `adapt(CuArray, kernel_setup(s))`, see kernel.jl). Box coordinates and scalar parameters stay
# on the host; numeric arrays move. Sparse transport matrices go through `sparse_to_device`, which a GPU package extension
# specializes (e.g. SparseMatrixCSC -> CuSparseMatrixCSR); by default they stay as they are.

using Adapt

"""
    sparse_to_device(to, A)

Hook for GPU extensions to convert the monthly transport matrices. The default keeps `A`.
"""
sparse_to_device(to, A) = A

function Adapt.adapt_structure(to, g::GlobalModel)
    a(x) = adapt(to, x)
    return GlobalModel(g.p, g.nb, g.ixBox, g.iyBox, g.izBox, g.nx, g.ny, g.nz, g.dznom,
        a(g.U0), a(g.L0), a(g.Tmat),
        [sparse_to_device(to, A) for A in g.Aexp], [sparse_to_device(to, A) for A in g.Aimp],
        g.ixSink, a(g.above), a(g.keep), a(g.gain), a(g.ixBottom), a(g.dzBottom), a(g.BCvalue), a(g.dvBox))
end
