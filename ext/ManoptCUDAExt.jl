"""
    ManoptCUDAExt

CUDA extension for Manopt.jl, enabling solvers to work transparently with
`CuArray`-backed manifold points.

"""
module ManoptCUDAExt

using Manopt
using ManifoldsBase

import Manopt: linesearch_backtrack!

using CUDA

# ArmijoLinesearchStepsize and NonmonotoneLinesearchStepsize call
# linesearch_backtrack!(M, candidate_point, f, p, ...) where candidate_point
# is a CPU Array and p is a CuArray. The workspace q is only used as scratch
# space, so we allocate a GPU buffer and forward the call.

function Manopt.linesearch_backtrack!(
    M::ManifoldsBase.AbstractManifold,
    q::Array,       # CPU workspace
    f::TF,
    p::CuArray,     # GPU iterate
    s,
    decrease,
    contract,
    η;
    kwargs...,
) where {TF}
    q_gpu = ManifoldsBase.allocate(M, p)
    return Manopt.linesearch_backtrack!(
        M, q_gpu, f, p, s, decrease, contract, η; kwargs...
    )
end

end # module
