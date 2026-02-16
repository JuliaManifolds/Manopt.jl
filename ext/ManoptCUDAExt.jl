"""
    ManoptCUDAExt

CUDA extension for Manopt.jl, enabling solvers to work transparently with
`CuArray`-backed manifold points.

## Problem

`ArmijoLinesearchStepsize` and `NonmonotoneLinesearchStepsize` pre-allocate a
`candidate_point` workspace via `allocate_result(M, rand)` at construction time.
This always returns a CPU `Array`, causing type mismatches when the solver iterate
is a `CuArray`.

## Solution

Override `linesearch_backtrack!` to detect CPU workspace / GPU iterate mismatches
and allocate a GPU scratch buffer on the fly.

A more complete upstream fix is tracked in JuliaManifolds/Manopt.jl#577, which
extends `_produce_type` to pass the iterate `p` into stepsize constructors so that
`candidate_point` can be allocated with the correct array type from the start.

## Companion extensions

GPU-aware `allocate` is provided by `ManifoldsBaseCUDAExt` (ManifoldsBase.jl).
GPU-compatible manifold operations are in `ManifoldsCUDAExt` (Manifolds.jl).
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
