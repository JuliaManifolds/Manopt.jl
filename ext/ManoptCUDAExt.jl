"""
    ManoptCUDAExt

CUDA extension for Manopt.jl, enabling solvers to work transparently with
`CuArray`-backed manifold points.

## Problem

Several linesearch stepsizes pre-allocate workspace arrays (e.g. `candidate_point`)
via `allocate_result(M, rand)` at construction time, which always returns CPU `Array`s.
When the optimization iterates are `CuArray`s, the type mismatch between CPU workspace
and GPU iterates causes `retract_fused!` and broadcasting to fail.

Since these workspace arrays are stored in parametric `mutable struct` fields
(e.g. `candidate_point::P` where `P = Vector{Float64}`), their types cannot
be changed after construction.

## Solution

Override `linesearch_backtrack!` (mutating) to detect CPU workspace / GPU point
mismatches and allocate a GPU scratch buffer. This covers
`ArmijoLinesearchStepsize` and `NonmonotoneLinesearchStepsize`.

GPU-aware `allocate` methods are provided by `ManifoldsBaseCUDAExt`
(from ManifoldsBase.jl). For GPU-compatible manifold operations, also load
`ManifoldsCUDAExt` (from Manifolds.jl).
"""
module ManoptCUDAExt

using Manopt
using ManifoldsBase

# Import functions we extend with new methods
import Manopt: linesearch_backtrack!

using CUDA

# === Fix linesearch_backtrack! for CPU workspace / GPU point mismatch ===

# ArmijoLinesearchStepsize and NonmonotoneLinesearchStepsize both call
# linesearch_backtrack!(M, candidate_point, f, p, ...) where candidate_point
# is a CPU Array (pre-allocated at stepsize construction) and p is a CuArray
# (the current iterate). The function only uses q as scratch space and returns
# the step size, so we can safely allocate a new GPU buffer for q.

function Manopt.linesearch_backtrack!(
    M::ManifoldsBase.AbstractManifold,
    q::Array,       # CPU workspace (from stepsize.candidate_point)
    f::TF,
    p::CuArray,     # GPU iterate
    s,
    decrease,
    contract,
    η;
    kwargs...,
) where {TF}
    q_gpu = ManifoldsBase.allocate(M, p)
    return Manopt.linesearch_backtrack!(M, q_gpu, f, p, s, decrease, contract, η; kwargs...)
end

# Note: WolfePowellLinesearchStepsize and CubicBracketingLinesearchStepsize
# use candidate_point directly (not via linesearch_backtrack!). GPU support
# for these stepsizes requires changes to the Manopt.jl base package to make
# the candidate_point field non-parametric or to add a dispatch hook. For now,
# use ArmijoLinesearch (the default for gradient_descent and
# conjugate_gradient_descent) or ConstantLength with CuArray points.

end # module
