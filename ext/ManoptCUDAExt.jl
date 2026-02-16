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

1. Override `ManifoldsBase.allocate` for `CuArray` points so that
   `linesearch_backtrack` (non-mutating) allocates GPU arrays.
2. Override `linesearch_backtrack!` (mutating) to detect CPU workspace /
   GPU point mismatches and allocate a GPU scratch buffer. This covers
   `ArmijoLinesearchStepsize` and `NonmonotoneLinesearchStepsize`.

## Companion extensions

For full GPU support, also load `ManifoldsBaseCUDAExt` (from ManifoldsBase.jl)
and `ManifoldsCUDAExt` (from Manifolds.jl). These provide GPU-aware allocation
and manifold operation overrides respectively.
"""
module ManoptCUDAExt

# === Imports (following Manopt extension conventions) ===
using Manopt
using ManifoldsBase
using ManifoldsBase: AbstractRetractionMethod, default_retraction_method,
    default_vector_transport_method
using LinearAlgebra

# Import functions we extend with new methods
import Manopt: linesearch_backtrack!

using CUDA

# === CuArray type union (following OMEinsum.jl pattern) ===

"""
    CUDAManifoldPoint{T,N}

Type union covering `CuArray` and its wrapped variants (`Transpose`, `Adjoint`)
for dispatch in GPU-aware methods.
"""
const CUDAManifoldPoint{T,N} = Union{
    CuArray{T,N},
    LinearAlgebra.Transpose{T,<:CuArray{T,N}},
    LinearAlgebra.Adjoint{T,<:CuArray{T,N}},
}

# === GPU-aware allocations ===

# Override allocate for CuArray manifold points.
# This fixes linesearch_backtrack (linesearch.jl:148) which calls allocate(M, p)
# and would otherwise create a CPU Array.

function ManifoldsBase.allocate(::ManifoldsBase.AbstractManifold, p::CuArray{T,N}) where {T,N}
    return CuArray{T,N}(undef, size(p))
end

function ManifoldsBase.allocate(
    ::ManifoldsBase.AbstractManifold, p::CuArray{T,N}, ::Type{S}
) where {T,N,S}
    return CuArray{S,N}(undef, size(p))
end

# Allocate without manifold argument (used by some ManifoldsBase paths)
function ManifoldsBase.allocate(p::CuArray{T,N}) where {T,N}
    return CuArray{T,N}(undef, size(p))
end

function ManifoldsBase.allocate(p::CuArray{T,N}, ::Type{S}) where {T,N,S}
    return CuArray{S,N}(undef, size(p))
end

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
