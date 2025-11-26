module ManoptManifoldsExt

using ManifoldsBase: exp, log, ParallelTransport, vector_transport_to
using Manopt
using Manopt: _math, _tex, _var, ManifoldDefaultsFactory, _produce_type
import Manopt:
    max_stepsize,
    get_gradient,
    get_gradient!,
    set_parameter!,
    reflect,
    reflect!,
    Rn,
    Rn_default
using LinearAlgebra: cholesky, det, diag, dot, Hermitian, qr, Symmetric, triu, I, Diagonal
import ManifoldsBase: copy, mid_point, mid_point!

using ManifoldDiff:
    adjoint_differential_shortest_geodesic_startpoint,
    adjoint_differential_shortest_geodesic_endpoint

using Manifolds

Rn(::Val{:Manifolds}, args...; kwargs...) = Euclidean(args...; kwargs...)

const NONMUTATINGMANIFOLDS = Union{Circle, PositiveNumbers, Euclidean{Tuple{}}}
include("manifold_functions.jl")
include("ChambollePockManifolds.jl")
include("test_examples.jl")
end
