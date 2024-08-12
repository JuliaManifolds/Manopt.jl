module ManoptManifoldsExt

using ManifoldsBase: exp, log, ParallelTransport, vector_transport_to
using Manopt
using Manopt:
    _l_refl,
    _l_retr,
    _kw_retraction_method_default,
    _kw_inverse_retraction_method_default,
    _kw_X_default
import Manopt:
    max_stepsize,
    alternating_gradient_descent,
    alternating_gradient_descent!,
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

if isdefined(Base, :get_extension)
    using Manifolds
else
    using ..Manifolds
end

Rn(::Val{:Manifolds}, args...; kwargs...) = Euclidean(args...; kwargs...)

const NONMUTATINGMANIFOLDS = Union{Circle,PositiveNumbers,Euclidean{Tuple{}}}
include("manifold_functions.jl")
include("ChambollePockManifolds.jl")
include("alternating_gradient.jl")
end
