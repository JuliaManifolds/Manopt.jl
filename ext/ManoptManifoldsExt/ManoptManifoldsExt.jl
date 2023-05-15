module ManoptManifoldsExt

if isdefined(Base, :get_extension)
    using Manifolds
    using ManifoldsBase
    using Manopt
    import Manopt:
        max_stepsize,
        prox_TV2,
        grad_TV2,
        alternating_gradient_descent,
        alternating_gradient_descent!,
        get_gradient,
        get_gradient!
    using StaticArrays
    using LinearAlgebra:
        cholesky, det, diag, dot, Hermitian, qr, Symmetric, triu, I, Diagonal
    import ManifoldsBase: copy, mid_point, mid_point!

    using ManifoldDiff:
        adjoint_differential_shortest_geodesic_startpoint,
        adjoint_differential_shortest_geodesic_endpoint
else
    # imports need to be relative for Requires.jl-based workflows:
    # https://github.com/JuliaArrays/ArrayInterface.jl/pull/387
    using ..Manifolds
    using ..ManifoldsBase
    using ..Manopt
    import ..Manopt:
        max_stepsize,
        prox_TV2,
        grad_TV2,
        alternating_gradient_descent,
        alternating_gradient_descent!,
        get_gradient,
        get_gradient!
    using ..StaticArrays
    using ..LinearAlgebra:
        cholesky, det, diag, dot, Hermitian, qr, Symmetric, triu, I, Diagonal
    import ..ManifoldsBase: copy, mid_point, mid_point!

    using ..ManifoldDiff:
        adjoint_differential_shortest_geodesic_startpoint,
        adjoint_differential_shortest_geodesic_endpoint
end

const NONMUTATINGMANIFOLDS = Union{Circle,PositiveNumbers,Euclidean{Tuple{}}}
include("manifold_functions.jl")
include("nonmutating_manifolds_functions.jl")
include("artificialDataFunctionsManifolds.jl")
include("ChambollePockManifolds.jl")
include("alternating_gradient.jl")

end
