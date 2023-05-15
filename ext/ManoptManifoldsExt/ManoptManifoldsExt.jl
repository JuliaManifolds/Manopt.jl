module ManoptManifoldsExt

if isdefined(Base, :get_extension)
    using Manifolds
    using ManifoldsBase
    using LinearAlgebra: cholesky, det, diag, dot, Hermitian, qr, Symmetric, triu
    import ManifoldsBase: copy
else
    # imports need to be relative for Requires.jl-based workflows:
    # https://github.com/JuliaArrays/ArrayInterface.jl/pull/387
    using ..Manifolds
    using ..ManifoldsBase
    using ..LinearAlgebra: cholesky, det, diag, dot, Hermitian, qr, Symmetric, triu
end

const NONMUTATINGMANIFOLDS = Union{Circle,PositiveNumbers,Euclidean{Tuple{}}}
include("manifold_functions.jl")
include("nonmutating_manifolds_functions.jl")

end
