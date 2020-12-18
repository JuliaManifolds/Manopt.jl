#
#   The joint diagonalization problem on the Stiefel manifold St(n,k)
#
using Manopt, Manifolds, ManifoldsBase, LinearAlgebra, Random
import Manifolds: vector_transport_to!
struct IdentityTransport <: AbstractVectorTransportMethod end
vector_transport_to!(::Stiefel, Y, p, X, q, ::IdentityTransport) = (Y .= project(M, q, X))
Random.seed!(42)
n = 12
k = 8
m = 512
A = randn(n, n, m)

for i in 1:m
    A[:, :, i] = diagm(n:-1:1) + 0.1 * (transpose(A[:, :, i]) + A[:, :, i])
end

M = Stiefel(n, k)
F(X::Array{Float64,2}) = -sum([norm(diag(X' * A[:, :, i] * X))^2 for i in 1:m])
function ∇F(X::Array{Float64,2})
    return project(
        M, X, -4 * sum([A[:, :, i] * X * norm(diag(X' * A[:, :, i] * X)) for i in 1:m])
    )
end
x = random_point(M)
@time quasi_Newton(
    M,
    F,
    ∇F,
    x;
    memory_size=32,
    cautious_update=true,
    vector_transport_method=IdentityTransport(),
    stopping_criterion=StopWhenGradientNormLess(norm(M, x, ∇F(x)) * 10^(-6)),
    debug=[:Iteration, " ", :Cost, "\n", 1, :Stop],
)
