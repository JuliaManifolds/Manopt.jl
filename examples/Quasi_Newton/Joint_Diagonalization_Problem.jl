#
#   The joint diagonalization problem on the Stiefel manifold St(n,k)
#
using Manopt, Manifolds, ManifoldsBase, LinearAlgebra, Random
import Manifolds: vector_transport_to!
struct IdentityTransport <: AbstractVectorTransportMethod end
vector_transport_to!(::Stiefel,Y,p,X,q,::IdentityTransport) = (Y .= X)
Random.seed!(42)
# Parameters
n = 5
k = 3
m = 50
A = randn(n,n,m)

for i = 1:m
    A[:,:,i] = 0.5*(transpose(A[:,:,i]) + A[:,:,i])
end

M = Stiefel(n,k)


F(X::Array{Float64,2}) = -sum([ norm(diag(transpose(X) * A[:, :, i] * X))^2 for i ∈ 1:m])

function ∇F(X::Array{Float64,2})
    g = zero_tangent_vector(M,X)
    for i = 1 : m
        g = g - 4 * A[:, :, i] * X * norm(diag(transpose(X) * A[:, :, i] * X))
    end
    project(M,X,g)
end

x = random_point(M)

quasi_Newton(M,F,∇F,x; vector_transport_method = IdentityTransport(), cautious_update = true, step_size = StrongWolfePowellLineseach(ExponentialRetraction(), IdentityTransport()))
