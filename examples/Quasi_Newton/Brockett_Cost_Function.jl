#
#   Brockett Cost Function on Stiefel(n,k)
#
using Manopt, Manifolds, ManifoldsBase, LinearAlgebra, Random
import Manifolds: vector_transport_to!
struct IdentityTransport <: AbstractVectorTransportMethod end
vector_transport_to!(::Stiefel,Y,p,X,q,::IdentityTransport) = (Y .= project(M, q, X))
Random.seed!(42)
n = 32
k = 32
M = Stiefel(n,k)
A = rand(n,n)
A = (A + A')/2
N = diagm(k:-1:1)
F(X::Array{Float64,2}) = tr(X' * A * X * N)
∇F(X::Array{Float64,2}) = project(M, X, 2 * A * X * N)
x = random_point(M)
quasi_Newton(M,F,∇F,x; vector_transport_method = IdentityTransport(), debug = [:Iteration, " ", :Cost, "\n", 1, :Stop])
